# -*- coding: utf-8 -*-
"""同じ音声を何度も起こしたとき、声のまとまりが動かないかを照合する。

README の「本文は揺れますが、声のまとまりは動きません」を測り直すための道具。
CI では回らない（測定データがリポジトリの外にあるため）。`report_verbatim.py`
などと同じ扱い。

    python tools\\verify_clusters.py <a.speakers.json> <b.speakers.json> ...
    python tools\\verify_clusters.py *.speakers.json --turns <.work_…\\diarize> ...

## 何を見るか

**照合 A — 話者分離の生データ（`--turns`）。**
`.work_<音声名>\\diarize\\turns.*.json` を 1 対 1 で突き合わせる。
話者分離は音声だけを見るので、本来これは実行ごとに動かないはず。
**ただしキャッシュを消すと生データも消える**ので、残っている回しか比べられない。

**照合 B — 転写区間に付いたまとまり。**
区間の数は実行ごとに違う（実測 779〜821）ので、**区間どうしを対応づけない。**
対応づけの規則を持ち込むと、その規則しだいで答えが変わってしまう。
かわりに **0.1 秒ごとに「その瞬間のまとまりは何か」を引いて、
時刻 → まとまり の対応表を作って比べる。**

## 2 つの一致率を必ず両方出す

| | 何に答えるか |
| --- | --- |
| **素の一致**（名前まで同じか） | **再実行しても割当をやり直さずに済むか** |
| **並べ替えを許した一致** | **まとまりの切り方が同じか** |

**素が一致せず並べ替えで一致するなら、切り方は同じだが名前が入れ替わっている**
——つまり割当はやり直しになる。片方だけ出すと、この区別が消える。

## 不一致は 3 つに分ける

  片方に区間が無い … 転写の区間が動いただけ。まとまりの付け方の話ではない
  どちらかが `?`  … まとまりを決められなかった時間
  **別のまとまり**   … 同じ瞬間に別の人が付いた。**これが本題**

実測（2026-08-31・7 回）では、素の一致 97.04% のうち内訳は
「片方に区間が無い」2.36% / `?` 0.43% / **別のまとまり 0.17%** だった。
**分けずに 97% だけ見ると、結論を間違える。**
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

STEP = 0.1          # 時間軸を刻む幅（秒）
NONE = "―"          # 区間の無い時刻
UNK = "?"           # まとまりを決められなかった


def load_segments(path: Path) -> list[dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return d["segments"] if isinstance(d, dict) else d


def timeline(segs: list[dict], n: int) -> list[str]:
    """時刻 → まとまり の対応表。区間を介さないので、数が違っても比べられる。"""
    arr = [NONE] * n
    for s in segs:
        cl = (s.get("cluster") or "").split(":")[-1] or UNK
        for i in range(max(0, int(s["start"] / STEP)), min(n, int(s["end"] / STEP))):
            arr[i] = cl
    return arr


def perm_best(a: list[str], b: list[str], n: int):
    """b の名前を付け替えて a に最も近づけたときの一致率と、その対応。

    4 万点を総当たりすると遅いので、**先に集計表を作ってから**並べ替えを試す。
    答えは同じで、桁が違う。
    """
    ct = Counter(zip(a, b))
    la = sorted({x for x, _ in ct} - {NONE})
    lb = sorted({y for _, y in ct} - {NONE})
    if not lb or len(la) > 8 or len(lb) > 8 or len(la) < len(lb):
        return None, None
    best, bmap = -1, None
    for pm in permutations(la, len(lb)):
        m = dict(zip(lb, pm))
        m[NONE] = NONE
        v = sum(c for (x, y), c in ct.items() if x == m.get(y))
        if v > best:
            best, bmap = v, m
    return best / n * 100, bmap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path,
                    help="比べる <音声名>.speakers.json（2 つ以上）")
    ap.add_argument("--turns", nargs="*", type=Path, default=[],
                    help="話者分離の生データがある diarize フォルダ")
    a = ap.parse_args()

    if len(a.files) < 2:
        print("2 つ以上のファイルを渡してください。")
        return 1
    for f in list(a.files) + list(a.turns):
        if not f.exists():
            print(f"見つかりません: {f}")
            return 1

    # ---- 照合 A ----------------------------------------------------
    if a.turns:
        print("=" * 72)
        print("照合 A — 話者分離の生データ")
        print("=" * 72)
        tv = []
        for d in a.turns:
            fs = sorted(Path(d).glob("turns.*.json"))
            if not fs:
                print(f"  turns.*.json がありません: {d}")
                continue
            j = json.loads(fs[0].read_text(encoding="utf-8"))
            tv.append((d, j))
            print(f"  {d.parent.parent.name[:34]:34s} {len(j['turns']):4d} turns "
                  f"/ num_speakers={j['num_speakers']}")
        for d, j in tv[1:]:
            same = j["turns"] == tv[0][1]["turns"]
            tm = ([(x["start"], x["end"]) for x in j["turns"]]
                  == [(x["start"], x["end"]) for x in tv[0][1]["turns"]])
            sp = ([x["speaker"] for x in j["turns"]]
                  == [x["speaker"] for x in tv[0][1]["turns"]])
            print(f"    1 本目と比べて  丸ごと同一 {same} / 時刻 {tm} / 話者 {sp}")
        print()

    # ---- 照合 B ----------------------------------------------------
    print("=" * 72)
    print(f"照合 B — 転写区間に付いたまとまり（{STEP} 秒刻み）")
    print("=" * 72)
    segs = [load_segments(f) for f in a.files]
    dur = max(s[-1]["end"] for s in segs if s)
    n = int(dur / STEP) + 1
    lines = [timeline(s, n) for s in segs]
    for f, s, ln in zip(a.files, segs, lines):
        cov = sum(1 for x in ln if x != NONE)
        print(f"  {f.name[:44]:44s} 区間 {len(s):4d} / 埋まり {cov / n * 100:4.1f}% "
              f"/ まとまり {sorted(set(ln) - {NONE})}")

    idx = range(len(a.files))
    print("\n  --- 素の一致（名前まで同じか / %）---")
    print("        " + "  ".join(f"{i+1:4d} " for i in idx))
    for i in idx:
        row = [f"{sum(1 for x, y in zip(lines[i], lines[j]) if x == y) / n * 100:5.1f}"
               for j in idx]
        print(f"  {i+1:3d}   " + "  ".join(row))

    print("\n  --- 並べ替えを許した一致（切り方が同じか / %）---")
    print("        " + "  ".join(f"{i+1:4d} " for i in idx))
    maps = {}
    for i in idx:
        row = []
        for j in idx:
            if i == j:
                row.append("100.0")
                continue
            v, m = perm_best(lines[i], lines[j], n)
            maps[(i, j)] = m
            row.append(f"{v:5.1f}" if v is not None else "  ―  ")
        print(f"  {i+1:3d}   " + "  ".join(row))

    print("\n  --- 名前が入れ替わっていないか（1 本目を基準に）---")
    for j in list(idx)[1:]:
        m = maps.get((0, j)) or {}
        ident = all(k == v for k, v in m.items())
        print(f"  {j+1:3d}  {'そのまま' if ident else '★入れ替わり'}   "
              + "  ".join(f"{k}→{v}" for k, v in sorted(m.items()) if k != NONE))

    print("\n  --- 不一致の内訳（時間の割合 %）---")
    print(f"  {'組':10s} {'一致':>7s} {'片方に区間なし':>15s} {'どちらか ?':>11s} "
          f"{'★別のまとまり':>14s} {'両方あるときの一致':>19s}")
    tot = Counter()
    for i in idx:
        for j in list(idx)[i + 1:]:
            c = Counter()
            for x, y in zip(lines[i], lines[j]):
                if x == y:
                    c["same"] += 1
                elif x == NONE or y == NONE:
                    c["gap"] += 1
                elif x == UNK or y == UNK:
                    c["unk"] += 1
                else:
                    c["diff"] += 1
            tot.update(c)
            d = c["same"] + c["diff"] or 1
            print(f"  {i+1:3d}-{j+1:<6d} {c['same']/n*100:7.2f} {c['gap']/n*100:15.2f} "
                  f"{c['unk']/n*100:11.2f} {c['diff']/n*100:14.2f} "
                  f"{c['same']/d*100:18.3f}%")
    m = sum(1 for i in idx for _ in list(idx)[i + 1:]) or 1
    d = tot["same"] + tot["diff"] or 1
    print(f"  {'合計':10s} {tot['same']/(n*m)*100:7.2f} {tot['gap']/(n*m)*100:15.2f} "
          f"{tot['unk']/(n*m)*100:11.2f} {tot['diff']/(n*m)*100:14.2f} "
          f"{tot['same']/d*100:18.3f}%")
    print(f"\n  ★別のまとまり = 1 組あたり平均 {tot['diff']/m*STEP:.1f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
