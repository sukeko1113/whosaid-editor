# -*- coding: utf-8 -*-
"""測定の前に、キャッシュを消して「1 回目」の状態に戻す。

    python clear_run_cache.py "<出力先フォルダ>"
    python clear_run_cache.py "<出力先フォルダ>" --yes     （確認を省く）

**7 回のうち 1 回でも消し忘れると、その回だけキャッシュから復元されて
結果が読めなくなる。**手で消すのをやめるためのもの。

----------------------------------------------------------------------
## 消すもの・残すもの

消す:
  transcripts/   転写のキャッシュ。**これが本体**（残ると転写が 0 秒になる）
  diarize/       話者分離の結果。残ると話者分離が数秒で終わる
  inspect/       聴く順番。作り直せる派生データ
  <音声名>.speakers.json  → **消さずに退避**（前回の割当を引き継いでしまうため）

残す:
  timings.jsonl  所要時間の記録。**追記なので消さない**（比較したいのはここ）
  chunks/        split_audio が毎回消して分割し直すので、触らなくてよい
                 （触らないほうが「分割」の時間が正しく測れる）

----------------------------------------------------------------------
## 守れる範囲・守れない範囲

**守れる**: 上のものが消えたこと。消したあとに残っていないことを、
消したあとにもう一度数えて確かめる。

**守れない**: **音声そのものが同じかどうか。**別の音声を置いていれば、
キャッシュを消しても比較にならない。指紋は転写のログに出るので、
そちらで突き合わせること。
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

CACHE_DIRS = ("transcripts", "diarize", "inspect")
KEEP = ("timings.jsonl", "chunks")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}TB"


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="転写の出力先（.work_… の親）")
    ap.add_argument("--yes", action="store_true", help="確認を省く")
    a = ap.parse_args()

    out = Path(a.out_dir)
    if not out.is_dir():
        print(f"フォルダがありません: {out}")
        return 1

    works = sorted(out.glob(".work_*"))
    projects = sorted(out.glob("*.speakers.json"))
    if not works and not projects:
        print(f"消すものがありません（既に 1 回目の状態です）: {out}")
        return 0

    # --- 何をするかを先に全部出す -------------------------------------
    plan_del: list[Path] = []
    plan_move: list[Path] = []
    print(f"対象: {out}\n")
    for w in works:
        print(f"  {w.name}/")
        for name in CACHE_DIRS:
            d = w / name
            if d.is_dir():
                plan_del.append(d)
                n = sum(1 for _ in d.rglob("*") if _.is_file())
                print(f"      消す   {name}/  {n} ファイル  {human(dir_size(d))}")
        for name in KEEP:
            k = w / name
            if k.exists():
                print(f"      残す   {name}")
    for p in projects:
        if ".bak." in p.name:
            continue
        plan_move.append(p)
        print(f"  退避     {p.name}  {human(p.stat().st_size)}"
              "   ← 消さない（前回の割当を引き継がせないため）")

    if not plan_del and not plan_move:
        print("\n消すものがありません（既に 1 回目の状態です）")
        return 0

    if not a.yes:
        try:
            ans = input("\n実行しますか? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans != "y":
            print("やめました。何も変えていません。")
            return 1

    # --- 実行 ---------------------------------------------------------
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for d in plan_del:
        shutil.rmtree(d, ignore_errors=True)
    for p in plan_move:
        p.replace(p.with_name(f"{p.stem}.{stamp}.bak.json"))

    # --- 消えたことを数え直して確かめる -------------------------------
    left = [d for d in plan_del if d.exists()] + [p for p in plan_move if p.exists()]
    if left:
        print("\n★ 残っています:")
        for x in left:
            print(f"    {x}")
        return 1
    print(f"\n消しました（キャッシュ {len(plan_del)} / 退避 {len(plan_move)}）。"
          "1 回目の状態です。")
    for w in works:
        t = w / "timings.jsonl"
        if t.is_file():
            n = sum(1 for ln in t.read_text(encoding="utf-8").splitlines() if ln)
            print(f"  {w.name}/timings.jsonl は残っています（{n} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
