"""逐語の正解に対して、複数のエンジンを同じ物差しで比べる(段階 2 の集計)。

    python tools\\report_verbatim.py <truth フォルダ> <名前=出力> [<名前=出力> ...]

    例:
      python tools\\report_verbatim.py C:\\dev\\01\\test-audio\\truth ^
        local=C:\\dev\\01\\test-audio\\01+02edited.speakers.json ^
        cloud=G:\\...\\01+02edited.speakers.json ^
        atrain=C:\\...\\transcription.srt

出力は `.speakers.json`(作業ファイル)か `.srt` を受け取る。

**指標の定義は測る前に固定してある**(src/evaluate.py)。出た数字を見てから
正規化の仕方や語のリストを変えれば、都合の良い値を選べてしまう。

測るもの:
  1. CER          … 正解の全文との文字誤り率
  2. 逐語保持率    … 正解のフィラー・相づちのうち、出力に残った割合
  3. 短発話再現率  … 正解の 2 秒未満の発話が、区間として現れる割合
  4. **脱落の再現** … 人が「拾われていない」と足した発話を、そのエンジンは
                      拾えているか。**この指標が段階 2 の中心**

4 について 2 つ断っておく。どちらも実データを 1 帯測ったあとで直した
(設計書 §1 の「測る前に定義を固定する」に反する)。理由は数字が気に入らなかった
からではなく、**人が耳で聴いて確定した事実と指標の答えが食い違ったから**で、
どちらも自社製品(local)に不利な向きの変更である。

  a) **回数で照合する。**含まれるかどうかだと、近くの区間に同じ語があるとき
     当たってしまう。別人の「はい」を探しているのに、その区間の話者自身の
     「はい」に当たった。
  b) **重なったものを分けて数える。**ほぼ同時に別の人が話している発話は、
     1 本の音声から 1 本の本文を書く仕組みである以上どのエンジンも拾えない。
     混ぜると全員 0 になり、エンジンの差が消える。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src import evaluate  # noqa: E402
from src.segments import Project, fmt_hms  # noqa: E402

_SRT_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+) --> (\d+):(\d+):(\d+)[,.](\d+)")


def load_engine(path: Path) -> list[tuple[float, float, str]]:
    """出力を (開始, 終了, 本文) の並びにする。"""
    if path.suffix.lower() == ".srt":
        out, block = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() == "":
                if block:
                    out.append(block)
                    block = []
            else:
                block.append(line)
        if block:
            out.append(block)
        segs = []
        for b in out:
            for i, line in enumerate(b):
                m = _SRT_TS.match(line.strip())
                if m:
                    g = [int(x) for x in m.groups()]
                    start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                    end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                    segs.append((start, end,
                                 " ".join(x.strip() for x in b[i + 1:]).strip()))
                    break
        # **必ず時間順に並べ直す。**ファイルの並びが時間順とは限らず、
        # 順序が狂うと本文を繋いだときに位置がずれ、CER が誤って悪化する
        # (合成データの検証で、正解と一致するはずのエンジンが 11.4% と出た)。
        return sorted(segs, key=lambda x: (x[0], x[1]))
    proj = Project.load(path)
    return [(s.start, s.end, s.text)
            for s in sorted(proj.segments, key=lambda s: (s.start, s.index))]


def in_band(segs, lo: float, hi: float):
    """帯に重なる区間だけを、時間順に返す。"""
    return [s for s in segs if s[1] > lo and s[0] < hi]


def text_in(segs, lo: float, hi: float) -> str:
    """[lo, hi] にあたる本文を、**はみ出した分を文字数で按分して**繋ぐ。

    区間ごと丸ごと取ると、**粗く区切るエンジンほど得をする。**Gemini は
    112 秒の区間を作るので、±2 秒の窓を見ているつもりで 500 字の干し草の山を
    探すことになり、短い相づちが当たりやすくなっていた（実測: 窓が拾う字数の
    中央値が turbo 42 字に対し gemini 103 字）。

    按分は「区間の中で話す速さは一定」という近似である。この製品は Gemini の
    時刻ドリフト対策で `redistribute_times()` が既に同じ仮定を置いている。
    区間が窓より短ければ、切り取りは起きない（Whisper 系はほぼこちら）。

    segs は (開始, 終わり, 本文) の並び。
    """
    out = []
    for s, e, t in segs:
        if e <= lo or s >= hi or not t:
            continue
        dur = e - s
        if dur <= 0:
            out.append(t)
            continue
        f0 = max(0.0, (lo - s) / dur)
        f1 = min(1.0, (hi - s) / dur)
        if f1 <= f0:
            continue
        out.append(t[int(f0 * len(t)):max(int(f0 * len(t)) + 1, int(f1 * len(t)))]
                   if (f0 > 0.0 or f1 < 1.0) else t)
    return "".join(out)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    truth_dir = Path(sys.argv[1])
    engines: dict[str, list] = {}
    for arg in sys.argv[2:]:
        name, _, p = arg.partition("=")
        path = Path(p)
        if not path.exists():
            print(f"見つかりません: {path}")
            return 1
        engines[name] = load_engine(path)
        print(f"読み込み: {name} … {len(engines[name])} 区間")

    files = sorted(truth_dir.glob("verbatim.*.json"))
    if not files:
        print(f"\n逐語の正解がありません: {truth_dir}\\verbatim.*.json")
        print("先に tools\\verbatim_truth.py で作ってください。")
        return 1

    grand: dict[str, dict] = {n: {"cer_num": 0.0, "cer_den": 0,
                                  "f_got": 0, "f_want": 0,
                                  "b_got": 0, "b_want": 0,
                                  "short_hit": 0, "short_all": 0,
                                  "miss_cl_hit": 0, "miss_cl_all": 0,
                                  "miss_ov_hit": 0, "miss_ov_all": 0}
                              for n in engines}

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("made_by") != "human-verbatim":
            print(f"\n飛ばします(人が作ったものではない): {f.name}")
            continue
        band = data["band"]
        lo, hi = float(band["start"]), float(band["end"])
        rows = data.get("segments", [])
        missing = data.get("missing", [])
        if not rows:
            continue

        # 正解の全文 = 直した本文 ＋ 足した発話(時間順に差し込む)。
        # 2 番目の要素で並びを決める: 同じ時刻なら本文が先、足した発話は
        # 足した順。これが無いと、同じ時刻の発話が本文の五十音順で並ぶ。
        pieces = [(float(r["start"]), 0, i, r["truth"])
                  for i, r in enumerate(rows)]
        pieces += [(float(m["at"]), 1, int(m.get("order", i)), m["text"])
                   for i, m in enumerate(missing)]
        truth_text = "".join(t for *_k, t in sorted(pieces, key=lambda p: p[:3]))

        print(f"\n{'='*62}\n【{band['name']}】{fmt_hms(lo)}〜{fmt_hms(hi)}"
              f"  正解 {len(rows)} 区間 / 足した発話 {len(missing)} 件"
              f" / {len(evaluate.normalize_for_cer(truth_text))} 字")
        print(f"{'エンジン':<10} {'CER':>7} {'フィラー':>18} {'相づち':>18}"
              f" {'短発話':>10}")

        for name, segs in engines.items():
            hyp_text = text_in(segs, lo, hi)
            g = grand[name]

            c = evaluate.cer(truth_text, hyp_text)
            g["cer_num"] += c * len(evaluate.normalize_for_cer(truth_text))
            g["cer_den"] += len(evaluate.normalize_for_cer(truth_text))

            fr, fg, fw = evaluate.retention_rate(
                truth_text, hyp_text, evaluate.FILLER_TERMS)
            br, bg, bw = evaluate.retention_rate(
                truth_text, hyp_text, evaluate.BACKCHANNEL_TERMS)
            g["f_got"] += fg
            g["f_want"] += fw
            g["b_got"] += bg
            g["b_want"] += bw

            # 短発話再現率: 正解の 2 秒未満の区間が、重なる区間の本文に現れるか
            short = [r for r in rows
                     if float(r["end"]) - float(r["start"])
                     < evaluate.SHORT_UTTERANCE_SECONDS and r["truth"].strip()]
            hit = 0
            for r in short:
                want = evaluate.normalize_for_cer(r["truth"])
                near = text_in(segs, float(r["start"]) - 1.0,
                               float(r["end"]) + 1.0)
                if want and want in evaluate.normalize_for_cer(near):
                    hit += 1
            g["short_hit"] += hit
            g["short_all"] += len(short)

            # 脱落の再現: 人が足した発話を、そのエンジンは拾えているか。
            # **含まれるかどうかではなく回数で見る。**近くの区間に同じ語が
            # あると当たってしまう——別人の「はい」を探しているのに、その
            # 区間の話者自身の「はい」に当たった(実データで起きた)。
            for m in missing:
                want = evaluate.normalize_for_cer(m["text"])
                if not want:
                    continue
                at = float(m["at"])
                near_hyp = evaluate.normalize_for_cer(
                    text_in(segs, at - 2.0, at + 2.0))
                # 正解の本文の側に既にある分は、エンジンの手柄にしない
                near_truth = evaluate.normalize_for_cer(text_in(
                    [(float(r["start"]), float(r["end"]), r["truth"])
                     for r in rows], at - 2.0, at + 2.0))
                # 同じ語を同じあたりに複数足していたら、その分も底上げする
                earlier = sum(
                    1 for o in missing
                    if o is not m
                    and evaluate.normalize_for_cer(o["text"]) == want
                    and abs(float(o["at"]) - at) < 2.0
                    and int(o.get("order", 0)) < int(m.get("order", 0)))
                got = near_hyp.count(want) > near_truth.count(want) + earlier
                # 重なったものは、どのエンジンも拾えない見込み。混ぜると
                # 差が消えるので別に数える
                key = "ov" if m.get("overlap") else "cl"
                g[f"miss_{key}_hit"] += 1 if got else 0
                g[f"miss_{key}_all"] += 1

            print(f"{name:<10} {c*100:>6.1f}% {fr*100:>9.1f}% ({fg:>3}/{fw:<3})"
                  f" {br*100:>9.1f}% ({bg:>3}/{bw:<3})"
                  f" {hit:>4}/{len(short):<4}")

    # ---- 合計 ----
    print(f"\n{'='*62}\n【合計】")
    print(f"{'エンジン':<10} {'CER':>7} {'フィラー保持':>14} {'相づち保持':>14}"
          f" {'短発話再現':>12} {'脱落の再現':>12}")
    for name in engines:
        g = grand[name]
        c = g["cer_num"] / g["cer_den"] if g["cer_den"] else 0.0
        fr = g["f_got"] / g["f_want"] if g["f_want"] else 0.0
        br = g["b_got"] / g["b_want"] if g["b_want"] else 0.0
        sr = g["short_hit"] / g["short_all"] if g["short_all"] else 0.0
        mr = (g["miss_cl_hit"] / g["miss_cl_all"]) if g["miss_cl_all"] else 0.0
        print(f"{name:<10} {c*100:>6.1f}% {fr*100:>13.1f}% {br*100:>13.1f}%"
              f" {sr*100:>11.1f}%"
              f" {mr*100:>10.1f}% ({g['miss_cl_hit']}/{g['miss_cl_all']})")

    # 重なったものは別枠。優劣には使わない
    ov_all = max((grand[n]["miss_ov_all"] for n in engines), default=0)
    if ov_all:
        print(f"\n【参考】重なって消えた発話 {ov_all} 件"
              "（ほぼ同時に別の人が話しているもの）")
        for name in engines:
            g = grand[name]
            print(f"  {name:<10} 拾えた {g['miss_ov_hit']}/{g['miss_ov_all']}")
        print("  ※ 1 本の音声から 1 本の本文を書く仕組みである以上、"
              "どのエンジンも拾えない見込み。"
              "\n    **エンジンの優劣には使わない。**"
              "人が手で足すしかない発話が何件あるかを見るための数字。")

    print("\n注: フィラー・相づちは部分一致で数えている(形態素解析はしない)。"
          "\n    粗さは 3 つのエンジンに共通に効く。保持率は 1.0 を超えうる"
          "(候補が正解より多く出している場合)。"
          "\n    「脱落の再現」は、人が聴いて足した発話のうち**重なっていない"
          "もの**を、そのエンジンが拾えている割合。"
          "\n    **本製品の small が落としたものを他のエンジンが拾えているかが、"
          "ここに出る。**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
