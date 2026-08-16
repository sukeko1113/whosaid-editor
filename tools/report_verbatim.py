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
                                  "miss_hit": 0, "miss_all": 0}
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

        # 正解の全文 = 直した本文 ＋ 足した発話(時間順に差し込む)
        pieces = [(float(r["start"]), r["truth"]) for r in rows]
        pieces += [(float(m["at"]), m["text"]) for m in missing]
        truth_text = "".join(t for _, t in sorted(pieces))

        print(f"\n{'='*62}\n【{band['name']}】{fmt_hms(lo)}〜{fmt_hms(hi)}"
              f"  正解 {len(rows)} 区間 / 足した発話 {len(missing)} 件"
              f" / {len(evaluate.normalize_for_cer(truth_text))} 字")
        print(f"{'エンジン':<10} {'CER':>7} {'フィラー':>18} {'相づち':>18}"
              f" {'短発話':>10}")

        for name, segs in engines.items():
            band_segs = in_band(segs, lo, hi)
            hyp_text = "".join(t for _, _, t in band_segs)
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
                near = "".join(t for s, e, t in segs
                               if e > float(r["start"]) - 1.0
                               and s < float(r["end"]) + 1.0)
                if want and want in evaluate.normalize_for_cer(near):
                    hit += 1
            g["short_hit"] += hit
            g["short_all"] += len(short)

            # 脱落の再現: 人が足した発話を、そのエンジンは拾えているか
            mhit = 0
            for m in missing:
                want = evaluate.normalize_for_cer(m["text"])
                near = "".join(t for s, e, t in segs
                               if e > float(m["at"]) - 2.0
                               and s < float(m["at"]) + 2.0)
                if want and want in evaluate.normalize_for_cer(near):
                    mhit += 1
            g["miss_hit"] += mhit
            g["miss_all"] += len(missing)

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
        mr = g["miss_hit"] / g["miss_all"] if g["miss_all"] else 0.0
        print(f"{name:<10} {c*100:>6.1f}% {fr*100:>13.1f}% {br*100:>13.1f}%"
              f" {sr*100:>11.1f}% {mr*100:>11.1f}%")
    print("\n注: フィラー・相づちは部分一致で数えている(形態素解析はしない)。"
          "\n    粗さは 3 つのエンジンに共通に効く。保持率は 1.0 を超えうる"
          "(候補が正解より多く出している場合)。"
          "\n    「脱落の再現」は、人が聴いて足した発話をそのエンジンが"
          "拾えている割合。**本製品の small が落としたものを"
          "\n    他のエンジンが拾えているかが、ここに出る。**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
