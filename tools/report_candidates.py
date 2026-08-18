"""「候補の一覧」(設計書 §10.3)が使いものになるかを、逐語正解で測る道具。

**作る前に天井を測るために書いた。**その結果、元の設計(「声はあるのに本文が
無い箇所」＝ turn と区間の差集合)は**この音声では 1 件も拾えない**ことが
分かった。脱落は他人の発言の最中に埋もれており、区間は存在するので
「すきま」にならない。定義を入れ替えた経緯は設計書 §10.3。

測るもの:
  天井    正解の脱落のうち、turn に覆われている件数 / すきまになる件数
  再現    入れ替えた定義の候補が、正解の脱落をどれだけ指すか
  適合    出した候補のうち、正解の脱落に当たるのは何個か
  比較    既存の「聴きどころ」(listen_order)が同じ脱落をどれだけ指せるか

**閾値は調整しない。**正解は音声 1 本・4 帯しかないので、ここで合わせこむと
この音声に当てるだけの数字になる。既定値のまま一度で出す。

再転写も再分離もしない(turn はキャッシュ済みのものを読む)。

    python tools/report_candidates.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import listen_order                       # noqa: E402
from src.diarize import SpeakerTurn                # noqa: E402
from src.evaluate import proportion_halfwidth      # noqa: E402
from src.segments import Project, Segment          # noqa: E402

TRUTH_DIR = Path(r"C:\dev\01\test-audio\truth")
BANDS = ["a-setsumei", "b-chuban", "c-ouchou", "d-missitsu"]
DEFAULT_PROJECT = Path(r"C:\dev\01\test-audio\01+02edited.speakers.json")
DEFAULT_TURNS = Path(
    r"C:\dev\01\test-audio\.work_01+02edited\diarize"
    r"\turns.ca1fb4d464e99c16.pyannote3-titanet.n9.v1.json")

# 脱落の時刻と突き合わせる窓。聴きどころと同じ 3 秒を使う
# (独自の値を置くと、そこだけ当たりやすい数字になる)
WINDOW = listen_order.WINDOW_SECONDS

# 候補として数える最小の重なり。短すぎる被りは息継ぎや漏れ込みで出るため
MIN_OVERLAP = 0.2


def load_bands() -> dict[str, dict]:
    out = {}
    for name in BANDS:
        d = json.loads((TRUTH_DIR / f"verbatim.{name}.json")
                       .read_text(encoding="utf-8"))
        out[name] = d["band"]
    return out


def load_missing() -> list[dict]:
    out: list[dict] = []
    for name in BANDS:
        d = json.loads((TRUTH_DIR / f"verbatim.{name}.json")
                       .read_text(encoding="utf-8"))
        for m in d.get("missing") or []:
            out.append({**m, "band": name})
    return out


def main_speakers(segments, turns) -> dict[int, int]:
    """区間ごとの「主たる話者」。最も長く重なる turn の話者。"""
    out: dict[int, int] = {}
    for s in segments:
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(s.end, t["end"]) - max(s.start, t["start"])
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        if best is not None:
            out[s.index] = best
    return out


def candidates(segments, turns) -> list[dict]:
    """**区間の中に、その区間の主たる話者と違う声がある箇所。**

    設計書 §10.3。元の「turn と区間の差集合」は再現 0/34 だった。
    """
    main = main_speakers(segments, turns)
    out: list[dict] = []
    for s in segments:
        for t in turns:
            ov = min(s.end, t["end"]) - max(s.start, t["start"])
            if (ov >= MIN_OVERLAP and s.index in main
                    and t["speaker"] != main[s.index]):
                out.append({"at": max(s.start, t["start"]),
                            "end": min(s.end, t["end"]),
                            "seg": s.index, "speaker": t["speaker"],
                            "overlap": ov})
    return out


def rate(n: int, total: int) -> str:
    if not total:
        return "  -"
    p = n / total
    return (f"{n:>3}/{total}  {p * 100:>5.1f}%  "
            f"±{proportion_halfwidth(p, total) * 100:.0f} 点")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=str(DEFAULT_PROJECT))
    ap.add_argument("--turns", default=str(DEFAULT_TURNS))
    args = ap.parse_args()

    proj = Project.load(args.project)
    turns_raw = json.loads(
        Path(args.turns).read_text(encoding="utf-8"))["turns"]
    bands = load_bands()
    missing = load_missing()

    print(f"正解の脱落 {len(missing)} 件 / turn {len(turns_raw)} / "
          f"区間 {len(proj.segments)}")
    print()

    # --- 天井 ---------------------------------------------------------
    turn_spans = [(t["start"], t["end"]) for t in turns_raw]
    seg_spans = [(s.start, s.end) for s in proj.segments]

    def covered(spans, at):
        return any(s <= at <= e for s, e in spans)

    in_turn = sum(1 for m in missing if covered(turn_spans, m["at"]))
    gap = sum(1 for m in missing
              if covered(turn_spans, m["at"]) and not covered(seg_spans, m["at"]))
    print("■ 天井")
    print("  turn に覆われている            " + rate(in_turn, len(missing)))
    print("  turn はあるが本文の区間が無い  " + rate(gap, len(missing)))
    print("    ↑ これが 0 なので、**元の設計(差集合)では何も拾えない**")
    print("  重なりと記録されたもの(取れない)"
          + rate(sum(1 for m in missing if m.get("overlap")), len(missing)))
    print()

    # --- 入れ替えた定義 -----------------------------------------------
    cands = candidates(proj.segments, turns_raw)
    in_band = [c for c in cands
               if any(b["start"] <= c["at"] <= b["end"] for b in bands.values())]

    def hits(c):
        return any(abs(c["at"] - m["at"]) <= WINDOW for m in missing)

    hit = [c for c in in_band if hits(c)]
    found = sum(1 for m in missing
                if any(abs(c["at"] - m["at"]) <= WINDOW for c in in_band))

    print("■ 入れ替えた定義: 区間の中に、主たる話者と違う声がある箇所")
    print(f"  候補(全体)  {len(cands)} 個 / 4 帯 8 分では {len(in_band)} 個")
    print("  再現(脱落を指せたか)  " + rate(found, len(missing)))
    print("  適合(候補が本物か)    " + rate(len(hit), len(in_band)))
    print()

    print("■ 帯ごと")
    for name in BANDS:
        b = bands[name]
        cb = [c for c in in_band if b["start"] <= c["at"] <= b["end"]]
        mb = [m for m in missing if m["band"] == name]
        h = sum(1 for m in mb
                if any(abs(c["at"] - m["at"]) <= WINDOW for c in cb))
        print(f"  {name:<12} 候補 {len(cb):>3} 個 / 脱落 {len(mb):>2} 件 "
              f"/ 当たり {h:>2} 件")
    print()

    # --- 比較 ---------------------------------------------------------
    turns = [SpeakerTurn.from_dict(t) for t in turns_raw]
    hints = listen_order.score_segments(proj.segments, turns)
    hi = [h for h in hints if h.score >= listen_order.HIGH_SCORE]
    hi_spans = [(h.start - WINDOW, h.start + WINDOW) for h in hi]
    print("■ 比較: 既存の「聴きどころ」")
    print(f"  高スコアの区間 {len(hi)} / {len(hints)} 件")
    print("  それが指す脱落  "
          + rate(sum(1 for m in missing if covered(hi_spans, m["at"])),
                 len(missing)))
    print()

    print("■ 覚え書き(数字を独り歩きさせない)")
    print(f"  幅は脱落 {len(missing)} 件・候補 {len(in_band)} 個ぶん。"
          "±を落として書かないこと。")
    print("  閾値は調整していない。正解は音声 1 本・4 帯しかないので、")
    print("  合わせこむとこの音声に当てるだけの数字になる。")
    print("  「その声が別の区間として本文になっているなら脱落ではない」という")
    print("  絞り込みは試して**退けた**(適合 64%・再現 20/34 に落ちた)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
