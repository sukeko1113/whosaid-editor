"""話者区間(diarize)を転写区間へ割り当て、作業量がどう変わるかを測る。

    python tools\\apply_turns.py <作業ファイル.speakers.json> <diarize.json> [重なりの閾値]

ローカル転写設計書 §6.2 の割当規則(最も重なる話者区間を採り、重なりが
閾値未満なら `?`)をそのまま実装し、実データで効果を見るための道具。
製品には含めない。

2026-08-15 の実測(67 分実会議・1219 区間・話者分離 611 区間):
  話者が付いた区間 97% / まとまり 6 個 / 割当の操作 1219 回 → 6 回 + 未判別 27 件
  細切れの解消: 514 組が連結可能 → 1219 → 705 区間
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.segments import Project  # noqa: E402

END_MARKS = "。．！？!?」』…"


def assign_turns(segments, turns, min_ratio: float):
    """区間ごとに、最も重なる話者を返す(重なりが足りなければ None)。"""
    out = []
    for s in segments:
        best_spk, best_ov = None, 0.0
        for t in turns:
            ov = min(s.end, t["end"]) - max(s.start, t["start"])
            if ov > best_ov:
                best_spk, best_ov = t["speaker"], ov
        span = max(0.01, s.end - s.start)
        out.append(best_spk if best_ov / span >= min_ratio else None)
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    proj = Project.load(Path(sys.argv[1]))
    data = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    min_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    turns = data["turns"]
    segs = sorted(proj.segments, key=lambda s: (s.start, s.index))
    covered = data.get("seconds", proj.duration)
    segs = [s for s in segs if s.start < covered]

    print(f"転写区間 {len(segs)} 件 / 話者区間 {len(turns)} 件 "
          f"/ 対象 {covered/60:.1f} 分 / 重なりの閾値 {min_ratio}")

    labels = assign_turns(segs, turns, min_ratio)
    named = sum(1 for x in labels if x is not None)
    print(f"\n話者が付いた区間: {named} / {len(segs)} ({named*100//len(segs)}%)")
    c = Counter(x for x in labels if x is not None)
    print(f"まとまりの数: {len(c)}")
    for spk, n in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  話者 {spk}: {n:>4} 区間")

    print("\n割当の操作回数(理想値)")
    print(f"  まとまり無し      : {len(segs):>5} 回")
    print(f"  話者分離あり      : {len(c):>5} 回 + 未判別 {len(segs)-named} 件")

    # 同じ話者で、文の途中で終わり、間隔が短い組は連結できる。
    # 話者ラベルが無いと安全に連結できない(merge_consecutive が擬似クラスタを
    # 連結しないのはこのため)。
    merge = 0
    for (a, la), (b, lb) in zip(zip(segs, labels), zip(segs[1:], labels[1:])):
        if la is None or la != lb:
            continue
        if a.text and a.text[-1] in END_MARKS:
            continue
        if b.start - a.end >= 0.3:
            continue
        merge += 1
    print(f"\n細切れの解消(同じ話者・文の途中・0.3 秒未満)")
    print(f"  連結できる組: {merge} 件 → 区間数 {len(segs)} → {len(segs)-merge}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
