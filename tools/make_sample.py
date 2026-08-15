"""正解ラベルを付ける区間を選び、測定前に固定する。

    python tools\\make_sample.py <作業ファイル.speakers.json> [出力フォルダ]

設計書 `claude/claude_正解ラベルの作り方_設計書.md` §2 の実装。
選び方(層・件数・種)は `src/evaluate.py` にあり、テストで固定されている。

**画面を作る前にこれを実行する。**あとから選び直せる状態で測ると、
「作りやすい区間を選んだ」という疑いを自分で否定できない。

出力は実名を含みうるのでリポジトリの外に置くこと(既定もそうしてある)。
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

from src import evaluate  # noqa: E402
from src.segments import Project, fmt_hms  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    proj_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else proj_path.parent / "truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sample.json"

    if out.exists():
        print(f"既にあります: {out}")
        print("測定前に固定するためのファイルなので、選び直しません。")
        print("やり直すには、このファイルを自分で消してください。")
        return 1

    proj = Project.load(proj_path)
    segs = proj.segments
    sizes = evaluate.stratum_sizes(segs)
    picked = evaluate.select_sample(segs)

    per = Counter(x.stratum for x in picked)
    print(f"母集団: {len(segs)} 区間 "
          f"(2 秒未満 {sizes.get(evaluate.STRATUM_SHORT, 0)} / "
          f"2 秒以上 {sizes.get(evaluate.STRATUM_LONG, 0)})")
    print(f"標本  : {len(picked)} 区間 "
          f"(2 秒未満 {per[evaluate.STRATUM_SHORT]} / "
          f"2 秒以上 {per[evaluate.STRATUM_LONG]})")
    print(f"種    : {evaluate.DEFAULT_SEED}")
    half = evaluate.proportion_halfwidth(0.7, len(picked), len(segs))
    print(f"見込みの精度: 95% 信頼区間 ±{half*100:.1f} ポイント "
          f"(70% の付近で {(0.70-half)*100:.0f}〜{(0.70+half)*100:.0f}% は判定不能)")

    out.write_text(json.dumps({
        "project": proj_path.name,
        "audio_sha256": proj.source_sha256,
        "seed": evaluate.DEFAULT_SEED,
        "short_seconds": evaluate.SHORT_SECONDS,
        "population": {"total": len(segs), **sizes},
        "samples": [
            {"index": x.index, "stratum": x.stratum,
             "start": round(x.start, 2), "duration": round(x.duration, 2)}
            for x in picked
        ],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n書き出し: {out}")

    print("\n先頭 5 件:")
    for x in picked[:5]:
        print(f"  #{x.index:>4}  {fmt_hms(x.start)}  {x.duration:>5.2f} 秒  {x.stratum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
