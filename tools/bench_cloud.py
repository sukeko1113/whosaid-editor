"""正解のある 4 帯だけを Gemini で起こして SRT にする（段階 2 の比較）。

    python tools\\bench_cloud.py <出す先> [モデル ID ...]

    例:
      python tools\\bench_cloud.py C:\\dev\\01\\test-audio\\bench gemini-3.1-pro-preview

**全長 67 分を通さない。**通しの出力は 292 秒に及ぶ区間を作り、2 分の帯で
切ろうとすると区間の本文を丸ごと拾ってしまう（正解 476 字に対して 3,110 字を
比べていた）。帯そのものを渡せば、その問題は起きない。タイムスタンプの
ドリフトも 2 分では溜まる余地が小さい。

**製品のクラウド経路（transcribe.transcribe_audio + parse_utterances）を
そのまま呼ぶ。**独自のプロンプトで測ると、製品の出力ではなくこの道具の
出力を測ることになる。

**録音は Gemini に送られる。**利用者が明示的に選んだときだけ動かすこと
（CLAUDE.md の原則）。この道具は測定用で、製品からは呼ばない。

音声は tools/bench_models.py が切り出した band.<帯>.wav を使う。
無ければ先にそちらを流す。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src import transcribe  # noqa: E402
from src.config import load_config  # noqa: E402
from src.pipeline import _make_client  # noqa: E402
from bench_models import ts  # noqa: E402
from verbatim_truth import BANDS  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    outdir = Path(sys.argv[1])
    models = sys.argv[2:] or ["gemini-3.1-pro-preview"]

    key = (load_config().get("api_key") or os.environ.get("GOOGLE_API_KEY")
           or os.environ.get("GEMINI_API_KEY") or "")
    if not key:
        print("API キーがありません（設定にも環境変数にも）。")
        return 1
    client = _make_client(key)

    bands = sorted(BANDS.items(), key=lambda kv: kv[1][0])
    cuts = []
    for name, (lo, hi) in bands:
        p = outdir / f"band.{name}.wav"
        if not p.exists():
            print(f"切り出しがありません: {p}\n"
                  "先に tools\\bench_models.py を流してください。")
            return 1
        cuts.append((name, lo, hi, p))

    for model in models:
        srt = outdir / f"{model}.srt"
        if srt.exists():
            print(f"\n[{model}] 既にあります: {srt.name}")
            continue
        print(f"\n[{model}]")
        rows: list[tuple[float, float, str]] = []
        t0 = time.monotonic()
        for name, lo, hi, path in cuts:
            print(f"  {name} …", end="", flush=True)
            t1 = time.monotonic()
            try:
                # 製品のクラウド経路と同じ設定。逐語で、時刻と話者の区別を出す
                text = transcribe.transcribe_audio(
                    client, path, model,
                    with_timestamps=True, with_diarization=True,
                    verbatim=True,
                )
            except Exception as e:
                print(f" 失敗: {e}")
                continue
            utts = transcribe.parse_utterances(text, chunk_seconds=hi - lo,
                                               verbatim=True)
            # **帯の開始秒を足して絶対時刻にする。**相対のままだと
            # report_verbatim.py がどの帯にも当てられない
            for u in utts:
                rows.append((lo + u.rel_start, lo + u.rel_end, u.text))
            print(f" {len(utts)} 区間 / {time.monotonic()-t1:.0f} 秒")

        if not rows:
            print("  区間が取れませんでした。SRT は作りません。")
            continue
        rows.sort(key=lambda r: (r[0], r[1]))
        srt.write_text(
            "\n".join(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n"
                      for i, (a, b, t) in enumerate(rows, 1)),
            encoding="utf-8")
        print(f"  → {srt.name}（{len(rows)} 区間 / {time.monotonic()-t0:.0f} 秒）")

    print(f"\n次:\n  python tools\\report_verbatim.py <truth> "
          + " ".join(f"{m}={outdir / (m + '.srt')}" for m in models))
    return 0


if __name__ == "__main__":
    sys.exit(main())
