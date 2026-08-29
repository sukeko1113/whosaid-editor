"""転写のモデルを取ってきて models/asr/<名前> に置く（同梱用）。

    python tools\\fetch_asr_model.py [モデル名] [置き場]
    python tools\\fetch_asr_model.py small          （既定）
    python tools\\fetch_asr_model.py large-v3

CI（`.github/workflows/build.yml`）と手作業の両方がこれを使う。
話者分離の `fetch_diarize_models.py` と同じ形にしてある。

**取得そのものは `src/asr_fetch.py` にある。**ここは同梱先へ置くための
薄い入口。画面（GPU 利用者への案内）も同じ実装を呼ぶ——二通りに書くと、
片方だけ直したときに「取れたはずなのに見つからない」になる。

**なぜ同梱するか。**faster-whisper はモデル名を渡すと Hugging Face へ
取りに行く。同梱していないと**新規インストール直後は通信が要る**——
画面の「ネットワークを遮断したままでも動きます」が嘘になる。
中核層は「録音を外に出せない」層で、**初回 DL ができない環境が現実にある**
（設計書 §9）。

`models/` は `.gitignore` で除外してある。取得を忘れやすいので、
`build.spec` は `small` が無ければビルドを止める。

**同梱するのは float16 のまま。**実行時に `compute_type="int8"` /
`"int8_float16"` を渡すので、読み込み時に量子化される。あらかじめ int8 に
変換すれば約半分になるが、変換に `torch` と `transformers` が要る
（ビルド環境が 2GB 重くなる）。**品質は変わらない**ので変換しない（§9）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.align import AVAILABLE_MODELS, DEFAULT_MODEL   # noqa: E402
from src.asr_fetch import REPOS, SIZES_MB, AsrFetchError, fetch  # noqa: E402


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    if model not in REPOS:
        print(f"知らないモデルです: {model}")
        print(f"選べるもの: {', '.join(REPOS)}")
        return 1
    if model not in AVAILABLE_MODELS:
        print(f"※ {model} は画面の選択肢にありません（align.AVAILABLE_MODELS）。")

    root = Path(sys.argv[2]) if len(sys.argv) > 2 else \
        Path(__file__).resolve().parent.parent / "models" / "asr"
    if (root / model / "model.bin").is_file():
        print(f"すでにあります: {root / model}")
        return 0

    print(f"{model}（約 {SIZES_MB.get(model, 0):,} MB）を取得します。")
    last = [-1]

    def progress(done: int, total: int) -> None:
        if not total:
            return
        pct = int(done * 100 / total)
        if pct != last[0]:                  # 1% ごとにしか出さない（CI のログ対策）
            last[0] = pct
            print(f"  {pct:3d}%  {done/1e6:,.0f} / {total/1e6:,.0f} MB",
                  flush=True)

    try:
        dest = fetch(model, root, on_log=lambda m: print(m, flush=True),
                     on_progress=progress)
    except AsrFetchError as e:
        print(str(e))
        return 1
    print(f"済: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
