"""転写のモデルを取ってきて models/asr/<名前> に置く。

    python tools\\fetch_asr_model.py [モデル名] [置き場]
    python tools\\fetch_asr_model.py small          （既定）
    python tools\\fetch_asr_model.py large-v3

CI（`.github/workflows/build.yml`）と手作業の両方がこれを使う。
話者分離の `fetch_diarize_models.py` と同じ形にしてある。

**なぜ同梱するか。**faster-whisper はモデル名を渡すと Hugging Face へ
取りに行く。同梱していないと**新規インストール直後は通信が要る**——
画面の「ネットワークを遮断したままでも動きます」が嘘になる。
中核層は「録音を外に出せない」層で、**初回 DL ができない環境が現実にある**
（設計書 §9）。

`models/` は `.gitignore` で除外してある（大きく、ライセンスも別なので
リポジトリに入れない）。取得を忘れやすいので、`build.spec` は
`small` が無ければビルドを止める。

**同梱するのは float16 のまま。**実行時に `compute_type="int8"` /
`"int8_float16"` を渡すので、読み込み時に量子化される。あらかじめ int8 に
変換すれば約半分（461MB → 約 230MB）になるが、変換に `torch` と
`transformers` が要る（ビルド環境が 2GB 重くなる）。**品質は変わらない**
ので、まずは変換なしで置く（設計書 §9 の注記）。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.align import AVAILABLE_MODELS, DEFAULT_MODEL  # noqa: E402

# faster-whisper が既定で見に行く配布元。**ここを変えるとモデルが別物に
# なる**ので、変えるときは実測し直すこと。
REPOS = {
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# **これが無いと動かない**という核。揃っているかを取得後に確かめる。
NEEDED = ("model.bin", "config.json", "tokenizer.json")

# **配布元によってファイル構成が違う。**small は vocabulary.txt、
# large-v3 は vocabulary.json + preprocessor_config.json を持つ。
# 決め打ちで並べると片方が落ちるので、**要らないものを除く**形にする。
SKIP = ("README.md", ".gitattributes")


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
    dest = root / model

    if (dest / "model.bin").is_file():
        size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
        print(f"すでにあります: {dest}（{size/1e6:.0f} MB）")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub が要ります（faster-whisper と一緒に入ります）。")
        print("    pip install huggingface_hub")
        return 1

    repo = REPOS[model]
    print(f"{repo} を取得します → {dest}")
    print("（初回は数分かかります。回線によってはもっと）")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        got = snapshot_download(
            repo_id=repo,
            # **説明文だけ除く。**float32 の重みや .safetensors が置かれる
            # ことがあるが、Systran の faster-whisper 系には無い。
            # 増えたときに黙って落とさないよう、除外は最小限にする。
            ignore_patterns=list(SKIP),
            local_dir=str(dest),
        )
    except Exception as e:
        print(f"取得できませんでした: {type(e).__name__}: {e}")
        print("通信が遮断された環境では、別の PC で取得したフォルダを")
        print(f"{dest} に置いてください。")
        return 1

    missing = [n for n in NEEDED if not (Path(got) / n).is_file()]
    if missing:
        # **足りないまま配らない。**利用者の端末で初めて分かるのでは遅い。
        print(f"取得したフォルダに足りないものがあります: {', '.join(missing)}")
        shutil.rmtree(dest, ignore_errors=True)
        return 1

    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"済: {dest}（{size/1e6:.0f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
