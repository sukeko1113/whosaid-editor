"""転写のモデルを取ってくる。**これが唯一の実装。**

`tools/fetch_asr_model.py`（ビルド用）と画面（GPU 利用者への案内）の両方が
ここを呼ぶ。二通りに実装すると、片方だけ直したときに「取れたはずなのに
見つからない」になる（`model_tag` を align に集めたのと同じ理由）。

**録音も本文も送らない。**通信するのは Hugging Face とだけで、送るのは
「このモデルをください」という要求だけ。CREDITS.txt にもその旨を書いてある。

**取得先は書き込める場所。**同梱先（PyInstaller の一時展開）は書けないし、
実行ファイルの隣は Program Files で管理者権限が要る。利用者ごとの
`%APPDATA%\\WhosaidEditor\\models\\asr\\<名前>` に置く。
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Callable, Optional

from .align import AVAILABLE_MODELS, downloaded_model_root

# faster-whisper が既定で見に行く配布元。**ここを変えるとモデルが別物に
# なる**ので、変えるときは実測し直すこと。
REPOS = {
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# **これが無いと動かない**という核。取得後に揃っているかを確かめる。
NEEDED = ("model.bin", "config.json", "tokenizer.json")

# **配布元によってファイル構成が違う。**small は vocabulary.txt、
# large-v3 は vocabulary.json + preprocessor_config.json を持つ。
# 決め打ちで並べると片方が落ちるので、**要らないものを除く**形にする。
SKIP = ("README.md", ".gitattributes")

# 目安の大きさ（MB）。**取得の前に利用者へ伝えるため。**
# 断りも入れずに GB 級を落とさない。
SIZES_MB = {"base": 145, "small": 486, "medium": 1530, "large-v3": 3090}


class AsrFetchError(RuntimeError):
    """取得できなかった。利用者にそのまま見せてよい文面を持つ。"""


def is_available(model: str) -> bool:
    from .align import find_bundled_model
    return find_bundled_model(model) is not None


def fetch(
    model: str,
    dest_root: Optional[Path] = None,
    *,
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """モデルを取ってきて、置いたフォルダを返す。

    on_progress: (取得済みバイト, 全体バイト) で呼ぶ。全体が分からない間は
    0 を渡す。**進み具合を出さないと、数分から十数分「固まった」ように
    見える。**3GB では致命的。
    """
    if model not in REPOS:
        raise AsrFetchError(f"知らないモデルです: {model}\n"
                            f"選べるもの: {', '.join(REPOS)}")
    root = Path(dest_root) if dest_root else downloaded_model_root()
    dest = root / model
    if (dest / "model.bin").is_file():
        return dest

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise AsrFetchError(
            "モデルの取得に huggingface_hub が必要です"
            "（faster-whisper と一緒に入ります）。\n"
            f"--- 詳細 ---\n{e}") from e

    if on_log:
        on_log(f"{REPOS[model]} を取得します（約 {SIZES_MB.get(model, 0):,} MB）。"
               "録音や本文は送りません。")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # **中断された取得を残さない。**途中で切れたフォルダが残ると、次に
    # 「あるのに動かない」になる。まず別名に取り、揃ってから置き換える。
    staging = root / f".{model}.part"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    # **進み具合はフォルダの大きさで見る。**huggingface_hub の tqdm を
    # 差し替える形にしたら、版によって total が 0 のまま流れてきて
    # 「0 / 0 MB」しか出せなかった（実機で確認・2026-08-21）。
    # 内部に頼らず、置かれたバイト数を数えるほうが壊れない。
    watcher = _watch(staging, SIZES_MB.get(model, 0) * 1_000_000, on_progress)
    try:
        got = Path(snapshot_download(
            repo_id=REPOS[model],
            ignore_patterns=list(SKIP),
            local_dir=str(staging),
        ))
    except Exception as e:
        watcher.set()
        shutil.rmtree(staging, ignore_errors=True)
        raise AsrFetchError(
            f"モデルを取得できませんでした。\n"
            "通信が遮断された環境では、別の PC で取得したフォルダを\n"
            f"{dest}\n に置いてください。\n"
            f"--- 詳細 ---\n{type(e).__name__}: {e}") from e

    watcher.set()
    missing = [n for n in NEEDED if not (got / n).is_file()]
    if missing:
        shutil.rmtree(staging, ignore_errors=True)
        raise AsrFetchError(
            "取得したものに足りないファイルがあります: " + "、".join(missing))

    shutil.rmtree(dest, ignore_errors=True)
    staging.replace(dest)
    if on_log:
        mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
        on_log(f"取得しました（{mb:,.0f} MB）: {dest}")
    return dest


def dir_size(path: Path) -> int:
    """フォルダの中身の合計バイト数。読めないものは 0 として数える。"""
    total = 0
    try:
        for f in path.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def _watch(path: Path, expected: int,
           on_progress: Optional[Callable[[int, int], None]]) -> "threading.Event":
    """置かれたバイト数を数えて on_progress に流す（別スレッド）。

    `expected` は目安（SIZES_MB）なので、実際と少しずれる。**100% を
    超えないように丸める**——「103%」は壊れて見える。
    """
    stop = threading.Event()
    if on_progress is None or expected <= 0:
        stop.set()
        return stop

    def loop() -> None:
        while not stop.wait(0.5):
            try:
                on_progress(min(dir_size(path), expected), expected)
            except Exception:
                return      # 表示のためだけ。ここで落として取得を止めない

    threading.Thread(target=loop, daemon=True).start()
    return stop


def available_models() -> list[str]:
    """画面に出す候補のうち、取得先が分かっているもの。"""
    return [m for m in AVAILABLE_MODELS if m in REPOS]
