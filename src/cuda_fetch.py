"""GPU で動かすための部品（cuBLAS）を取ってくる。

**同梱しない。**`nvidia-cublas-cu12` は `LicenseRef-NVIDIA-Proprietary` で、
再配布には条件が付く（実質的な追加機能があること、配布条件が NVIDIA の
契約と矛盾しないこと等）。本体は MIT なので、**MIT のものに proprietary を
同梱すると受け取った人が「自由に再配布できる」と誤解する。**

そこで**利用者の PC が配布元（NVIDIA が上げた PyPI の wheel）から直接
取ってくる**形にした。こちらが配るのは「取ってくる仕組み」だけで、
再配布にあたらない。CPU だけで使う人には 1 バイトも増えない。

**要るのは 2 ファイルだけ**（実測・2026-08-21）。nvidia パッケージ一式は
2.0GB あるが、cuDNN(1.0GB) も nvrtc(178MB) も無しで large-v3 の転写が
最後まで通った（7 分のチャンクで 71 区間・1,760 字）。

置き場は `%APPDATA%\\WhosaidEditor\\cuda\\`（利用者ごと・書き込める場所）。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

# **版を留める。**新しい版が出たら黙って別物を配るのではなく、
# こちらで実測してから上げる。ここに書いてあるのは動作確認した実体。
CUBLAS_VERSION = "12.9.2.10"
CUBLAS_WHEEL = (
    f"nvidia_cublas_cu12-{CUBLAS_VERSION}-py3-none-win_amd64.whl")
CUBLAS_SHA256 = (
    "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661")
WHEEL_SIZE_MB = 553          # 圧縮された wheel の大きさ（取得前に伝える）
UNPACKED_MB = 736            # 取り出したあとの大きさ

# wheel の中の、要るものだけ。**全部展開すると 2GB になる。**
WANT = ("cublas64_12.dll", "cublasLt64_12.dll")

PYPI_JSON = "https://pypi.org/pypi/nvidia-cublas-cu12/{v}/json"


class CudaFetchError(RuntimeError):
    """取得できなかった。利用者にそのまま見せてよい文面を持つ。"""


def cuda_dir() -> Path:
    """取ってきた DLL の置き場。"""
    from .config import config_dir
    return config_dir() / "cuda"


def is_available() -> bool:
    """GPU で動かす部品が手元にあるか。"""
    d = cuda_dir()
    return all((d / n).is_file() for n in WANT)


def _wheel_url() -> str:
    """配布元の URL を PyPI に聞く。**URL を直書きしない**——

    files.pythonhosted.org のパスにはハッシュが入っており、直書きすると
    版を上げたときに必ず古いままになる。
    """
    try:
        with urllib.request.urlopen(
                PYPI_JSON.format(v=CUBLAS_VERSION), timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        raise CudaFetchError(
            "配布元(PyPI)に問い合わせできませんでした。\n"
            f"--- 詳細 ---\n{type(e).__name__}: {e}") from e
    for f in data.get("urls", []):
        if f.get("filename") == CUBLAS_WHEEL:
            return str(f["url"])
    raise CudaFetchError(
        f"配布元に {CUBLAS_WHEEL} が見つかりませんでした。"
        "版が取り下げられた可能性があります。")


def fetch(
    dest: Optional[Path] = None,
    *,
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """cuBLAS を取ってきて、置いたフォルダを返す。"""
    out = Path(dest) if dest else cuda_dir()
    if all((out / n).is_file() for n in WANT):
        return out

    if on_log:
        on_log(f"GPU で動かすための部品を取得します"
               f"（約 {WHEEL_SIZE_MB:,} MB / 展開後 約 {UNPACKED_MB:,} MB）。"
               "取得元は NVIDIA が PyPI に置いたものです。"
               "録音や本文は送りません。")

    url = _wheel_url()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        whl = Path(tmp) / CUBLAS_WHEEL
        _download(url, whl, on_progress)

        # **中身を確かめてから使う。**配布元の資産が差し替わったら、
        # 黙って別物を動かすのではなく止まってほしい。
        got = _sha256(whl)
        if got != CUBLAS_SHA256:
            raise CudaFetchError(
                "取得したファイルが想定と違います（改ざん、または版の差し替え）。\n"
                f"期待: {CUBLAS_SHA256}\n実際: {got}")

        staging = Path(tmp) / "out"
        staging.mkdir()
        try:
            with zipfile.ZipFile(whl) as z:
                for name in z.namelist():
                    base = name.rsplit("/", 1)[-1]
                    if base in WANT:
                        with z.open(name) as src, \
                                open(staging / base, "wb") as dst:
                            shutil.copyfileobj(src, dst)
        except (zipfile.BadZipFile, OSError) as e:
            raise CudaFetchError(
                f"取り出せませんでした。\n--- 詳細 ---\n{e}") from e

        missing = [n for n in WANT if not (staging / n).is_file()]
        if missing:
            raise CudaFetchError(
                "中に必要なものがありませんでした: " + "、".join(missing))
        for n in WANT:
            shutil.move(str(staging / n), str(out / n))

    if on_log:
        mb = sum((out / n).stat().st_size for n in WANT) / 1e6
        on_log(f"取得しました（{mb:,.0f} MB）: {out}")
    return out


def _download(url: str, dest: Path,
              on_progress: Optional[Callable[[int, int], None]]) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r, \
                open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if on_progress:
                    try:
                        on_progress(done, total or done)
                    except Exception:
                        pass    # 表示のためだけ。ここで落として取得を止めない
    except (urllib.error.URLError, OSError) as e:
        raise CudaFetchError(
            "GPU で動かすための部品を取得できませんでした。\n"
            "取得できなくても、CPU では今までどおり動きます。\n"
            f"--- 詳細 ---\n{type(e).__name__}: {e}") from e


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
