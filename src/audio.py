"""ffmpeg を呼び出して音声ファイルを分割・調査するモジュール"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def find_ffmpeg() -> str:
    """ffmpeg のパスを返す。
    1) PyInstaller でフリーズされている場合: 同梱の ffmpeg.exe を返す
    2) システムの PATH にあれば、そちらを返す
    3) 見つからなければ FileNotFoundError
    """
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        for cand in (base / "ffmpeg.exe", base / "ffmpeg"):
            if cand.exists():
                return str(cand)

    found = shutil.which("ffmpeg")
    if found:
        return found

    # 開発時の利便: プロジェクトルート直下の ffmpeg/ffmpeg.exe も探す
    here = Path(__file__).resolve().parent.parent
    for cand in (here / "ffmpeg" / "ffmpeg.exe", here / "ffmpeg" / "ffmpeg"):
        if cand.exists():
            return str(cand)

    raise FileNotFoundError(
        "ffmpeg が見つかりません。アプリに同梱されていない可能性があります。"
    )


def hide_console_kwargs() -> dict:
    """Windows でサブプロセスのコンソールを隠すための kwargs"""
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return {"startupinfo": si, "creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}


# 後方互換(旧名)
_hide_console_kwargs = hide_console_kwargs


def audio_fingerprint(input_path: Path, on_progress=None) -> str:
    """音声ファイルの中身から指紋(16桁)を作る。

    ファイル名だけで作業ファイルやキャッシュを識別すると、
    「音声を編集したのに名前が同じ」というときに古い結果を再利用してしまう。
    中身が 1 バイトでも変われば別物として扱えるように、全体をハッシュする。

    1GB の WAV でも数秒で終わる。取得できない場合は空文字を返す
    (指紋なしとして扱い、再利用の判断は継続時間の比較にフォールバックする)。
    """
    try:
        size = input_path.stat().st_size
    except OSError:
        return ""

    h = hashlib.blake2b(digest_size=8)
    h.update(str(size).encode("ascii"))
    read = 0
    try:
        with open(input_path, "rb") as f:
            while True:
                block = f.read(4 * 1024 * 1024)
                if not block:
                    break
                h.update(block)
                read += len(block)
                if on_progress and size:
                    on_progress(read, size)
    except OSError:
        return ""
    return h.hexdigest()


_DURATION_PATTERN = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2})(?:\.(\d+))?")


def probe_duration(input_path: Path) -> float:
    """入力音声の総再生時間(秒)を返す。

    ffprobe は同梱していないため、`ffmpeg -i <file>` の stderr に出る
    "Duration: HH:MM:SS.ss" を読む。取得できなければ 0.0 を返す。
    """
    try:
        res = subprocess.run(
            [find_ffmpeg(), "-hide_banner", "-i", str(input_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            **hide_console_kwargs(),
        )
    except Exception:
        return 0.0
    m = _DURATION_PATTERN.search((res.stderr or "") + (res.stdout or ""))
    if not m:
        return 0.0
    h, mm, ss, frac = m.group(1), m.group(2), m.group(3), m.group(4) or "0"
    return int(h) * 3600 + int(mm) * 60 + int(ss) + float(f"0.{frac}")


def split_audio(
    input_path: Path,
    output_dir: Path,
    chunk_seconds: int,
) -> list[Path]:
    """音声ファイルを chunk_seconds 秒ごとに分割し、生成されたチャンクを返す。

    出力は AAC モノラル 16kHz 64kbps の m4a。
    既に output_dir にチャンクがある場合はクリアしてから分割する。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # 既存チャンクを削除
    for old in output_dir.glob("chunk_*.m4a"):
        old.unlink()

    pattern = output_dir / "chunk_%04d.m4a"
    cmd = [
        find_ffmpeg(),
        "-y",
        "-i", str(input_path),
        "-vn",                        # 映像トラックは無視
        "-ac", "1",                   # モノラル
        "-ar", "16000",               # 16kHz
        "-c:a", "aac",
        "-b:a", "64k",
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1",
        str(pattern),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_hide_console_kwargs(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg による分割に失敗しました\n--- stderr ---\n{result.stderr}"
        )

    chunks = sorted(output_dir.glob("chunk_*.m4a"))
    if not chunks:
        raise RuntimeError("分割結果のチャンクが生成されませんでした。")
    return chunks
