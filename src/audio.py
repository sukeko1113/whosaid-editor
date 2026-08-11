"""ffmpeg を呼び出して音声ファイルを分割・調査するモジュール"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from array import array
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


# 波形を見るためだけの取り込みなので、音質より速さを取る。
# 8kHz あれば「どこで声が途切れたか」は十分読み取れる。
PEAK_SAMPLE_RATE = 8000


def extract_peaks(
    audio_path: Path | str,
    start: float,
    end: float,
    buckets: int,
) -> list[tuple[int, int]]:
    """指定範囲の波形を buckets 個の (最小, 最大) にまとめて返す。

    区間の分割点を決めるときに使う。発言の切れ目は無音の谷として見えるので、
    人はそこをクリックして境界を合わせられる。

    ffmpeg が無い、範囲が読めない、といった場合は空リストを返す。
    呼び出し側は波形なしでも操作できるようにしておくこと。

    numpy は使わない(このアプリの依存に numpy は無く、波形を描くためだけに
    増やす価値はない)。8kHz モノラルなら 1 分でも 48 万サンプルで、
    array('h') と素朴なループで十分間に合う。
    """
    start = max(0.0, float(start))
    duration = float(end) - start
    if duration <= 0 or buckets <= 0:
        return []

    try:
        ffmpeg = find_ffmpeg()
    except FileNotFoundError:
        return []

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        # -ss を -i の前に置くと入力側でシークするので、長い音声でも速い
        "-ss", f"{start:.3f}",
        "-i", str(audio_path),
        "-t", f"{duration:.3f}",
        "-vn",
        "-ac", "1",                       # モノラル
        "-ar", str(PEAK_SAMPLE_RATE),
        "-f", "s16le",                    # ヘッダなしの生 PCM
        "-",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, **hide_console_kwargs())
    except Exception:
        return []
    if res.returncode != 0 or not res.stdout:
        return []

    raw = res.stdout
    if len(raw) % 2:
        raw = raw[:-1]                    # 半端なサンプルは捨てる
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()                # s16le を読んでいるので
    if not samples:
        return []

    peaks: list[tuple[int, int]] = []
    n = len(samples)
    for i in range(buckets):
        lo = n * i // buckets
        hi = n * (i + 1) // buckets
        if hi <= lo:
            peaks.append((0, 0))          # サンプルより多くのバケツを求められた
            continue
        part = samples[lo:hi]
        peaks.append((min(part), max(part)))
    return peaks


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
