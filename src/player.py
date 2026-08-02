"""区間音声プレイヤー。

話者割当エディタから「この区間だけを聴く」ために使う。
再生手段は次の優先順で自動選択する。

1. ffplay.exe (同梱 or PATH) … シークが速く、再生速度も変えられる。既定。
2. ffmpeg で一時 wav に切り出し → winsound で再生 (Windows のみ)
   ffplay が無いビルドでも最低限動くようにするためのフォールバック。

いずれも「元音声ファイルの絶対時刻でシーク」する。チャンク分割済みの
ファイルは使わないので、チャンク境界を意識せずに任意区間を再生できる。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

from .audio import find_ffmpeg, hide_console_kwargs


# ----------------------------------------------------------------------
# ffplay の探索
# ----------------------------------------------------------------------

def find_ffplay() -> Optional[str]:
    """ffplay の実行ファイルパス。見つからなければ None。"""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        for cand in (base / "ffplay.exe", base / "ffplay"):
            if cand.exists():
                return str(cand)
        # onedir 配布では exe と同じ階層にも置かれる
        side = Path(sys.executable).parent
        for cand in (side / "ffplay.exe", side / "ffplay"):
            if cand.exists():
                return str(cand)

    found = shutil.which("ffplay")
    if found:
        return found

    here = Path(__file__).resolve().parent.parent
    for cand in (here / "ffmpeg" / "ffplay.exe", here / "ffmpeg" / "ffplay"):
        if cand.exists():
            return str(cand)
    return None


def _atempo_chain(speed: float) -> str:
    """atempo は 0.5〜2.0 しか受け付けないので、範囲外は多段にする。"""
    if abs(speed - 1.0) < 1e-3:
        return ""
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


class SegmentPlayer:
    """区間再生を担当。GUI スレッドから play/stop を呼ぶ。

    on_finished は再生が自然終了 or 停止したときに呼ばれる(別スレッド)。
    GUI 側で after() 経由に載せ替えて使うこと。
    """

    def __init__(self, on_finished: Optional[Callable[[], None]] = None) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._token = 0
        self._tempdir = Path(tempfile.gettempdir()) / "GeminiTranscriberPlay"
        self._winsound_active = False
        self.on_finished = on_finished
        self.ffplay = find_ffplay()

    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        if self.ffplay:
            return "ffplay"
        if sys.platform == "win32":
            return "winsound"
        return "none"

    def is_playing(self) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return True
            return self._winsound_active

    # ------------------------------------------------------------------
    def play(
        self,
        audio_path: Path | str,
        start: float,
        end: float,
        speed: float = 1.0,
        pre_roll: float = 0.4,
        post_roll: float = 0.3,
        volume: int = 100,
    ) -> None:
        """[start-pre_roll, end+post_roll] を再生する。既存の再生は止める。"""
        self.stop()
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {audio_path}")

        ss = max(0.0, float(start) - pre_roll)
        dur = max(0.35, float(end) + post_roll - ss)

        with self._lock:
            self._token += 1
            token = self._token

        if self.ffplay:
            self._play_ffplay(audio_path, ss, dur, speed, volume, token)
        elif sys.platform == "win32":
            self._play_winsound(audio_path, ss, dur, speed, token)
        else:
            raise RuntimeError(
                "再生に使えるプログラムが見つかりません。ffplay を同梱するか PATH に通してください。"
            )

    # ------------------------------------------------------------------
    def _play_ffplay(
        self,
        audio_path: Path,
        ss: float,
        dur: float,
        speed: float,
        volume: int,
        token: int,
    ) -> None:
        cmd = [
            self.ffplay or "ffplay",
            "-nodisp",
            "-autoexit",
            "-hide_banner",
            "-loglevel", "error",
            "-ss", f"{ss:.3f}",
            "-t", f"{dur:.3f}",
        ]
        chain = _atempo_chain(speed)
        if volume != 100:
            vol = f"volume={max(0, min(300, volume)) / 100:.2f}"
            chain = f"{chain},{vol}" if chain else vol
        if chain:
            cmd += ["-af", chain]
        cmd.append(str(audio_path))

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            **hide_console_kwargs(),
        )
        with self._lock:
            self._proc = proc
        threading.Thread(target=self._watch_proc, args=(proc, token), daemon=True).start()

    def _watch_proc(self, proc: subprocess.Popen, token: int) -> None:
        try:
            proc.wait()
        except Exception:
            pass
        self._notify_finished(token)

    # ------------------------------------------------------------------
    def _play_winsound(
        self,
        audio_path: Path,
        ss: float,
        dur: float,
        speed: float,
        token: int,
    ) -> None:
        """ffmpeg で一時 wav を作って winsound で鳴らす。"""
        import winsound  # noqa: PLC0415  (Windows 専用)

        self._tempdir.mkdir(parents=True, exist_ok=True)
        self._cleanup_temp()
        out = self._tempdir / f"seg_{uuid.uuid4().hex[:8]}.wav"

        cmd = [
            find_ffmpeg(),
            "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{ss:.3f}",
            "-t", f"{dur:.3f}",
            "-i", str(audio_path),
            "-vn", "-ac", "1", "-ar", "22050",
        ]
        chain = _atempo_chain(speed)
        if chain:
            cmd += ["-af", chain]
        cmd.append(str(out))

        res = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", **hide_console_kwargs(),
        )
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"区間の切り出しに失敗しました\n{res.stderr}")

        with self._lock:
            self._winsound_active = True
        winsound.PlaySound(str(out), winsound.SND_FILENAME | winsound.SND_ASYNC)

        # 再生完了の通知は概算タイマーで行う(winsound に完了通知が無いため)
        def _timer() -> None:
            import time
            time.sleep(dur / max(0.1, speed) + 0.2)
            with self._lock:
                if self._token == token:
                    self._winsound_active = False
            self._notify_finished(token)

        threading.Thread(target=_timer, daemon=True).start()

    def _cleanup_temp(self) -> None:
        try:
            for f in self._tempdir.glob("seg_*.wav"):
                try:
                    f.unlink()
                except OSError:
                    pass  # 再生中のファイルは消せない。次回に回す。
        except Exception:
            pass

    # ------------------------------------------------------------------
    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            self._token += 1
            was_ws = self._winsound_active
            self._winsound_active = False
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        if was_ws and sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    def _notify_finished(self, token: int) -> None:
        with self._lock:
            current = self._token
        if current != token:
            return  # 別の再生に置き換わっている
        if self.on_finished:
            try:
                self.on_finished()
            except Exception:
                pass

    def close(self) -> None:
        self.stop()
        self._cleanup_temp()
