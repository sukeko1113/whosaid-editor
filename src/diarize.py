"""話者分離(sherpa-onnx)の薄いアダプタ。

音声を「誰がいつ話したか」の区間列に変える。返すのは時刻と話者番号だけで、
区間への割り当ては segments.py、画面は assign_gui が担う
(claude/claude_話者分離_設計書.md)。

**ここが付けるのは「声のまとまり」であって話者ではない。**実名は人が聴いて
確定する。この結果で reviewed(✓)を立てることは決してない。

torch は引かない(sherpa-onnx は onnxruntime を静的に同梱している)。
sherpa_onnx と numpy は関数の中で import する。トップレベルに置くと、
話者分離を使わない利用者や CI でも import した時点で落ちる(align.py と同じ)。

音声は外に出さない。ここは完全にローカルで動く。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .align import model_tag


# 実装のバージョン。上げると話者分離のキャッシュを作り直す。
DIARIZE_VER = 1

# 分割の下限。短すぎる断片を話者区間として立てない(sherpa-onnx の既定に合わせる)。
MIN_DURATION_ON = 0.3
MIN_DURATION_OFF = 0.5

# 名簿が空のときに使う話者数。自動(-1)にすると 246 人に割れることを実測した
# ため、自動は選ばない(設計書 §7)。
DEFAULT_NUM_SPEAKERS = 6

# モデルの置き場の候補。上から順に探す。
SEG_NAME = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMB_NAME = "nemo_en_titanet_small.onnx"


class DiarizeUnavailable(RuntimeError):
    """話者分離ができない(部品が無い・モデルが無い)。

    利用者に何をすれば動くのかを伝えるための例外。呼び出し側はそのまま
    メッセージを見せてよい。原因は環境の不足なので、音声を変えても直らない
    = 途中で打ち切ってよい種類の失敗。
    """


class _Cancelled(Exception):
    """内部用。sherpa-onnx の進捗コールバックから抜けるために使う。"""


@dataclass(frozen=True)
class SpeakerTurn:
    """誰かが話していた 1 区間。時刻は音声全体での絶対秒。

    **重なりうる。**pyannote の分割は同時発話を出せるので、別々の話者の
    区間が時間的に重なることがある(実測: 611 件中 317 組)。
    """

    start: float
    end: float
    speaker: int

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "speaker": self.speaker}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpeakerTurn":
        return cls(start=float(d["start"]), end=float(d["end"]),
                   speaker=int(d["speaker"]))


def model_dirs() -> list[Path]:
    """モデルを探す場所を、優先順に返す。"""
    out: list[Path] = []
    env = os.environ.get("DIARIZE_MODELS")
    if env:
        out.append(Path(env))
    if getattr(sys, "frozen", False):           # PyInstaller で固めた実行形式
        out.append(Path(getattr(sys, "_MEIPASS", ".")) / "models" / "diarize")
        out.append(Path(sys.executable).parent / "models" / "diarize")
    else:
        out.append(Path(__file__).resolve().parent.parent / "models" / "diarize")
    return out


def resolve_models(model_dir: Optional[Path | str] = None) -> tuple[Path, Path]:
    """分割モデルと埋め込みモデルの実体を探す。

    見つからないときは、どこを探したかを添えて知らせる。「モデルがありません」
    だけでは、利用者は何を置けばよいのか分からない。
    """
    candidates = [Path(model_dir)] if model_dir else model_dirs()
    for base in candidates:
        seg, emb = base / SEG_NAME, base / EMB_NAME
        if seg.exists() and emb.exists():
            return seg, emb
    looked = "\n".join(f"    {p}" for p in candidates)
    raise DiarizeUnavailable(
        "話者分離のモデルが見つかりません。次の場所を探しました:\n"
        f"{looked}\n"
        "この下に以下の 2 つが要ります:\n"
        f"    {SEG_NAME}\n"
        f"    {EMB_NAME}\n"
        "環境変数 DIARIZE_MODELS で置き場を指定することもできます。"
    )


def turns_cache_path(
    work_dir: Path | str,
    fingerprint: str,
    num_speakers: int,
    model_dir: Optional[Path | str] = None,
) -> Optional[Path]:
    """話者分離のキャッシュの置き場(設計書 §11)。指紋が無ければ None。

    音声全長に対して 1 つ(チャンク単位ではない)。**話者数の指定もキーに入れる**
    ——指定が変われば結果が変わるため。10 分かかる処理なので、再実行のたびに
    回さない。
    """
    if not fingerprint:
        return None
    tag = model_tag("pyannote3-titanet", model_dir)
    return Path(work_dir) / "diarize" / (
        f"turns.{fingerprint}.{tag}.n{num_speakers}.v{DIARIZE_VER}.json"
    )


def load_turns(path: Optional[Path]) -> Optional[list[SpeakerTurn]]:
    """キャッシュを読む。無い・壊れている・別バージョンなら None。"""
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("diarize_ver", -1)) != DIARIZE_VER:
            return None
        return [SpeakerTurn.from_dict(t) for t in data.get("turns", [])]
    except Exception:
        return None         # 壊れたキャッシュは無かったことにして取り直す


def save_turns(path: Optional[Path], turns: list[SpeakerTurn], *,
               num_speakers: int, fingerprint: str, duration: float) -> None:
    """キャッシュを書く。書けなくても分離自体は続けられるので握りつぶす。"""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "diarize_ver": DIARIZE_VER,
            "num_speakers": num_speakers,
            "fingerprint": fingerprint,
            "duration": duration,
            "turns": [t.to_dict() for t in turns],
        }, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _load_16k_mono(audio_path: Path):
    """ffmpeg で 16kHz モノラルに落として読む(sherpa-onnx の要求)。"""
    import numpy as np

    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "x.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-vn", "-ac", "1",
                 "-ar", "16000", "-c:a", "pcm_s16le", str(wav),
                 "-loglevel", "error"],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as e:
            raise DiarizeUnavailable(
                f"音声を読み込めませんでした({audio_path.name})。\n"
                "ffmpeg が使える状態か確認してください。\n"
                f"--- 詳細 ---\n{e}"
            ) from e
        with wave.open(str(wav), "rb") as w:
            raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def ensure_available(model_dir: Optional[Path | str] = None) -> None:
    """部品が揃っているかだけを先に確かめる(モデルは読まない)。

    音声を分割し終えてから「部品がありません」では遅い、というのは
    ローカル転写で学んだとおり(local_asr.ensure_available と同じ考え)。
    """
    try:
        import sherpa_onnx      # noqa: F401
    except ImportError as e:
        raise DiarizeUnavailable(
            "話者分離には sherpa-onnx が必要です。\n"
            f"いま動いている Python: {sys.executable}\n"
            "この Python に対して\n"
            "    pip install sherpa-onnx\n"
            "で導入するか、導入済みの環境から起動してください。\n"
            f"--- 詳細 ---\n{e}"
        ) from e
    resolve_models(model_dir)


def diarize(
    audio_path: Path | str,
    *,
    num_speakers: int = DEFAULT_NUM_SPEAKERS,
    work_dir: Optional[Path | str] = None,
    model_dir: Optional[Path | str] = None,
    fingerprint: Optional[str] = None,
    on_log=None,
    on_progress=None,
    is_cancelled=None,
    force: bool = False,
) -> Optional[list[SpeakerTurn]]:
    """音声を話者ごとの区間に分ける。中断されたら None(途中結果は残さない)。

    num_speakers: 名簿の人数。**正解の話者数である必要はない**——上限を与えて
        過剰分割を防ぐのが目的で、自動(-1)にすると 246 人に割れることを
        実測した(設計書 §7)。
    on_progress: (処理済み秒, 全体秒) で呼ぶ。
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise DiarizeUnavailable(f"音声ファイルが見つかりません: {audio_path}")
    if num_speakers <= 0:
        num_speakers = DEFAULT_NUM_SPEAKERS

    if fingerprint is None:
        from .audio import audio_fingerprint
        fingerprint = audio_fingerprint(audio_path)
    cache = (turns_cache_path(work_dir, fingerprint, num_speakers, model_dir)
             if work_dir else None)

    if not force:
        cached = load_turns(cache)
        if cached is not None:
            if on_log:
                on_log(f"話者の分離結果をキャッシュから読みました"
                       f"({len(cached)} 区間)。")
            return cached

    got = _run(audio_path, num_speakers=num_speakers, model_dir=model_dir,
               on_log=on_log, on_progress=on_progress,
               is_cancelled=is_cancelled)
    if got is None:
        return None
    turns, total = got
    save_turns(cache, turns, num_speakers=num_speakers,
               fingerprint=fingerprint, duration=total)
    return turns


def _diarize_here(
    audio_path: Path,
    *,
    num_speakers: int,
    model_dir: Optional[Path | str] = None,
    on_log=None,
    on_progress=None,
    is_cancelled=None,
) -> Optional[tuple[list[SpeakerTurn], float]]:
    """**この場で**話者分離を回す(キャッシュには触らない)。

    子プロセス側からも呼ぶ。計算中は sherpa-onnx が GIL を握るので、
    親の画面から直接呼ぶと固まる(実測: 7 分の音声で 17 秒間まったく
    描けない・2026-08-22)。親は `_run` 経由で子プロセスに任せる。
    """
    ensure_available(model_dir)
    seg_model, emb_model = resolve_models(model_dir)
    import sherpa_onnx as so

    if on_log:
        on_log(f"話者を分けています(話者数 {num_speakers} を上限として指定)...")
    samples = _load_16k_mono(audio_path)
    total = len(samples) / 16000.0

    config = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(seg_model)),
            num_threads=1,
        ),
        embedding=so.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model), num_threads=1),
        clustering=so.FastClusteringConfig(num_clusters=num_speakers),
        min_duration_on=MIN_DURATION_ON,
        min_duration_off=MIN_DURATION_OFF,
    )
    if not config.validate():
        raise DiarizeUnavailable(
            "話者分離の設定が不正です。モデルの置き場を確認してください:\n"
            f"    {seg_model}\n    {emb_model}"
        )

    def progress(done: int, num_total: int) -> int:
        # sherpa-onnx 側から抜ける手段が無いので、例外で脱出する。
        if is_cancelled and is_cancelled():
            raise _Cancelled()
        if on_progress and num_total:
            on_progress(min(done / num_total * total, total), total)
        return 0

    try:
        sd = so.OfflineSpeakerDiarization(config)
        result = sd.process(samples, callback=progress)
    except _Cancelled:
        if on_log:
            on_log("話者の分離を中止しました。")
        return None

    turns = [SpeakerTurn(start=float(t.start), end=float(t.end),
                         speaker=int(t.speaker))
             for t in result.sort_by_start_time()]
    if on_progress:
        on_progress(total, total)
    if on_log:
        found = len({t.speaker for t in turns})
        on_log(f"話者の分離が終わりました({len(turns)} 区間 / {found} 人ぶん)。")
    return turns, total


# ======================================================================
# 子プロセスで回す
#
# **sherpa-onnx は計算中に GIL を握る。**同じプロセスで回すと、画面を
# 描く側がまったく動けない。実測(2026-08-22):
#
#   音声 1 分 → 処理 7.3 秒 / 最長 2.3 秒まったく描けない
#   音声 7 分 → 処理 58 秒  / 最長 17.2 秒まったく描けない
#
# 空白は音声の長さに比例するので、67 分では 3 分以上になる。Windows は
# 5 秒で「応答なし」を付けるため、利用者には**落ちたようにしか見えない**
# (実機で「途中で落ちたようです」と報告を受けた)。進捗表示を足しても
# 直らない——描く側が動けないため。**別プロセスに出すしかない。**
#
# やり取りはファイルで行う。標準出力は使わない——画面付きで固めた
# 実行ファイルでは `sys.stdout` が None になり、書いた時点で落ちる。
# ======================================================================

# 子が進捗を書く間隔。細かすぎるとディスクを叩きすぎる。
_PROGRESS_EVERY = 0.25


def _child_argv(audio: Path, out: Path, num_speakers: int,
                model_dir: Optional[Path | str]) -> list[str]:
    """子プロセスの起動コマンド。

    固めた実行ファイルなら自分自身を、開発環境なら `python -m src.main` を
    呼ぶ。**同じソースが動く**ので、親と子で結果がずれることはない。
    """
    argv = ([sys.executable] if getattr(sys, "frozen", False)
            else [sys.executable, "-m", "src.main"])
    argv += ["--diarize", str(audio), str(out), "--speakers", str(num_speakers)]
    if model_dir:
        argv += ["--model-dir", str(model_dir)]
    return argv


def _run_child(
    audio_path: Path,
    *,
    num_speakers: int,
    model_dir: Optional[Path | str] = None,
    on_progress=None,
    is_cancelled=None,
) -> Optional[tuple[list[SpeakerTurn], float]]:
    """子プロセスで回す。起動できなければ None(呼び出し側が同居で走らせる)。"""
    with tempfile.TemporaryDirectory(prefix="whosaid-diarize-") as d:
        out = Path(d) / "turns.json"
        prog = out.with_suffix(".progress")
        err = out.with_suffix(".err")
        kw: dict[str, Any] = {}
        if sys.platform == "win32":
            # 黒い窓を出さない(固めた実行ファイルでも開発環境でも)
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(
                _child_argv(audio_path, out, num_speakers, model_dir),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parent.parent), **kw)
        except OSError:
            return None                 # 起動できない = 同居で走らせる

        try:
            while True:
                try:
                    proc.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if is_cancelled and is_cancelled():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return None
                if on_progress:
                    got = _read_progress(prog)
                    if got:
                        on_progress(*got)
        finally:
            if proc.poll() is None:
                proc.kill()

        if is_cancelled and is_cancelled():
            return None
        if proc.returncode != 0 or not out.exists():
            msg = ""
            try:
                msg = err.read_text(encoding="utf-8").strip()
            except OSError:
                pass
            if not msg:
                # **起動しただけで落ちた場合は、同居で走らせ直す。**
                # 配布の形が噛み合わないときに機能ごと失わないための逃げ道
                # (固まるが、動きはする)。
                return None
            raise DiarizeUnavailable(msg)
        data = json.loads(out.read_text(encoding="utf-8"))
        turns = [SpeakerTurn.from_dict(t) for t in data.get("turns", [])]
        return turns, float(data.get("duration", 0.0))


def _read_progress(path: Path) -> Optional[tuple[float, float]]:
    """子が書いた進捗を読む。読めなければ None(次の周回で読み直す)。"""
    try:
        done, total = path.read_text(encoding="utf-8").split()
        return float(done), float(total)
    except (OSError, ValueError):
        return None


def _run(
    audio_path: Path,
    *,
    num_speakers: int,
    model_dir: Optional[Path | str] = None,
    on_log=None,
    on_progress=None,
    is_cancelled=None,
) -> Optional[tuple[list[SpeakerTurn], float]]:
    """子プロセスで回す。無理なら同じプロセスで回す。"""
    if os.environ.get("WHOSAID_DIARIZE_INPROC"):
        return _diarize_here(audio_path, num_speakers=num_speakers,
                             model_dir=model_dir, on_log=on_log,
                             on_progress=on_progress,
                             is_cancelled=is_cancelled)
    got = _run_child(audio_path, num_speakers=num_speakers,
                     model_dir=model_dir, on_progress=on_progress,
                     is_cancelled=is_cancelled)
    if got is not None:
        if on_log:
            found = len({t.speaker for t in got[0]})
            on_log(f"話者の分離が終わりました"
                   f"({len(got[0])} 区間 / {found} 人ぶん)。")
        return got
    if is_cancelled and is_cancelled():
        return None
    if on_log:
        on_log("  ※ 別プロセスで動かせなかったので、この画面で処理します"
               "(終わるまで画面が止まって見えます)。")
    return _diarize_here(audio_path, num_speakers=num_speakers,
                         model_dir=model_dir, on_log=on_log,
                         on_progress=on_progress, is_cancelled=is_cancelled)


def child_main(argv: list[str]) -> int:
    """子プロセスの入口。`main.py` が `--diarize` を見つけたら呼ぶ。

    **画面は作らない。**結果はファイルに書く(標準出力は使えない)。
    """
    import argparse
    ap = argparse.ArgumentParser(prog="--diarize", add_help=False)
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--speakers", type=int, default=DEFAULT_NUM_SPEAKERS)
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out)
    prog = out.with_suffix(".progress")
    last = [0.0]

    def on_progress(done: float, total: float) -> None:
        import time
        now = time.monotonic()
        if now - last[0] < _PROGRESS_EVERY:
            return
        last[0] = now
        try:
            prog.write_text(f"{done} {total}", encoding="utf-8")
        except OSError:
            pass                # 進捗が書けなくても分離は続ける

    try:
        got = _diarize_here(Path(args.audio), num_speakers=args.speakers,
                            model_dir=args.model_dir, on_progress=on_progress)
        if got is None:
            return 1
        turns, total = got
        out.write_text(json.dumps({
            "diarize_ver": DIARIZE_VER,
            "duration": total,
            "turns": [t.to_dict() for t in turns],
        }, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception as e:
        try:
            out.with_suffix(".err").write_text(str(e), encoding="utf-8")
        except OSError:
            pass
        return 1
