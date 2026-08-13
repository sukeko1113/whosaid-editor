"""faster-whisper を呼び出して、単語ごとの時刻を取る薄いアダプタ。

自動点検(Step 1-3)は「Gemini の転写に付いた時刻」と「実音声の時刻」を突き合わせて
ずれを見つける。その実測側をここが担う。返すのは単語と時刻の並びだけで、
照合の理屈は anchor.py、提案の組み立ては inspect.py にある。

薄くしてあるのは差し替えを想定しているため。精度が足りなければアライナを
入れ替える(WhisperX など)が、そのときもこのモジュールの返り値の形は変えない。
features.extract を差し替え可能にしたのと同じ流儀。

faster_whisper は関数の中で import する。トップレベルに置くと、点検を使わない
利用者や CI でも import した時点で落ちる(transcribe.py が google.genai を
トップレベル import していて素の Python では動かない、あの状態を増やさない)。

音声は外に出さない。ここは完全にローカルで動く。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .audio import audio_fingerprint


# 実装のバージョン。上げると words キャッシュを作り直す。
ALIGN_VER = 1

# 既定は small。base は速いが精度が落ち、medium は CPU では重い(§9)。
DEFAULT_MODEL = "small"
AVAILABLE_MODELS = ("base", "small", "medium")

# CPU で回す前提。int8 量子化で実用速度になる(torch は要らない)。
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

LANGUAGE = "ja"


class AlignUnavailable(RuntimeError):
    """点検の下ごしらえができない(部品が無い・モデルが無い)。

    利用者に何をすれば動くのかを伝えるための例外。呼び出し側は
    そのままメッセージを見せてよい。
    """


@dataclass
class Word:
    """アライナが返す 1 単語。これが anchor.py への唯一の入口。"""

    text: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Word":
        return cls(text=str(d.get("text", "")),
                   start=float(d.get("start", 0.0)),
                   end=float(d.get("end", 0.0)))


def resolve_model(model: str = DEFAULT_MODEL,
                  model_dir: Optional[Path | str] = None) -> str:
    """faster-whisper に渡すモデルの指定を決める。

    手動で置いたフォルダが最優先(§9)。ローカル完結が要る現場(研究倫理審査・
    機密案件)ではオンライン取得そのものができないので、「別 PC で取った
    モデルフォルダを置けば動く」経路を必ず残す。

    フォルダ指定が無ければモデル名を返す。faster-whisper が既定の置き場に
    無ければ取りに行く(初回のみ)。
    """
    if model_dir:
        p = Path(model_dir)
        if not p.is_dir():
            raise AlignUnavailable(
                f"モデルフォルダが見つかりません: {p}\n"
                "別の PC で取得したモデルフォルダを指定するか、指定を外して"
                "自動取得に任せてください。"
            )
        if not (p / "model.bin").exists():
            raise AlignUnavailable(
                f"モデルフォルダの中身が足りません: {p}\n"
                "CTranslate2 形式のフォルダ(model.bin と config.json などが"
                "入ったもの)を指定してください。"
            )
        return str(p)
    return model


def words_cache_path(work_dir: Path | str, fingerprint: str, model: str,
                     model_dir: Optional[Path | str] = None) -> Optional[Path]:
    """words キャッシュの置き場(§7)。指紋が無いときは None。

    キーは「音声の指紋 + モデルの素性 + 実装バージョン」。どれが欠けても
    別物の転写を使い回す事故になるので、指紋が取れなかった音声は
    そもそもキャッシュしない(毎回取り直すほうが安全)。

    モデルの素性には model_dir も含める。手動配置のフォルダを差し替えると
    モデル名が同じでも中身は別物で、名前だけをキーにすると差し替え前の
    実測を使い回してしまう。フォルダはパスの短縮ハッシュで区別する
    (パスが同じで中身だけ入れ替えた場合は拾えない。§7 に既知の限界として記載)。

    モデル名の区切り文字(HF のリポ ID に含まれる「/」等)は「-」に無害化する。
    そのままだとキャッシュのファイル名が下位フォルダに割れてしまう。
    """
    if not fingerprint:
        return None
    tag = re.sub(r"[\\/:]+", "-", model)
    if model_dir:
        digest = hashlib.blake2b(
            str(Path(model_dir).resolve()).encode("utf-8"),
            digest_size=4).hexdigest()
        tag += f".d{digest}"
    return Path(work_dir) / "align" / f"words.{fingerprint}.{tag}.a{ALIGN_VER}.json"


def load_words(path: Optional[Path]) -> Optional[list[Word]]:
    """キャッシュを読む。無い・壊れている・別バージョンなら None。"""
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("align_ver", -1)) != ALIGN_VER:
            return None
        return [Word.from_dict(w) for w in data.get("words", [])]
    except Exception:
        return None         # 壊れたキャッシュは無かったことにして取り直す


def save_words(path: Optional[Path], words: list[Word], *,
               model: str, fingerprint: str, duration: float) -> None:
    """キャッシュを書く。書けなくても点検自体は続けられるので握りつぶす。"""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "align_ver": ALIGN_VER,
            "model": model,
            "fingerprint": fingerprint,
            "duration": duration,
            "words": [w.to_dict() for w in words],
        }, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def transcribe_words(
    audio_path: Path | str,
    *,
    work_dir: Optional[Path | str] = None,
    model: str = DEFAULT_MODEL,
    model_dir: Optional[Path | str] = None,
    fingerprint: Optional[str] = None,
    on_log=None,
    on_progress=None,
    is_cancelled=None,
    force: bool = False,
) -> Optional[list[Word]]:
    """音声を逐語で起こし、単語ごとの時刻を返す。

    重いのはここだけなので結果はキャッシュする(§7)。照合と提案づくりは
    軽いので毎回やり直す。redistribute_times を毎回計算し直すのと同じ考え方。

    work_dir: `.work_<音声名>` のパス。省略するとキャッシュしない。
    on_progress: (処理済み秒, 全体秒) で呼ぶ。
    is_cancelled: True を返したら中断し None を返す(途中結果は残さない)。
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise AlignUnavailable(f"音声ファイルが見つかりません: {audio_path}")

    if fingerprint is None:
        fingerprint = audio_fingerprint(audio_path)
    cache = (words_cache_path(work_dir, fingerprint, model, model_dir)
             if work_dir else None)

    if not force:
        cached = load_words(cache)
        if cached is not None:
            if on_log:
                on_log(f"実測の単語時刻をキャッシュから読みました({len(cached)} 語)。")
            return cached

    target = resolve_model(model, model_dir)
    try:
        from faster_whisper import WhisperModel      # 重いのでここで読む
    except ImportError as e:
        raise AlignUnavailable(
            "点検には faster-whisper が必要です。\n"
            "    pip install faster-whisper\n"
            "で導入してから、もう一度お試しください。"
        ) from e

    if on_log:
        on_log(f"実測用の転写を始めます(モデル {model} / CPU)。"
               "初回はモデルの取得に時間がかかります。")
    try:
        whisper = WhisperModel(target, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        raise AlignUnavailable(
            f"モデルを読み込めませんでした({model})。\n"
            "オンラインで取得できない環境では、別の PC で取得したモデル"
            "フォルダを指定してください。\n"
            f"--- 詳細 ---\n{e}"
        ) from e

    segments, info = whisper.transcribe(
        str(audio_path),
        language=LANGUAGE,
        word_timestamps=True,       # これが目的。単語ごとの start/end が付く
        # VAD は使わない。無音を飛ばすと速くなるが、短い相づちを落とす
        # 恐れがある。落ちた区間は「照合不能」になって提案が出せなくなる。
        vad_filter=False,
    )
    total = float(getattr(info, "duration", 0.0) or 0.0)

    words: list[Word] = []
    for seg in segments:            # ここで初めて実際の転写が走る(遅延評価)
        if is_cancelled and is_cancelled():
            if on_log:
                on_log("実測用の転写を中止しました。")
            return None
        for w in (getattr(seg, "words", None) or []):
            text = (w.word or "").strip()
            if not text:
                continue
            words.append(Word(text=text, start=float(w.start), end=float(w.end)))
        if on_progress and total:
            on_progress(min(float(seg.end), total), total)

    if on_progress and total:
        on_progress(total, total)
    if on_log:
        on_log(f"実測の単語時刻を {len(words)} 語ぶん取りました。")
    save_words(cache, words, model=model, fingerprint=fingerprint, duration=total)
    return words
