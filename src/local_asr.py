"""faster-whisper で本文を作る、ローカル転写のアダプタ。

align.py と同じエンジンを使うが、役割が違う。align.py は「時刻を測る物差し」で
本文には触れない。こちらは**本文そのもの**を作る。録音を外に出さずに反訳を
完成させるための経路(claude/claude_ローカル転写_設計書.md)。

返すのは Utterance の並び(チャンク先頭からの相対秒 + 本文)と、同じ 1 回の
転写から取れた単語時刻。テキストの行形式(`[MM:SS] 【A】 本文`)には落とさない。
落とすと時刻が秒に丸められ、さらに redistribute_times の文字数按分で
作り直されてしまう。按分は Gemini のドリフト対策であって、実測時刻に
かけるものではない(設計書 §4.1)。

モデルの解決とキャッシュのキーは align.py の部品をそのまま使う。二重に
実装すると、片方だけ直したときに別物の転写が同じキーを共有する事故になる。

faster_whisper は関数の中で import する(align.py と同じ理由)。
音声は外に出さない。ここは完全にローカルで動く。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .align import (
    AlignUnavailable,
    COMPUTE_TYPE,
    DEFAULT_MODEL,
    DEVICE,
    LANGUAGE,
    Word,
    model_tag,
    resolve_model,
)
from .segments import PSEUDO_UNKNOWN, Utterance


# 実装のバージョン。上げるとローカル転写のキャッシュを作り直す。
LOCAL_ASR_VER = 1


class LocalAsrUnavailable(AlignUnavailable):
    """ローカル転写ができない(部品が無い・モデルが無い)。

    利用者に何をすれば動くのかを伝えるための例外。呼び出し側はそのまま
    メッセージを見せてよい。原因はモデルやライブラリの不足なので、
    チャンクを変えても直らない = 途中で打ち切ってよい種類の失敗。
    """


@dataclass(frozen=True)
class ChunkResult:
    """1 チャンクを起こした結果。

    utterances と words は同じ 1 回の転写から取れる。別々に転写すると
    同じ音声に 2 通りの結果ができてしまうし、時間も倍かかる(設計書 §5.3)。
    時刻はどちらもチャンク先頭からの相対秒。
    """

    utterances: list[Utterance] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)
    duration: float = 0.0


def chunk_cache_path(
    cache_dir: Path | str,
    chunk_stem: str,
    *,
    fingerprint: str,
    chunk_seconds: float,
    model: str,
    model_dir: Optional[Path | str] = None,
) -> Optional[Path]:
    """ローカル転写のチャンクキャッシュの置き場(設計書 §7)。

    キーは「音声の指紋 + チャンク長 + モデルの素性 + 実装バージョン」。
    クラウド側(.cluster.<指紋>.c<秒>[.vb].txt)にモデルを足した形で、
    どれが欠けても別物の転写を使い回す事故になる。指紋が取れなかった音声は
    そもそもキャッシュしない(毎回取り直すほうが安全)。

    逐語フラグ(.vb)は付けない。ローカルに逐語モードは無いので、付けると
    同じ転写が 2 つできるだけになる(§5.4)。

    中身はテキストではなく JSON。行形式に落とすと時刻が秒に丸められる。
    """
    if not fingerprint:
        return None
    tag = model_tag(model, model_dir)
    return Path(cache_dir) / (
        f"{chunk_stem}.local.{fingerprint}.c{int(chunk_seconds)}"
        f".{tag}.v{LOCAL_ASR_VER}.json"
    )


def load_chunk(path: Optional[Path]) -> Optional[ChunkResult]:
    """キャッシュを読む。無い・壊れている・別バージョンなら None。"""
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("local_asr_ver", -1)) != LOCAL_ASR_VER:
            return None
        return ChunkResult(
            utterances=[
                Utterance(rel_start=float(u["rel_start"]),
                          rel_end=float(u["rel_end"]),
                          text=str(u.get("text", "")),
                          cluster=str(u.get("cluster", PSEUDO_UNKNOWN)))
                for u in data.get("utterances", [])
            ],
            words=[Word.from_dict(w) for w in data.get("words", [])],
            duration=float(data.get("duration", 0.0)),
        )
    except Exception:
        return None         # 壊れたキャッシュは無かったことにして取り直す


def save_chunk(path: Optional[Path], result: ChunkResult, *, model: str) -> None:
    """キャッシュを書く。書けなくても転写自体は続けられるので握りつぶす。"""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "local_asr_ver": LOCAL_ASR_VER,
            "model": model,
            "duration": result.duration,
            "utterances": [{
                "rel_start": u.rel_start,
                "rel_end": u.rel_end,
                "text": u.text,
                "cluster": u.cluster,
            } for u in result.utterances],
            # 単語時刻も残す。同じ 1 回の転写から取れているので、あとで
            # 自動点検が使い回せる(§5.3)。捨てると測り直しになる。
            "words": [w.to_dict() for w in result.words],
        }, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


class LocalTranscriber:
    """モデルを 1 回だけ読み込んで、チャンクを順に起こす。

    クラウド経路が `client = _make_client(api_key)` を 1 つ作って各チャンクに
    使い回すのと同じ形にしてある。チャンクごとに読み直すと、2 時間音声(8 チャンク)
    でモデルの読み込みだけを 8 回繰り返すことになる。
    """

    def __init__(self, model: str = DEFAULT_MODEL,
                 model_dir: Optional[Path | str] = None) -> None:
        self.model = model
        self.model_dir = model_dir
        # 置き場の間違い(フォルダが無い・model.bin が無い)はここで分かる。
        # 音声を分割し終わってから気づくのでは遅い。
        self.target = resolve_model(model, model_dir)
        self._whisper = None

    def _load(self, on_log=None):
        """初回の転写のときにモデルを読む(生成そのものは軽くしておく)。"""
        if self._whisper is not None:
            return self._whisper
        try:
            from faster_whisper import WhisperModel      # 重いのでここで読む
        except ImportError as e:
            raise LocalAsrUnavailable(
                "ローカル転写には faster-whisper が必要です。\n"
                "    pip install faster-whisper\n"
                "で導入してから、もう一度お試しください。"
            ) from e

        if on_log:
            on_log(f"ローカル転写の準備をしています(モデル {self.model} / CPU)。")
        try:
            self._whisper = WhisperModel(
                self.target, device=DEVICE, compute_type=COMPUTE_TYPE)
        except Exception as e:
            raise LocalAsrUnavailable(
                f"モデルを読み込めませんでした({self.model})。\n"
                "オンラインで取得できない環境では、別の PC で取得したモデル"
                "フォルダを指定してください。\n"
                f"--- 詳細 ---\n{e}"
            ) from e
        return self._whisper

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        on_log=None,
        on_progress=None,
        is_cancelled=None,
    ) -> Optional[ChunkResult]:
        """1 チャンクを起こす。中断されたら None(途中結果は残さない)。

        on_progress: (処理済み秒, チャンク全体の秒) で呼ぶ。
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise LocalAsrUnavailable(f"音声ファイルが見つかりません: {audio_path}")

        whisper = self._load(on_log)

        segments, info = whisper.transcribe(
            str(audio_path),
            language=LANGUAGE,
            # 本文と一緒に単語時刻も取る。自動点検が使う実測値がただで手に入る。
            word_timestamps=True,
            # VAD は使わない。無音を飛ばすと速くなるが、短い相づちを落とす
            # 恐れがある。ここは本文を作る経路なので、落ちればその発言は
            # 記録から消える(align.py の「照合不能」より重い)。
            vad_filter=False,
        )
        total = float(getattr(info, "duration", 0.0) or 0.0)

        utterances: list[Utterance] = []
        words: list[Word] = []
        for seg in segments:        # ここで初めて実際の転写が走る(遅延評価)
            if is_cancelled and is_cancelled():
                if on_log:
                    on_log("ローカル転写を中止しました。")
                return None

            for w in (getattr(seg, "words", None) or []):
                wt = (w.word or "").strip()
                if not wt:
                    continue
                words.append(Word(text=wt, start=float(w.start), end=float(w.end)))

            text = (seg.text or "").strip()
            # 中身が空の区間だけ落とす。短い相づち(「はい」等)は落とさない。
            # 同意の意思表示が記録から消えるため(CLAUDE.md の設計原則)。
            if text:
                start = float(seg.start)
                # まれに終わりが始まりを下回る。負の長さを作らないだけに留め、
                # 独自の下限を足したりはしない(閾値を増やすと較正が要る)。
                end = max(float(seg.end), start)
                utterances.append(Utterance(
                    rel_start=round(start, 3),
                    rel_end=round(end, 3),
                    text=text,
                    # 話者分離が入るまでは全区間が「判別不能」。誰の声かを
                    # 決める者がこの経路にはまだいない(設計書 §6)。
                    cluster=PSEUDO_UNKNOWN,
                ))

            if on_progress and total:
                on_progress(min(float(seg.end), total), total)

        if on_progress and total:
            on_progress(total, total)
        if on_log:
            on_log(f"  → {len(utterances)} 区間 / {len(words)} 語")
        return ChunkResult(utterances=utterances, words=words, duration=total)
