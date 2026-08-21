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
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from .align import (
    AlignUnavailable,
    COMPUTE_TYPE,
    DEFAULT_MODEL,
    DEVICE,
    LANGUAGE,
    Word,
    default_model,
    model_tag,
    pick_device,
    resolve_model,
)
from .segments import PSEUDO_UNKNOWN, Utterance


# 実装のバージョン。上げるとローカル転写のキャッシュを作り直す。
LOCAL_ASR_VER = 1

# **プロンプトの版。変えたら必ず上げること**(キャッシュキーに入る)。
# p1 = 句読点の例文 + この種の会議の用語(2026-08-20)。
PROMPT_VER = 1

# **large-v3 は句読点をほとんど付けない。**実測(2026-08-20・逐語正解 4 帯)で、
# 説明が続く帯では 1 つも付かなかった(1 万字あたり 正解 1138 に対し 0)。
# 区間が細かい(725 → 1298)ぶん断片が文の途中で終わるため。そのままだと
# Word 出力で「同じ話者の連続区間」を 1 段落にまとめたときに文が繋がる。
#
# 例文を与えると戻る(0 → 971)。**用語は固有名詞には効かない**ことも
# 測定済みだが(initial_prompt では同窓会 0/5 のまま)、誤字全体は下がり、
# 脱落と長さも改善するので併せて置く。
#
# 代償: 掛け合いの帯では読点が過剰になる(正解 142 に対し 1274)。意味は
# 壊れないので許容した。「建て替え」が 1 回落ちるのも確認済みで、
# これは［語句をまとめて直す...］で直せる範囲。
STYLE_PROMPT = (
    "本日は、お忙しい中お集まりいただき、ありがとうございます。"
    "それでは、お手元の資料に沿って、順にご説明いたします。"
    "この点につきまして、何かご質問がございましたら、お願いいたします。"
)


# **同じ本文が何区間続いたら「暴走」とみなすか。**
# whisper が同じ文を延々と繰り返し、その間の会話が丸ごと失われる既知の
# 事故。実データで large-v3 が 00:49:16 から **20 回**繰り返し、4 分ぶんが
# 消えた(実機の指摘・2026-08-20)。small では起きなかった。
#
# 3 にしたのは、本物の繰り返しを巻き込まないため。同じ音声で「はい。」が
# 3 回、「吉沢さん。」が 4 回続く箇所があり、**どちらも実際にそう言っている**。
# ここは検出であって削除ではないので、拾いすぎても起こし直すだけで済む。
LOOP_RUN_THRESHOLD = 3


def max_repeat_run(utterances: Sequence[Utterance]) -> int:
    """同じ本文が連続した最大の回数。

    **区間ごとに数えること。**本文を 1 本に繋いでから正規表現で探すと、
    区間をまたいだ繰り返しを拾えない(それで一度見逃した)。
    """
    best = run = 0
    prev = None
    for u in utterances:
        t = (u.text or "").strip()
        if t and t == prev:
            run += 1
        else:
            run, prev = 1, t
        best = max(best, run)
    return best


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


# **キャッシュには「結果を固定する」役目もある**(設計書 §9.5.7)。
# 同じ音声・同じ設定でも、**キャッシュを消して再実行すると本文が変わる**。
# faster-whisper が「うまく起こせていない」と判断した 30 秒を温度を上げて
# 引き直すためで、その復号は乱数を使う。実測(2026-08-20):
#   2 分の帯を 3 回     → 1 文字も違わない
#   7 分のチャンクを 2 回 → 1,793 字 / 1,708 字 と違う
#   引き直しを止めると    → 完全に同一
# **揺れるのは転写が苦しいところだけ。**引き直しは §9.5.6 の暴走から
# 抜けるための仕組みなので、止めない。


def chunk_cache_path(
    cache_dir: Path | str,
    chunk_stem: str,
    *,
    fingerprint: str,
    chunk_seconds: float,
    model: str,
    model_dir: Optional[Path | str] = None,
    compute_type: str = "",
    prompt_ver: int = 0,
) -> Optional[Path]:
    """ローカル転写のチャンクキャッシュの置き場(設計書 §7)。

    キーは「音声の指紋 + チャンク長 + モデルの素性 + **精度** +
    **プロンプトの版** + 実装バージョン」。クラウド側
    (.cluster.<指紋>.c<秒>[.vb].txt)にモデルを足した形で、どれが欠けても
    別物の転写を使い回す事故になる。指紋が取れなかった音声はそもそも
    キャッシュしない(毎回取り直すほうが安全)。

    **精度とプロンプトの版は 2026-08-20 に足した。**
    - `int8`(CPU) と `int8_float16`(GPU) では出力が変わりうる
    - 用語と句読点のプロンプトを変えると転写が変わる。実測で句読点の密度が
      1 万字あたり 452 → 1055 に動いた。版を入れないと古い転写を使い回す
      (CLAUDE.md「どれかを欠くと古い転写の使い回し事故になる」)

    逐語フラグ(.vb)は付けない。ローカルに逐語モードは無いので、付けると
    同じ転写が 2 つできるだけになる(§5.4)。

    中身はテキストではなく JSON。行形式に落とすと時刻が秒に丸められる。
    """
    if not fingerprint:
        return None
    tag = model_tag(model, model_dir)
    if compute_type:
        tag += "." + re.sub(r"[^A-Za-z0-9_]+", "-", compute_type)
    if prompt_ver:
        tag += f".p{int(prompt_ver)}"
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

    def __init__(self, model: Optional[str] = None,
                 model_dir: Optional[Path | str] = None,
                 condition_on_previous_text: bool = True,
                 prefer_gpu: bool = True,
                 style_prompt: Optional[str] = STYLE_PROMPT) -> None:
        """model を省くと、装置に合った既定を選ぶ(GPU なら large-v3)。

        condition_on_previous_text: 直前までの本文を次の 30 秒に引き継ぐか。

        **既定は True。faster-whisper の既定と同じで、これまでの挙動を変えない。**
        False にすると各 30 秒が独立に起きるので、言い淀みが整えられて落ちる
        度合いが変わる(段階 2 の測定で、同じ small でも渡す長さを変えると
        フィラー保持が 9.0% → 20.9% と動いた)。

        **ここを変えたまま pipeline に載せるなら、キャッシュキーに足すこと。**
        いまはキー(音声指紋 + チャンク長 + 逐語フラグ)に入っていないので、
        設定違いの転写が同じキーを共有してしまう。だから今は測定専用で、
        pipeline からは渡していない。
        """
        # **装置を先に決める。**モデルの既定が装置で変わる(GPU なら
        # large-v3、CPU なら small。CPU で large-v3 は実用外)。
        self.device, self.compute_type = pick_device(prefer_gpu)
        self.model = model or default_model(self.device)
        self.model_dir = model_dir
        self.condition_on_previous_text = condition_on_previous_text
        self.style_prompt = style_prompt or None
        self.prompt_ver = PROMPT_VER if self.style_prompt else 0
        # 置き場の間違い(フォルダが無い・model.bin が無い)はここで分かる。
        # 音声を分割し終わってから気づくのでは遅い。
        self.target = resolve_model(self.model, model_dir)
        self._whisper = None

    def ensure_available(self) -> None:
        """部品が揃っているかだけを先に確かめる(モデルは読まない)。

        音声を分割し終えてから「部品がありません」と言われるのでは遅い。
        2 時間の音声だと分割だけで数分かかる。
        """
        try:
            import faster_whisper       # noqa: F401
        except ImportError as e:
            raise LocalAsrUnavailable(
                "ローカル転写には faster-whisper が必要です。\n"
                f"いま動いている Python: {sys.executable}\n"
                "この Python に対して\n"
                "    pip install faster-whisper\n"
                "で導入するか、faster-whisper を入れてある環境から起動して"
                "ください。\n"
                f"--- 詳細 ---\n{e}"
            ) from e

    def _load(self, on_log=None):
        """初回の転写のときにモデルを読む(生成そのものは軽くしておく)。"""
        if self._whisper is not None:
            return self._whisper
        self.ensure_available()
        from faster_whisper import WhisperModel      # 重いのでここで読む

        where = "GPU" if self.device == "cuda" else "CPU"
        if on_log:
            on_log(f"ローカル転写の準備をしています"
                   f"(モデル {self.model} / {where})。")

        def build(device: str, ctype: str):
            m = WhisperModel(self.target, device=device, compute_type=ctype)
            # **試し撃ちをする。**CUDA のドライバがあれば device_count は 1 を
            # 返すが、cuBLAS の DLL が無いと**最初の推論で**落ちる(実機で発生・
            # 2026-08-20)。1 チャンク起こしたあとで気づくのでは遅いので、
            # ここで無音 1 秒を通して確かめる。
            if device == "cuda":
                import numpy as np
                list(m.transcribe(np.zeros(16000, dtype=np.float32),
                                  language=LANGUAGE,
                                  without_timestamps=True)[0])
            return m

        try:
            self._whisper = build(self.device, self.compute_type)
        except Exception as first:
            if self.device == "cuda":
                # **GPU が駄目でも止めない。**CPU に落とす。既定モデルも
                # 戻す——CPU で large-v3 は実時間比が数倍で実用外になる。
                if on_log:
                    on_log(f"GPU では動かせませんでした({first})。"
                           "CPU に切り替えます。")
                self.device, self.compute_type = DEVICE, COMPUTE_TYPE
                self.model = default_model(self.device)
                self.target = resolve_model(self.model, self.model_dir)
                try:
                    self._whisper = build(self.device, self.compute_type)
                    return self._whisper
                except Exception as e:
                    first = e
            raise LocalAsrUnavailable(
                f"モデルを読み込めませんでした({self.model})。\n"
                "オンラインで取得できない環境では、別の PC で取得したモデル"
                "フォルダを指定してください。\n"
                f"--- 詳細 ---\n{first}"
            ) from first
        return self._whisper

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        on_log=None,
        on_progress=None,
        is_cancelled=None,
        word_timestamps: bool = True,
    ) -> Optional[ChunkResult]:
        """1 チャンクを起こす。中断されたら None(途中結果は残さない)。

        on_progress: (処理済み秒, チャンク全体の秒) で呼ぶ。
        word_timestamps: 単語時刻も取るか。**製品経路は常に既定(True)。**
        False は測定用——CT2 変換が不完全なモデル(kotoba-whisper-v2.0-faster)は
        単語時刻の位置合わせでネイティブ層ごと落ちる(0xC0000005・実測)。
        本文だけの測定ならこれで回避できる。
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise LocalAsrUnavailable(f"音声ファイルが見つかりません: {audio_path}")

        result = self._once(audio_path, self.condition_on_previous_text,
                            on_log=on_log, on_progress=on_progress,
                            is_cancelled=is_cancelled,
                            word_timestamps=word_timestamps)
        if result is None:
            return None

        # **暴走していないか見る。**同じ本文が続くのは、その間の会話が
        # 丸ごと失われているということ。**黙って通してはいけない。**
        n = max_repeat_run(result.utterances)
        if n < LOOP_RUN_THRESHOLD or not self.condition_on_previous_text:
            return result

        if on_log:
            on_log(f"  ※ 同じ本文が {n} 区間続いています(転写の暴走)。"
                   "引き継ぎを切って起こし直します。")
        retry = self._once(audio_path, False, on_log=None,
                           on_progress=on_progress, is_cancelled=is_cancelled,
                           word_timestamps=word_timestamps)
        if retry is None:
            return None
        n2 = max_repeat_run(retry.utterances)
        before = sum(len(u.text) for u in result.utterances)
        after = sum(len(u.text) for u in retry.utterances)
        # **繰り返しが減っただけでは採らない。本文が減っていないことも見る。**
        # 引き継ぎを切ると、壊れていない箇所では脱落が増える（実測・2026-08-21:
        # 逐語正解 4 帯で脱落 10.4% → 16.6%、落とした発言の回収 18 → 10 件）。
        # 閾値ちょうど（3 区間）で発動したチャンクを起こし直したら、繰り返しは
        # 3 → 2 に減ったが**本文が 1,758 字 → 1,669 字に減った**。
        # 本当に壊れていた chunk_0007 は 1,288 字 → 1,723 字と**増えている**。
        # **増えたときだけ採る**なら、両方とも正しく分かれる。
        if n2 < n and after >= before:
            if on_log:
                on_log(f"  → 直りました({n} 区間 → {n2} 区間の繰り返し / "
                       f"本文 {before:,} 字 → {after:,} 字)。")
            return retry
        if n2 < n:
            # 繰り返しは減ったが本文も減った。**元を採る**——起こし直しの
            # ほうが失っているので、繰り返しを消すために本文を捨てることになる。
            if on_log:
                on_log(f"  → 起こし直すと本文が減る"
                       f"({before:,} 字 → {after:,} 字)ので、元のままにします"
                       f"(繰り返し {n} 区間)。本物の相づちかもしれません。")
            return result
        # **直らなかったことを必ず伝える。**この事故は本文が消えるので、
        # 気づかないまま納品されるのがいちばん悪い。
        if on_log:
            on_log(f"  ※ 起こし直しても直りませんでした({n2} 区間)。"
                   "この付近は本文が失われている可能性があります。"
                   "音声を聴いて確かめてください。")
        return result

    def _once(
        self,
        audio_path: Path,
        condition_on_previous_text: bool,
        *,
        on_log=None,
        on_progress=None,
        is_cancelled=None,
        word_timestamps: bool = True,
    ) -> Optional[ChunkResult]:
        """1 回だけ起こす。**暴走の判定はしない**(呼び出し側が見る)。"""
        whisper = self._load(on_log)

        segments, info = whisper.transcribe(
            str(audio_path),
            language=LANGUAGE,
            # 本文と一緒に単語時刻も取る。自動点検が使う実測値がただで手に入る。
            word_timestamps=word_timestamps,
            # VAD は使わない。無音を飛ばすと速くなるが、短い相づちを落とす
            # 恐れがある。ここは本文を作る経路なので、落ちればその発言は
            # 記録から消える(align.py の「照合不能」より重い)。
            vad_filter=False,
            condition_on_previous_text=condition_on_previous_text,
            # **句読点のための例文**(STYLE_PROMPT の注記)。変えたら
            # PROMPT_VER を上げること——キャッシュキーに入っている。
            initial_prompt=self.style_prompt,
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
