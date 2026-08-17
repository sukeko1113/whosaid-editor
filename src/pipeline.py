"""分割 → 文字起こし → docx 結合 のパイプライン全体を制御するモジュール。

GUI から別スレッドで run_pipeline() を呼ぶ想定。
進捗とログはコールバック関数経由で UI スレッドへ通知する。
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from google import genai
from google.genai import types as genai_types

from . import diarize as diarize_mod
from . import listen_order
from . import local_asr
from .align import DEFAULT_MODEL as DEFAULT_LOCAL_MODEL
from .audio import audio_fingerprint, audio_hashes, probe_duration, split_audio
from .config import APP_VERSION
from .segments import (
    ENGINE_CLOUD,
    ENGINE_LABELS,
    ENGINE_LOCAL,
    Project,
    Segment,
    Speaker,
    Utterance,
    assign_speaker_clusters,
    fmt_hms,
    merge_same_speaker,
    parse_roster,
    utterances_to_segments,
)
from .transcribe import (
    DIARIZATION_NOTE,
    ROSTER_NOTE,
    CancelledError,
    FatalTranscriptionError,
    parse_segments,
    parse_utterances,
    shift_timestamps,
    transcribe_audio,
    write_docx,
)


LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]   # (current, total)
CancelFn = Callable[[], bool]


# 1回の HTTP リクエストの上限(ミリ秒)。
# これが無いと、通信が滞留したとき応答を無限に待ち続けて
# 処理が「止まったように見える」問題が起きる(2026-07 実戦投入で確認)。
# タイムアウトすると例外になり、transcribe_audio 側の再試行に乗る。
REQUEST_TIMEOUT_MS = 8 * 60 * 1000  # 8分

# クラウド経路の既定モデル(gui.MODELS の先頭と揃える)
DEFAULT_CLOUD_MODEL = "gemini-2.5-flash"


@dataclass(frozen=True)
class EngineSpec:
    """どのエンジンで転写するか。

    既定はローカル。クラウドは利用者が明示的に選んだときだけ使う
    (録音を外へ出す判断を既定にしない。事業計画 4.3)。

    api_key はクラウドのときだけ要る。ローカルは端末内で完結するので、
    キーの入力を求めない(求めると「ローカルなのに鍵が要る」ことになる)。
    """

    mode: str = ENGINE_LOCAL
    model: str = ""
    api_key: str = ""
    model_dir: Optional[str] = None
    # 話者分離(ローカル経路のみ)。クラウドは Gemini が A/B/C を出すので当てない
    # (話者分離設計書 §10)。
    diarize: bool = True
    # 話者数の上限。0 なら名簿の人数を使う。名簿も空なら既定値。
    # **正解の話者数である必要はない**——上限を与えて過剰分割を防ぐのが目的。
    num_speakers: int = 0

    @property
    def is_local(self) -> bool:
        return self.mode == ENGINE_LOCAL

    @property
    def wants_diarize(self) -> bool:
        return self.is_local and self.diarize

    @property
    def label(self) -> str:
        return ENGINE_LABELS.get(self.mode, self.mode)

    def resolved_model(self) -> str:
        """モデル名。指定が無ければ経路ごとの既定。"""
        if self.model:
            return self.model
        return DEFAULT_LOCAL_MODEL if self.is_local else DEFAULT_CLOUD_MODEL

    def record(self) -> dict:
        """作業ファイルに残す engine の中身(Word の検証要約に出る処理経路)。

        第三者が「どの経路のどのモデルで作られた記録か」を読めるようにする。
        """
        rec = {
            "mode": self.mode,
            "model": self.resolved_model(),
            "app_version": APP_VERSION,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if self.is_local:
            rec["compute_type"] = local_asr.COMPUTE_TYPE
            if self.model_dir:
                rec["model_dir"] = str(self.model_dir)
        return rec


def _make_client(api_key: str) -> genai.Client:
    """タイムアウトを設定した Gemini クライアントを作る。

    Client の生成箇所が複数あるので、設定漏れを防ぐために関数にしてある。
    """
    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


def _unique_path(base: Path) -> Path:
    """同名ファイルが既にあれば '<name> (1).docx' のように退避する"""
    if not base.exists():
        return base
    stem, suffix, parent = base.stem, base.suffix, base.parent
    i = 1
    while True:
        cand = parent / f"{stem} ({i}){suffix}"
        if not cand.exists():
            return cand
        i += 1


# 逐語プロンプトの版。**上げるとクラウドの逐語キャッシュを作り直す。**
# 1: v1.3.0 以来。相づちを前の行に含めてよい／区切りは話者交代のみ
# 2: 相づちを独立した行にし、同じ話者でも 1 行 20 秒以内に分ける(2026-08-16)
VERBATIM_PROMPT_VER = 2


def _cache_suffix(
    with_timestamps: bool,
    with_diarization: bool,
    verbatim: bool,
    roster: str,
    chunk_seconds: int = 0,
    fingerprint: str = "",
) -> str:
    """設定の組み合わせごとに別キャッシュにする(混在を防ぐ)。

    v1.3.0: 逐語モードと名簿の内容もキャッシュキーに含める。
    名簿を書き換えたのに古い結果が再利用される事故を防ぐため、
    名簿本文のハッシュ(先頭8桁)をサフィックスに埋め込む。

    v2.0.0: チャンク長と音声の指紋も含める。チャンクのファイル名は長さに
    よらず chunk_0000.m4a なので、含めないと別の長さ・別の中身の転写を
    使い回してしまう(音声を編集してもファイル名が同じなら再利用される)。
    """
    parts: list[str] = []
    if fingerprint:
        parts.append(fingerprint)
    if chunk_seconds:
        parts.append(f"c{chunk_seconds}")
    if with_diarization:
        parts.append("diar")
    elif with_timestamps:
        parts.append("ts")
    if verbatim:
        # **プロンプトの版も入れる。**逐語の指示を変えると同じ音声・同じ設定でも
        # 別物の転写になる。版を入れないと、古い結果を使い回してしまう
        # (CLAUDE.md「どれかを欠くと古い転写の使い回し事故になる」)。
        # p2 = 相づちを独立した行にし、1 行 20 秒以内に分ける(2026-08-16)。
        parts.append(f"vb{VERBATIM_PROMPT_VER}")
    if with_diarization and roster.strip():
        h = hashlib.md5(roster.strip().encode("utf-8")).hexdigest()[:8]
        parts.append(h)
    if not parts:
        return ".txt"
    return "." + ".".join(parts) + ".txt"


def run_pipeline(
    audio_path: Path,
    output_dir: Path,
    api_key: str,
    model: str,
    chunk_minutes: int,
    on_log: LogFn,
    on_progress: ProgressFn,
    is_cancelled: CancelFn,
    with_timestamps: bool = False,
    with_diarization: bool = False,
    roster: str = "",
    verbatim: bool = False,
    force_retranscribe: bool = False,
) -> Optional[Path]:
    """音声ファイル → docx を生成。キャンセル時は None を返す。"""
    # 話者識別が ON の場合、タイムスタンプも自動的に ON にする
    if with_diarization:
        with_timestamps = True

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / f".work_{audio_path.stem}"
    chunks_dir = work_dir / "chunks"
    cache_dir = work_dir / "transcripts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_path = _unique_path(output_dir / f"{audio_path.stem}.docx")

    on_log(f"出力先: {output_path}")
    if with_diarization:
        if roster.strip():
            n = len([ln for ln in roster.strip().splitlines() if ln.strip()])
            on_log(f"話者識別: 有効(参加者名簿 {n} 行を使用)")
        else:
            on_log("話者識別: 有効(名簿なし・発言者A/B/C方式)")
    elif with_timestamps:
        on_log("タイムスタンプ付き出力: 有効")
    if verbatim:
        on_log("逐語モード: 有効(フィラー・言い直しを保持し、整文しません)")
    on_log(f"音声を {chunk_minutes} 分単位で分割します...")
    chunks = split_audio(audio_path, chunks_dir, chunk_minutes * 60)
    on_log(f"{len(chunks)} 個のチャンクに分割しました。")

    if is_cancelled():
        on_log("キャンセルされました。")
        return None

    client = _make_client(api_key)
    transcripts: list[str] = []
    on_progress(0, len(chunks))

    chunk_seconds = chunk_minutes * 60
    cache_suffix = _cache_suffix(
        with_timestamps, with_diarization, verbatim, roster, chunk_seconds,
        audio_fingerprint(audio_path))

    # docx 冒頭の注意書きの選択
    note: str | None = None
    if with_diarization:
        note = ROSTER_NOTE if roster.strip() else DIARIZATION_NOTE

    for i, chunk in enumerate(chunks):
        if is_cancelled():
            on_log("キャンセルされました。")
            return None

        cache_path = cache_dir / f"{chunk.stem}{cache_suffix}"
        label = f"[{i + 1}/{len(chunks)}] {chunk.name}"
        offset = i * chunk_seconds

        if cache_path.exists() and not force_retranscribe:
            on_log(f"{label} (キャッシュから復元)")
            text = cache_path.read_text(encoding="utf-8")
        else:
            on_log(f"{label} 文字起こし中...")
            try:
                raw = transcribe_audio(
                    client, chunk, model,
                    with_timestamps=with_timestamps,
                    with_diarization=with_diarization,
                    roster=roster,
                    verbatim=verbatim,
                    on_log=on_log,
                    is_cancelled=is_cancelled,
                )
                # チャンク内相対時刻 [MM:SS] を絶対時刻 [HH:MM:SS] に変換
                text = shift_timestamps(raw, offset) if with_timestamps else raw
                cache_path.write_text(text, encoding="utf-8")
            except CancelledError:
                on_log("キャンセルされました。(完了済みチャンクはキャッシュに保存されています)")
                return None
            except FatalTranscriptionError:
                raise      # 残高切れ・キー不正は続けても同じ結果になる
            except Exception as e:
                on_log(f"  失敗: {e}")
                text = f"【文字起こし失敗: {chunk.name} - {e}】"

        transcripts.append(text)
        # 都度保存(途中で落ちてもここまでは残る)
        write_docx(transcripts, output_path, audio_path.stem, note=note)
        on_progress(i + 1, len(chunks))

    on_log(f"完了: {output_path.name}")
    return output_path


# ======================================================================
# v2.0.0: 手動割当モード用パイプライン
#
#   分割 → (声質だけの)話者クラスタ付き文字起こし → セグメント JSON
#
# ここでは話者の実名を一切推定しない。名簿は「候補者リスト」として
# JSON に持たせるだけで、Gemini には渡さない(その分だけ速く・安定する)。
# 実名の確定はこのあと GUI(assign_gui)でユーザーが行う。
# ======================================================================

MATCH_TOLERANCE_SECONDS = 2.0

# 指紋が無い(古い形式の)作業ファイルについて、
# 「別の音声に差し替わった」と判断する継続時間の差
DURATION_MISMATCH_SECONDS = 1.0


def _backup_stale_project(json_path: Path, on_log: LogFn) -> None:
    """中身の違う音声の作業ファイルを退避する。上書きで消さない。"""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup = json_path.with_name(f"{json_path.stem}.{stamp}.bak.json")
    try:
        json_path.replace(backup)
        on_log(f"以前の作業ファイルは {backup.name} に退避しました。")
    except OSError as e:
        on_log(f"以前の作業ファイルを退避できませんでした({e})。上書きします。")


def _is_same_audio(old: Project, fingerprint: str, duration: float) -> bool:
    """作業ファイルが、いま処理している音声と同じ中身のものか。

    指紋があれば指紋で判定する。無い(古い形式の)場合は継続時間で代用する。
    """
    if old.audio_fingerprint and fingerprint:
        return old.audio_fingerprint == fingerprint
    if old.duration and duration:
        return abs(old.duration - duration) <= DURATION_MISMATCH_SECONDS
    return True     # どちらも判定材料が無いときは従来どおり引き継ぐ


def _merge_speakers(old_speakers: list[Speaker], roster: str) -> list[Speaker]:
    """既存の話者 ID を保ったまま、名簿テキストの内容を反映する。

    ID を振り直してはいけない。segment.speaker_id は ID を指しているので、
    振り直すと「以前の割当」が別人を指してしまう(名簿の並びを変えただけで
    全員の名前が入れ替わる、という事故になる)。
    """
    wanted = parse_roster(roster)
    if not wanted:
        return list(old_speakers)

    remaining: dict[str, list[Speaker]] = {}
    for sp in old_speakers:
        remaining.setdefault(sp.name, []).append(sp)

    used_ids = {sp.id for sp in old_speakers}
    result: list[Speaker] = []

    for w in wanted:
        pool = remaining.get(w.name)
        if pool:
            src = pool.pop(0)                  # 同名が複数いても 1 人ずつ対応付ける
            src.note = w.note or src.note
            result.append(src)
        else:
            i = 1
            while f"sp{i:02d}" in used_ids:
                i += 1
            sid = f"sp{i:02d}"
            used_ids.add(sid)
            result.append(Speaker(id=sid, name=w.name, note=w.note))

    # 名簿から消えた人も、割当が残っているかもしれないので保持する
    for pool in remaining.values():
        result.extend(pool)

    for i, sp in enumerate(result):
        sp.order = i
    assert len({sp.id for sp in result}) == len(result), "話者 ID が重複しています"
    return result


def _carry_over_assignments(
    old: Project, new_segments: list[Segment], on_log: LogFn
) -> list[Segment]:
    """再実行時に、以前の割当・本文修正・時刻修正を新しいセグメントへ引き継ぐ。

    突き合わせの鍵は orig_start(パイプラインが出した元の時刻)を使う。
    ユーザーが直したあとの start で照合すると、時刻のずれを数秒ぶん直した区間が
    再実行のたびに迷子になる。orig_start はユーザーが何をしても動かないので、
    キャッシュが効いていれば再生成側と完全に一致する。
    本文は編集されている可能性があるので鍵に使わない。
    話者 ID は _merge_speakers が ID を保存するので、そのまま移してよい。

    戻り値は新しい区間リスト。分割・結合を復元するために区間が増減するので、
    呼び出し側はこの戻り値で置き換えること。
    """
    # **人が足した区間は突き合わせから完全に外す**(相づちを足す設計書 §4)。
    # これらの orig_start は独自の値なので、
    #   - 近くの再生成区間に誤って照合され、time_edited があればその区間を
    #     置き換えて消す(1 秒の相づちが 8 秒の発言を上書きする)
    #   - どこにも当たらなければ、相づちのほうが黙って捨てられる(実測はこちら)
    # どちらにせよ人の作業が消える。照合にも吸収範囲にも参加させず、
    # 最後に時間順の位置へ挿し直す。
    added_keys = old.added_utterance_keys()
    added = [s for s in old.segments
             if round(float(s.orig_start if s.orig_start is not None
                            else s.start), 3) in added_keys]

    kept = [s for s in old.segments
            if (s.speaker_id or s.text_edited or s.time_edited)
            and s not in added]
    if not kept and not added:
        return new_segments

    # 同じ orig_start を共有する旧区間は、1 つの区間を分割した兄弟(ファミリー)。
    # まとめて 1 つの再生成区間に対応付ける。
    families: dict[float, list[Segment]] = {}
    for s in kept:
        families.setdefault(round(float(s.orig_start), 3), []).append(s)
    pool = [
        sorted(fam, key=lambda s: (s.start, s.index))
        for _, fam in sorted(families.items())
    ]
    used: set[int] = set()
    matched: dict[int, list[Segment]] = {}      # new_segments の位置 → ファミリー

    for pos, seg in enumerate(new_segments):
        best_i = -1
        best_gap = MATCH_TOLERANCE_SECONDS + 1e-9
        for i, fam in enumerate(pool):
            if i in used:
                continue
            gap = abs(float(fam[0].orig_start) - seg.start)
            if gap < best_gap:
                best_i, best_gap = i, gap
        if best_i >= 0:
            used.add(best_i)
            matched[pos] = pool[best_i]

    # 結合して 1 つにした区間が、再実行で再び 2 つに戻るのを防ぐ。
    # 判定は再生成区間を差し替える「前」に済ませる。差し替えて入る旧区間の
    # start はユーザーが直した実時刻なので、パイプライン時刻と混ざると誤判定する。
    # 範囲は照合できたファミリーのぶんだけ使う(照合できなかった旧区間の範囲まで
    # 使うと、旧区間は入らないのに再生成区間だけ消えて穴が開く)。
    absorb = [
        (float(s.orig_start), float(s.orig_end))
        for fam in matched.values()
        for s in fam
        if s.time_edited
    ]

    result: list[Segment] = []
    carried = carried_text = replaced = 0
    absorbed: list[float] = []

    for pos, seg in enumerate(new_segments):
        fam = matched.get(pos)
        if fam is None:
            if any(lo < seg.start < hi for lo, hi in absorb):
                absorbed.append(seg.start)
                continue
            result.append(seg)
            continue

        if len(fam) == 1 and not fam[0].time_edited:
            # 時刻に手が入っていない区間。再生成側の時刻改善を活かし、
            # 人が入れた情報だけを移す(従来どおりの動作)。
            src = fam[0]
            if src.speaker_id:
                seg.speaker_id = src.speaker_id
                seg.reviewed = src.reviewed
                carried += 1
            seg.note = src.note
            if src.text_edited:
                seg.text = src.text          # ユーザーの手直しを潰さない
                seg.text_edited = True
                carried_text += 1
            result.append(seg)
        else:
            # 時刻を直した、または分割した区間。時刻も本文も構造も旧側が正しい。
            result.extend(fam)
            replaced += len(fam)
            carried += sum(1 for s in fam if s.speaker_id)
            carried_text += sum(1 for s in fam if s.text_edited)

    # 人が足した区間を時間順の位置へ挿し直す。照合には参加していないので
    # 再生成区間を置き換えることも、消されることもない。
    if added:
        result.extend(added)
        result.sort(key=lambda s: (s.start, s.end))

    for n, seg in enumerate(result):
        seg.index = n

    if added:
        on_th = "、".join(fmt_hms(s.start) for s in added[:5])
        more = " ほか" if len(added) > 5 else ""
        on_log(f"人が足した発話 {len(added)} 件をそのまま残しました"
               f"({on_th}{more})。")
    if carried:
        on_log(f"以前の割当 {carried} 区間を引き継ぎました。")
    if carried_text:
        on_log(f"手直しした本文 {carried_text} 区間を復元しました。")
    if replaced:
        on_log(f"時刻を直した・分割した {replaced} 区間はそのまま残しました。")
    if absorbed:
        heads = "、".join(fmt_hms(t) for t in absorbed[:5])
        more = " ほか" if len(absorbed) > 5 else ""
        on_log(
            f"結合済みの区間に重なる再生成区間 {len(absorbed)} 個を取り込みました"
            f"({heads}{more})。"
        )
    return result


def run_segment_pipeline(
    audio_path: Path,
    output_dir: Path,
    engine: EngineSpec,
    chunk_minutes: int,
    on_log: LogFn,
    on_progress: ProgressFn,
    is_cancelled: CancelFn,
    verbatim: bool = False,
    roster: str = "",
    force_retranscribe: bool = False,
) -> Optional[Project]:
    """音声 → セグメント JSON(話者未確定)を生成して Project を返す。

    engine で経路を選ぶ。既定はローカル(端末内で完結し、録音も本文も
    外へ出さない)。クラウドは利用者が明示的に選んだときだけ。

    force_retranscribe=True なら、キャッシュを無視して必ず転写し直す。
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / f".work_{audio_path.stem}"
    chunks_dir = work_dir / "chunks"
    cache_dir = work_dir / "transcripts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    json_path = Project.default_json_path(output_dir, audio_path)
    model = engine.resolved_model()

    on_log(f"作業ファイル: {json_path}")
    on_log(f"処理経路: {engine.label}(モデル {model})")
    if engine.is_local:
        on_log("ローカル転写は端末内で完結します。録音も本文も外へ出しません。")
        if engine.diarize:
            on_log("転写のあと、声のまとまりを端末内で分けます"
                   "(会議全体で 1 つのまとまりになるので、割当がまとめて行えます)。")
        else:
            on_log("※ 話者分離は使いません。全区間が未判別として取り込まれ、"
                   "割当画面で 1 件ずつ確定することになります。")
        if verbatim:
            # 逐語は書式の指定ではなくプロンプトの指示。whisper に対応する
            # 指示は無いので、選べるように見せない(設計書 §5.4)。
            on_log("※ 逐語モードはローカルでは指定できません(書式を指示する"
                   "仕組みがありません)。相づち・言い淀みは脱落することがあります。")
    else:
        on_log("方式: 話者は A/B/C… に分けるだけで、実名は後の割当画面で確定します。")
        if verbatim:
            on_log("逐語モード: 有効(フィラー・言い直しを保持し、整文しません)")
    if force_retranscribe:
        on_log("キャッシュを使わず、最初から転写し直します。")

    # エンジンが使える状態かを、重い処理に入る前に確かめる。
    # 370MB の音声だと、ハッシュと分割だけで数分かかる。それを終えてから
    # 「部品がありません」と言われるのでは、待った時間がまるごと無駄になる。
    transcriber = None
    if engine.is_local:
        transcriber = local_asr.LocalTranscriber(
            model=model, model_dir=engine.model_dir)
        transcriber.ensure_available()

    # 話者分離も先に確かめる。ただし**使えなくても止めない**——転写は 25 分
    # かかる。それを終えてから「話者分離の部品がありません」で全部を捨てるのは
    # 割に合わない。まとまりの無い状態でも作業はできる(1 件ずつになるだけ)。
    do_diarize = engine.wants_diarize
    if do_diarize:
        try:
            diarize_mod.ensure_available()
        except diarize_mod.DiarizeUnavailable as e:
            do_diarize = False
            on_log("※ 話者分離は使えません。転写だけ行います"
                   "(声のまとまりが付かないため、割当は 1 件ずつになります)。")
            on_log(f"  理由: {str(e).splitlines()[0]}")

    duration = probe_duration(audio_path)
    if duration:
        on_log(f"音声長: {int(duration // 60)}分{int(duration % 60)}秒")

    # 音声の中身から指紋(キャッシュ同一性)と SHA-256(第三者の検算用)を、
    # 1 回の読みで同時に取る。ファイル名が同じでも中身が変わっていれば
    # 別物として扱い、古い転写や割当を引き継がない。
    on_log("音声の内容を確認しています...")
    fingerprint, source_sha = audio_hashes(audio_path)
    if fingerprint:
        on_log(f"音声の指紋: {fingerprint}")
    if source_sha:
        on_log(f"元音声の SHA-256: {source_sha}")

    on_log(f"音声を {chunk_minutes} 分単位で分割します...")
    chunks = split_audio(audio_path, chunks_dir, chunk_minutes * 60)
    on_log(f"{len(chunks)} 個のチャンクに分割しました。")

    if is_cancelled():
        on_log("キャンセルされました。")
        return None

    # 各チャンクの実際の長さを測り、開始位置を積み上げる。
    # 「i × チャンク長」で決め打つと、分割の端数(AAC のフレーム境界)が
    # 積み上がって後半ほど時刻がずれる。ずれ自体は 0.1 秒未満だが、
    # 測るのは一瞬なので決め打ちにする理由がない。
    chunk_starts: list[float] = []
    chunk_lengths: list[float] = []
    acc = 0.0
    for c in chunks:
        chunk_starts.append(acc)
        d = probe_duration(c) or float(chunk_minutes * 60)
        chunk_lengths.append(d)
        acc += d

    client = None if engine.is_local else _make_client(engine.api_key)
    on_progress(0, len(chunks))

    chunk_seconds = chunk_minutes * 60
    # キャッシュ名には「音声の指紋」と「チャンク長」を含める。
    #   - 指紋が無いと、音声を編集してもファイル名が同じなら古い転写を再利用する
    #   - チャンク長が無いと、分割サイズを変えたとき音声とテキストがずれる
    cache_suffix = (
        f".cluster{'.' + fingerprint if fingerprint else ''}"
        f".c{chunk_seconds}{'.vb' if verbatim else ''}.txt"
    )

    all_segments: list[Segment] = []
    failed_chunks = 0
    last_failure: str = ""
    for i, chunk in enumerate(chunks):
        if is_cancelled():
            on_log("キャンセルされました。")
            return None

        label = f"[{i + 1}/{len(chunks)}] {chunk.name}"
        offset = chunk_starts[i]

        # このチャンクの実際の長さ(最終チャンクは短い)
        this_len = chunk_lengths[i]
        if duration:
            this_len = max(1.0, min(this_len, duration - offset))

        if engine.is_local:
            cache_path = local_asr.chunk_cache_path(
                cache_dir, chunk.stem, fingerprint=fingerprint,
                chunk_seconds=chunk_seconds, model=model,
                model_dir=engine.model_dir)
            cached = None if force_retranscribe else local_asr.load_chunk(cache_path)
            if cached is not None:
                on_log(f"{label} (キャッシュから復元)")
                utterances = cached.utterances
            else:
                on_log(f"{label} 文字起こし中...")
                result = transcriber.transcribe(
                    chunk, on_log=on_log, is_cancelled=is_cancelled)
                if result is None:      # 中断(途中結果は残さない)
                    on_log("キャンセルされました。"
                           "(完了済みチャンクはキャッシュに保存されています)")
                    return None
                local_asr.save_chunk(cache_path, result, model=model)
                utterances = result.utterances
        else:
            cache_path = cache_dir / f"{chunk.stem}{cache_suffix}"
            if cache_path.exists() and not force_retranscribe:
                on_log(f"{label} (キャッシュから復元)")
                raw_text = cache_path.read_text(encoding="utf-8")
            else:
                on_log(f"{label} 文字起こし中...")
                try:
                    raw_text = transcribe_audio(
                        client, chunk, model,
                        with_timestamps=True,
                        with_diarization=True,
                        roster="",
                        verbatim=verbatim,
                        on_log=on_log,
                        cluster_only=True,
                        is_cancelled=is_cancelled,
                    )
                    cache_path.write_text(raw_text, encoding="utf-8")
                except CancelledError:
                    on_log("キャンセルされました。"
                           "(完了済みチャンクはキャッシュに保存されています)")
                    return None
                except FatalTranscriptionError:
                    # 残高切れ・キー不正。続けても全チャンク同じ結果になるので、
                    # 中途半端な結果を作らずにここで止める。
                    raise
                except Exception as e:
                    on_log(f"  失敗: {e}")
                    failed_chunks += 1
                    last_failure = str(e)
                    raw_text = f"[00:00] 【?】 【文字起こし失敗: {chunk.name}】"

            # 逐語では連結の上限を短くする。**渡さないと、同じ人が話し続ける
            # 限り連結が進み、割り当てられない長さの区間ができる。**
            utterances = parse_utterances(raw_text, this_len, verbatim=verbatim)

        segs = utterances_to_segments(
            utterances,
            chunk_index=i,
            offset_seconds=offset,
            start_index=len(all_segments),
        )
        all_segments.extend(segs)
        on_log(f"  → {len(segs)} 区間")
        on_progress(i + 1, len(chunks))

    if not all_segments:
        raise RuntimeError("発言区間が 1 つも取得できませんでした。音声とモデル設定を確認してください。")

    if failed_chunks == len(chunks):
        # 全滅。割当画面を開いてもエラー文言が並ぶだけなので、開かせない。
        raise RuntimeError(
            f"すべてのチャンク({len(chunks)}個)で文字起こしに失敗しました。\n"
            "割当画面は開きません。原因を解消してから実行し直してください。\n\n"
            f"最後のエラー:\n{last_failure}"
        )
    if failed_chunks:
        on_log(
            f"※ {len(chunks)} 個中 {failed_chunks} 個のチャンクで文字起こしに失敗しました。"
            "該当区間は【文字起こし失敗】と表示されます。"
            "原因を解消して再実行すると、失敗したぶんだけ取得し直します。"
        )

    # ---- 話者分離(ローカル経路のみ・話者分離設計書 §4・§8) ----------------
    # 全長で 1 回走らせる。チャンクをまたいで同じ人を同じまとまりにできるのが
    # 要点で、Gemini のチャンク内クラスタ(実測 70 個)が 6 個になる。
    roster_speakers = parse_roster(roster)
    diarize_record: Optional[dict] = None
    turns = None                      # 聴く順番の計算にも使う(下の §listen)
    if do_diarize and all_segments:
        n_speakers = (engine.num_speakers or len(roster_speakers)
                      or diarize_mod.DEFAULT_NUM_SPEAKERS)
        try:
            turns = diarize_mod.diarize(
                audio_path,
                num_speakers=n_speakers,
                work_dir=work_dir,
                model_dir=engine.model_dir,
                fingerprint=fingerprint,
                on_log=on_log,
                on_progress=lambda d, t: on_progress(int(d), int(t) or 1),
                is_cancelled=is_cancelled,
                force=force_retranscribe,
            )
        except diarize_mod.DiarizeUnavailable as e:
            # ここまで来て落ちても転写は捨てない(上と同じ理由)
            on_log(f"※ 話者分離に失敗しました。転写はそのまま使えます: {e}")
            turns = None
        if turns is None and is_cancelled():
            on_log("キャンセルされました。")
            return None
        if turns:
            all_segments = sorted(all_segments, key=lambda s: (s.start, s.index))
            for seg, cluster in zip(
                    all_segments, assign_speaker_clusters(all_segments, turns)):
                seg.cluster = cluster
            before = len(all_segments)
            all_segments = merge_same_speaker(all_segments)
            found = len({s.cluster for s in all_segments
                         if not s.is_pseudo_cluster})
            on_log(f"声のまとまりを {found} 種類に整理しました"
                   f"(区間 {before} → {len(all_segments)})。")
            diarize_record = {
                "model": "pyannote-segmentation-3.0 + nemo-titanet-small",
                "num_speakers": n_speakers,
            }

    # 通し番号を振り直す
    for n, seg in enumerate(all_segments):
        seg.index = n

    # §listen: 聴く順番を出す。**必ず番号を振り直したあとで計算する**
    # ——sidecar は index で区間を指すので、先に計算すると全部ずれる。
    #
    # 話者分離が無ければ出さない(手がかりが turns なので)。失敗しても
    # 転写は捨てない。順番が無くても作業は成立する。
    listen_path = listen_order.hints_path(work_dir, fingerprint)
    if turns and all_segments:
        try:
            hints = listen_order.score_segments(all_segments, turns)
            listen_order.save_hints(listen_path, hints)
            high, total = listen_order.coverage(hints)
            on_log(f"聴く順番を付けました(手がかりの強い区間 {high}/{total})。"
                   "上位から聴くと取りこぼしを見つけやすくなります"
                   "(印の無い区間にも取りこぼしはあります)。")
        except Exception as e:                    # 順番が出せなくても続ける
            on_log(f"※ 聴く順番は出せませんでした: {e}")
    elif listen_path is not None and listen_path.exists():
        # **前回の順番を残さない。**話者分離を切って作り直したのに古い順番が
        # 残ると、いまの区間と対応しない案内が出る。作り直せる派生データ
        # なので消してよい(話者分離を入れ直せば戻る)。
        try:
            listen_path.unlink()
            on_log("話者分離が無いので、前回の「聴く順番」は消しました。")
        except OSError:
            pass

    speakers = roster_speakers
    # 再実行しても文書の履歴(版・編集履歴)は消さない。ただし音声の中身が
    # 変わっていた場合は別の文書の系譜なので引き継がない。
    carried_revision = 0
    carried_log: list[dict] = []
    if json_path.exists():
        try:
            old = Project.load(json_path)
            # 経路の記録が無い作業ファイル(schema 4 以前)はクラウド製。
            # ローカル転写はまだ存在しなかったので、そう読んで間違いない。
            old_mode = str((old.engine or {}).get("mode") or ENGINE_CLOUD)
            if not _is_same_audio(old, fingerprint, duration):
                # 同じファイル名でも中身が違う。前回の割当を引き継ぐと
                # 別の音声の話者が乗ってしまうので、作り直す。
                on_log(
                    "同じ名前の作業ファイルがありますが、音声の内容が変わっています。"
                    "前回の割当は引き継がず、新しく作り直します。"
                )
                _backup_stale_project(json_path, on_log)
                # 出席者(候補者リスト)だけは引き継ぐ。録り直しでも顔ぶれは
                # 同じことが多く、入力し直す手間だけが増えるため。
                # 区間の割当は引き継がない。
                speakers = _merge_speakers(old.speakers, roster)
            elif old_mode != engine.mode:
                # 経路が変われば区間の切れ目も本文も別物になる。それでも
                # _carry_over_assignments は orig_start が 2 秒以内なら
                # 対応づけてしまい、人が聴いて確定した ✓ を、その人が一度も
                # 見ていない区間へ移してしまう(設計書 §8.1)。
                # engine は作業ファイルに 1 つしか持てないので、本文が
                # 混ざると検証要約の処理経路も実態と食い違う。
                on_log(
                    f"前回は{ENGINE_LABELS.get(old_mode, old_mode)}転写の"
                    f"作業ファイルです。{engine.label}では区間の切れ目が"
                    "変わるため、前回の割当・本文修正・時刻修正は引き継ぎません。"
                )
                _backup_stale_project(json_path, on_log)
                speakers = _merge_speakers(old.speakers, roster)
            else:
                # 話者リストを先に統合してから割当を移す(ID を保存するのが要点)
                speakers = _merge_speakers(old.speakers, roster)
                # 分割・結合を復元するぶん区間が増減するので、戻り値で置き換える
                all_segments = _carry_over_assignments(old, all_segments, on_log)
                carried_revision = old.doc_revision
                carried_log = old.edit_log
        except Exception as e:  # 壊れた JSON は無視して作り直す
            on_log(f"既存の作業ファイルを読めませんでした({e})。新規作成します。")
            speakers = parse_roster(roster)

    proj = Project(
        audio_path=str(audio_path),
        duration=duration or (len(chunks) * chunk_seconds),
        chunk_seconds=chunk_seconds,
        model=model,
        # ローカルに逐語モードは無いので、選ばれていても記録しない(§5.4)。
        # 記録すると「逐語で作った」という誤った履歴になる。
        verbatim=verbatim and not engine.is_local,
        audio_fingerprint=fingerprint,
        source_sha256=source_sha,
        # 話者分離を使ったかどうかも処理経路の一部。第三者が「どうやって
        # まとまりが付いたのか」を読めるようにする。
        engine={**engine.record(),
                **({"diarize": diarize_record} if diarize_record else {})},
        doc_revision=carried_revision,
        edit_log=carried_log,
        speakers=speakers,
        segments=all_segments,
    )
    proj.save(json_path)
    real = {s.cluster for s in all_segments if not s.is_pseudo_cluster}
    if real:
        on_log(f"完了: {len(all_segments)} 区間 / 声のまとまり {len(real)} 種類")
    else:
        on_log(f"完了: {len(all_segments)} 区間(話者は全て未判別)")
    return proj
