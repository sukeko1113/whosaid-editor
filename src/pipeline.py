"""分割 → 文字起こし → docx 結合 のパイプライン全体を制御するモジュール。

GUI から別スレッドで run_pipeline() を呼ぶ想定。
進捗とログはコールバック関数経由で UI スレッドへ通知する。
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from google import genai
from google.genai import types as genai_types

from .audio import audio_fingerprint, audio_hashes, probe_duration, split_audio
from .config import APP_VERSION
from .segments import Project, Segment, Speaker, fmt_hms, parse_roster
from .transcribe import (
    PROMPT_LANG,
    DIARIZATION_NOTE,
    ROSTER_NOTE,
    CancelledError,
    FatalTranscriptionError,
    parse_segments,
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

    【英語テスト用ブランチ】プロンプトの言語も含める。これが無いと、同じ音声を
    main(日本語プロンプト)と feature/en-test(英語プロンプト)で流したとき、
    指紋もチャンク長も逐語フラグも一致するので、黙ってもう一方の言語の転写が
    返ってくる。転写の中身がキーに現れていない、という点で他のキー欠落と
    同じ事故になる。transcribe.PROMPT_LANG を見るので、あちらを "ja" に
    戻せばキーも一緒に戻る。
    """
    parts: list[str] = []
    if fingerprint:
        parts.append(fingerprint)
    if PROMPT_LANG != "ja":
        parts.append(PROMPT_LANG)
    if chunk_seconds:
        parts.append(f"c{chunk_seconds}")
    if with_diarization:
        parts.append("diar")
    elif with_timestamps:
        parts.append("ts")
    if verbatim:
        parts.append("vb")
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
    kept = [s for s in old.segments if s.speaker_id or s.text_edited or s.time_edited]
    if not kept:
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

    for n, seg in enumerate(result):
        seg.index = n

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
    api_key: str,
    model: str,
    chunk_minutes: int,
    on_log: LogFn,
    on_progress: ProgressFn,
    is_cancelled: CancelFn,
    verbatim: bool = False,
    roster: str = "",
    force_retranscribe: bool = False,
) -> Optional[Project]:
    """音声 → セグメント JSON(話者未確定)を生成して Project を返す。

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

    on_log(f"作業ファイル: {json_path}")
    on_log("方式: 話者は A/B/C… に分けるだけで、実名は後の割当画面で確定します。")
    if verbatim:
        on_log("逐語モード: 有効(フィラー・言い直しを保持し、整文しません)")
    if force_retranscribe:
        on_log("キャッシュを使わず、最初から転写し直します。")

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

    client = _make_client(api_key)
    on_progress(0, len(chunks))

    chunk_seconds = chunk_minutes * 60
    # キャッシュ名には「音声の指紋」と「チャンク長」を含める。
    #   - 指紋が無いと、音声を編集してもファイル名が同じなら古い転写を再利用する
    #   - チャンク長が無いと、分割サイズを変えたとき音声とテキストがずれる
    #
    # 【英語テスト用ブランチ】プロンプトの言語も含める。ここは _cache_suffix()
    # とは別に自前で組み立てているので、あちらだけ直しても効かない
    # (実際にこの取りこぼしを踏んだ。転写 3 チャンクぶんが言語を含まない鍵で
    #  保存され、main で同じ音声を流せば英語の転写が返るところだった)。
    cache_suffix = (
        f".cluster{'.' + fingerprint if fingerprint else ''}"
        f"{'.' + PROMPT_LANG if PROMPT_LANG != 'ja' else ''}"
        f".c{chunk_seconds}{'.vb' if verbatim else ''}.txt"
    )

    all_segments: list[Segment] = []
    failed_chunks = 0
    last_failure: str = ""
    for i, chunk in enumerate(chunks):
        if is_cancelled():
            on_log("キャンセルされました。")
            return None

        cache_path = cache_dir / f"{chunk.stem}{cache_suffix}"
        label = f"[{i + 1}/{len(chunks)}] {chunk.name}"
        offset = chunk_starts[i]

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
                on_log("キャンセルされました。(完了済みチャンクはキャッシュに保存されています)")
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

        # このチャンクの実際の長さ(最終チャンクは短い)
        this_len = chunk_lengths[i]
        if duration:
            this_len = max(1.0, min(this_len, duration - offset))

        parsed = parse_segments(
            raw_text,
            chunk_index=i,
            offset_seconds=offset,
            chunk_seconds=this_len,
            start_index=len(all_segments),
        )
        all_segments.extend(Segment(**p) for p in parsed)
        on_log(f"  → {len(parsed)} 区間")
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

    # 通し番号を振り直す
    for n, seg in enumerate(all_segments):
        seg.index = n

    speakers = parse_roster(roster)
    # 再実行しても文書の履歴(版・編集履歴)は消さない。ただし音声の中身が
    # 変わっていた場合は別の文書の系譜なので引き継がない。
    carried_revision = 0
    carried_log: list[dict] = []
    if json_path.exists():
        try:
            old = Project.load(json_path)
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
        verbatim=verbatim,
        audio_fingerprint=fingerprint,
        source_sha256=source_sha,
        engine={
            "mode": "cloud",
            "model": model,
            "app_version": APP_VERSION,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        doc_revision=carried_revision,
        edit_log=carried_log,
        speakers=speakers,
        segments=all_segments,
    )
    proj.save(json_path)
    on_log(f"完了: {len(all_segments)} 区間 / 声のまとまり {len(proj.clusters())} 種類")
    return proj
