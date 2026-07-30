"""分割 → 文字起こし → docx 結合 のパイプライン全体を制御するモジュール。

GUI から別スレッドで run_pipeline() を呼ぶ想定。
進捗とログはコールバック関数経由で UI スレッドへ通知する。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional

from google import genai

from .audio import probe_duration, split_audio
from .segments import Project, Segment, parse_roster
from .transcribe import (
    DIARIZATION_NOTE,
    ROSTER_NOTE,
    parse_segments,
    shift_timestamps,
    transcribe_audio,
    write_docx,
)


LogFn = Callable[[str], None]
ProgressFn = Callable[[int, int], None]   # (current, total)
CancelFn = Callable[[], bool]


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
) -> str:
    """設定の組み合わせごとに別キャッシュにする(混在を防ぐ)。

    v1.3.0: 逐語モードと名簿の内容もキャッシュキーに含める。
    名簿を書き換えたのに古い結果が再利用される事故を防ぐため、
    名簿本文のハッシュ(先頭8桁)をサフィックスに埋め込む。
    """
    parts: list[str] = []
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

    client = genai.Client(api_key=api_key)
    transcripts: list[str] = []
    on_progress(0, len(chunks))

    chunk_seconds = chunk_minutes * 60
    cache_suffix = _cache_suffix(with_timestamps, with_diarization, verbatim, roster)

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

        if cache_path.exists():
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
                )
                # チャンク内相対時刻 [MM:SS] を絶対時刻 [HH:MM:SS] に変換
                text = shift_timestamps(raw, offset) if with_timestamps else raw
                cache_path.write_text(text, encoding="utf-8")
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

def _carry_over_assignments(old: Project, new_segments: list[Segment], on_log: LogFn) -> int:
    """再実行時に、以前の割当結果を新しいセグメントへ引き継ぐ。

    「開始秒(±2秒)と本文先頭が一致する区間」を同一とみなす。
    キャッシュがある限り本文は変わらないので、実際にはほぼ全件が一致する。
    """
    buckets: dict[tuple[int, str], list[Segment]] = {}
    for seg in old.segments:
        if not seg.speaker_id:
            continue
        key = (int(seg.start // 2), seg.text[:16])
        buckets.setdefault(key, []).append(seg)

    carried = 0
    for seg in new_segments:
        for k in ((int(seg.start // 2), seg.text[:16]),
                  (int(seg.start // 2) - 1, seg.text[:16]),
                  (int(seg.start // 2) + 1, seg.text[:16])):
            pool = buckets.get(k)
            if pool:
                src = pool.pop(0)
                seg.speaker_id = src.speaker_id
                seg.reviewed = src.reviewed
                seg.note = src.note
                carried += 1
                break
    if carried:
        on_log(f"以前の割当 {carried} 区間を引き継ぎました。")
    return carried


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
) -> Optional[Project]:
    """音声 → セグメント JSON(話者未確定)を生成して Project を返す。"""
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

    duration = probe_duration(audio_path)
    if duration:
        on_log(f"音声長: {int(duration // 60)}分{int(duration % 60)}秒")

    on_log(f"音声を {chunk_minutes} 分単位で分割します...")
    chunks = split_audio(audio_path, chunks_dir, chunk_minutes * 60)
    on_log(f"{len(chunks)} 個のチャンクに分割しました。")

    if is_cancelled():
        on_log("キャンセルされました。")
        return None

    client = genai.Client(api_key=api_key)
    on_progress(0, len(chunks))

    chunk_seconds = chunk_minutes * 60
    cache_suffix = ".cluster.vb.txt" if verbatim else ".cluster.txt"

    all_segments: list[Segment] = []
    for i, chunk in enumerate(chunks):
        if is_cancelled():
            on_log("キャンセルされました。")
            return None

        cache_path = cache_dir / f"{chunk.stem}{cache_suffix}"
        label = f"[{i + 1}/{len(chunks)}] {chunk.name}"
        offset = i * chunk_seconds

        if cache_path.exists():
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
                )
                cache_path.write_text(raw_text, encoding="utf-8")
            except Exception as e:
                on_log(f"  失敗: {e}")
                raw_text = f"[00:00] 【?】 【文字起こし失敗: {chunk.name} - {e}】"

        # このチャンクの実際の長さ(最終チャンクは短い)
        this_len = chunk_seconds
        if duration:
            this_len = max(1.0, min(chunk_seconds, duration - offset))

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

    # 通し番号を振り直す
    for n, seg in enumerate(all_segments):
        seg.index = n

    speakers = parse_roster(roster)
    if json_path.exists():
        try:
            old = Project.load(json_path)
            _carry_over_assignments(old, all_segments, on_log)
            if not speakers:
                speakers = old.speakers
            else:
                # 既存の話者 ID を保ちながら、名簿に無い人は残す
                known = {sp.name for sp in speakers}
                speakers = speakers + [sp for sp in old.speakers if sp.name not in known]
                for n, sp in enumerate(speakers):
                    sp.order = n
        except Exception as e:  # 壊れた JSON は無視して作り直す
            on_log(f"既存の作業ファイルを読めませんでした({e})。新規作成します。")

    proj = Project(
        audio_path=str(audio_path),
        duration=duration or (len(chunks) * chunk_seconds),
        chunk_seconds=chunk_seconds,
        model=model,
        verbatim=verbatim,
        speakers=speakers,
        segments=all_segments,
    )
    proj.save(json_path)
    on_log(f"完了: {len(all_segments)} 区間 / 声のまとまり {len(proj.clusters())} 種類")
    return proj
