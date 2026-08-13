"""自動点検: 実測と突き合わせて「直したほうがよさそうな所」を提案する。

点検は**提案しか出さない**。本体データは書き換えない。承認するかどうかは
人が決め、適用は画面側の既存の時刻編集の経路を通す(点検専用の書き込み経路は
作らない)。ここが壊れると「誰が言ったかの検証済み記録」という製品の芯が
壊れるので、境界をはっきりさせてある。

    align.py    音声 → 単語と時刻(重い。キャッシュする)
    anchor.py   本文 × 単語 → 引き直した時刻と被覆率(純粋関数)
    inspection  そこから提案を組み立て、作業ディレクトリに置く  ← ここ

モジュール名が設計書の `inspect.py` と違うのは、標準ライブラリの inspect と
名前が衝突するため。PyInstaller は src/main.py を入口にするので src/ が
top-level の探索パスに入り、凍結アプリの中で標準の inspect を隠しかねない
(faster-whisper 系は内部で inspect を使う)。保存先のフォルダ名は設計書
どおり inspect/ のままにしてある。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .align import ALIGN_VER, Word
from .anchor import Measured, measure_segments
from .segments import (
    MIN_SEGMENT_SECONDS,
    Project,
    Segment,
    audio_span,
    fmt_hms_frac,
)


# 被覆率がこれ未満なら提案しない。偶然の数文字一致から時刻を作らないための壁。
MIN_COVERAGE = 0.60

# 開始・終了のどちらかがこれ以上ずれていたら提案する。これ未満は誤差の範囲。
TIME_DELTA = 0.75

PROPOSAL_TIME = "time"


@dataclass
class Proposal:
    """1 件の提案。作業ディレクトリに置く sidecar の 1 要素でもある。"""

    id: str
    type: str                   # いまは "time" だけ。分割は Step 1-3b で足す
    # 対象の指し方は orig_start(パイプラインが出した元の時刻)。index は
    # 分割で振り直るので使えない。
    target_orig_start: float
    payload: dict[str, Any]     # 時刻提案なら {"start": ..., "end": ...}
    evidence: str               # なぜそう言えるのか(画面にそのまま出す)
    confidence: float           # 0〜1。被覆率をそのまま使う
    status: str = "pending"     # pending / accepted / rejected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Proposal":
        return cls(
            id=str(d.get("id", "")),
            type=str(d.get("type", PROPOSAL_TIME)),
            target_orig_start=float(d.get("target_orig_start", 0.0)),
            payload=dict(d.get("payload") or {}),
            evidence=str(d.get("evidence", "")),
            confidence=float(d.get("confidence", 0.0)),
            status=str(d.get("status", "pending")),
        )


@dataclass
class InspectResult:
    """点検の結果。提案と、出さなかった理由の内訳。

    内訳を残すのは、閾値が効きすぎているのか実測が悪いのかを、あとから
    切り分けられるようにするため(§12 のキャリブレーションで使う)。
    """

    proposals: list[Proposal] = field(default_factory=list)
    reviewed: int = 0           # 人が耳で確定済み。機械は口を出さない
    unmatched: int = 0          # 照合できなかった
    low_coverage: int = 0       # 一致はしたが乗りが足りない
    close_enough: int = 0       # ずれが小さいので出す必要がない
    notes: list[str] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return (len(self.proposals) + self.reviewed + self.unmatched
                + self.low_coverage + self.close_enough)


def _proposal_id(kind: str, orig_start: float, start: float, end: float) -> str:
    """同じ根拠なら同じ id になるようにする。

    却下した提案を再点検で出し直さないために使う。実測が変われば
    別の提案として出てくる(そのときは改めて判断してもらう)。
    """
    seed = f"{kind}|{orig_start:.3f}|{start:.2f}|{end:.2f}"
    return hashlib.blake2b(seed.encode("utf-8"), digest_size=6).hexdigest()


def _evidence(m: Measured, d_start: float, d_end: float) -> str:
    return (f"一致 {m.matched}/{m.total} 文字(被覆 {m.coverage:.0%})・"
            f"ずれ 開始 {d_start:+.1f} 秒 / 終了 {d_end:+.1f} 秒")


def inspect_times(
    proj: Project,
    words: Sequence[Word],
    *,
    min_coverage: float = MIN_COVERAGE,
    time_delta: float = TIME_DELTA,
    **anchor_kw: Any,
) -> InspectResult:
    """実測と突き合わせて、時刻の提案を作る(§3)。

    出さない区間の規則(§3.4):
      - `time_reviewed` が立っている区間は提案しない。人が耳で確定した時刻を
        機械が上書き提案しない。差分は notes に記録するだけ。
      - `time_edited` でも `time_reviewed` でない区間は提案する。過去に推定を
        当てただけなので、より新しい実測で上書きしてよい。
      - 話者の `reviewed` はスキップ理由にしない。時刻を直しても話者は
        変わらないため。
    """
    result = InspectResult()
    if not proj.segments or not words:
        return result

    # 照合には実音声の時刻を渡す(保存値にはずれ補正が乗っていない区間がある)
    spans = [(s.text, *audio_span(s, proj.time_offset)) for s in proj.segments]
    measured = measure_segments(spans, words, **anchor_kw)

    for seg, m in zip(proj.segments, measured):
        if m is None:
            result.unmatched += 1
            continue

        now_start, now_end = audio_span(seg, proj.time_offset)
        d_start = m.start - now_start
        d_end = m.end - now_end

        if seg.time_reviewed:
            # 人が確定した時刻は動かさない。ただし記録は残す(実測がずれて
            # いるなら、照合そのものを疑う手がかりになる)
            result.reviewed += 1
            if abs(d_start) > time_delta:
                result.notes.append(
                    f"確認済みの区間 {seg.index + 1} "
                    f"({fmt_hms_frac(seg.start)}) は実測と {d_start:+.1f} 秒ちがいます。"
                )
            continue

        if m.coverage < min_coverage:
            result.low_coverage += 1
            continue

        if abs(d_start) < time_delta and abs(d_end) < time_delta:
            result.close_enough += 1
            continue

        result.proposals.append(Proposal(
            id=_proposal_id(PROPOSAL_TIME, float(seg.orig_start), m.start, m.end),
            type=PROPOSAL_TIME,
            target_orig_start=float(seg.orig_start),
            payload={"start": round(m.start, 2), "end": round(m.end, 2)},
            evidence=_evidence(m, d_start, d_end),
            confidence=m.coverage,
        ))
    return result


# ----------------------------------------------------------------------
# 適用のための計算(書き込みは画面側の既存経路が行う)
# ----------------------------------------------------------------------

def clip_to_neighbours(start: float, end: float,
                       prev_end: Optional[float],
                       next_start: Optional[float]) -> Optional[tuple[float, float]]:
    """隣の区間と重なる提案を、接点で切り詰める(§3.2)。

    1 件ずつ承認しても、まとめて適用しても重ならないようにするため。
    切り詰めた結果が最短の長さを割るなら None(この提案は当てられない)。
    潰れた区間を作るくらいなら、適用しないほうがいい。
    """
    if prev_end is not None:
        start = max(start, prev_end)
    if next_start is not None:
        end = min(end, next_start)
    if end - start < MIN_SEGMENT_SECONDS:
        return None
    return start, end


def target_segment(proj: Project, proposal: Proposal) -> Optional[Segment]:
    """提案が指している区間を返す。分割で index が変わっても見失わない。

    同じ orig_start を持つ区間が複数あるのは、その区間を分割した兄弟。
    そのときは先頭(前側)を対象にする。
    """
    same = [s for s in proj.segments
            if abs(float(s.orig_start) - proposal.target_orig_start) < 1e-6]
    if not same:
        return None
    return min(same, key=lambda s: (s.start, s.index))


# ----------------------------------------------------------------------
# sidecar(§6.1)
# ----------------------------------------------------------------------

def proposals_path(work_dir: Path | str, fingerprint: str) -> Optional[Path]:
    """提案の置き場。指紋が無い音声では保存しない(取り違えを防ぐ)。"""
    if not fingerprint:
        return None
    return Path(work_dir) / "inspect" / f"proposals.{fingerprint}.a{ALIGN_VER}.json"


def load_proposals(path: Optional[Path]) -> list[Proposal]:
    """保存してある提案を読む。無い・壊れていれば空。"""
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Proposal.from_dict(d) for d in data.get("proposals", [])]
    except Exception:
        return []


def save_proposals(path: Optional[Path], proposals: Iterable[Proposal]) -> None:
    """提案を書く。派生データなので、書けなくても作業は続けられる。"""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "align_ver": ALIGN_VER,
            "proposals": [p.to_dict() for p in proposals],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass


def merge_history(fresh: Sequence[Proposal],
                  history: Sequence[Proposal]) -> list[Proposal]:
    """前回の判断を引き継ぐ。

    却下した提案は、同じ根拠なら再点検でも出さない(§6.1)。承認済みも
    出さない(すでに反映されている)。判断していないものだけを残す。
    """
    decided = {p.id: p.status for p in history
               if p.status in ("accepted", "rejected")}
    return [p for p in fresh if p.id not in decided]
