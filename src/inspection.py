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

# 一致した文字がこれより少ない提案は出さない。3〜4 文字の偶然一致は被覆率が
# 100% になり得るので、被覆率では防げない。
# 実会議の聴き取り(2026-08-13・15 件): 完全に外れた 2 件はどちらも一致 3 文字。
MIN_MATCHED = 8

# 開始がこれ以上ずれていたら提案する。これ未満は誤差の範囲。
# 終了のずれでは提案しない。記録に出るのも頭出しに効くのも開始時刻で、
# 終了は照合の中で最も精度が弱い(末尾の言い回しほど聞き取りが揺れる)。
TIME_DELTA = 0.75

# 本文の末尾がこれ以上アンカーに乗らなかったら、終了時刻を信用せず
# いまの区間の長さを保つ。終了は「最後に一致した文字」から取るので、
# 末尾が欠けると手前に出て発言の途中で切れる。
# 実会議の聴き取り: 欠け 3 字以上の 2 件はどちらも終了が外れ、2 字以下は全部当たり。
TAIL_GAP_LIMIT = 3

# 提案に付ける境界の余白(秒)。whisper の単語時刻は発話の立ち上がりを
# 少し遅く取る癖があるため、開始側は厚めに取る。
# 聴き取り 2 回目: 頭の欠けゼロの区間でも「わずかに遅く始まる」と判定された。
START_PAD = 0.30
END_PAD = 0.20

# 一致した文字の密度(字/秒)。正常な発話は実測で 4〜11 字/秒。これを大きく
# 下回る「一致」は、広い範囲に散らばった偶然の一致を拾っている。
# 聴き取り 2 回目: 本文が発話と食い違っていた区間は 0.8 字/秒だった。
MIN_DENSITY = 1.5

# 頭の欠けの秒換算がこれを超えたら、開始そのものが測れていないとみなして
# 提案しない(補正が実測より大きくなったら、それはもう推定であって実測ではない)。
HEAD_COMP_MAX = 6.0

# 実測の範囲が、いまの区間の長さのこの倍(＋1 秒)を超えて伸びていたら、
# 照合が遠くまで届きすぎている。区間の中に無いはずの沈黙や別の発言まで
# 抱き込んでいるので、時刻も分割候補も信用できない。
# 実測(67 分会議): 保存 4.7 秒の区間が 25.7 秒に伸びていた例がある。
# 判定を比にするのは、長い区間の数十秒の伸びは正常だから
# (保存 201 秒に対する +18.8 秒の区間は、聴き取りで当たりだった)。
STRETCH_RATIO = 1.5

# 提案どうしの実測範囲が、短いほうの半分を超えて重なっていたら「取り合い」。
# 挨拶やお礼の応酬(「よろしくお願いします」の連発)では、複数の区間が同じ
# 音声位置に当たる。正しいのは高々 1 つで、どれかは機械には決められない。
# 実測(67 分会議): 3 区間が全く同じ時刻に当たった組があった。
CONFLICT_OVERLAP = 0.5

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
    short_match: int = 0        # 一致した文字が少なすぎる(偶然かもしれない)
    scattered: int = 0          # 一致が広い範囲に散らばりすぎている
    stretched: int = 0          # 実測の範囲が区間より大きく伸びている
    head_lost: int = 0          # 頭の欠けが大きすぎて開始を測れない
    close_enough: int = 0       # 開始のずれが小さいので出す必要がない
    conflicted: int = 0         # 他の提案と同じ音声位置を取り合った
    notes: list[str] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return (len(self.proposals) + self.reviewed + self.unmatched
                + self.low_coverage + self.short_match + self.scattered
                + self.stretched + self.head_lost + self.close_enough
                + self.conflicted)

    @property
    def weak(self) -> int:
        """根拠が弱くて出さなかった数(画面の内訳表示用)。"""
        return (self.low_coverage + self.short_match + self.scattered
                + self.stretched + self.head_lost + self.conflicted)


def _proposal_id(kind: str, orig_start: float, start: float, end: float) -> str:
    """同じ根拠なら同じ id になるようにする。

    却下した提案を再点検で出し直さないために使う。実測が変われば
    別の提案として出てくる(そのときは改めて判断してもらう)。
    """
    seed = f"{kind}|{orig_start:.3f}|{start:.2f}|{end:.2f}"
    return hashlib.blake2b(seed.encode("utf-8"), digest_size=6).hexdigest()


def _evidence(m: Measured, d_start: float, d_end: float, keep_tail: bool,
              head_comp: float = 0.0) -> str:
    base = (f"一致 {m.matched}/{m.total} 文字(被覆 {m.coverage:.0%})・"
            f"ずれ 開始 {d_start:+.1f} 秒")
    if not keep_tail:
        base += f" / 終了 {d_end:+.1f} 秒"
    else:
        base += "・終わりは測れず(いまの長さを保つ)"
    if head_comp >= 0.1:
        base += f"・頭の欠け {m.head_gap} 字を {head_comp:.1f} 秒手前に補正"
    return base


def inspect_times(
    proj: Project,
    words: Sequence[Word],
    *,
    min_coverage: float = MIN_COVERAGE,
    min_matched: int = MIN_MATCHED,
    min_density: float = MIN_DENSITY,
    time_delta: float = TIME_DELTA,
    tail_gap_limit: int = TAIL_GAP_LIMIT,
    head_comp_max: float = HEAD_COMP_MAX,
    stretch_ratio: float = STRETCH_RATIO,
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
    raw_spans: list[tuple[float, float]] = []   # 余白を付ける前の実測範囲

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

        if m.matched < min_matched:
            result.short_match += 1
            continue

        if m.coverage < min_coverage:
            result.low_coverage += 1
            continue

        # 一致の密度(字/秒)。低すぎる「一致」は広い範囲に散らばった偶然で、
        # 本文が発話と食い違っているときにこうなる(実測 0.8 字/秒の例)。
        rate = m.matched / max(0.1, m.end - m.start)
        if rate < min_density:
            result.scattered += 1
            continue

        # 実測が区間の長さより大きく伸びていたら、照合が遠くまで届きすぎ。
        # 区間の外の沈黙や別の発言まで抱き込んでいるので信用しない。
        if (m.end - m.start) > (now_end - now_start) * stretch_ratio + 1.0:
            result.stretched += 1
            continue

        # 本文の頭が聞き取られていない分、開始は実際より遅く出る。欠けた
        # 文字数をこの区間の実測発話速度で秒に換算し、手前に戻す。
        # 聴き取り 2 回目: 換算 5.3 秒の区間を人は「約 5 秒切れている」と
        # 判定しており、ほぼ一致した。
        head_comp = m.head_gap / rate
        if head_comp > head_comp_max:
            result.head_lost += 1
            continue
        start = max(0.0, m.start - head_comp)
        d_start = start - now_start

        if abs(d_start) < time_delta:
            result.close_enough += 1
            continue

        # 末尾が乗っていない区間の終了は信用せず、いまの長さを保って
        # 開始だけ動かす(区間全体が同じ量ずれているのが典型的なドリフト)。
        # 開始側の余白は厚め(whisper は立ち上がりを遅く取る)。終了の余白は
        # 実測できたときだけ足す(測っていない値に余白を足す意味はない)。
        keep_tail = m.tail_gap >= tail_gap_limit
        pad_start = max(0.0, start - START_PAD)
        if keep_tail:
            pad_end = pad_start + (now_end - now_start)
        else:
            # 上限未満の小さな尻欠けも、頭と同じ換算で先へ伸ばす。
            # 聴き取り 3 回目: 尻欠け 2 字(換算 0.8 秒)の区間で、
            # 最後の「はい」が切れていた。
            pad_end = m.end + m.tail_gap / rate + END_PAD

        result.proposals.append(Proposal(
            id=_proposal_id(PROPOSAL_TIME, float(seg.orig_start),
                            pad_start, pad_end),
            type=PROPOSAL_TIME,
            target_orig_start=float(seg.orig_start),
            payload={"start": round(pad_start, 2), "end": round(pad_end, 2)},
            evidence=_evidence(m, d_start, d_end, keep_tail, head_comp),
            confidence=m.coverage,
        ))
        raw_spans.append((start, start + (now_end - now_start) if keep_tail
                          else m.end))

    # 取り合いの判定は余白を付ける前の実測範囲で行う。余白込みで比べると、
    # 隣り合う正当な発言どうしが余白のぶんだけ重なって巻き添えになる。
    result.proposals, result.conflicted = drop_conflicts(
        result.proposals, spans=raw_spans)
    return result


def drop_conflicts(proposals: Sequence[Proposal],
                   overlap_ratio: float = CONFLICT_OVERLAP,
                   spans: Optional[Sequence[tuple[float, float]]] = None,
                   ) -> tuple[list[Proposal], int]:
    """同じ音声位置を取り合う提案を、全部引っ込める。

    実測の範囲が重なる提案どうしは、正しくても高々 1 つ。どれが正しいかを
    機械が選ぶと、外れたときに間違った時刻が「実測」の顔をして残る。
    全部引っ込めて、その区間は人の耳に任せる。

    隣り合う発言が接しているだけ(重なりゼロか僅か)の組は正当なので残す。
    判定は「短いほうの長さの半分を超えて重なるか」。
    spans を渡すと payload の代わりにその範囲で判定する(余白を付ける前の
    実測範囲で比べるため。余白込みだと隣どうしが巻き添えになる)。
    """
    if spans is None:
        spans = [(float(p.payload.get("start", 0.0)),
                  float(p.payload.get("end", 0.0))) for p in proposals]
    order = sorted(range(len(proposals)), key=lambda i: spans[i][0])
    bad: set[int] = set()
    for a_pos, i in enumerate(order):
        s1, e1 = spans[i]
        for j in order[a_pos + 1:]:
            s2, e2 = spans[j]
            if s2 >= e1:
                break               # start 順なので、これ以降は重ならない
            overlap = min(e1, e2) - s2
            shorter = max(MIN_SEGMENT_SECONDS, min(e1 - s1, e2 - s2))
            if overlap > overlap_ratio * shorter:
                bad.add(i)
                bad.add(j)
    kept = [p for k, p in enumerate(proposals) if k not in bad]
    return kept, len(bad)


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
