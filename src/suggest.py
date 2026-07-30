"""話者候補の並べ替え(学習)エンジン。

ユーザーが区間を 1 つ確定するたびに統計を更新し、次の区間で
「その人である可能性が高い順」に候補を並べ替える。学習に使う手がかりは 4 つ。

1. クラスタ投票 (最強)
   AI は声質で発言者A/B/C…に分けている。同じクラスタで既に「佐藤」と
   確定していれば、そのクラスタの残りもほぼ佐藤である。

2. 直前・直後の話者からの遷移 (会話の流れ)
   「議長が振ったら次はこの人が答える」といった実際の遷移を数え、
   直前に確定した話者からの遷移確率を加点する。
   データがまだ薄いうちは「直前と同じ人が続けて話す」可能性を軽く減点する。

3. 全体の発言頻度
   よく話す人を上位に置く。

4. 名簿の並び順
   同点のときの決定的なタイブレーク(毎回同じ順で出るようにする)。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

from .segments import (
    Project,
    SPECIAL_MULTI,
    SPECIAL_NOISE,
    SPECIAL_UNKNOWN,
    Speaker,
)


# 各手がかりの重み(合計スコアの目安: 0〜150)
W_CLUSTER = 100.0
W_TRANSITION = 25.0
W_BACK_TRANSITION = 15.0
W_FREQUENCY = 10.0
W_ROSTER = 1.0

# 遷移統計がこの件数未満のときは「直前と同じ話者」を軽く減点する
THIN_TRANSITION_SAMPLES = 6
REPEAT_PENALTY = 8.0

# 統計が少ないうちは、その手がかりの効きを弱める(1〜2件の偶然に引きずられないため)。
# 例: 遷移の観測が 2 件しかなければ、遷移スコアは満点の 2/8 しか効かない。
CONFIDENCE_SAMPLES_TRANSITION = 8
CONFIDENCE_SAMPLES_FREQUENCY = 10


def _confidence(samples: int, full: int) -> float:
    return min(1.0, samples / full) if full > 0 else 1.0

# 「不明」などの特別ラベルは学習に混ぜない
_EXCLUDED_FROM_STATS = {SPECIAL_UNKNOWN, SPECIAL_MULTI, SPECIAL_NOISE}


@dataclass
class Candidate:
    speaker: Speaker
    score: float
    reasons: list[str]

    @property
    def reason_text(self) -> str:
        return " / ".join(self.reasons) if self.reasons else ""


class SpeakerSuggester:
    """Project の確定済み区間から統計を作り、候補を並べ替える。

    確定内容を変えたら refresh() を呼ぶ(区間数が数千でも十分速い)。
    """

    def __init__(self, project: Project) -> None:
        self.project = project
        self.cluster_votes: dict[str, Counter] = defaultdict(Counter)
        self.forward: dict[str, Counter] = defaultdict(Counter)
        self.backward: dict[str, Counter] = defaultdict(Counter)
        self.frequency: Counter = Counter()
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self.cluster_votes = defaultdict(Counter)
        self.forward = defaultdict(Counter)
        self.backward = defaultdict(Counter)
        self.frequency = Counter()

        segs = self.project.segments
        for seg in segs:
            sid = seg.speaker_id
            if not sid:
                continue
            self.cluster_votes[seg.cluster][sid] += 1
            if sid not in _EXCLUDED_FROM_STATS:
                self.frequency[sid] += 1

        # 隣接する確定済みペアから遷移を数える(間に未確定があっても、
        # 直前/直後の確定済みどうしは繋がっていないので数えない)
        for prev, cur in zip(segs, segs[1:]):
            a, b = prev.speaker_id, cur.speaker_id
            if not a or not b:
                continue
            if a in _EXCLUDED_FROM_STATS or b in _EXCLUDED_FROM_STATS:
                continue
            self.forward[a][b] += 1
            self.backward[b][a] += 1

    # ------------------------------------------------------------------
    def _prev_assigned(self, index: int, max_back: int = 3) -> Optional[str]:
        segs = self.project.segments
        for i in range(index - 1, max(-1, index - 1 - max_back), -1):
            sid = segs[i].speaker_id
            if sid and sid not in _EXCLUDED_FROM_STATS:
                return sid
        return None

    def _next_assigned(self, index: int, max_fwd: int = 3) -> Optional[str]:
        segs = self.project.segments
        for i in range(index + 1, min(len(segs), index + 1 + max_fwd)):
            sid = segs[i].speaker_id
            if sid and sid not in _EXCLUDED_FROM_STATS:
                return sid
        return None

    # ------------------------------------------------------------------
    def rank(self, index: int, include_special: bool = False) -> list[Candidate]:
        """index の区間について、話者候補を可能性の高い順に返す。"""
        segs = self.project.segments
        if not (0 <= index < len(segs)):
            return []
        seg = segs[index]
        speakers = list(self.project.speakers)
        if not speakers:
            return []

        n = len(speakers)
        scores: dict[str, float] = {sp.id: 0.0 for sp in speakers}
        reasons: dict[str, list[str]] = {sp.id: [] for sp in speakers}

        # --- 1. 同一クラスタの投票 -------------------------------------
        votes = self.cluster_votes.get(seg.cluster, Counter())
        # 自分自身の確定は手がかりから除く(付け替え時に固着させない)
        votes = Counter({k: v for k, v in votes.items() if k not in _EXCLUDED_FROM_STATS})
        if seg.speaker_id in votes:
            votes[seg.speaker_id] -= 1
            if votes[seg.speaker_id] <= 0:
                del votes[seg.speaker_id]
        total_votes = sum(votes.values())
        if total_votes:
            for sid, c in votes.items():
                if sid in scores:
                    scores[sid] += W_CLUSTER * c / total_votes
                    reasons[sid].append(f"同じ声のまとまり({seg.cluster_label})で{c}回確定")

        # --- 2. 直前の話者からの遷移 -----------------------------------
        prev = self._prev_assigned(index)
        if prev:
            trans = self.forward.get(prev, Counter())
            t = sum(trans.values())
            prev_name = self.project.speaker_name(prev)
            if t:
                conf = _confidence(t, CONFIDENCE_SAMPLES_TRANSITION)
                for sid, c in trans.items():
                    if sid in scores:
                        scores[sid] += W_TRANSITION * conf * c / t
                        reasons[sid].append(f"{prev_name}の次に話した実績{c}回")
            if t < THIN_TRANSITION_SAMPLES and prev in scores:
                scores[prev] -= REPEAT_PENALTY

        # --- 3. 直後の話者からの逆向き遷移 -----------------------------
        nxt = self._next_assigned(index)
        if nxt:
            back = self.backward.get(nxt, Counter())
            t = sum(back.values())
            nxt_name = self.project.speaker_name(nxt)
            if t:
                conf = _confidence(t, CONFIDENCE_SAMPLES_TRANSITION)
                for sid, c in back.items():
                    if sid in scores:
                        scores[sid] += W_BACK_TRANSITION * conf * c / t
                        reasons[sid].append(f"{nxt_name}の前に話した実績{c}回")

        # --- 4. 全体頻度 -----------------------------------------------
        total_freq = sum(self.frequency.values())
        if total_freq:
            conf = _confidence(total_freq, CONFIDENCE_SAMPLES_FREQUENCY)
            for sid, c in self.frequency.items():
                if sid in scores:
                    scores[sid] += W_FREQUENCY * conf * c / total_freq

        # --- 5. 名簿順(タイブレーク) ---------------------------------
        for i, sp in enumerate(speakers):
            scores[sp.id] += W_ROSTER * (n - i) / n

        cands = [
            Candidate(speaker=sp, score=scores[sp.id], reasons=reasons[sp.id])
            for sp in speakers
        ]
        cands.sort(key=lambda c: (-c.score, c.speaker.order))

        if include_special:
            from .segments import SPECIAL_SPEAKERS
            for sid, name in SPECIAL_SPEAKERS.items():
                cands.append(
                    Candidate(speaker=Speaker(id=sid, name=name, order=999), score=-1.0, reasons=[])
                )
        return cands

    # ------------------------------------------------------------------
    def cluster_summary(self, cluster: str) -> str:
        """クラスタの確定状況を短く説明する文字列。"""
        votes = self.cluster_votes.get(cluster, Counter())
        total = len(self.project.cluster_segments(cluster))
        if not votes:
            return f"{total}区間 / 未確定"
        parts = [
            f"{self.project.speaker_name(sid) or sid}×{c}"
            for sid, c in votes.most_common()
        ]
        return f"{total}区間 / " + ", ".join(parts)

    def dominant_speaker(self, cluster: str) -> Optional[str]:
        """クラスタ内で最も多く確定している話者 ID(なければ None)。"""
        votes = Counter({
            k: v for k, v in self.cluster_votes.get(cluster, Counter()).items()
            if k not in _EXCLUDED_FROM_STATS
        })
        if not votes:
            return None
        return votes.most_common(1)[0][0]


def next_unassigned(project: Project, from_index: int, forward: bool = True) -> Optional[int]:
    """from_index の次(前)にある未確定区間の位置を返す。"""
    segs = project.segments
    rng = range(from_index + 1, len(segs)) if forward else range(from_index - 1, -1, -1)
    for i in rng:
        if not segs[i].speaker_id:
            return i
    return None
