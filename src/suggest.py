"""話者候補の並べ替え(学習)エンジン。

ユーザーが区間を 1 つ確定するたびに統計を更新し、次の区間で
「その人である可能性が高い順」に候補を並べ替える。

手がかりは 6 つ。上ほど強い。

1. クラスタ投票
   AI は声質で発言者A/B/C…に分けている。同じクラスタで既に「佐藤」と
   確定していれば、そのクラスタの残りもほぼ佐藤である。

2. チャンク境界の連続性
   音声は 10 分などの固定長で機械的に切っているので、境界は発言の途中に
   落ちることがほとんど。前チャンク末尾の話者は、次チャンク先頭と同一人物で
   ある可能性が高い。クラスタはチャンクごとに振り直されるため、この手がかりが
   「新しいチャンクの最初の 1 手」を助ける。

3. 同じ記号の引き継ぎ
   Gemini はチャンクごとに A から振り直すが、A は概ね「最初に/よく話す声」に
   当たる。前チャンクの A が誰だったかは弱い事前分布になる。

4. 直前・直後の話者からの遷移 (会話の流れ)
   「議長が振ったら次はこの人が答える」といった実際の遷移を数える。
   データがまだ薄いうちは「直前と同じ人が続けて話す」可能性を軽く減点する。

5. 全体の発言頻度

6. 名簿の並び順(同点時の決定的なタイブレーク)

なお【?】【*】(判別不能・複数人同時)は「声のまとまり」ではなく雑多な
寄せ集めなので、クラスタ投票・記号引き継ぎのどちらからも除外する。
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
W_BOUNDARY = 45.0        # チャンク境界の連続性
W_LETTER = 20.0          # 前チャンクの同じ記号(A/B/C)が誰だったか
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
CONFIDENCE_SAMPLES_LETTER = 4


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
        self.letter_votes: dict[str, Counter] = defaultdict(Counter)
        self.forward: dict[str, Counter] = defaultdict(Counter)
        self.backward: dict[str, Counter] = defaultdict(Counter)
        self.frequency: Counter = Counter()
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        self.cluster_votes = defaultdict(Counter)
        self.letter_votes = defaultdict(Counter)
        self.forward = defaultdict(Counter)
        self.backward = defaultdict(Counter)
        self.frequency = Counter()

        segs = self.project.segments
        for seg in segs:
            sid = seg.speaker_id
            if not sid:
                continue
            # 擬似クラスタ(?/*)は「同じ声」ではないので投票に入れない
            if not seg.is_pseudo_cluster:
                self.cluster_votes[seg.cluster][sid] += 1
                if sid not in _EXCLUDED_FROM_STATS:
                    self.letter_votes[seg.cluster_tail][sid] += 1
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

    def _boundary_partner(self, index: int) -> Optional[str]:
        """チャンク境界をまたいで隣接する区間の話者(確定済みなら)を返す。

        自分がチャンクの先頭なら「前チャンクの末尾」、末尾なら「次チャンクの先頭」。
        両方に当てはまる(そのチャンクに区間が1つしかない)場合は前を優先する。
        """
        segs = self.project.segments
        if not (0 <= index < len(segs)):
            return None
        seg = segs[index]

        if index > 0 and segs[index - 1].chunk != seg.chunk:
            sid = segs[index - 1].speaker_id
            if sid and sid not in _EXCLUDED_FROM_STATS:
                return sid
        if index + 1 < len(segs) and segs[index + 1].chunk != seg.chunk:
            sid = segs[index + 1].speaker_id
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
        # 擬似クラスタ(判別不能・複数人同時)は寄せ集めなので投票を使わない
        votes = Counter() if seg.is_pseudo_cluster else self.cluster_votes.get(seg.cluster, Counter())
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

        # --- 2. チャンク境界の連続性 -----------------------------------
        # 音声は固定長で機械的に切っているので、境界は発言の途中に落ちやすい。
        # 境界をまたいで隣り合う区間は同一話者である可能性が高い。
        boundary_sid = self._boundary_partner(index)
        if boundary_sid and boundary_sid in scores:
            scores[boundary_sid] += W_BOUNDARY
            reasons[boundary_sid].append("チャンク境界で発言が続いている可能性")

        # --- 3. 前チャンクの同じ記号 -----------------------------------
        # 【A】はチャンクごとに振り直されるが、概ね同じ役回りの声に当たる。
        if not seg.is_pseudo_cluster:
            letters = Counter({
                k: v for k, v in self.letter_votes.get(seg.cluster_tail, Counter()).items()
            })
            # 同一クラスタ内の票は 1 で数え済みなので、それは差し引く
            for sid, c in self.cluster_votes.get(seg.cluster, Counter()).items():
                if sid in letters:
                    letters[sid] -= c
            letters = Counter({k: v for k, v in letters.items() if v > 0})
            lt = sum(letters.values())
            if lt:
                conf = _confidence(lt, CONFIDENCE_SAMPLES_LETTER)
                for sid, c in letters.items():
                    if sid in scores:
                        scores[sid] += W_LETTER * conf * c / lt
                        reasons[sid].append(
                            f"別チャンクの 【{seg.cluster_tail}】 もこの人だった({c}回)")

        # --- 4. 直前の話者からの遷移 -----------------------------------
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
            # 「直前と同じ人がすぐ続けて話す」は稀。ただし境界連続の場合は逆に
            # 同一話者が濃厚なので減点しない。
            if (t < THIN_TRANSITION_SAMPLES and prev in scores
                    and prev != boundary_sid):
                scores[prev] -= REPEAT_PENALTY

        # --- 5. 直後の話者からの逆向き遷移 -----------------------------
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

        # --- 6. 全体頻度 -----------------------------------------------
        total_freq = sum(self.frequency.values())
        if total_freq:
            conf = _confidence(total_freq, CONFIDENCE_SAMPLES_FREQUENCY)
            for sid, c in self.frequency.items():
                if sid in scores:
                    scores[sid] += W_FREQUENCY * conf * c / total_freq

        # --- 7. 名簿順(タイブレーク) ---------------------------------
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
        total = len(self.project.cluster_segments(cluster))
        tail = cluster.partition(":")[2] or cluster
        if tail in ("?", "*"):
            kind = "判別不能" if tail == "?" else "複数人が同時"
            return f"{total}区間 / {kind}のため一括適用は不可"
        votes = self.cluster_votes.get(cluster, Counter())
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
    return next_matching(project, from_index, lambda s: not s.speaker_id, forward)


def next_unreviewed(project: Project, from_index: int, forward: bool = True) -> Optional[int]:
    """次の「未確認」区間。

    未確定(まだ誰も入っていない)か、一括適用で埋めただけで音声を聴いていない区間。
    一括適用で一気に埋めたあと、それを見直すための移動に使う。
    """
    return next_matching(project, from_index, lambda s: not (s.speaker_id and s.reviewed), forward)


def next_matching(project: Project, from_index: int, pred, forward: bool = True) -> Optional[int]:
    segs = project.segments
    rng = range(from_index + 1, len(segs)) if forward else range(from_index - 1, -1, -1)
    for i in rng:
        if pred(segs[i]):
            return i
    return None
