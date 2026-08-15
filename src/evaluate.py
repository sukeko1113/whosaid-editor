"""正解ラベルに対して話者分離の質を測る(検証用の純粋な計算)。

`claude/claude_正解ラベルの作り方_設計書.md` の §1・§2 をそのまま実装する。
測る前に手順を固定するための道具なので、**ここの定義を後から変えない**
(変えるときは設計書の版を上げ、理由を残す)。

画面もモデルも要らない純粋な関数だけを置く。標本を選ぶ処理をここに入れて
あるのは、「作りやすい区間を選んだ」という疑いを自分で否定できるようにする
ため——種と手順が決まっていれば、誰が実行しても同じ 200 件になる。
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import Any, Iterable, Optional, Sequence


# 層の境目。設計書 §2.2 で「2 秒未満 / 2 秒以上」と決めた。
SHORT_SECONDS = 2.0

STRATUM_SHORT = "short"
STRATUM_LONG = "long"

# 標本の既定(層ごと)。合計 200 件で全体の 95% 信頼区間はおよそ ±6 ポイント。
DEFAULT_PER_STRATUM = 100

# 種。固定しないと標本が実行のたびに変わり、測定前に固定したことにならない。
DEFAULT_SEED = 20260815


@dataclass(frozen=True)
class Sampled:
    """標本に選ばれた 1 区間。"""

    index: int
    stratum: str
    start: float
    duration: float


def stratum_of(duration: float) -> str:
    """区間の長さから層を決める。"""
    return STRATUM_SHORT if duration < SHORT_SECONDS else STRATUM_LONG


def stratum_sizes(segments: Iterable[Any]) -> dict[str, int]:
    """母集団の層ごとの件数(全体値の重みに使う)。"""
    c: Counter[str] = Counter()
    for s in segments:
        c[stratum_of(s.duration)] += 1
    return dict(c)


def select_sample(
    segments: Sequence[Any],
    per_stratum: int = DEFAULT_PER_STRATUM,
    seed: int = DEFAULT_SEED,
) -> list[Sampled]:
    """層ごとに単純無作為抽出する。同じ種なら必ず同じ結果になる。

    層の中の件数が要求に満たないときは、その層は全件を採る(母数が小さい層で
    水増しをしない)。並びは時間順に整える——人が会話を追いながら付けるため。
    """
    by_stratum: dict[str, list[Any]] = defaultdict(list)
    for s in segments:
        by_stratum[stratum_of(s.duration)].append(s)

    picked: list[Sampled] = []
    for name in sorted(by_stratum):                 # 層の順序も固定する
        pool = sorted(by_stratum[name], key=lambda s: (s.start, s.index))
        rng = random.Random(f"{seed}:{name}")       # 層ごとに独立した流れ
        take = pool if len(pool) <= per_stratum else rng.sample(pool, per_stratum)
        picked.extend(
            Sampled(index=s.index, stratum=name, start=s.start,
                    duration=s.duration)
            for s in take
        )
    picked.sort(key=lambda x: (x.start, x.index))
    return picked


def cluster_purity(
    pairs: Iterable[tuple[Any, Optional[str]]]
) -> tuple[float, dict[Any, tuple[Optional[str], int, int]]]:
    """クラスタ純度(設計書 §1.1)。

    pairs は (話者分離が付けたまとまり, 正解の話者 ID)。正解が None の区間
    (分からない・複数人同時)は除く。

    まとまりごとに「正解で最も多い話者」を対応づけ、そこに一致する区間の
    割合を返す。候補学習に依存しないので、話者分離そのものの質を測れる。

    戻り値は (全体の純度, {まとまり: (対応づけた話者, 一致数, 総数)})。
    """
    groups: dict[Any, list[str]] = defaultdict(list)
    for cluster, truth in pairs:
        if truth is None:
            continue
        groups[cluster].append(truth)

    detail: dict[Any, tuple[Optional[str], int, int]] = {}
    hit = total = 0
    for cluster, truths in groups.items():
        c = Counter(truths)
        # 同数のときは話者 ID の順で決める(実行のたびに変わらないように)
        best = min(sorted(c.items()), key=lambda kv: (-kv[1], kv[0]))[0]
        detail[cluster] = (best, c[best], len(truths))
        hit += c[best]
        total += len(truths)
    return (hit / total if total else 0.0), detail


def weighted_rate(rates: dict[str, float], sizes: dict[str, int]) -> float:
    """層ごとの値を、母集団の層の大きさで重み付けして全体値にする(§2.3)。

    層ごとに同じ件数を採っているので、単純平均では母集団の構成からずれる。
    """
    total = sum(sizes.get(k, 0) for k in rates)
    if not total:
        return 0.0
    return sum(rates[k] * sizes.get(k, 0) for k in rates) / total


def proportion_halfwidth(p: float, n: int, population: Optional[int] = None) -> float:
    """割合の 95% 信頼区間の半幅(有限母集団の補正つき)。

    「70% を上回ったか」を言えるかどうかの判断に使う。設計書 §2.3 は
    64〜76% を判定不能と定めた。
    """
    if n <= 0:
        return 1.0
    half = 1.96 * sqrt(max(0.0, p * (1 - p)) / n)
    if population and population > n:
        half *= sqrt((population - n) / (population - 1))
    return half


def verdict(p: float, n: int, threshold: float = 0.70,
            population: Optional[int] = None) -> str:
    """合否を言えるかどうか。言えないときは「判定不能」と返す。

    測る前にこの規則を決めておかないと、出た数字を見てから
    「もう少し測れば届きそう」と動かしてしまう。
    """
    half = proportion_halfwidth(p, n, population)
    if p - half >= threshold:
        return "達成"
    if p + half < threshold:
        return "未達"
    return "判定不能"
