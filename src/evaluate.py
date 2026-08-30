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


# 層の境目(秒)。設計書 §2.2。1 秒未満を独立させているのは、声の手がかりが
# 最も乏しく話者分離が苦しい帯だから。実データで 285 件(23%)ある。
VERY_SHORT_SECONDS = 1.0
SHORT_SECONDS = 2.0

STRATUM_VERY_SHORT = "vshort"     # 1 秒未満
STRATUM_SHORT = "short"           # 1〜2 秒
STRATUM_LONG = "long"             # 2 秒以上
STRATA = (STRATUM_VERY_SHORT, STRATUM_SHORT, STRATUM_LONG)

# 1 巡ぶん(層ごと)。3 層 × 100 = 300 件を 1 巡とする。
DEFAULT_PER_STRATUM = 100

# 巡の数。1 巡ごとに集計でき、途中で止めても全長に散らばった標本になる。
DEFAULT_ROUNDS = 2

# 種。固定しないと標本が実行のたびに変わり、測定前に固定したことにならない。
DEFAULT_SEED = 20260815


@dataclass(frozen=True)
class Sampled:
    """標本に選ばれた 1 区間。"""

    index: int
    stratum: str
    start: float
    duration: float
    round: int = 1


def stratum_of(duration: float) -> str:
    """区間の長さから層を決める。"""
    if duration < VERY_SHORT_SECONDS:
        return STRATUM_VERY_SHORT
    if duration < SHORT_SECONDS:
        return STRATUM_SHORT
    return STRATUM_LONG


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
    rounds: int = DEFAULT_ROUNDS,
) -> list[Sampled]:
    """層ごとに無作為抽出し、巡(round)に割り振る。同じ種なら必ず同じ結果。

    **巡に分けるのが要点。**時間順に付ける以上、途中でやめると会議の前半だけの
    標本になってしまう。1 巡を「全長に散らばった層別標本」にしておけば、
    どこで止めても偏らず、巡ごとに集計できる。

    層の中は一度だけ混ぜてから前から切り出すので、巡を増やしても
    前の巡の中身は変わらない(1 巡目を固定したまま 2 巡目を足せる)。

    層の件数が足りないときは、その層は全件を使う(水増ししない)。
    並びは時間順——人が会話を追いながら付けるため。
    """
    by_stratum: dict[str, list[Any]] = defaultdict(list)
    for s in segments:
        by_stratum[stratum_of(s.duration)].append(s)

    picked: list[Sampled] = []
    for name in STRATA:
        pool = sorted(by_stratum.get(name, []), key=lambda s: (s.start, s.index))
        random.Random(f"{seed}:{name}").shuffle(pool)   # 層ごとに独立した流れ
        for r in range(rounds):
            chunk = pool[r * per_stratum:(r + 1) * per_stratum]
            picked.extend(
                Sampled(index=s.index, stratum=name, start=s.start,
                        duration=s.duration, round=r + 1)
                for s in chunk
            )
    picked.sort(key=lambda x: (x.round, x.start, x.index))
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


def purity_chance_level(
    pairs: Sequence[tuple[Any, Optional[str]]],
    trials: int = 200,
    seed: int = DEFAULT_SEED,
) -> float:
    """でたらめな正解でもクラスタ純度がどれだけ出るか（下駄）。

    純度は「まとまりごとに多数派を採る」ので、正解と何の関係もなくても
    1/話者数 より高い値が出る。実測 200 件で 21.7% 出た（話者 9 名）。
    この下駄を併記しないと、純度 60% を実力と読み違える。

    正解ラベルの並びだけを混ぜ、まとまりの分け方はそのままにして測り直す。
    """
    truths = [t for _, t in pairs if t is not None]
    clusters = [c for c, t in pairs if t is not None]
    if not truths:
        return 0.0
    rng = random.Random(seed)
    total = 0.0
    for _ in range(trials):
        rng.shuffle(truths)
        p, _detail = cluster_purity(zip(clusters, truths))
        total += p
    return total / trials


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


# ----------------------------------------------------------------------
# 逐語の正解に対する指標(段階 2)
#
# **定義は測る前に固定する。**出た数字を見てから正規化の仕方や語のリストを
# 変えれば、都合の良い値を選べてしまう。
# ----------------------------------------------------------------------

# CER を測る前に落とすもの。句読点と空白だけ。
# **フィラーは落とさない** —— 測りたい対象そのものだから。
_CER_STRIP = "、。，．・「」『』（）()　 \t\n!！?？…ー"

# 言い淀み(フィラー)と相づちは別物として数える。
# 相づちは「同意の意思表示」で、記録から消してはいけないもの(CLAUDE.md)。
FILLER_TERMS = ("えー", "えっと", "ええと", "あのー", "あの", "まあ", "うーん",
                "んー", "そのー", "なんか", "ちょっと")
BACKCHANNEL_TERMS = ("はい", "ええ", "うん", "なるほど", "そうですね",
                     "わかりました")

SHORT_UTTERANCE_SECONDS = 2.0


def normalize_for_cer(text: str) -> str:
    """CER を測る前の正規化。句読点と空白だけ落とす。"""
    return "".join(c for c in (text or "") if c not in _CER_STRIP)


def edit_distance(a: str, b: str) -> int:
    """挿入・削除・置換の最小回数(レーベンシュタイン距離)。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(truth: str, hyp: str) -> float:
    """文字誤り率。正規化してから、正解の長さで割る。

    1.0 を超えることがある(候補が正解より長く、全部違う場合)。頭打ちにしない
    ——頭打ちにすると「ひどく間違えた」と「まあまあ間違えた」が同じに見える。
    """
    t, h = normalize_for_cer(truth), normalize_for_cer(hyp)
    if not t:
        return 0.0 if not h else 1.0
    return edit_distance(t, h) / len(t)


def edit_ops(truth: str, hyp: str) -> dict:
    """編集操作の内訳。`cer()` と**同じ正規化・同じ整列**で数える。

    返すのは {"len": 正解長, "sub": 置換, "dele": 削除, "ins": 挿入}。
    `(sub + dele + ins) / len` は `cer()` と一致する（検査で縛ってある）。

    **なぜ要るか。**画面には「誤字」という指標が出ているが（`gui.py:850`
    ほか）、**それを計算する実装がリポジトリに無い**（HANDOFF「画面に出て
    いる『(実測)』の 1 つは、誰も再現できない」）。定義を決め直すための
    材料として、まず内訳を出せるようにする。

    **この関数は内訳を返すだけで、「誤字」を定義しない。**どれを誤字と
    呼ぶかは測ってから決める——先に定義を確定させると、結果を見てから
    定義を選び直したくなる。
    """
    a, b = normalize_for_cer(truth), normalize_for_cer(hyp)
    n, m = len(a), len(b)
    # dp[j] = (コスト, 置換, 削除, 挿入)。cer() と同じ重み(すべて 1)。
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * m
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = min(
                    (prev[j - 1][0] + 1, prev[j - 1][1] + 1, prev[j - 1][2], prev[j - 1][3]),
                    (prev[j][0] + 1, prev[j][1], prev[j][2] + 1, prev[j][3]),
                    (cur[j - 1][0] + 1, cur[j - 1][1], cur[j - 1][2], cur[j - 1][3] + 1),
                )
        prev = cur
    _cost, sub, dele, ins = prev[m]
    return {"len": n, "sub": sub, "dele": dele, "ins": ins}


def count_terms(text: str, terms: Iterable[str]) -> int:
    """語の出現回数を数える(単純な部分一致)。

    形態素解析は使わない。「あの」が「あのー」にも数えられる等の粗さは残るが、
    **3 つのエンジンに同じ物差しを当てる**ことが目的なので、粗さは共通に効く。
    この限界は報告に併記する。
    """
    body = text or ""
    return sum(body.count(t) for t in terms)


def retention_rate(truth_text: str, hyp_text: str,
                   terms: Iterable[str]) -> tuple[float, int, int]:
    """正解に含まれる語のうち、候補にも現れる割合。

    戻り値は (保持率, 候補の出現数, 正解の出現数)。**1.0 を超えうる**
    ——候補が正解より多く出していれば超える(幻覚か、語のリストの粗さ)。
    頭打ちにせず、そのまま報告する。
    """
    want = count_terms(truth_text, terms)
    got = count_terms(hyp_text, terms)
    return (got / want if want else 0.0), got, want


def stratified_halfwidth(rates: dict[str, float], sizes: dict[str, int],
                         counts: dict[str, int]) -> float:
    """層別標本での全体値の 95% 信頼区間の半幅。

    層ごとに同じ件数を採るので、全体の精度は単純な n だけでは決まらない。
    層の重み(母集団に占める割合)の二乗で効く。

    rates: 層ごとの実測値 / sizes: 母集団の層の大きさ / counts: 標本の層の件数
    """
    total = sum(sizes.get(k, 0) for k in rates)
    if not total:
        return 1.0
    var = 0.0
    for k, p in rates.items():
        big, n = sizes.get(k, 0), counts.get(k, 0)
        if not big or n <= 0:
            continue
        w = big / total
        fpc = (big - n) / (big - 1) if big > 1 else 0.0
        var += (w ** 2) * (max(0.0, p * (1 - p)) / n) * max(0.0, fpc)
    return 1.96 * sqrt(var)


def verdict(p: float, halfwidth: float, threshold: float = 0.70) -> str:
    """合否を言えるかどうか。言えないときは「判定不能」と返す。

    測る前にこの規則を決めておかないと、出た数字を見てから
    「もう少し測れば届きそう」と基準のほうを動かしてしまう。
    """
    if p - halfwidth >= threshold:
        return "達成"
    if p + halfwidth < threshold:
        return "未達"
    return "判定不能"
