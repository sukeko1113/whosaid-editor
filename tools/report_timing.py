"""時刻の質を測り、あわせて保持率の内訳を出す(評価スパイク・段階2 の判定2/判定1)。

**このファイルは名前より広い。**timing(時刻)と名乗っているが、
**§8 判定1 の「保持率の超過分の内訳」と「語の誤カウントの切り分け」も
ここで算出する。**`src/evaluate.py` は測定前に定義を固定した資産で変更できず、
どこかに置く必要があったため同居させている。ファイルを増やすよりよいという判断。
名前が中身より狭いことを、最初に知っておくこと。

    役割1: 時刻の精度(判定2)     … 外れ量・単調性の破れ・重なり・ドリフト・打ち切り
    役割2: 保持率の内訳(判定1)   … 二重計上・指示語/副詞の候補出し・超過分の内訳

**このモジュールは純粋関数と定数だけを置く。**ファイルも標準入出力も触らない。
実データを見る前に定義を固定するためで、入出力と整形は別の単位で足す
(`claude/claude_文字起こしバックエンド評価_指示_v4_2.md` §7-4)。

**`src/evaluate.py` と `src/lang.py` を変更しない。**既測定 6 エンジンの数値との
比較可能性が失われる。切り分けはこちら側だけで行い、元の `retention_rate` の値も
併記すること(§8 判定1)。

---

## 物差しについて(役割1・必読)

外れ量は「正解の時刻との差」ではない。**時刻の正解データは存在しない。**
`truth/verbatim.*.json` は本文(`truth`)だけを人が直したもので、`start`/`end` は
製品のローカル実行(faster-whisper small)の出力がそのまま残っている。

そこで **`src/align.py` の単語時刻を物差しにする。**音声から直接得られ、
評価対象のエンジンとは独立している。ただし次の 2 点を**報告に必ず併記する。**

1. **物差しは faster-whisper であり、local と同じ ASR である。**
   local の外れ量が小さく出るのは当然で、これは偶然の誤差ではなく**系統的な偏り**。
   「local は物差しと同じエンジン由来なので有利」と書かずに数字だけ出すと、
   後から読む人が local を過大評価する。
2. **物差し自身のカバー率を出す。**align が全長を処理できていなければ、
   外れ量を測れない区間が出る。**何件測れて何件測れなかったかを必ず報告する**
   (`DeviationSet.measurable` / `.unmeasurable`)。

したがってこの指標は**エンジン間の相対比較にしか使えない。**
「真の時刻からのずれ」の絶対値ではない。

---

## `anchor.py` の使い方(役割1・必読)

**照合の実体は `anchor.measure()` を使う。**自前の突き合わせを書かない
(同じ SRT を別の読み方で解釈しないため)。ただし **`measure_segments()` は使わない。**
あちらは単調掃引 `floor` を持っており(`anchor.py:270`/`:283`)、
**単調性の破れを測る道具が同じ罠にはまる。**`measure()` 自体は範囲を引数で受け取る
だけで floor を持たないので、範囲をこちらで決めれば使える。

**範囲を「申告された時刻の周り」で切らない。**測りたいのは窓幅を超える外れ量
なので、申告時刻で窓を切ったら原理的に測れない。代わりに**本文から位置を当てる**
(`locate()`)。時刻を一切見ずに候補を決め、その周りに余裕を取って `measure()` に渡す。

### ただし全長照合はできない(anchor.py 冒頭の警告)

`anchor.py` の docstring はこう書いている。

> 全文どうしを difflib に掛けると 4 万文字(52 分の会議)で 66 秒かかる。
> O(n^2) なので 2 時間の録音では 5 分を超える。…速さ以上に効くのは誤マッチが
> 構造的に起きなくなること。全文照合では「はい」が 3 分先の「はい」に
> 当たり得るが、窓を切ればそもそも届かない。

**両方とも本当の問題である。**500 区間 × 4 万文字を素の difflib に掛けると終わらず、
短い区間は遠くの同じ語に当たる。**だからこう分ける。**

- **位置決め(`locate`)は k-gram の投票で行う。**文字列探索なので速く、
  時刻を見ないので外れ量に上限を作らない
- **候補が一意に決まらない区間は、数字を出さずに「測定不能」にする。**
  当てずっぽうの外れ量を出すくらいなら測れなかったと言う——`anchor.measure()` が
  照合不能に None を返すのと同じ思想
- したがって**外れ量は全区間では出ない。**短い区間・ありふれた本文の区間は落ちる。
  **これは偏りなので、測れた件数と落ちた理由を必ず併記する**

**「測れなかった」は、そのエンジンの欠陥ではない。**本文が短いだけかもしれない。
測定不能率をエンジンの評価に使わないこと。
"""
from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

# **ここだけは import 時に走る。**`src` を引くために要る(他の tools/ と同じ書き方)。
# 「純粋関数だけ」の方針に対する唯一の例外で、パスを足す以外は何もしない。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 純粋計算だけを引く。どちらも import 時の副作用が無いことを確認済み。
from src import anchor  # noqa: E402
from src.align import Word  # noqa: E402

# (開始秒, 終了秒, 本文)。`report_verbatim.load_engine()` と同じ形。
Seg = tuple[float, float, str]


# ---------------------------------------------------------------------------
# 定数 — **実データを見る前に固定する。**出た数字を見てから動かさないこと。
# ---------------------------------------------------------------------------

# 「極端に短い区間」の線。A では「37 文字が 1 秒」という機械的な割り振りが
# 起きていた(§8 判定2)。evaluate.SHORT_UTTERANCE_SECONDS(2.0)とは別物で、
# あちらは「短い発話」、こちらは「短すぎる区間」。混同しないこと。
SHORT_SEGMENT_SECONDS = 1.0

# 打ち切り検出のカバー率の下限(§7-3)。これを下回ったら中断する。
COVERAGE_FLOOR = 0.95

# 超過分の内訳を見る窓(役割2・群3)。`report_verbatim.py` の脱落の再現が
# 使っている ±2 秒に揃える。**独自の幅を作らない。**
OVERAGE_WINDOW_SECONDS = 2.0

# --- 位置決め(locate)の閾値 ---

# k-gram の k。照合の最短一致長に合わせる(日本語 3・英語 10)。
# **既定引数に固定しない。**lang.use() の切り替えが効かなくなる
# (anchor.py が MIN_BLOCK で踏んだ罠と同じ)。呼ぶ時に引く。
def gram_size() -> int:
    """位置決めに使う k-gram の長さ。現在の言語プロファイルから引く。"""
    from src import lang
    return lang.current().matching.min_block


# これ未満の正規化文字数の区間は、外れ量を測らない。
# 「はい」「うん」は全長に何十回も現れ、どこに当てても支持が同じになる。
LOCATE_MIN_CHARS = 12

# 1 位の候補が 2 位の何倍の支持を集めていれば一意とみなすか。
# 割れているときは測定不能にする(当てずっぽうを出さない)。
LOCATE_MARGIN_RATIO = 1.5

# **本文の曖昧さの門番。**2 位の候補が、本文の k-gram のうちこの割合以上を
# 説明できるなら、**1 位との差がどれだけ開いていても測定不能**にする。
#
# **相対の余裕(`LOCATE_MARGIN_RATIO`)だけでは足りない。**実測で
# 「遅刻して申し訳ございません。」(14 文字)が 3,192 秒離れた別の出現に当たった。
# 会議には定型句が複数回現れ、**文字数を増やしても定型句は長いので止まらない。**
# 言い回しが少し違えば 1 位は 1.5 倍の票を取れてしまう。**曖昧さを直接見るしかない。**
#
# **0.5 の根拠。**「本文の半分以上を別の場所でも説明できるなら、どちらか決められない」
# という線。**外れ量の分布を一切見ずに決まる**——本文が物差しの中で何回説明できるかは、
# 時刻とも申告値とも無関係な、本文だけの性質である。
# (外れ量を見てから閾値を動かすのとは種類が違う。あれは都合のいい値を選べてしまう)
LOCATE_AMBIGUOUS_SUPPORT = 0.5

# 位置決めのあと、`anchor.measure()` に渡す範囲の余裕(秒)。
# **申告時刻ではなく、当てた位置の周りに取る。**したがって外れ量に
# 上限を作らない。単語時刻の粗さと区間長のぶんだけ見ておけば足りる。
LOCATE_SLACK_SECONDS = 60.0

# 照合が薄い区間は測定不能にする。
#
# **この値は `inspection.py` からの流用である。目的が違うことを承知で使っている。**
#   inspection.py の 0.60 … 「**提案を採用するか**」の壁
#   ここでの 0.60       … 「**位置を確信できるか**」の壁
# 揃えておけば「なぜこの値か」を説明できるので現状は流用しているが、
# **両者は別の問いである。将来この値を触る場合はこの差を踏まえること。**
LOCATE_MIN_COVERAGE = 0.60

# --- 語の誤カウント(役割2) ---

# **(1) 同じカテゴリの中での二重計上。**`evaluate.FILLER_TERMS` に
# 「あのー」と「あの」が両方あり、`count_terms()` は部分一致の総和なので
# 「あのー」は必ず 2 回数えられる。**語リストの作り方の誤りであって、
# 音声の性質でも言語の難しさでもない。**引き算だけで確定する。
#   (長い方, 短い方) の順。長い方の出現数が、短い方に丸ごと乗っている。
FILLER_DOUBLE_COUNTED = (("あのー", "あの"),)

# **(2) カテゴリをまたぐ二重計上。**「ええと」(フィラー)は「ええ」(相づち)を
# 含む。フィラー側と相づち側で同じ文字列を数えている。
#   (含む語, 含まれる語, 含まれる語のカテゴリ)
CROSS_COUNTED = (("ええと", "ええ", "backchannel"),)

# **(3) 人の判断が要る語。**自動分類しない。形態素解析を入れると
# `evaluate.py` の「形態素解析は使わない」という設計判断を測定側から崩し、
# 既測定 6 エンジンとの比較可能性に影響する。候補を出して人が決める。
AMBIGUOUS_TERMS = (
    # フィラーであると同時に指示語。「あの資料」「あの件」「あの時」
    "あの",
    # フィラーであると同時に副詞「少し」。「ちょっと長くなった」
    "ちょっと",
)

# 候補一覧に添える前後の文字数。
CANDIDATE_CONTEXT_CHARS = 12


# ---------------------------------------------------------------------------
# 役割1 — 時刻の精度(判定2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Coverage:
    """1 リクエストぶんの被覆。打ち切りの検出に使う(§7-3)。"""

    audio_seconds: float        # そのリクエストに渡した音声の実測長
    last_end: float             # 応答の最終区間の終了(リクエスト内相対)
    ratio: float                # last_end / audio_seconds
    max_gap: float              # 隣接区間の最大の空き(中抜けの手がかり)
    truncated: bool             # ratio < COVERAGE_FLOOR

    @property
    def missing_seconds(self) -> float:
        return max(0.0, self.audio_seconds - self.last_end)


def coverage(segs: Sequence[Seg], audio_seconds: float,
             floor: float = COVERAGE_FLOOR) -> Coverage:
    """出力が入力音声の全長をカバーしているか。**リクエストごとに呼ぶこと。**

    **結合後に一度だけ呼んではいけない。**3 分割のうち 2 本目が切れても、
    3 本目が音声末尾まで届いていれば結合後の比は 100% 近くになり、
    **真ん中の穴が素通りする**(§7-3)。

    `audio_seconds` は**そのリクエストに渡したファイルの実測長**
    (`src.audio.probe_duration`)。名目のチャンク長を使わないこと——
    切り出しの端数が乗って分母がずれる。
    """
    if audio_seconds <= 0:
        return Coverage(audio_seconds, 0.0, 0.0, 0.0, True)
    if not segs:
        return Coverage(audio_seconds, 0.0, 0.0, audio_seconds, True)
    last_end = max(e for _s, e, _t in segs)
    ratio = last_end / audio_seconds
    return Coverage(
        audio_seconds=audio_seconds,
        last_end=last_end,
        ratio=ratio,
        max_gap=max_gap(segs),
        truncated=ratio < floor,
    )


def max_gap(segs: Sequence[Seg]) -> float:
    """隣接区間の最大の空き。中抜けの手がかり(自動で中断はしない)。

    **無音は実在するので、これで打ち切りを判定しない。**値を出して人が見る。
    時間順に並べてから測る(出力順のままだと逆転が巨大な空きに化ける)。
    """
    if len(segs) < 2:
        return 0.0
    ordered = sorted(segs, key=lambda s: (s[0], s[1]))
    return max((b[0] - a[1] for a, b in zip(ordered, ordered[1:])), default=0.0)


def monotonicity_breaks(segs: Sequence[Seg]) -> int:
    """単調性の破れ。**出力順のまま・開始時刻で数える。**

    **`report_verbatim.load_engine()` の戻り値を渡さないこと。**
    あちらは最後に (start, end) で並べ替えるので、**破れが消える。**
    順序を保存して読んだ並びを渡すこと。

    **開始時刻で見る理由。**`anchor.measure_segments()` の単調掃引 `floor` は
    一致位置を前へ進め、区間は出力順に処理される。前進を起こすのは start。
    end で見ると「重なり」(`overlap_count`)と混ざるが、両者は別の現象である——
    **重なりは無害(合成データで 200/200 正しい)、破れは危険。**混ぜないこと。
    """
    return sum(1 for a, b in zip(segs, segs[1:]) if b[0] < a[0])


def max_forward_jump(segs: Sequence[Seg]) -> float:
    """出力順で、次の区間の開始が前より何秒先へ跳んだかの最大。

    調査 v01 §5 の「1 区間だけ時刻が 5 分先」に対応する量。
    **破れが 1 件でも、跳躍が大きければ後続が軒並み範囲外になる**ので、
    件数と一緒に見ること。
    """
    if len(segs) < 2:
        return 0.0
    return max((b[0] - a[0] for a, b in zip(segs, segs[1:])), default=0.0)


def overlap_count(segs: Sequence[Seg]) -> int:
    """隣接区間の重なり(前の終わりより前に次が始まる)。時間順に並べて数える。

    **これ自体は無害である。**合成データでは重なり 5 秒・40 秒のいずれでも
    照合は 200/200 正しかった(調査 v01 §5)。**この件数を根拠に時刻を
    悪く評価しないこと。**危険なのは `monotonicity_breaks` のほう。
    """
    ordered = sorted(segs, key=lambda s: (s[0], s[1]))
    return sum(1 for a, b in zip(ordered, ordered[1:]) if b[0] < a[1])


def short_segment_count(segs: Sequence[Seg],
                        threshold: float = SHORT_SEGMENT_SECONDS) -> int:
    """極端に短い区間の件数。A では「37 文字が 1 秒」が起きていた。"""
    return sum(1 for s, e, _t in segs if (e - s) < threshold)


def length_duration_correlation(segs: Sequence[Seg]) -> Optional[float]:
    """文字数と区間長の相関(Pearson)。機械的に均等割りされていないかを見る。

    **1.0 に近いほど疑わしい。**本文の長さから時刻を作っているなら、
    実測ではなく計算値である。`redistribute_times()` が文字数按分をかけた
    出力はここが高く出るはずで、**それが期待値である**(A の出力が対照)。

    区間が 2 件未満、または分散が 0 のときは None(相関が定義できない)。
    """
    xs = [len((t or "").strip()) for _s, _e, t in segs]
    ys = [max(0.0, e - s) for s, e, _t in segs]
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def tail_drift(segs: Sequence[Seg], audio_seconds: float) -> Optional[float]:
    """音声の長さと、最後の区間の終了時刻のずれ(進行性ドリフトの有無)。

    正なら出力が音声より短く終わっている(打ち切りか、末尾の取りこぼし)。
    負なら音声より先へ出ている(ドリフト)。
    **`coverage()` と分母が同じなので、両方を別々に報告すること**——
    打ち切りは「切れた」、ドリフトは「ずれた」で、原因が違う。
    """
    if not segs or audio_seconds <= 0:
        return None
    return audio_seconds - max(e for _s, e, _t in segs)


# --- 外れ量 -----------------------------------------------------------------

@dataclass(frozen=True)
class Deviation:
    """1 区間ぶんの外れ量。測れなかった場合は `reason` に理由が入る。"""

    index: int
    declared_start: float           # エンジンが申告した開始
    matched_start: Optional[float]  # 物差しの上で本文が見つかった開始
    gap: Optional[float]            # |declared - matched|
    coverage: Optional[float]       # 照合の被覆率
    reason: str = ""                # 測れなかった理由(測れたときは空)

    @property
    def measurable(self) -> bool:
        return self.gap is not None


@dataclass
class DeviationSet:
    """外れ量の集計。**測れた件数と落ちた理由を必ず一緒に報告する。**"""

    items: list[Deviation] = field(default_factory=list)

    @property
    def measured(self) -> list[Deviation]:
        return [d for d in self.items if d.measurable]

    @property
    def measurable(self) -> int:
        return len(self.measured)

    @property
    def unmeasurable(self) -> int:
        return len(self.items) - self.measurable

    @property
    def reasons(self) -> dict[str, int]:
        """落ちた理由の内訳。**エンジンの評価に使わないこと**——
        本文が短いだけかもしれない(モジュール冒頭の注意)。"""
        return dict(Counter(d.reason for d in self.items if not d.measurable))

    @property
    def max_gap(self) -> Optional[float]:
        gaps = [d.gap for d in self.measured if d.gap is not None]
        return max(gaps) if gaps else None

    @property
    def median_gap(self) -> Optional[float]:
        gaps = sorted(d.gap for d in self.measured if d.gap is not None)
        if not gaps:
            return None
        n = len(gaps)
        return gaps[n // 2] if n % 2 else (gaps[n // 2 - 1] + gaps[n // 2]) / 2

    @property
    def p90_gap(self) -> Optional[float]:
        gaps = sorted(d.gap for d in self.measured if d.gap is not None)
        if not gaps:
            return None
        return gaps[min(len(gaps) - 1, int(0.9 * len(gaps)))]


def build_gram_index(text: str, k: int) -> dict[str, list[int]]:
    """物差しの本文に k-gram の索引を張る。位置決め(`locate`)の下ごしらえ。

    **時刻を一切見ない。**だから外れ量に上限を作らない。
    """
    idx: dict[str, list[int]] = defaultdict(list)
    for i in range(max(0, len(text) - k + 1)):
        idx[text[i:i + k]].append(i)
    return idx


@dataclass(frozen=True)
class Located:
    """位置決めの結果。棄却したときは `reason` に理由が入る。"""

    offset: Optional[int]   # 物差しの文字位置。棄却なら None
    support: int            # 1 位の票
    runner_up: int          # 2 位の票
    grams: int              # 本文の k-gram 数(票の上限)
    reason: str = ""        # 採用時は空

    @property
    def runner_up_share(self) -> float:
        """2 位が本文のどれだけを説明できるか。曖昧さの尺度。"""
        return self.runner_up / self.grams if self.grams else 0.0


def locate_detail(norm: str, index: dict[str, list[int]], k: int,
                  *, margin: float = LOCATE_MARGIN_RATIO,
                  ambiguous: float = LOCATE_AMBIGUOUS_SUPPORT) -> Located:
    """位置決めの本体。**票の内訳まで返す**(棄却の理由づけに要る)。

    門番は 2 段。**どちらも時刻を見ない。**

      1. **相対**: 1 位が 2 位の `margin` 倍の票を取れていなければ棄却
      2. **絶対**: 2 位が本文の `ambiguous` 以上を説明できるなら棄却
         (1 位との差がどれだけ開いていても)

    2 が要る理由は `LOCATE_AMBIGUOUS_SUPPORT` の説明にある。
    """
    grams = max(0, len(norm) - k + 1)
    if grams <= 0:
        return Located(None, 0, 0, grams, "本文が短い")
    votes: Counter[int] = Counter()
    for i in range(grams):
        for pos in index.get(norm[i:i + k], ()):
            votes[pos - i] += 1
    if not votes:
        return Located(None, 0, 0, grams, "物差しに無い")
    ranked = votes.most_common(2)
    best, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else 0
    if s2 and s1 < s2 * margin:
        return Located(None, s1, s2, grams, "位置が割れる")
    if grams and s2 / grams >= ambiguous:
        return Located(None, s1, s2, grams, "本文が曖昧(別の場所でも説明できる)")
    return Located(best, s1, s2, grams, "")


def locate(norm: str, index: dict[str, list[int]], k: int,
           *, margin: float = LOCATE_MARGIN_RATIO) -> Optional[int]:
    """本文が物差しのどこにあるかを、**時刻を見ずに**当てる。

    k-gram の投票。各 k-gram が「自分の出現位置 − 自分の本文内での位置」に
    1 票入れる。同じ場所に本文が丸ごと乗っていれば、そこに票が集まる。

    **一意に決まらなければ None を返す。**1 位が 2 位の `margin` 倍の票を
    集めていなければ割れているとみなす。「はい」が 3 分先の「はい」に
    当たる問題(`anchor.py` 冒頭の警告)を、**当てずっぽうを出さないことで**
    避ける。窓で届かなくするのではない——窓を切ると外れ量が測れなくなる。
    """
    return locate_detail(norm, index, k, margin=margin).offset


def deviation(index_no: int, seg: Seg, track: anchor.Track,
              gram_index: dict[str, list[int]], k: int,
              *, min_chars: int = LOCATE_MIN_CHARS,
              slack: float = LOCATE_SLACK_SECONDS,
              min_coverage: float = LOCATE_MIN_COVERAGE) -> Deviation:
    """1 区間の外れ量。**申告時刻は最後に引き算するだけで、探索には使わない。**

    手順:
      1. 本文を正規化する(`anchor.normalize` — 照合と同じ正規化を使う)
      2. 短すぎる本文は測らない(`min_chars`)。「はい」はどこにでも当たる
      3. `locate()` で位置を当てる。**時刻を見ない**
      4. 当てた位置の**周り**に `slack` 秒の余裕を取り、`anchor.measure()` に渡す。
         **申告時刻の周りではない。**だから外れ量に上限がつかない
      5. 被覆率が壁を下回れば測定不能(当てずっぽうの時刻を返さない)
    """
    declared = seg[0]
    norm, _map = anchor.normalize(seg[2] or "")
    if len(norm) < min_chars:
        return Deviation(index_no, declared, None, None, None, "本文が短い")

    loc = locate_detail(norm, gram_index, k)
    if loc.offset is None:
        return Deviation(index_no, declared, None, None, None, loc.reason)
    at = max(0, min(loc.offset, len(track) - 1))

    # **当てた位置の周りに範囲を取る。**申告時刻は使わない。
    t0 = max(0.0, track.starts[at] - slack)
    t1 = track.ends[min(len(track) - 1, at + len(norm))] + slack

    m = anchor.measure(seg[2] or "", track, t0, t1)
    if m is None:
        return Deviation(index_no, declared, None, None, None, "照合できない")
    if m.coverage < min_coverage:
        return Deviation(index_no, declared, m.start, None, m.coverage,
                         "被覆率が低い")
    return Deviation(index_no, declared, m.start, abs(declared - m.start),
                     m.coverage, "")


def deviations(segs: Sequence[Seg], words: Sequence[Word],
               **kw) -> DeviationSet:
    """区間の並びぜんぶの外れ量。`words` は `src.align` が出した単語時刻。

    **`anchor.measure_segments()` を使わない。**あちらは単調掃引 `floor` を
    持つので、単調性が壊れた出力を測ると道具自身が同じ罠にはまる
    (モジュール冒頭)。ここは 1 区間ずつ独立に測る。
    """
    track = anchor.prepare(words)
    k = gram_size()
    gram_index = build_gram_index(track.text, k)
    return DeviationSet([
        deviation(i, s, track, gram_index, k, **kw) for i, s in enumerate(segs)
    ])


# ---------------------------------------------------------------------------
# 役割2 — 保持率の内訳と語の誤カウント(判定1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoubleCount:
    """引き算だけで確定する過大計上(群1)。**人の判断を挟まない。**"""

    longer: str         # 含む側(「あのー」)
    shorter: str        # 含まれる側(「あの」)
    longer_count: int   # 含む側の出現数 = そのまま過大計上の件数
    category: str = "filler"


def double_counts(text: str,
                  pairs: Sequence[tuple[str, str]] = FILLER_DOUBLE_COUNTED,
                  category: str = "filler") -> list[DoubleCount]:
    """同じカテゴリの中での二重計上(群1)。

    `evaluate.count_terms()` は `body.count(t)` の総和なので、語リストに
    「あのー」と「あの」が両方あると「あのー」は必ず 2 回数えられる。
    **過大計上の件数は、長い側の出現数そのもの。**文脈を見る必要がない。
    """
    body = text or ""
    return [DoubleCount(lo, sh, body.count(lo), category) for lo, sh in pairs]


def cross_counts(text: str,
                 triples: Sequence[tuple[str, str, str]] = CROSS_COUNTED
                 ) -> list[DoubleCount]:
    """カテゴリをまたぐ二重計上(群1)。

    「ええと」は `FILLER_TERMS` にあり、その中の「ええ」は
    `BACKCHANNEL_TERMS` にある。**フィラー側と相づち側で同じ文字列を
    数えている。**実測で `ええと、そうですね` は相づち 2 件と数えられた。
    ここで数えるのは**相づち側の過大計上**である。
    """
    body = text or ""
    return [DoubleCount(lo, sh, body.count(lo), cat)
            for lo, sh, cat in triples]


@dataclass(frozen=True)
class Candidate:
    """人の判断が要る 1 件(群2)。**自動で分類しない。**"""

    term: str           # 「あの」「ちょっと」
    position: int       # 本文中の文字位置
    before: str         # 前の文脈
    after: str          # 後ろの文脈

    @property
    def snippet(self) -> str:
        return f"{self.before}【{self.term}】{self.after}"


def candidates(text: str,
               terms: Sequence[str] = AMBIGUOUS_TERMS,
               context: int = CANDIDATE_CONTEXT_CHARS) -> list[Candidate]:
    """曖昧な語の出現を、前後の文脈つきで全部並べる(群2)。

    **分類はしない。**「あの」はフィラーであると同時に指示語、
    「ちょっと」はフィラーであると同時に副詞「少し」である。
    形態素解析を入れれば自動化できるが、それは `evaluate.py` の
    「形態素解析は使わない」という設計判断を測定側から崩すことになり、
    既測定 6 エンジンとの比較可能性に影響する。
    **正解データでは「あの」23 件・「ちょっと」7 件。目視で足りる。**

    長い語を先に消費しないので、「あのー」は「あの」としても 1 件挙がる。
    **それでよい**——群1 で別に数えるので、人は素の出現を見ればよい。
    """
    body = text or ""
    out: list[Candidate] = []
    for term in terms:
        start = 0
        while True:
            i = body.find(term, start)
            if i < 0:
                break
            out.append(Candidate(
                term=term,
                position=i,
                before=body[max(0, i - context):i],
                after=body[i + len(term):i + len(term) + context],
            ))
            start = i + 1
    return sorted(out, key=lambda c: c.position)


# 群2 の判定を残す JSON の形(入出力は別の単位で足す)。
#
#     {
#       "judged_by":  "<判定した人>",       ← 必須
#       "judged_at":  "2026-08-30T14:00:00+09:00",  ← 必須
#       "source":     "<正解データの識別。音声の SHA-256 など>",
#       "items": [
#         {"term": "あの", "position": 412, "snippet": "…【あの】子ども…",
#          "verdict": "demonstrative", "note": "指示語。直後が名詞"}
#       ]
#     }
#
# **判定した人と判定日を必ず残す。**後から「なぜこの件を指示語と判定したか」を
# 追えるようにするため。`reviewed`(✓)と同じ思想で、**人が判断したという
# 記録そのものが成果物の価値**である(CLAUDE.md)。
JUDGEMENT_REQUIRED_KEYS = ("judged_by", "judged_at", "items")

VERDICT_FILLER = "filler"                # 言い淀み。保持率の分母に数えてよい
VERDICT_DEMONSTRATIVE = "demonstrative"  # 指示語。数えてはいけない
VERDICT_ADVERB = "adverb"                # 副詞「少し」。数えてはいけない
VERDICT_UNSURE = "unsure"                # 判別できない。幅として報告する
VERDICTS = (VERDICT_FILLER, VERDICT_DEMONSTRATIVE,
            VERDICT_ADVERB, VERDICT_UNSURE)


@dataclass(frozen=True)
class Overage:
    """100% を超えた分の内訳(群3)。**位置合わせが要るのはここだけ。**"""

    term_total_truth: int   # 正解側の出現数
    term_total_hyp: int     # 出力側の出現数
    matched: int            # 正解と同じあたりに出ている分(本物の逐語性)
    inserted: int           # 正解に対応が無い分(**捏造**)

    @property
    def rate(self) -> float:
        return (self.term_total_hyp / self.term_total_truth
                if self.term_total_truth else 0.0)


def overage(truth_spans: Sequence[Seg], hyp_spans: Sequence[Seg],
            terms: Iterable[str], lo: float, hi: float,
            window: float = OVERAGE_WINDOW_SECONDS) -> Overage:
    """保持率の超過分を「一致」と「挿入」に分ける(群3)。

    **`retention_rate` は位置の一致を見ていない。**正解のフィラーを全部落として
    別の場所に同数を捏造しても 100% になる。**逐語録では捏造は脱落より重い**——
    落ちた発話は「記録が不完全」だが、無かった発話が入るのは「記録が虚偽」で、
    `reviewed`(✓)で担保しようとしている価値そのものを裏切る。しかも人が
    聴いて直すとき、**脱落は気づけるが捏造はもっともらしく読めて気づけない。**

    やり方: 帯 [lo, hi] を `window` 秒ごとに刻み、窓ごとに正解側と出力側の
    該当語数を比べる。出力が多い分を「挿入」、正解に届いている分を「一致」とする。

    本文の切り出しは **`report_verbatim.text_in()` をそのまま使う。**
    按分の仕様を二重に持たないため(§7-4)。**関数の中で import する**のは、
    あちらが import 時に標準出力を再構成するため——このモジュールを
    「純粋関数だけ」に保つ。

    **`normalize_for_cer()` を通さない。**あれは `_CER_STRIP` に長音「ー」を
    含むので、通すと「えー」「あのー」「うーん」「そのー」が**丸ごと消える**
    (実測で確認)。CER を測るときは正しい正規化だが、**語を数える前にかけては
    いけない。**`report_verbatim.py` の `retention_rate` も素の本文に当てており、
    ここもそれに揃える——揃えないと超過分の内訳と保持率の分母が別物になる。
    """
    from report_verbatim import text_in  # noqa: E402  (冒頭の注記を参照)
    from src import evaluate

    matched = inserted = t_all = h_all = 0
    edges = _windows(lo, hi, window)
    for a, b in edges:
        t_n = evaluate.count_terms(text_in(truth_spans, a, b), terms)
        h_n = evaluate.count_terms(text_in(hyp_spans, a, b), terms)
        t_all += t_n
        h_all += h_n
        matched += min(t_n, h_n)
        inserted += max(0, h_n - t_n)
    return Overage(t_all, h_all, matched, inserted)


def _windows(lo: float, hi: float, width: float) -> list[tuple[float, float]]:
    """[lo, hi] を width 秒ごとに刻む。端数は最後の窓に足す。"""
    if width <= 0 or hi <= lo:
        return [(lo, hi)]
    out: list[tuple[float, float]] = []
    a = lo
    while a < hi:
        b = min(hi, a + width)
        out.append((a, b))
        a = b
    return out


# ===========================================================================
# 入出力と整形(単位2)
#
# **ここから下だけが外の世界に触れる。**上の定義は純粋なまま保つこと。
# `report_verbatim` の import を関数の中に置いているのも同じ理由
# (あちらは import 時に標準出力を再構成する)。
# ===========================================================================

def load_ordered(path) -> list[Seg]:
    """出力を **ファイルの並びのまま** 読む。時間順に並べ替えない。

    **`report_verbatim.load_engine()` を使えない理由。**あちらは最後に
    `sorted(segs, key=(start, end))` で並べ替える。本文を繋ぐ用途では正しいが、
    **並べ替えると単調性の破れが消える**(`monotonicity_breaks` の説明)。

    **パーサは 1 つに保つ。**正規表現は `report_verbatim` から借り、
    違いは「最後にソートしないこと」だけにする。同じ SRT を別の読み方で
    解釈しないため(§7-4)。
    """
    from pathlib import Path
    import report_verbatim as rv

    p = Path(path)
    if p.suffix.lower() != ".srt":
        # 作業ファイル(.speakers.json)は index の順が出力の順。
        from src.segments import Project
        proj = Project.load(p)
        return [(s.start, s.end, s.text)
                for s in sorted(proj.segments, key=lambda s: s.index)]

    out, block = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            if block:
                out.append(block)
                block = []
        else:
            block.append(line)
    if block:
        out.append(block)

    segs: list[Seg] = []
    for b in out:
        for i, line in enumerate(b):
            m = rv._SRT_TS.match(line.strip())
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                body = [x.strip() for x in b[i + 1:]
                        if not rv._SRT_SPEAKER.match(x)]
                segs.append((start, end, " ".join(body).strip()))
                break
    return segs                      # ← ソートしない。ここが load_engine との差


def load_words_json(path) -> list[Word]:
    """`src.align` が保存した単語時刻を読む(物差し)。"""
    from src.align import load_words
    words = load_words(Path(path))
    if words is None:
        raise SystemExit(f"単語時刻を読めません: {path}")
    return words


def load_local_production_order(work_dir, pattern: str) -> list[Seg]:
    """local の**組み立て前**の出力を、生成順のまま読む(判定2 の基準線)。

    **`.speakers.json` を使えない理由。**話者分離が有効なとき、
    `pipeline.py:959` が `sorted(all_segments, key=(start, index))` で
    **開始時刻に並べ替え**、`pipeline.py:974` が `index` を振り直す。
    したがって `.speakers.json` の `index` 順は**時刻順であって生成順ではない**。
    `segments.py:90` も「**`index` は鍵にしてはいけない**——分割・結合・
    再実行で振り直る」と警告している。**premerge でも救われない**——
    ソート(959)は `merge_same_speaker`(964)より前にある。

    そこで `.work_<音声>/transcripts/` に残る**チャンクごとのキャッシュ**を読む。
    これは組み立ての前段で、ソートを一度も通っていない。

    **チャンクの並びはファイル名の連番で決める。**これは実データで確かめてある
    (10 本すべて、期待位置との差 0.0 秒・累積長 4036.9 秒 対 音声 4036 秒)。
    `split_audio()` が `sorted(glob("chunk_*.m4a"))` を返し、`pipeline.py:807`
    が同じ順で長さを積み上げるのと一致する。

    `pattern` は系列を選ぶ glob(同じ音声に複数の設定の系列が残っている)。
    例: `chunk_*.local.ca1fb4d464e99c16.c420.small.v1.json`
    """
    import json
    out: list[Seg] = []
    offset = 0.0
    for p in sorted(Path(work_dir).glob(pattern)):
        d = json.loads(p.read_text(encoding="utf-8"))
        for u in d.get("utterances", []):
            out.append((offset + float(u["rel_start"]),
                        offset + float(u["rel_end"]), u.get("text", "")))
        offset += float(d.get("duration", 0.0))
    return out                       # ← 生成順のまま。並べ替えない


def load_local_words(work_dir, pattern: str) -> list[Word]:
    """チャンクキャッシュに残る `local_asr` 由来の単語時刻(物差しの相互検証用)。

    align とは**別経路**(`local_asr` はチャンクごと、align は全長を 1 本で)。
    同じ `small` でも値が揃うとは限らない。**その差が物差し自身の揺れの下限**に
    なるので、外れ量を解釈するときの基準に使う。
    """
    import json
    out: list[Word] = []
    offset = 0.0
    for p in sorted(Path(work_dir).glob(pattern)):
        d = json.loads(p.read_text(encoding="utf-8"))
        for w in d.get("words", []):
            out.append(Word(text=w.get("text", ""),
                            start=offset + float(w["start"]),
                            end=offset + float(w["end"])))
        offset += float(d.get("duration", 0.0))
    return out


def ruler_meta(path) -> dict:
    """物差しキャッシュの素性を読む(モデル・音声の指紋・実装版・長さ)。

    **`load_words()` は words だけを返して素性を捨てる**ので、
    ここで生の JSON から読む。**外れ量を解釈するのに、どのモデルで作った
    物差しかが要る。**将来モデルを変えれば物差しも変わり、過去の数字と
    比べられなくなる。

    **装置(cuda/cpu)と compute_type はキャッシュに入っていない**
    (`align.save_words()` が保存するのは align_ver / model / fingerprint /
     duration の 4 つだけ)。**実行した人が別に控えること。**
    """
    import json
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        return {"読めない": str(e)}
    return {k: d.get(k) for k in ("align_ver", "model", "fingerprint",
                                  "duration")} | {"語数": len(d.get("words", []))}


def print_ruler(path, words) -> None:
    """物差しの素性を出す。**外れ量を出すなら必ず一緒に出すこと。**"""
    m = ruler_meta(path)
    print("\n" + "=" * 78)
    print("物差し(align の単語時刻)")
    print("=" * 78)
    for k, v in m.items():
        print(f"  {k:<12}: {v}")
    if words:
        print(f"  {'語の範囲':<12}: {words[0].start:.2f} 〜 {words[-1].end:.2f} 秒")
        dur = m.get("duration") or 0.0
        if dur:
            print(f"  {'物差しの被覆':<12}: {words[-1].end / dur:.3f}"
                  "  ← 1.0 に近くなければ、末尾の区間は測れない")
    print("  ※ 装置(cuda/cpu)と compute_type はキャッシュに入っていない。"
          "実行時に控えた値を併記すること。")


# **外れ量の表を出すたびに、必ずこの直後に添える文言。**
# 表だけ切り取られたときに最も危険な数字なので、表と同じ場所に置く。
DEVIATION_CAVEAT = """\
  ※ **local の外れ量は定義上ほぼ 0 になる。判定に使わないこと。**
     物差しは faster-whisper であり、local と同じ ASR である。
     「local の時刻が正確」なのではなく、**local を local 自身で測っている。**
     物差しを align に替えても同じ small 系列なので、この偏りは残る。
  ※ **最大値は上限であり、偽マッチを含みうる。単独で判定に使わないこと。**
     主指標は p90。中央値と p90 は分布の本体を見るので信用できる。
     **物差し自身が発話を落とし・崩すことがあり、その場合は偽マッチが生じる。
     閾値では防げない**——情報が物差しに無いためである。"""

# 物差しの素性を報告に必ず書くための文言。**省略しないこと。**
RULER_CAVEAT = """\
【物差しについて — 数字を読む前に】
  外れ量は「正解の時刻との差」ではない。**時刻の正解データは存在しない**
  (truth/verbatim.*.json は本文だけを人が直したもので、start/end は
   製品のローカル実行の出力がそのまま残っている)。
  代わりに src/align.py の単語時刻を物差しにしている。

  **物差しは faster-whisper であり、local と同じ ASR である。**
  したがって **local の外れ量が小さく出るのは当然**で、これは偶然の誤差では
  なく系統的な偏りである。**local を過大評価しないこと。**
  この指標はエンジン間の相対比較にしか使えず、真の時刻からのずれの
  絶対値ではない。"""


def rejection_breakdown(segs, *, min_chars: int = LOCATE_MIN_CHARS
                        ) -> dict[str, int]:
    """閾値でどれだけ落ちるかの内訳。**物差しが無くても出せる分だけ。**

    `LOCATE_MIN_CHARS` は区間の本文だけで決まるので、単語時刻が無くても
    数えられる。残りの 2 つ(票の優位比・被覆率)は物差しが要るので、
    `deviations()` を通したあとの `DeviationSet.reasons` で見る。
    """
    total = len(segs)
    short = sum(1 for _s, _e, t in segs
                if len(anchor.normalize(t or "")[0]) < min_chars)
    return {"総区間": total, "本文が短い(測れない)": short,
            "残り(物差しがあれば測れる候補)": total - short}


def _fmt(x, nd=2, dash="—"):
    return dash if x is None else f"{x:.{nd}f}"


def timing_rows(name: str, segs, audio_seconds: float,
                words=None, bands: bool = False) -> dict:
    """1 エンジンぶんの時刻の指標をまとめる。`words` が無ければ外れ量は空。

    `bands=True` は「この出力は正解のある 4 帯だけを繋いだもの」の意味。
    **帯モードでは打ち切り検出と前方跳躍が意味を持たない。**
      - カバー率: 帯は全長の一部なので、全長を分母にすると必ず低く出る
        (4 帯の最後は 3540 秒で、全長 4036 秒に対して 0.877)。**打ち切りではない。**
      - 前方跳躍: 帯と帯の間が丸ごと跳躍として出る(1260 秒など)。
        **帯の設計によるもので、エンジンの欠陥ではない。**
    どちらも「全長を 1 本で処理した出力」でのみ読むこと。
    """
    row = {
        "エンジン": name,
        "区間": len(segs),
        "単調性の破れ": monotonicity_breaks(segs),
        "前方跳躍の最大": max_forward_jump(segs),
        "重なり": overlap_count(segs),
        "1秒未満": short_segment_count(segs),
        "相関": length_duration_correlation(segs),
        "末尾のずれ": tail_drift(segs, audio_seconds),
        "最大ギャップ": max_gap(segs),
    }
    cov = coverage(segs, audio_seconds)
    row["カバー率"] = cov.ratio
    # **帯モードでは打ち切り判定を出さない。**全長を分母にすると必ず低く出るだけで、
    # 切れたわけではない(docstring 参照)。false positive を出すほうが有害。
    row["打ち切り"] = cov.truncated and not bands
    row["帯モード"] = bands
    if words:
        ds = deviations(segs, words)
        row["_dev"] = ds
        row["測れた"] = ds.measurable
        row["測れず"] = ds.unmeasurable
        row["最大外れ量"] = ds.max_gap
        row["中央外れ量"] = ds.median_gap
        row["p90外れ量"] = ds.p90_gap
    return row


def print_timing(rows: list[dict], words_present: bool) -> None:
    print("\n" + "=" * 78)
    print("判定2: 時刻の精度")
    print("=" * 78)
    print(f"{'エンジン':<24}{'区間':>6}{'破れ':>6}{'跳躍':>8}"
          f"{'重なり':>7}{'1秒未満':>8}{'相関':>7}{'カバー率':>9}")
    for r in rows:
        print(f"{r['エンジン']:<24}{r['区間']:>6}{r['単調性の破れ']:>6}"
              f"{_fmt(r['前方跳躍の最大'], 1):>8}{r['重なり']:>7}"
              f"{r['1秒未満']:>8}{_fmt(r['相関']):>7}"
              f"{_fmt(r['カバー率'], 3):>9}"
              + ("  [帯]" if r.get("帯モード") else "")
              + ("  ← 打ち切りの疑い" if r["打ち切り"] else ""))
    print("\n  破れ = 出力順で start が前より戻った件数。**重なりとは別物**"
          "(重なりは無害・破れは危険)")
    print("  相関 = 文字数と区間長の Pearson。**1.0 に近いほど機械的な按分の疑い**")
    if any(r.get("帯モード") for r in rows):
        print("\n  [帯] 印の行は正解のある 4 帯だけを繋いだ出力。"
              "**カバー率と跳躍を読まないこと。**")
        print("       カバー率が低いのは帯が全長の一部だからで、切れたのではない。")
        print("       跳躍は帯と帯の間(約 1260 秒)がそのまま出ているだけである。")
    if any(r["単調性の破れ"] == 0 for r in rows):
        print("\n  **破れ 0 を「壊れていない」と読まないこと。**"
              "既存の tools/bench_*.py は")
        print("       書き出し前に rows.sort() しているので"
              "(bench_cloud.py:100 / bench_models.py:124 /")
        print("       bench_amivoice.py:176)、**その SRT では破れが原理的に 0 になる。**")
        print("       新しいランナーは並べ替えずに書くこと。")

    if not words_present:
        print("\n  外れ量は測っていない(物差しの単語時刻が未指定)。"
              "--words を渡すと出る。")
        return
    print("\n" + "-" * 78)
    # **p90 が主指標。**最大値は上限として右端に置き、単独で読ませない。
    print(f"{'エンジン':<24}{'測れた':>8}{'測れず':>8}"
          f"{'中央':>8}{'p90(主)':>10}{'最大(上限)':>12}")
    for r in rows:
        print(f"{r['エンジン']:<24}{r['測れた']:>8}{r['測れず']:>8}"
              f"{_fmt(r['中央外れ量']):>8}{_fmt(r['p90外れ量']):>10}"
              f"{_fmt(r['最大外れ量']):>12}")
    print(DEVIATION_CAVEAT)
    print("\n  棄却の理由:")
    for r in rows:
        print(f"    {r['エンジン']:<24}{r['_dev'].reasons}")
    print("\n" + RULER_CAVEAT)


def main() -> int:
    import sys
    from pathlib import Path

    args = [a for a in sys.argv[1:]]
    if not args or "-h" in args or "--help" in args:
        print(__doc__)
        print("\n使い方:\n"
              "  python tools\report_timing.py --audio <秒> "
              "[--words <words.json>] <名前=出力> [<名前=出力> ...]\n")
        return 2

    audio_seconds = 0.0
    words = None
    words_path = None
    bands = False
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--audio":
            audio_seconds = float(args[i + 1]); i += 2
        elif a == "--words":
            words_path = args[i + 1]
            words = load_words_json(words_path); i += 2
        elif a == "--bands":
            bands = True; i += 1
        elif "=" in a:
            n, _, p = a.partition("=")
            pairs.append((n, p)); i += 1
        else:
            print(f"読めない引数: {a}"); return 2

    if audio_seconds <= 0:
        print("--audio に音声の実測長(秒)を渡してください。"
              "**名目のチャンク長を使わないこと**(§7-3)。")
        return 2

    rows = []
    print("\n" + "=" * 78)
    print("閾値による棄却(物差しが無くても出せる分)")
    print("=" * 78)
    for name, path in pairs:
        segs = load_ordered(path)
        rb = rejection_breakdown(segs)
        pct = 100.0 * rb["本文が短い(測れない)"] / max(1, rb["総区間"])
        print(f"  {name:<24} 総 {rb['総区間']:>5} / "
              f"短くて測れない {rb['本文が短い(測れない)']:>5} ({pct:5.1f}%) / "
              f"候補 {rb['残り(物差しがあれば測れる候補)']:>5}")
        rows.append(timing_rows(name, segs, audio_seconds, words, bands))
    print(f"\n  閾値 LOCATE_MIN_CHARS = {LOCATE_MIN_CHARS} 文字(正規化後)。"
          "\n  **落ちすぎるなら標本の偏りが大きく、判定2 が成立しない。**"
          "\n  値を変えるなら実データの外れ量を見る前に決めること"
          "(見てから動かすと都合のいい値を選べてしまう)。")

    if words is not None:
        print_ruler(words_path, words)
    print_timing(rows, words is not None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
