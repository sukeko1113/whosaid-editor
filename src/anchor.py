"""Gemini の本文と、実測の単語時刻を突き合わせる(照合)。

やること:
    1. 正規化(NFKC・記号落とし)と、元の位置へ戻る写像表づくり
    2. 区間ごとに、その時刻の周りの単語だけを見て文字を突き合わせる
    3. 一致した文字の時刻から、区間の始まりと終わりを引き直す
    4. どれだけ乗ったか(被覆率)を返す。低ければ呼び出し側が提案を諦める

ここは純粋関数だけで書いてある。音声もモデルも要らないので、テキストと
単語のフィクスチャだけで全部テストできる。

**区間ごとに窓を切る理由**(設計書 §3.1 からの変更・2026-08-13 に決定):
全文どうしを difflib に掛けると 4 万文字(52 分の会議)で 66 秒かかる。
O(n^2) なので 2 時間の録音では 5 分を超える。区間の時刻の周りだけを見れば
n が千文字弱に落ちて 0.16 秒で済む(いずれも実測)。
速さ以上に効くのは誤マッチが構造的に起きなくなること。全文照合では
「はい」が 3 分先の「はい」に当たり得るが、窓を切ればそもそも届かない。
窓から外れた区間は黙って間違えず「照合できなかった」として返る。

渡す時刻は**実音声の時刻**(再生位置)であること。作業ファイルの保存値には
ずれ補正が乗っていない区間があるので、呼び出し側で補正を足してから渡す。
"""
from __future__ import annotations

import bisect
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional, Sequence

from . import lang
from .align import Word


# 一致とみなす最短の長さ。**言語で変わる**ので lang.py が持つ。
#   日本語 3  … 1〜2 文字の一致が偶然でも頻発する
#   英語  10  … 3 文字では the / and / for が当たり続ける
# 値の根拠(実測)は lang.py の各プロファイルに書いてある。
#
# **この定数は互換のために残してある。**import した時点の値で固定されるので、
# lang.use() の切り替えが効かない。新しい呼び出しは min_block=None のまま
# 渡し、関数の中で現在のプロファイルから引くこと。
MIN_BLOCK = lang.JA.matching.min_block

# 区間の時刻の前後、何秒ぶんの単語を探すか。実測のドリフトは +6.8 秒程度
# だったので、4 倍以上の余裕を取ってある(§12 のキャリブレーション対象)。
WINDOW = 30.0

# まず狭く探し、十分に当たらなければ WINDOW まで広げる。
# 会議では同じ言い回しが何度も出る(「そうですね」「はい、わかりました」)。
# difflib は窓の中で最も手前の一致を選ぶので、いきなり広く探すと
# 30 秒前の同じ言葉に当たり、ずれていない区間にずれた提案を出してしまう。
NEAR_WINDOW = 5.0

# 狭い窓でこれだけ乗ったら、広げずにそれを採る。
GOOD_ENOUGH = 0.8

# 上の窓で見つからなかった区間だけ、もう一度広く探すときの幅。
# 数が少ないので広げても安い。
WIDE_WINDOW = 120.0

# 単調に進ませるときの戻り幅(秒)。区間の切れ目は両者で少しずれるので、
# 直前の一致の終わりから少しだけ戻ることは許す。
SWEEP_SLACK = 1.0

# これだけ乗った一致でなければ、掃引位置を進めない。
# 偶然 3〜4 文字だけ当たった区間で先へ進んでしまうと、その後ろの区間が
# まとめて範囲外になり、1 件の誤マッチがファイル全体を壊す。
# 進めないだけで、その区間の測定結果自体は返す(呼び出し側が被覆率で弾く)。
SWEEP_MIN_COVERAGE = 0.5


@dataclass
class Measured:
    """1 区間ぶんの実測結果。"""

    start: float            # 引き直した開始(余白込み)
    end: float              # 引き直した終了(余白込み)
    coverage: float         # 区間の文字のうち、一致に乗った割合(0〜1)
    matched: int            # 一致した文字数
    total: int              # 区間の文字数(正規化後)
    window: float           # どの幅の窓で見つけたか(根拠に出す)
    # 本文の頭・末尾それぞれ何文字がアンカーに乗らなかったか。
    # 開始は最初の一致から、終了は最後の一致から取るので、端が欠けている
    # ほどその側の時刻は信用できない(実測: 末尾 3 字以上欠けた提案は
    # 終了が外れ、2 字以下は全部当たりだった)。
    head_gap: int = 0
    tail_gap: int = 0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Track:
    """実測側を 1 本の文字列として扱えるようにしたもの。

    文字ごとに時刻を持つ。単語の中は文字数で按分する(逐語の粒度なら
    これで十分。実測では 45 語中 35 語が 1 文字だった)。
    """

    text: str
    starts: list[float]     # 昇順であること(range() が二分探索する)
    ends: list[float]       # 同上

    def __len__(self) -> int:
        return len(self.text)

    def range(self, t0: float, t1: float) -> tuple[int, int]:
        """[t0, t1] に掛かる文字の範囲(半開区間)を返す。"""
        lo = bisect.bisect_left(self.ends, t0)
        hi = bisect.bisect_right(self.starts, t1)
        return lo, max(lo, hi)


def _droppable(ch: str) -> bool:
    """照合の邪魔になる文字か(空白・句読点・記号・制御文字)。

    実測側は句読点が単語にくっついて返る('ありがとうございます。')。
    Gemini 側の句読点の打ち方とは揃わないので、両方から落として比べる。
    """
    return unicodedata.category(ch)[0] in ("Z", "C", "P", "S")


def normalize(text: str) -> tuple[str, list[int]]:
    """NFKC 正規化して記号を落とす。

    戻り値は (正規化後の文字列, 各文字が元の何文字目から来たか)。
    写像表を持つのは、分割候補(§4)で音声側の切れ目を本文の位置に
    戻す必要があるため。

    1 文字ずつ正規化するので、全体を NFKC に掛けた場合と結果が違うことが
    ある(「か」+濁点のように、分かれて書かれた濁点は合体しない)。
    位置の対応を正確に保つほうを取っている。合体しなかった濁点は
    1 文字ぶん一致しないだけで、被覆率に吸収される。
    """
    out: list[str] = []
    origin: list[int] = []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch):
            if _droppable(c):
                continue
            out.append(c)
            origin.append(i)
    return "".join(out), origin


def prepare(words: Sequence[Word]) -> Track:
    """単語の並びを、文字ごとに時刻を持つ 1 本の文字列にする。

    時刻順に並べ直してから作る。faster-whisper は前から順に返すが、
    アライナを差し替えたときに順序が崩れていると、窓を切る二分探索が
    静かに嘘をつく(そこだけは崩れてはいけない)。
    終了時刻も同じ理由で、前の文字より前に戻らないようにしておく
    (単語どうしが重なって返る場合。faster-whisper では起きない)。
    """
    ordered = sorted(words, key=lambda w: (w.start, w.end))
    raw: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    for w in ordered:
        n = len(w.text)
        if n == 0:
            continue
        span = max(0.0, w.end - w.start)
        for i in range(n):
            starts.append(w.start + span * i / n)
            end = w.start + span * (i + 1) / n
            ends.append(max(end, ends[-1]) if ends else end)
        raw.append(w.text)

    text, origin = normalize("".join(raw))
    return Track(text=text,
                 starts=[starts[i] for i in origin],
                 ends=[ends[i] for i in origin])


def measure(
    text: str,
    track: Track,
    t0: float,
    t1: float,
    *,
    min_block: Optional[int] = None,
) -> Optional[Measured]:
    """1 区間ぶんを、[t0, t1] の範囲の実測と突き合わせる。

    見つからなければ None(照合不能)。当てずっぽうの時刻を返すくらいなら、
    照合できなかったと言うほうがいい。

    始まりは「最初に一致した文字」の開始、終わりは「最後に一致した文字」の
    終了から取る(§3.2)。**測ったままの生の値**で、余白は付けない。
    余白や頭の欠けの補正は提案を作る側(inspection.py)の方針であり、
    実測と混ぜると較正のときに切り分けられなくなる。
    区間の頭が丸ごと聞き取れていない場合、始まりはその分だけ後ろに出る。
    head_gap と被覆率を一緒に返すので、呼び出し側で補正・棄却できる。
    """
    # **既定は import 時ではなく呼ばれた時に決める。**既定引数に書くと
    # 値が import 時に固定され、lang.use() の切り替えが効かない
    # (feature/en-test では、この罠のせいで検査が min_block を明示的に
    #  渡す必要があった)。
    if min_block is None:
        min_block = lang.current().matching.min_block
    norm, _ = normalize(text)
    if not norm:
        return None
    lo, hi = track.range(t0, t1)
    window = track.text[lo:hi]
    if not window:
        return None

    sm = SequenceMatcher(None, norm, window, autojunk=False)
    blocks = [m for m in sm.get_matching_blocks() if m.size >= min_block]
    if not blocks:
        return None

    first, last = blocks[0], blocks[-1]
    start = track.starts[lo + first.b]
    end = track.ends[lo + last.b + last.size - 1]
    matched = sum(m.size for m in blocks)
    return Measured(
        start=max(0.0, start),
        end=max(start, end),
        coverage=matched / len(norm),
        matched=matched,
        total=len(norm),
        window=t1 - t0,
        head_gap=first.a,
        tail_gap=len(norm) - (last.a + last.size),
    )


def measure_segments(
    spans: Sequence[tuple[str, float, float]],
    words: Sequence[Word],
    *,
    window: float = WINDOW,
    near_window: float = NEAR_WINDOW,
    wide_window: float = WIDE_WINDOW,
    good_enough: float = GOOD_ENOUGH,
    min_block: Optional[int] = None,
) -> list[Optional[Measured]]:
    """全区間ぶんまとめて実測と突き合わせる。

    spans は (本文, いまの開始, いまの終了)。時刻は**実音声の時刻**で渡す
    こと(ずれ補正を足したあとの値)。

    まず近く(near_window)を探し、十分に当たらなければ window まで広げる。
    同じ言い回しが窓の中に何度も出てくるとき、difflib は最も手前の一致を
    選ぶので、いきなり広く探すと手前の同じ言葉に当たってしまう。

    前から順に見て、次の区間は直前の区間が一致した位置より後ろから探す。
    同じ文言が前にもあるとき(「はい」「そうですね」)に前へ戻って当たるのを
    防ぐ。ここで弾かれるのは時刻が前後する一致なので、設計書 §3.1 の
    時刻単調性フィルタと同じ役目を、窓を切る側でまとめて果たしている。
    ただし掃引位置を進めるのは、十分に乗った一致だけ(SWEEP_MIN_COVERAGE)。
    数文字しか当たっていない区間で先へ進むと、その後ろが軒並み範囲外になる。

    1 度目で見つからなかった区間だけ、広い窓でもう一度探す。数が少ないので
    広げても安く、ドリフトが大きい帯を取りこぼさずに済む。
    """
    if min_block is None:
        min_block = lang.current().matching.min_block
    track = prepare(words)
    results: list[Optional[Measured]] = [None] * len(spans)
    if not track.text:
        return results

    floor = 0.0            # ここより前は探さない(単調に進ませる)
    steps = [w for w in (near_window, window) if w <= window] or [window]
    for i, (text, start, end) in enumerate(spans):
        best: Optional[Measured] = None
        for w in steps:
            got = measure(text, track, max(floor, start - w), end + w,
                          min_block=min_block)
            if got is not None and (best is None or got.coverage > best.coverage):
                best = got
            if best is not None and best.coverage >= good_enough:
                break       # 近くで十分に当たった。わざわざ広げない
        if best is not None:
            results[i] = best
            if best.coverage >= SWEEP_MIN_COVERAGE:
                floor = max(floor, best.end - SWEEP_SLACK)

    # 取りこぼしを広い窓で拾い直す。ここでは単調の制約を「前後の一致の間」
    # に限る(前後が決まっていれば、その間から出ることはない)。
    for i, (text, start, end) in enumerate(spans):
        if results[i] is not None:
            continue
        lo = max(0.0, start - wide_window)
        hi = end + wide_window
        prev = next((results[j] for j in range(i - 1, -1, -1) if results[j]), None)
        nxt = next((results[j] for j in range(i + 1, len(spans)) if results[j]), None)
        if prev is not None:
            lo = max(lo, prev.end - SWEEP_SLACK)
        if nxt is not None:
            hi = min(hi, nxt.start + SWEEP_SLACK)
        if hi <= lo:
            continue
        results[i] = measure(text, track, lo, hi, min_block=min_block)
    return results
