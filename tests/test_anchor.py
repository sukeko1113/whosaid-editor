"""anchor.py(照合)の検査。モデルも音声も要らないので CI でも回る。

    .venv\\Scripts\\python.exe tests\\test_anchor.py

実物の単語列(tests/fixtures/words_sapi_sample.json)も使う。これは
test_align_integration.py が実 whisper から作ったもので、句読点が単語に
くっついてくる癖や、「第一号」が「第1号」と返る癖がそのまま入っている。
作り物のフィクスチャだけで固めると、実物で外れる。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.align import Word                                      # noqa: E402
from src.anchor import (                                        # noqa: E402
    Track,
    measure,
    measure_segments,
    normalize,
    prepare,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "words_sapi_sample.json"

# 【英語テスト用ブランチ】anchor.MIN_BLOCK の既定は英語向けに 10 へ上げた。
# 以下の日本語フィクスチャは「照合アルゴリズムそのもの」を見るためのもので、
# 閾値の値とは無関係。当時の値を呼び出し側に明示して意味を保つ
# (閾値を緩めて通しているのではなく、日本語の検査には日本語の閾値を渡している)。
JA_MIN_BLOCK = 3


def words(*specs: tuple[str, float, float]) -> list[Word]:
    return [Word(text=t, start=s, end=e) for t, s, e in specs]


def evenly(text: str, start: float, per_char: float = 0.2) -> list[Word]:
    """1 文字ずつ等間隔に喋った、という単語列を作る(実測の粒度に近い)。"""
    return [Word(text=ch, start=start + i * per_char,
                 end=start + (i + 1) * per_char)
            for i, ch in enumerate(text)]


# ======================================================================
# 正規化
# ======================================================================

def test_normalize_drops_marks_and_space():
    got, origin = normalize("はい、そうです。 ね!")
    assert got == "はいそうですね"
    # 落とした分だけ元の位置がずれる(写像表で戻せる)
    assert [origin[0], origin[1], origin[2]] == [0, 1, 3]
    assert origin[-1] == len("はい、そうです。 ")


def test_normalize_folds_width_and_digits():
    got, _ = normalize("第１号　ＡＢＣ")
    assert got == "第1号ABC"


def test_normalize_keeps_position_map_after_folding():
    got, origin = normalize("Ａ１")
    assert got == "A1"
    assert origin == [0, 1]


def test_normalize_empty_when_only_marks():
    got, origin = normalize("。、…!?　")
    assert got == "" and origin == []


# ======================================================================
# 実測側の下ごしらえ(Track)
# ======================================================================

def test_prepare_shares_time_inside_a_word():
    track = prepare(words(("あいう", 0.0, 3.0)))
    assert track.text == "あいう"
    assert track.starts == [0.0, 1.0, 2.0]
    assert track.ends == [1.0, 2.0, 3.0]


def test_prepare_drops_marks_but_keeps_times():
    # 実測は句読点を単語にくっつけて返す('ありがとうございます。')
    track = prepare(words(("はい。", 0.0, 3.0), ("どうも", 3.0, 6.0)))
    assert track.text == "はいどうも"
    assert track.starts[0] == 0.0
    assert abs(track.starts[2] - 3.0) < 1e-9        # 句点の次は次の単語
    assert abs(track.ends[-1] - 6.0) < 1e-9


def test_prepare_sorts_out_of_order_words():
    """アライナを差し替えたときに順序が崩れても、二分探索が嘘をつかない。"""
    track = prepare(words(("うえ", 4.0, 6.0), ("あい", 0.0, 2.0)))
    assert track.text == "あいうえ"
    assert track.starts == sorted(track.starts)
    assert track.ends == sorted(track.ends)


def test_prepare_ignores_empty_words():
    track = prepare(words(("", 0.0, 1.0), ("あ", 1.0, 2.0)))
    assert track.text == "あ"


def test_track_range_picks_the_window():
    track = prepare(evenly("あいうえおかきくけこ", 0.0, 1.0))    # 1 文字 1 秒
    lo, hi = track.range(3.0, 5.0)
    # 端に掛かっているだけの文字も入れる(切り落とすと境界の文字を落とす)
    assert track.text[lo:hi] == "うえおか"
    lo, hi = track.range(100.0, 200.0)
    assert lo == hi                             # 範囲外は空


# ======================================================================
# 1 区間の照合
# ======================================================================

def test_measure_exact_match():
    track = prepare(evenly("きょうはあついですね", 10.0, 0.5))
    got = measure("きょうはあついですね", track, 0.0, 100.0)
    assert got is not None
    assert abs(got.start - 10.0) < 1e-9
    assert abs(got.end - 15.0) < 1e-9
    assert got.coverage == 1.0 and got.matched == got.total == 10


def test_measure_ignores_punctuation_difference():
    """本文の句読点の打ち方は両者で揃わない。落としてから比べる。"""
    track = prepare(words(("はい、そうですね。", 5.0, 8.0)))
    got = measure("はいそうですね", track, 0.0, 100.0, min_block=JA_MIN_BLOCK)
    assert got is not None and got.coverage == 1.0


def test_measure_short_coincidence_is_not_an_anchor():
    """日本語は 1〜2 文字の一致が偶然でも起きる。3 文字未満は拾わない。"""
    track = prepare(evenly("かきくけこさしすせそ", 0.0, 0.5))
    assert measure("かき", track, 0.0, 100.0,
                   min_block=JA_MIN_BLOCK) is None          # 2 文字だけ一致
    assert measure("かきく", track, 0.0, 100.0,
                   min_block=JA_MIN_BLOCK) is not None       # 3 文字なら拾う


def test_measure_reports_partial_coverage():
    """フィラーや誤認識で一致しない分は被覆率に出る。"""
    track = prepare(evenly("それでは議事に入ります", 0.0, 0.5))
    got = measure("えーとあのーそれでは議事に入ります", track, 0.0, 100.0)
    assert got is not None
    assert got.matched == 11 and got.total == 17
    assert 0.6 < got.coverage < 0.7
    # 始まりは「一致した最初の文字」から取る(フィラーの分は含まない)
    assert got.start == 0.0
    # 頭の 6 文字(えーとあのー)が乗らなかった、と分かる
    assert got.head_gap == 6 and got.tail_gap == 0


def test_measure_reports_tail_gap():
    """末尾が聞き取れていないことも分かる(終了時刻の信用判断に使う)。"""
    track = prepare(evenly("それでは議事に入ります", 0.0, 0.5))
    got = measure("それでは議事に入りますけれども", track, 0.0, 100.0)
    assert got is not None
    assert got.head_gap == 0 and got.tail_gap == 4


def test_measure_outside_the_window_is_unmatched():
    track = prepare(evenly("あいうえお", 500.0, 0.5))
    assert measure("あいうえお", track, 0.0, 60.0) is None


def test_measure_empty_text_is_unmatched():
    track = prepare(evenly("あいうえお", 0.0, 0.5))
    assert measure("", track, 0.0, 100.0) is None
    assert measure("。、…", track, 0.0, 100.0) is None


# ======================================================================
# 全区間の照合(窓と単調揃え)
# ======================================================================

def _three_segments() -> tuple[list[tuple[str, float, float]], list[Word]]:
    """3 区間ぶん。実測は 0 秒から 1 文字 0.5 秒で並ぶ。"""
    a, b, c = "おはようございます", "きょうはさむいですね", "そうですねまったく"
    ws = (evenly(a, 0.0, 0.5) + evenly(b, 6.0, 0.5) + evenly(c, 12.0, 0.5))
    spans = [(a, 0.0, 4.5), (b, 6.0, 11.0), (c, 12.0, 16.5)]
    return spans, ws


def test_measure_segments_matches_each_span():
    spans, ws = _three_segments()
    got = measure_segments(spans, ws, min_block=JA_MIN_BLOCK)
    assert all(g is not None for g in got)
    assert got[0].start == 0.0
    assert abs(got[1].start - 6.0) < 1e-9
    assert abs(got[2].start - 12.0) < 1e-9
    assert all(g.coverage == 1.0 for g in got)


def test_measure_segments_finds_drifted_times():
    """本文の時刻が数秒ずれていても、窓の中なら見つかる(これが目的)。"""
    spans, ws = _three_segments()
    drifted = [(t, s + 6.8, e + 6.8) for t, s, e in spans]
    got = measure_segments(drifted, ws, min_block=JA_MIN_BLOCK)
    assert all(g is not None for g in got)
    assert abs(got[1].start - 6.0) < 1e-9      # 実測の位置に戻る


def test_measure_segments_narrow_window_then_wide():
    """窓から外れた区間は 1 度目で落ち、広い窓で拾い直す。"""
    spans, ws = _three_segments()
    far = list(spans)
    far[1] = (spans[1][0], spans[1][1] + 90.0, spans[1][2] + 90.0)
    assert measure(far[1][0], prepare(ws), far[1][1] - 30.0,
                   far[1][2] + 30.0) is None           # 1 度目の窓では届かない
    got = measure_segments(far, ws)
    assert got[1] is not None                          # 2 度目で拾える
    assert abs(got[1].start - 6.0) < 1e-9


def test_measure_segments_does_not_go_backwards():
    """同じ文言が前にもあるとき、前へ戻って当たらない。"""
    phrase = "そうですねたしかに"
    ws = evenly(phrase, 0.0, 0.5) + evenly("べつのはなしです", 10.0, 0.5) \
        + evenly(phrase, 20.0, 0.5)
    spans = [(phrase, 0.0, 4.5),
             ("べつのはなしです", 10.0, 14.0),
             (phrase, 20.0, 24.5)]
    got = measure_segments(spans, ws, min_block=JA_MIN_BLOCK)
    assert got[0] is not None and got[2] is not None
    assert got[0].start < got[2].start
    assert abs(got[2].start - 20.0) < 1e-9     # 後ろの方に当たる


def test_measure_segments_prefers_the_near_occurrence():
    """同じ言い回しが窓の中に 2 つあるとき、近いほうを採る。

    会議では「そうですね」「はい、わかりました」が何度も出る。difflib は
    窓の中で最も手前の一致を選ぶので、近くを先に見ないと 20 秒前の同じ
    言葉に当たり、ずれていない区間にずれた提案が出る。
    """
    phrase = "そうですねたしかに"
    ws = evenly(phrase, 0.0, 0.5) + evenly("あいだのはつげんです", 8.0, 0.5) \
        + evenly(phrase, 20.0, 0.5)
    # 20 秒側の発言。窓を広く取ると 0 秒側にも届く
    got = measure_segments([(phrase, 20.0, 24.5)], ws, min_block=JA_MIN_BLOCK)
    assert got[0] is not None
    assert abs(got[0].start - 20.0) < 1e-9


def test_measure_segments_still_finds_far_matches():
    """近くに無ければ、ちゃんと広げて探す(近さを優先しすぎない)。"""
    phrase = "ここにしかないはつげん"
    ws = evenly(phrase, 40.0, 0.5)
    got = measure_segments([(phrase, 20.0, 24.5)], ws)      # 20 秒ずれている
    assert got[0] is not None
    assert abs(got[0].start - 40.0) < 1e-9


def test_measure_segments_without_words():
    spans, _ = _three_segments()
    assert measure_segments(spans, []) == [None, None, None]


def test_measure_segments_keeps_unmatched_as_none():
    """実測に無い区間(転写から落ちた発言)は、当てずっぽうを返さない。"""
    spans, ws = _three_segments()
    spans.insert(1, ("ぜんぜんべつのはなし", 4.6, 5.9))
    got = measure_segments(spans, ws)
    assert got[1] is None


def test_measure_segments_flags_a_lucky_short_match():
    """たまたま数文字だけ一致することはある。被覆率が低く出れば弾ける。

    これが MIN_COVERAGE(呼び出し側の閾値)が要る理由。照合できた/できないの
    2 値では、偶然の一致から作った時刻を提案してしまう。
    """
    spans, ws = _three_segments()       # 3 つ目に「まったく」が含まれる
    spans.insert(1, ("まったくちがうはつげんです", 4.6, 5.9))
    got = measure_segments(spans, ws, min_block=JA_MIN_BLOCK)
    assert got[1] is not None and got[1].matched == 4
    assert got[1].coverage < 0.4        # 13 文字中 4 文字しか乗っていない


def test_a_lucky_short_match_does_not_block_the_rest():
    """数文字しか当たっていない区間で掃引位置を進めない。

    進めてしまうと、後ろの区間が軒並み範囲外になり、1 件の誤マッチで
    ファイル全体の照合が壊れる。
    """
    spans, ws = _three_segments()
    # この区間は 3 つ目の「まったく」に当たる(実測ではずっと後ろ)
    spans.insert(1, ("まったくちがうはつげんです", 4.6, 5.9))
    got = measure_segments(spans, ws, min_block=JA_MIN_BLOCK)
    assert got[2] is not None and got[3] is not None, "後ろの区間が巻き添えになった"
    assert abs(got[2].start - 6.0) < 1e-9
    assert abs(got[3].start - 12.0) < 1e-9


# ======================================================================
# 実物の単語列で照合する
# ======================================================================

def test_real_words_from_whisper():
    if not FIXTURE.exists():
        print("    (フィクスチャが無いので飛ばす: "
              "python tests\\test_align_integration.py で作れる)")
        return
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ws = [Word(**w) for w in data["words"]]

    # Gemini 側の本文のつもりで、喋らせた文をそのまま区間に切る。
    # 実測側は「お集まりいただき → を集まり頂き」「第一号 → 第1号」と
    # 返っているので、ここは完全一致にはならない。
    spans = [
        ("本日はお忙しい中お集まりいただきありがとうございます。", 0.0, 3.5),
        ("それでは第一号議案について事務局から説明をお願いします。", 4.5, 9.5),
        ("はい。お手元の資料の三ページをご覧ください。", 10.0, 14.0),
    ]
    got = measure_segments(spans, ws, min_block=JA_MIN_BLOCK)
    assert all(g is not None for g in got), "実物で照合できない区間がある"
    # 表記ゆれがあっても 6 割は乗る(MIN_COVERAGE の初期値が 0.60)
    assert all(g.coverage >= 0.6 for g in got), [g.coverage for g in got]
    # 前から順に並び、隣とぶつからない
    assert got[0].start < got[1].start < got[2].start
    assert got[0].end <= got[1].start + 1.0
    # 1 つ目は音声の先頭から始まる
    assert got[0].start < 0.3


def test_real_words_survive_drifted_input():
    if not FIXTURE.exists():
        return
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ws = [Word(**w) for w in data["words"]]
    spans = [
        ("本日はお忙しい中お集まりいただきありがとうございます。", 6.8, 10.3),
        ("それでは第一号議案について事務局から説明をお願いします。", 11.3, 16.3),
    ]
    got = measure_segments(spans, ws, min_block=JA_MIN_BLOCK)
    assert all(g is not None for g in got)
    assert got[0].start < 0.3           # ずれた入力でも実測の位置に戻る


# ======================================================================
# 【英語テスト用ブランチ】英語の既定値(MIN_BLOCK = 10)を押さえる
#
# 照合器が見るのは normalize() 後の**空白を落とした**文字列で、これは
# 両側に掛かる(anchor 側も align 側も w.word.strip() で空白を捨てている)。
# つまり英語も 1 本の文字列として衝突するので、日本語より偶然一致が長い。
# ======================================================================

def en_words(text: str, start: float, per_char: float = 0.09) -> list[Word]:
    """英語を 1 文字ずつ喋った単語列にする(約 11 字/秒 = 実際の話速)。"""
    return [Word(text=ch, start=start + i * per_char,
                 end=start + (i + 1) * per_char)
            for i, ch in enumerate(text)]


def test_en_short_coincidence_is_not_an_anchor():
    """英語は 3 文字の一致が偶然でも頻発する。既定の 10 文字未満は拾わない。"""
    track = prepare(en_words("the government should adopt this policy", 0.0))
    # "the" は 3 文字。無関係な区間でも当たってしまうので拾わない
    assert measure("the", track, 0.0, 100.0) is None
    assert measure("and", track, 0.0, 100.0) is None
    # 9 文字でもまだ拾わない(実測の偶然一致の中央値〜p95 の帯)
    assert measure("governmen", track, 0.0, 100.0) is None
    # 10 文字で初めてアンカーになる
    assert measure("government", track, 0.0, 100.0) is not None


def test_en_unrelated_text_does_not_anchor():
    """無関係な英語どうしは、既定値なら当たらない。

    MIN_BLOCK = 3 のままだと the / and / for が当たり、measure() が
    blocks[0] と blocks[-1] から時刻を取るため区間の境界がそこへ動く。
    """
    spoken = ("ladies and gentlemen the negative team believes "
              "the proposal is unworkable")
    track = prepare(en_words(spoken, 0.0))
    unrelated = "we would ask the affirmative to explain how they finance this"
    assert measure(unrelated, track, 0.0, 100.0) is None
    # 当時の日本語向けの値では、偶然の一致を拾ってしまう
    assert measure(unrelated, track, 0.0, 100.0, min_block=JA_MIN_BLOCK) is not None


def test_en_genuine_match_still_lands():
    """本物の一致は既定値でもちゃんと乗る(閾値を上げすぎていない)。"""
    spoken = "our first contention is that the current system fails to protect"
    track = prepare(en_words(spoken, 5.0))
    got = measure(spoken, track, 0.0, 100.0)
    assert got is not None
    assert got.coverage == 1.0
    assert abs(got.start - 5.0) < 1e-9


def test_en_short_response_is_unmatched_not_deleted():
    """短い応答は「照合不能」になる。これは意図した挙動。

    "Yes" は正規化して 3 文字で閾値を下回り、measure() は None を返す。
    本文には一切触れない(呼び出し側は unmatched として数えるだけ)。
    日本語の「はい」= 2 文字も MIN_BLOCK = 3 の下で同じだった。

    短い区間だけ閾値を下げる仕掛けは入れていない。入れると "Yes" が
    30 秒の窓の中のどの yes にも当たり、自信を持って間違った時刻を出す。
    """
    spoken = ("does the study say that yes it does and we have shown "
              "you the page reference already")
    track = prepare(en_words(spoken, 0.0))
    for short in ("Yes", "No", "Yes.", "No."):
        assert measure(short, track, 0.0, 100.0) is None, short


def test_en_measure_segments_finds_drifted_times():
    """英語でも、時刻がずれた区間が実測の位置に戻る(点検の目的そのもの)。"""
    a = "thank you mister chairperson we stand in firm affirmation"
    b = "point of information about the implementation cost"
    ws = en_words(a, 0.0) + en_words(b, 10.0)
    spans = [(a, 6.8, 12.0), (b, 16.8, 22.0)]       # 6.8 秒ドリフト
    got = measure_segments(spans, ws)
    assert all(g is not None for g in got)
    assert abs(got[0].start - 0.0) < 1e-6
    assert abs(got[1].start - 10.0) < 1e-6


# ======================================================================
# pytest が無い環境向けの簡易ランナー
# ======================================================================

if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except Exception as e:
                failures += 1
                import traceback
                print(f"  FAIL {name}: {e}")
                traceback.print_exc()
    print(f"\n{'FAILED' if failures else 'ALL PASSED'} ({failures} failures)")
    sys.exit(1 if failures else 0)
