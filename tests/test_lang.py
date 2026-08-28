"""言語プロファイル(src/lang.py)の検査。

    .venv\\Scripts\\python.exe tests\\test_lang.py

この器の目的は「言語で変わる設定の入れ忘れを止める」こと。だから
**埋め忘れが落ちること**と、**切り替えが実際に効くこと**を見る。

切り替えは値ではなく**挙動**で確かめる。定数を読んで比べるだけでは、
その定数が使われていない経路を見逃す(2026-08-27 にキャッシュキーで
実際に見逃した)。
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src import lang                                        # noqa: E402


# ======================================================================
# 埋め忘れが落ちること
# ======================================================================

def test_no_field_has_a_default():
    """**既定値を持たせない。**

    既定があると、新しい言語を足したときの埋め忘れが「静かに日本語の値」に
    なる。埋めるまで構築できない形にしておく。
    """
    groups = [lang.LanguageProfile, lang.Matching, lang.Speech,
              lang.TextJoin, lang.Labels, lang.Asr, lang.Prompt]
    for cls in groups:
        for f in dataclasses.fields(cls):
            assert f.default is dataclasses.MISSING, f"{cls.__name__}.{f.name}"
            assert f.default_factory is dataclasses.MISSING, \
                f"{cls.__name__}.{f.name}"


def test_incomplete_profile_cannot_be_built():
    """項目を 1 つ落としたら構築できないこと(器の要点)。"""
    try:
        lang.Matching(min_block=10, min_matched=22, tail_gap_limit=8)
    except TypeError:
        pass
    else:
        raise AssertionError("min_density を落としても構築できてしまった")


def test_every_profile_is_complete():
    for code, p in lang.PROFILES.items():
        assert p.code == code, f"{code} の code が一致しない"
        for group in ("matching", "speech", "text", "labels", "asr", "prompt"):
            assert getattr(p, group) is not None, f"{code}.{group}"


def test_calibrated_by_is_filled():
    """**閾値がどの測定から来たかを必ず書く。**

    空欄を許すと、他の言語の値をコピーして済ませられてしまう。
    """
    for code, p in lang.PROFILES.items():
        assert p.calibrated_by and len(p.calibrated_by) > 20, (
            f"{code} の calibrated_by が書かれていません。"
            "閾値は測って決めた値で、出所が分からないと使えません。")


def test_english_profile_says_it_is_n_of_one():
    """英語は 1 試合しか測っていない。その事実を器に残しておく。"""
    assert "n=1" in lang.EN.calibrated_by


# ======================================================================
# 未知の言語は落ちること(既定へ落とさない)
# ======================================================================

def test_unknown_language_raises():
    for code in ("zh", "", "JA", "en-US"):
        try:
            lang.get(code)
        except KeyError as e:
            assert "代用しないこと" in str(e), str(e)
        else:
            raise AssertionError(f"{code!r} が通ってしまった")


def test_unknown_language_does_not_fall_back_to_japanese():
    """**フォールバックは親切に見えて、静かに壊れる形。**"""
    try:
        got = lang.get("zh")
    except KeyError:
        return
    raise AssertionError(f"既定に落ちている: {got.code}")


# ======================================================================
# 切り替えが効くこと — 値ではなく挙動で見る
# ======================================================================

def test_default_is_japanese():
    assert lang.DEFAULT == "ja"
    assert lang.current().code == "ja"


def test_use_switches_and_restores():
    before = lang.current().code
    try:
        assert lang.use("en").code == "en"
        assert lang.current().code == "en"
        assert lang.use("ja").code == "ja"
        assert lang.current().code == "ja"
    finally:
        lang.use(before)


def test_current_reflects_the_switch():
    """`from .lang import CURRENT` の罠を、器の側で塞いでいること。

    current() は呼ぶたびに今の値を返す。モジュール変数を直接 import すると
    束縛された時点の値のままになる(既定引数と同じ罠)。
    """
    before = lang.current().code
    try:
        lang.use("en")
        assert lang.current().matching.min_block == 10
        lang.use("ja")
        assert lang.current().matching.min_block == 3
    finally:
        lang.use(before)


# ======================================================================
# 値そのもの(実測で決めたもの)
# ======================================================================

def test_measured_values_match_the_report():
    """2026-08-27 の実測レポートの値と一致すること。"""
    assert (lang.JA.matching.min_block, lang.EN.matching.min_block) == (3, 10)
    assert (lang.JA.matching.min_matched, lang.EN.matching.min_matched) == (8, 22)
    assert (lang.JA.matching.tail_gap_limit,
            lang.EN.matching.tail_gap_limit) == (3, 8)
    assert (lang.JA.matching.min_density,
            lang.EN.matching.min_density) == (1.5, 4.0)
    assert (lang.JA.speech.chars_per_second,
            lang.EN.speech.chars_per_second) == (4.5, 11.0)


def test_join_rules_differ():
    assert lang.JA.text.word_separator == ""
    assert lang.EN.text.word_separator == " "
    assert lang.JA.text.fragment_comma == "、"
    assert lang.EN.text.fragment_comma == ", "


def test_label_vocabularies_do_not_overlap_between_kinds():
    """同じ語が「複数」と「不明」の両方に入っていないこと。"""
    for p in lang.PROFILES.values():
        both = set(p.labels.multi_words) & set(p.labels.unknown_words)
        assert not both, f"{p.code}: {both}"


def test_asr_language_codes():
    assert lang.JA.asr.whisper_language == "ja"
    assert lang.EN.asr.whisper_language == "en"


def test_english_style_prompt_is_english():
    """日本語の例文が英語プロファイルに残っていないこと。

    英語音声に日本語の例文を initial_prompt で与えると害になる。
    """
    got = lang.EN.asr.style_prompt
    assert got, "英語の例文が空"
    assert not any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿"
                   for ch in got), "英語プロファイルに日本語が混じっている"


# ======================================================================
# ここに置かないと決めたもの
# ======================================================================

def test_min_coverage_is_not_in_the_profile():
    """**MIN_COVERAGE は帯の軸であって、言語の軸ではない。**

    置くと「言語で決まる」という誤った模型が固定される。2026-08-27 の実測で、
    同じ英語の中で質疑帯 0.84 / 演説帯 0.96 と割れている。
    理由はファイル冒頭に書いてある。
    """
    names = {f.name for f in dataclasses.fields(lang.Matching)}
    assert "min_coverage" not in names, (
        "MIN_COVERAGE を言語プロファイルに入れないこと。"
        "帯ごとに変える設計へ進む道を塞ぐ。lang.py 冒頭の説明を読むこと。")


def test_backend_specific_values_are_not_in_the_profile():
    """**merge / redistribute の上限はバックエンドの軸。**"""
    all_names = set()
    for cls in (lang.Matching, lang.Speech, lang.TextJoin, lang.Labels,
                lang.Asr):
        all_names |= {f.name for f in dataclasses.fields(cls)}
    for banned in ("merge_max_seconds", "merge_max_gap", "max_estimated_seconds"):
        assert banned not in all_names, (
            f"{banned} はバックエンドで変わる値。言語プロファイルに入れると"
            "Day 30 のバックエンド抽象化でほどけなくなる。")


def test_the_reasons_are_written_down():
    """**置かない理由がファイルに残っていること。**

    実測を見ていない人が読むと「なぜこれだけ別扱いなのか」が分からず、
    器に足してしまう。理由が消えたらここで落とす。
    """
    src = (Path(__file__).resolve().parent.parent / "src" / "lang.py"
           ).read_text(encoding="utf-8")
    assert "MIN_COVERAGE" in src and "帯の軸であって、言語の軸ではない" in src
    assert "merge_consecutive" in src and "バックエンドの軸" in src
    assert "0.84" in src and "0.96" in src, "実測の数値が根拠として残っていない"


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
