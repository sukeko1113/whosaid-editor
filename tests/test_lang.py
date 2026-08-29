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
# 切り替えが**実際の処理**に効くこと
#
# **値を読んで比べるだけでは足りない。**その値が使われていない経路があると
# 見逃す(2026-08-27 にキャッシュキーで実際に見逃した)。処理を通して確かめる。
#
# 既定引数は import 時に束縛されるので、`min_block: int = MIN_BLOCK` と
# 書いてあると lang.use() が効かない。その罠が塞がっていることも、ここで見る。
# ======================================================================

def _en_words(text, per_char=0.09, start=0.0):
    from src.align import Word
    return [Word(text=ch, start=start + i * per_char,
                 end=start + (i + 1) * per_char)
            for i, ch in enumerate(text)]


def test_switch_changes_anchor_matching():
    """anchor.measure() が、切り替えたあとの min_block で動くこと。"""
    from src import anchor

    before = lang.current().code
    try:
        track = anchor.prepare(_en_words(
            "the government should adopt this policy"))
        lang.use("ja")
        assert anchor.measure("the", track, 0.0, 100.0) is not None, \
            "日本語(3)なら 3 文字で当たるはず"
        lang.use("en")
        assert anchor.measure("the", track, 0.0, 100.0) is None, \
            "英語(10)に切り替えたのに 3 文字で当たっている。" \
            "**既定引数が import 時に束縛されていないか。**"
        assert anchor.measure("government", track, 0.0, 100.0) is not None, \
            "10 文字なら英語でも当たるはず"
    finally:
        lang.use(before)


def test_switch_changes_inspection_thresholds():
    """inspect_times() が、切り替えたあとの min_matched で動くこと。

    正規化後 14 文字の本文は、日本語(8)なら提案が出て、英語(22)なら
    「一致が短い」として出ない。
    """
    from src.inspection import inspect_times
    from src.segments import Project, Segment

    text = "yes it does say so"          # 正規化 14 字
    proj = Project(audio_path="a.m4a", duration=60.0)
    proj.segments = [Segment(index=0, start=6.8, end=9.8, text=text,
                             cluster="0:A")]
    words = _en_words(text, per_char=0.125)

    before = lang.current().code
    try:
        lang.use("ja")
        ja = inspect_times(proj, words)
        lang.use("en")
        en = inspect_times(proj, words)
    finally:
        lang.use(before)

    assert ja.short_match == 0, "日本語(8)では一致 14 字は十分なはず"
    assert en.short_match == 1, (
        "英語(22)に切り替えたのに「一致が短い」で弾かれていない。"
        "**既定引数が import 時に束縛されていないか。**")


def test_explicit_argument_still_wins():
    """呼び出し側が明示的に渡したら、そちらが勝つこと(検査が使う)。"""
    from src import anchor

    before = lang.current().code
    try:
        track = anchor.prepare(_en_words("the government should adopt"))
        lang.use("en")
        assert anchor.measure("the", track, 0.0, 100.0) is None
        assert anchor.measure("the", track, 0.0, 100.0, min_block=3) is not None
    finally:
        lang.use(before)


def test_no_module_constant_is_used_as_a_default_argument():
    """**言語で変わる定数を既定引数に書かないこと。**

    書くと import 時に値が固定され、切り替えが静かに効かなくなる。
    grep で見つけて落とす(次に触る人が同じ書き方に戻すのを防ぐ)。
    """
    root = Path(__file__).resolve().parent.parent / "src"
    banned = ("= MIN_BLOCK", "= MIN_MATCHED", "= MIN_DENSITY",
              "= TAIL_GAP_LIMIT", "= SPEECH_CHARS_PER_SECOND")
    found = []
    for path in sorted(root.glob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#")[0]
            if any(b in code for b in banned) and ":" in code and "def " not in code:
                # 定義そのもの(MIN_BLOCK = ...)は除く
                if code.strip().startswith(tuple(b.strip("= ") for b in banned)):
                    continue
                found.append(f"{path.name}:{n} {line.strip()}")
    assert not found, (
        "言語で変わる定数が既定引数に使われています。import 時に束縛され、"
        "lang.use() の切り替えが効きません:\n  " + "\n  ".join(found))


# ======================================================================
# プロンプト — 短い応答の保護
#
# **これがこの一連の作業の主目的そのもの。**ディベートでは「答えなかった」
# 「沈黙した」「聞き返した」こと自体が判定材料になる。
#
# 2026-08-27 の実測(英語ディベート決勝 45:12)で効いていることを確認した:
#   Yes 28 / No 30 / Right 16 / Yeah 5 回、25 文字以下の独立区間 48 件。
#   遮られて切れた語("How")も、言い淀んだまま終わった応答("No, uh")も、
#   相づち扱いで前の発言に併合されずに独立した区間として残っていた。
#
# **書き換えるときに薄まらないよう、ここで固定する。**
# ======================================================================

def _en_prompt(**kw):
    from src.transcribe import build_prompt

    before = lang.current().code
    try:
        lang.use("en")
        return build_prompt(True, True, **kw)
    finally:
        lang.use(before)


def test_en_verbatim_protects_short_responses():
    p = _en_prompt(verbatim=True, cluster_only=True)
    assert "Keep every short response in full" in p
    for token in ("Yes", "No", "I don't know", "It does not say that",
                  "Could you repeat the question"):
        assert token in p, token
    assert "never merge them into a neighbouring turn" in p
    assert "never treat them as backchannel noise" in p


def test_en_verbatim_marks_silence_and_no_answer():
    """無回答・沈黙・遮り・聞き取り不能を、省略せず書かせること。"""
    p = _en_prompt(verbatim=True, cluster_only=True)
    for token in ("(no response)", "(silence)", "(interrupted)", "(inaudible)"):
        assert token in p, token


def test_en_cluster_gives_short_responses_their_own_line():
    p = _en_prompt(verbatim=True, cluster_only=True)
    assert "A short response from a different voice always gets its own line" in p


def test_en_diar_verbatim_also_protects_short_responses():
    """**この部品は本線に後から入ったもので、英語版は新規に書き起こした。**

    en-test の時点では存在しなかったので、実測で確認されていない。
    保護の指示が薄まっていないことだけは、ここで固定しておく。
    """
    p = _en_prompt(verbatim=True)
    assert "Every short response gets its own line, always." in p
    assert "Never put them on the previous speaker's line." in p
    for token in ("Yes", "No", "I don't know", "Could you repeat the question"):
        assert token in p, token
    # 1 行が長くなりすぎない指示(区間に話者を割り当てられなくなるため)
    assert "Keep each line under 20 seconds." in p


def test_en_examples_demonstrate_short_responses():
    """**例は指示より強く効く。**沈黙と短い応答が例に出ていること。"""
    cluster = _en_prompt(verbatim=True, cluster_only=True)
    assert "【A】 (silence)" in cluster
    assert "【A】 I don't know." in cluster
    diar = _en_prompt(verbatim=True)
    assert "(no response)" in diar
    assert "【Speaker B】 Yes." in diar


def test_en_verbatim_keeps_english_fillers():
    p = _en_prompt(verbatim=True, cluster_only=True)
    for filler in ("um", "uh", "you know", "I mean", "like"):
        assert filler in p, filler
    assert "Never summarise, paraphrase, correct or tidy" in p


def test_en_prompt_has_no_japanese_left():
    """**英語の指示に日本語が混じっていないこと。**

    単位 6 の前は EN.prompt が JA.prompt を指す仮置きだった。その状態で
    英語の転写を走らせると、日本語の指示が Gemini に飛ぶ。
    仮置きに戻したらここで落ちる。
    """
    import dataclasses

    for f in dataclasses.fields(lang.Prompt):
        v = getattr(lang.EN.prompt, f.name)
        if callable(v):
            v = v("Sato")
        bad = [c for c in v if "぀" <= c <= "ヿ" or "一" <= c <= "鿿"]
        assert not bad, f"EN.prompt.{f.name} に日本語が混じっている: {set(bad)}"


def test_en_and_ja_prompts_are_different_objects():
    assert lang.EN.prompt is not lang.JA.prompt
    assert lang.EN.prompt.opening != lang.JA.prompt.opening


def test_prompt_parts_are_not_duplicated_in_transcribe():
    """**プロンプトの部品を transcribe.py に書き戻さないこと。**

    2 箇所に散ると、言語を足す人がどちらかを見落とす。この一連の作業で
    4 回起こした形。
    """
    src = (Path(__file__).resolve().parent.parent / "src" / "transcribe.py"
           ).read_text(encoding="utf-8")
    for banned in ("_RULES_VERBATIM = ", "_RULES_CLUSTER = ",
                   "_EXAMPLE_CLUSTER = ", "_RULES_DIAR_VERBATIM = "):
        assert banned not in src, (
            f"{banned.strip(' =')} が transcribe.py に戻っています。"
            "プロンプトの部品は src/lang.py にまとめること。")


# ======================================================================
# ローカル転写 — **2 経路が同時に切り替わること**
#
# LANGUAGE の参照は align.py(単語時刻の取得)と local_asr.py(本文の転写)の
# 2 箇所にある。**片方だけ切り替えると、本文と物差しの言語が食い違う。**
# ======================================================================

def test_both_asr_paths_switch_together():
    from src import align, local_asr

    before = lang.current().code
    try:
        for code in ("ja", "en"):
            lang.use(code)
            want = lang.current().asr.whisper_language
            assert align.whisper_language() == want, f"align が {code} で切り替わらない"
            assert local_asr.whisper_language() == want,                 f"local_asr が {code} で切り替わらない(本文と物差しが食い違う)"
    finally:
        lang.use(before)


def test_style_prompt_switches_with_the_language():
    """英語音声に日本語の例文を与えないこと。"""
    from src import local_asr

    before = lang.current().code
    try:
        lang.use("en")
        t = local_asr.LocalTranscriber(model="small")
        assert t.style_prompt
        assert not any("぀" <= c <= "ヿ" or "一" <= c <= "鿿"
                       for c in t.style_prompt),             "英語なのに日本語の例文が渡っている"
        lang.use("ja")
        t = local_asr.LocalTranscriber(model="small")
        assert "本日は" in t.style_prompt
    finally:
        lang.use(before)


def test_style_prompt_can_still_be_turned_off():
    """明示的に None を渡せば「例文を使わない」になること。

    「指定なし(プロファイルを使う)」と区別が要る。
    """
    from src import local_asr

    t = local_asr.LocalTranscriber(model="small", style_prompt=None)
    assert t.style_prompt is None
    assert t.prompt_ver == 0, "例文を使わないのに版が付いている"


def test_language_constant_is_not_referenced_directly():
    """**LANGUAGE を直接参照しないこと。**

    import 時に束縛されるので、切り替えが静かに効かなくなる。
    whisper_language() を通すこと。
    """
    root = Path(__file__).resolve().parent.parent / "src"
    found = []
    for path in sorted(root.glob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#")[0]
            if "LANGUAGE" not in code:
                continue
            if code.strip().startswith("LANGUAGE = "):
                continue            # 定義そのもの(互換のため残している)
            found.append(f"{path.name}:{n} {line.strip()}")
    assert not found, (
        "LANGUAGE を直接参照しています。import 時に束縛され、切り替えが"
        "効きません。whisper_language() を使ってください:\n  "
        + "\n  ".join(found))


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
