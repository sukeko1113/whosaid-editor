"""言語ごとに変わる設定を、1 箇所にまとめた器。

**言語を足すときに見る場所は、このファイルだけ。**

散らばった設定を入れ忘れる事故を、2026 年 8 月だけで 4 回起こしている
(キャッシュキーの要素・モデルフォルダ・言語・モデル名)。同じ形を言語でも
繰り返さないために、言語で変わるものは値もプロンプトも全部ここに置く。

`feature/en-test`(2026-08-27)は定数を直接書き換えた。それだと日本語が
同時に壊れるので `main` に入れられない。**両方の言語・両方のエンジンを測る**
以上、切り替えられる器が要る。

---

## このファイルは他の src モジュールを import しない

`transcribe.py` は `google.genai` をトップレベル import しているので、
プロンプトをあちらに置いたままここから参照すると、`lang.py` が genai を
引き込む。すると `anchor.py` や `segments.py` から使えなくなる
(`segments.py` が `transcribe` を import しないよう避けてあるのと同じ理由)。

だからプロンプト本文もここに置く。**このファイルが長いのはそのため**で、
大半は文字列であってロジックではない。

---

## ここに置かないもの — 「入れられるから入れる」をしない

### `inspection.MIN_COVERAGE`(0.60) — **帯の軸であって、言語の軸ではない**

置きたくなるが、**置くと間違った模型が固定される。**

2026-08-27 の実測(英語ディベート 45:12)では、**同じ言語の中で**帯によって
これだけ割れた。

    質疑(速い応酬・発話の重なり)  照合 83 件  被覆中央値 0.84  最小 0.09  閾値未満 14
    演説(長い連続発言)            照合 25 件  被覆中央値 0.96  最小 0.74  閾値未満  0

0.6 は演説帯には緩すぎ、質疑帯では 14 件を落とす。**言語を切り替えても
この差は消えない。**正しい設計は「帯ごと・発言の性質ごとに変える」であり、
言語別テーブルに入れると、その設計に進む道を塞ぐ。

日本語でも同じ差が出る可能性がある(議論が白熱した箇所と、報告を読み上げる
箇所)。**日本語側でも未検証の論点**として残してある。

### `merge_consecutive` / `redistribute_times` の上限 — **バックエンドの軸**

`MERGE_MAX_SECONDS` `MERGE_MAX_GAP` と、時刻の按分。これらは
**Gemini が行を細かく割る・時刻をドリフトさせる**という固有の欠陥への補正で、
単語時刻を返すバックエンド(faster-whisper)に通すとむしろ壊す。

言語ではなくバックエンドで変わる。ここに混ぜると Day 30 の
バックエンド抽象化でほどけなくなる。

**`SPEECH_CHARS_PER_SECOND` だけは、この 2 軸が交差している。**按分の根拠で
あると同時に話速でもある。ここには言語としての値を置き、バックエンド側の
都合は将来 Capabilities 側で扱う。

### その他

- GUI の日本語表示・Word のフォント … アプリの UI は日本語のまま
- `parse_hms` の全角数字変換 … 変換するだけで ASCII を弾かないので無害
- `is_degenerate` の閾値 … 英語で誤検出しないことを確認済み(2026-08-27)
- `Segment.preview` の幅 … 効いているのは 340px の列幅であって定数ではない
  (Tk の実測: 340px に英語 63 字 / 日本語 34 字。上限 60・70 は届いていない)
- `VERBATIM_PROMPT_VER` / `local_asr.PROMPT_VER` … キャッシュ鍵の版であって
  言語の値ではない。言語は鍵に別途入るので、版は言語共通で据え置く
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


# ======================================================================
# 器
#
# **どのフィールドにも既定値を持たせない。**新しい言語を足したとき、
# 埋め忘れが「静かに日本語の値」になるのを防ぐ。埋めるまで構築できない。
# ======================================================================

@dataclass(frozen=True)
class Matching:
    """照合の閾値(anchor / inspection)。**すべて測って決めた値。**"""

    min_block: int          # 一致とみなす最短の長さ
    min_matched: int        # 提案を出す最低の一致文字数
    tail_gap_limit: int     # 末尾の欠けがこれ以上なら終了を信用しない
    min_density: float      # 一致の密度(字/秒)。これ未満は散らばり


@dataclass(frozen=True)
class Speech:
    """発話の見積もり(transcribe)。"""

    chars_per_second: float


@dataclass(frozen=True)
class TextJoin:
    """断片の連結(transcribe / segments)。"""

    word_separator: str     # 区間どうしを繋ぐとき語の間に入れるもの
    fragment_comma: str     # 行が割れた断片の間に補うもの
    sentence_ends: str      # これで終わっていれば上を補わない
    clinging: str           # 右がこれで始まるときは前の語に直付けする


@dataclass(frozen=True)
class Labels:
    """Gemini が返す擬似話者ラベルの語彙(transcribe)。"""

    multi_words: tuple[str, ...]      # → 【*】複数人が同時
    unknown_words: tuple[str, ...]    # → 【?】判別不能


@dataclass(frozen=True)
class Asr:
    """ローカル転写(align / local_asr)。"""

    whisper_language: str   # faster-whisper に渡す言語コード
    style_prompt: str       # initial_prompt の例文(句読点を出させる)
    style_prompt_ver: int   # 変えたら上げる。キャッシュ鍵に入る


@dataclass(frozen=True)
class Prompt:
    """Gemini のプロンプト部品(transcribe)。"""

    opening: str
    rules_heading: str
    rules_cleanup: str
    rules_verbatim: str
    rules_ts: str
    rules_diar: str
    rules_diar_verbatim: str
    rules_cluster: str
    label_abc: str
    verbatim_punct: str
    tail: str
    example_cluster: str
    example_diar: str
    example_diar_verbatim: str
    example_ts: str
    roster_block: Callable[[str], str]


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    label: str
    # **その値がどの測定から来たかを必ず書く。**閾値は測って決めた値で、
    # 別の言語にコピーしても意味がない。空欄では検査が落ちる。
    calibrated_by: str
    matching: Matching
    speech: Speech
    text: TextJoin
    labels: Labels
    asr: Asr
    prompt: Prompt


# ======================================================================
# 日本語
# ======================================================================

_JA_RULES_CLEANUP = """- 内容は正確に一言一句書き起こす(情報の欠落・改変なし)
- フィラー(「えー」「あのー」「えっと」「まあ」「そのー」など)は適宜削除
- 言い淀み・不要な繰り返しは整理し、読みやすい日本語に整える
- 話者の意図・固有名詞・数字は正確に保つ
- 聞き取れなかった箇所は [不明] と記す"""

_JA_RULES_VERBATIM = """【逐語(一言一句)ルール・最重要】
- 発言は一切要約・整文しない。話されたとおりに書き起こす
- 「えー」「あのー」「えっと」等の言いよどみ(フィラー)、言い直し、繰り返しも省略せずそのまま残す
- 聞き取れない箇所は推測で補完せず、必ず「(聴取不能)」と表記する
- 音声に存在しない語句を付け加えない。文法的に不自然でも直さない
- 固有名詞が不明瞭な場合は、聞こえたとおりに表記する(勝手に「正しい」名前に直さない)
- 実際に発話された内容だけを書く。音声上で繰り返されていない限り、出力で同じ言葉を繰り返さない"""

_JA_RULES_TS = """- **段落の先頭に必ず [MM:SS] 形式のタイムスタンプを入れる**
  (音声内のその段落が始まる時刻、ゼロ埋め2桁、例: [00:00], [03:45])
- 段落は話題のまとまり・明確な区切り・または30秒~2分ごとに分ける"""

_JA_RULES_DIAR = """- **話者が変わるごとに新しい行として書き出す**
- **各行の冒頭に必ず以下の形式を入れる**:
  [MM:SS] 【話者ラベル】 本文...
  (時刻はゼロ埋め2桁の分:秒。ミリ秒や1/100秒は付けない)
- 短い相づち(「はい」「うん」程度)は前の発言と同じ行に含めてよい"""

_JA_RULES_DIAR_VERBATIM = """- **話者が変わるごとに新しい行として書き出す**
- **各行の冒頭に必ず以下の形式を入れる**:
  [MM:SS] 【話者ラベル】 本文...
  (時刻はゼロ埋め2桁の分:秒。ミリ秒や1/100秒は付けない)
- **短い相づち(「はい」「うん」「なるほど」等)も必ず独立した行にする。**
  前の発言の行に含めてはならない。誰がいつ同意したかが記録として要る
- **同じ話者が続く場合も、文の切れ目や間(ま)で行を分ける。
  1 行が 20 秒を超えないようにする**"""

_JA_RULES_CLUSTER = """- **行を分けるのは「話す人が変わったとき」だけ**
- **各行は必ず次の形式で始める**:
  [MM:SS] 【A】 本文...
  (時刻はこの音声の先頭からの 分:秒。ゼロ埋め2桁。ミリ秒は付けない)
- 話者ラベルは A, B, C, ... のアルファベット1文字のみ。**名前や役職は絶対に書かない**
- 同じ声の人物には常に同じラベルを使う。声が変われば必ず別のラベルにする
- **同じ人が話し続けている間は、絶対に行を分けない**。
  息継ぎ・間・読点・言いよどみで区切ってはいけない。
  文がいくつ続いても、次の人が話し始めるまでは 1 行に書く。
  (悪い例: 「工程表」「えー、財源計画」「年度別資金繰り」と 3 行に分ける
   → 正しくは 1 行に「工程表、えー、財源計画、年度別資金繰り…」と続ける)
- 同じ人が 3 分以上話し続けた場合のみ、話題の切れ目で行を分けてよい
- 短い相づち(「はい」「うん」程度)も、別の人物の声なら独立した行にする
- 誰の声か判別できない場合は 【?】、複数人が同時に話している場合は 【*】 とする"""

_JA_EXAMPLE_CLUSTER = """出力例:
[00:00] 【A】 本日はお忙しい中お集まりいただきありがとうございます。それでは議事を始めます。まず、お手元の資料をご覧ください。えー、本日の議題は、予算と、あの、人事の二点になります。
[00:25] 【B】 すみません、確認ですが、前回の議事録は配布済みですか?
[00:32] 【A】 はい、配布済みです。修正点も反映されています。
[00:40] 【C】 その件、私からも補足させてください。"""

_JA_LABEL_ABC = """- 話者ラベル(発言者A, 発言者B, 発言者C...)は声の特徴で識別し、同じ人物には常に同じラベルを使う
- 話者を特定できない場合は【発言者不明】とする"""

_JA_EXAMPLE_DIAR = """出力例:
[00:00] 【発言者A】 本日はお忙しい中お集まりいただきありがとうございます。それでは議事を始めます。
[00:25] 【発言者B】 すみません、確認ですが、前回の議事録は配布済みですか?
[00:32] 【発言者A】 はい、配布済みです。修正点も反映されています。"""

_JA_EXAMPLE_DIAR_VERBATIM = """出力例:
[00:00] 【発言者A】 えー、本日はお忙しい中お集まりいただき、あの、ありがとうございます。
[00:08] 【発言者B】 はい。
[00:09] 【発言者A】 それでは、えーと、議事を始めます。
[00:14] 【発言者C】 うん。
[00:15] 【発言者B】 すみません、確認ですが、前回の議事録は配布済みですか?
[00:20] 【発言者A】 はい、配布済みです。えー、修正点も反映されています。"""

_JA_EXAMPLE_TS = """出力例:
[00:00] 本日の会議を開始します。まず議題ですが、予算と人事の二点になります。
[00:42] それでは一つ目、来期予算について議論を始めます。
[03:15] 続いて人事についての検討に移ります。"""


def _ja_roster_block(roster: str) -> str:
    return f"""この音声の参加者は以下のとおり事前に判明しています。話者の判別には、声質に加えて、
発言内容(名乗り・指名・役職に応じた発言内容)も手がかりにしてください。

【参加者名簿】
{roster.strip()}

- 話者ラベルは名簿にある呼称を【】で囲んで用いる(例: 【議長(理事長)】【佐藤理事】)
- どうしても特定できない場合のみ【発言者不明】、複数人の発話が重なる場合は【発言者複数・重複】とする。
  無理に名簿の誰かに割り当てるより、不明と明示するほうが望ましい"""


# **large-v3 は句読点をほとんど付けない。**実測(2026-08-20・逐語正解 4 帯)で、
# 説明が続く帯では 1 万字あたり 正解 1138 に対し 0 だった。例文を与えると
# 戻る(0 → 971)。用語は固有名詞には効かないことも測定済みだが、誤字全体は
# 下がり、脱落と長さも改善するので併せて置く。
_JA_STYLE_PROMPT = (
    "本日は、お忙しい中お集まりいただき、ありがとうございます。"
    "それでは、お手元の資料に沿って、順にご説明いたします。"
    "この点につきまして、何かご質問がございましたら、お願いいたします。"
)


JA = LanguageProfile(
    code="ja",
    label="日本語",
    calibrated_by=(
        "実会議 2026-08-13〜19(聴き取り 3 回・実データ 01+02edited 67 分)。"
        "逐語正解 4 帯 8 分 175 区間"),
    matching=Matching(
        # 日本語は 1〜2 文字の一致が偶然でも頻発する。無関係な日本語どうしの
        # 最長偶然一致は 2 文字だった(2026-08-27 の実測)。
        min_block=3,
        # 3〜4 文字の偶然一致は被覆率 100% になり得るので被覆率では防げない。
        # 実会議の聴き取り 15 件で、完全に外れた 2 件はどちらも一致 3 文字。
        min_matched=8,
        # 欠け 3 字以上の 2 件はどちらも終了が外れ、2 字以下は全部当たり。
        tail_gap_limit=3,
        # 正常な発話は実測で 4〜11 字/秒。本文が発話と食い違っていた区間は
        # 0.8 字/秒だった。
        min_density=1.5,
    ),
    speech=Speech(
        # 逐語では言いよどみが多く実測 4〜6 文字/秒。短く見積もると発言が
        # 途中で切れるので、範囲の下端を採る。
        chars_per_second=4.5,
    ),
    text=TextJoin(
        # 日本語には語の区切りが無いので、繋ぐときに何も入れない。
        word_separator="",
        # 行が割れた断片をそのまま繋ぐと「けどもえー」と読みにくい。
        # 逐語での「間」の表現であり、発話内容そのものは変えない。
        fragment_comma="、",
        sentence_ends="。、．，!?！？」』）)…・ー~〜-",
        # 句読点が必ず前の語に付くので、この区別が要らない。
        clinging="",
    ),
    labels=Labels(
        multi_words=("複数", "重複", "同時"),
        unknown_words=("不明", "不詳", "unknown"),
    ),
    asr=Asr(
        whisper_language="ja",
        style_prompt=_JA_STYLE_PROMPT,
        style_prompt_ver=1,
    ),
    prompt=Prompt(
        opening="この音声を日本語で書き起こしてください。",
        rules_heading="ルール:",
        rules_cleanup=_JA_RULES_CLEANUP,
        rules_verbatim=_JA_RULES_VERBATIM,
        rules_ts=_JA_RULES_TS,
        rules_diar=_JA_RULES_DIAR,
        rules_diar_verbatim=_JA_RULES_DIAR_VERBATIM,
        rules_cluster=_JA_RULES_CLUSTER,
        label_abc=_JA_LABEL_ABC,
        verbatim_punct="- 句読点は聞こえたとおりの区切りで付けてよい(内容の変更は不可)",
        tail="- 説明や前置き、Markdown 装飾は不要。書き起こし本文のみを出力する。",
        example_cluster=_JA_EXAMPLE_CLUSTER,
        example_diar=_JA_EXAMPLE_DIAR,
        example_diar_verbatim=_JA_EXAMPLE_DIAR_VERBATIM,
        example_ts=_JA_EXAMPLE_TS,
        roster_block=_ja_roster_block,
    ),
)


# ======================================================================
# 英語
#
# 単位 6(プロンプト)と単位 7(ローカル転写)で中身を入れる。
# ここではまず器と、測って決めた値だけを置く。
# ======================================================================

EN = LanguageProfile(
    code="en",
    label="English",
    calibrated_by=(
        "英語ディベート決勝 1 試合 45:12(2026-08-27・n=1)。"
        "**他の試合・他の話者層では違う可能性がある**"),
    matching=Matching(
        # 照合器が見るのは正規化後の**空白を落とした**文字列(両側に掛かる)。
        # 無関係なディベート英語どうしの最長偶然一致(2026-08-27 の実測):
        #     区間  40 字 × 窓 375 字  中央値 5 / p95  9
        #     区間 100 字 × 窓 375 字  中央値 8 / p95 10
        #     区間 200 字 × 窓 375 字  中央値 8 / p95 17
        # 同じ試験を日本語で行うと最長 2 文字。10 は現実的な長さでの p95 を
        # 上回り、英語 2 語ぶんに当たる。
        min_block=10,
        # 話速の比(11.0 / 4.5 ≒ 2.4)ぶん引き上げないと同じ「一致の重み」に
        # ならない。加えて 8 は min_block=10 を下回り、一度も発火しない。
        min_matched=22,
        # 英語の 3 文字は 1 語に満たず、語尾が欠けただけで終了を捨ててしまう。
        tail_gap_limit=8,
        # 英語は空白を落とした状態で約 11 字/秒あり、1.5 では選別にならない。
        # 日本語で「正常の下端の約 1/3」だった比を保つ。
        min_density=4.0,
    ),
    speech=Speech(
        # 150 wpm = 2.5 語/秒、1 語平均 5 文字(空白込み)で約 12.5 文字/秒。
        # 12.5 ではなく 11.0(約 132 wpm)にしてあるのは、この定数が意図的に
        # 低めへ倒してあるため(日本語も範囲の下端を採っている)。加えて日本の
        # 高校英語ディベートは原稿を読む形式が主で、150 wpm より遅い。
        chars_per_second=11.0,
    ),
    text=TextJoin(
        word_separator=" ",
        fragment_comma=", ",
        sentence_ends=".!?,;:…—-\"')]",
        clinging=".!?,;:%)]}",
    ),
    labels=Labels(
        # これが無いと 【Multiple】 が「先頭の英字 1 文字」に落ち、クラスタ
        # "M" という実在しない話者が立つ。しかも merge_consecutive が
        # その連続を 1 人の発言として連結する。
        multi_words=("multiple", "several", "overlap", "crosstalk",
                     "cross talk", "simultaneous", "both speak", "everyone"),
        unknown_words=("unknown", "unclear", "unidentified", "inaudible",
                       "not sure", "cannot tell", "can't tell", "unsure"),
    ),
    asr=Asr(
        whisper_language="en",
        # **未検証。**日本語の例文は実測で効果を確認してある(句読点 0 → 971)が、
        # 英語版は置いただけで、効果は 1 本目の測定で初めて分かる。
        # runs.csv の params_note に「STYLE_PROMPT: en-v1(未検証)」と記録する。
        style_prompt=(
            "Thank you, Mr. Chairperson. We stand in firm affirmation of "
            "today's motion. Our first contention is that the current system "
            "fails to protect the most vulnerable members of our society. "
            "Could you repeat the question, please?"
        ),
        style_prompt_ver=1,
    ),
    # 単位 6 で入れる。それまでは日本語のプロンプトを指しておく
    # (英語で走らせると日本語の指示が出るので、単位 6 の前に使わないこと)。
    prompt=JA.prompt,
)


# ======================================================================
# 登録と切り替え
# ======================================================================

PROFILES: dict[str, LanguageProfile] = {"ja": JA, "en": EN}

DEFAULT = "ja"

# いま使っているプロファイル。**既定は日本語**——何もしなければ従来どおり。
CURRENT: LanguageProfile = PROFILES[DEFAULT]


def get(code: str) -> LanguageProfile:
    """言語コードからプロファイルを引く。

    **既定へのフォールバックはしない。**知らない言語を日本語の値で処理すると、
    閾値が意味を成さないまま静かに壊れる。落ちるほうがよい。
    """
    try:
        return PROFILES[code]
    except KeyError:
        raise KeyError(
            f"言語プロファイルがありません: {code!r}。"
            f"あるのは {sorted(PROFILES)}。"
            "**日本語や英語の値で代用しないこと**——照合の閾値は測って決めた"
            "値で、別の言語では意味がありません。lang.py に profile を足し、"
            "calibrated_by にどの測定から来たかを書いてください。"
        ) from None


def use(code: str) -> LanguageProfile:
    """使う言語を切り替える。"""
    global CURRENT
    CURRENT = get(code)
    return CURRENT


def current() -> LanguageProfile:
    """いまのプロファイル。

    **モジュール変数を直接読まず、これを通すこと。**
    `from .lang import CURRENT` は import した時点の値を束縛するので、
    あとで use() しても切り替わらない(既定引数と同じ罠)。
    """
    return CURRENT
