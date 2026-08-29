# 英語ディベート測定 — 集計 CSV の雛形

10 本の測定結果を貯める CSV の**列定義**を置く。ここにあるのは見出し行だけで、
実データは入れない。

**実データの置き場**: `C:\dev\01\test-audio\measure\`（リポジトリ外）
録音の素性と大会名を含むため、正解ファイル（`test-audio\truth\`）と同じ扱いにする。

```
tools/measure/sources.header.csv   → C:\dev\01\test-audio\measure\sources.csv
tools/measure/runs.header.csv      → C:\dev\01\test-audio\measure\runs.csv
tools/measure/stages.header.csv    → C:\dev\01\test-audio\measure\stages.csv
tools/measure/truth.header.csv     → C:\dev\01\test-audio\measure\truth.csv
```

**1 本目を流す前に列を確定させる。**途中で足すと前半と後半が揃わない。
列が合っているかは `tools/measure/check_header.py` で確かめる。

---

## なぜ 4 枚か

同じ音声を複数のエンジンで測り、さらに帯ごとに正解を作るので、粒度が 3 段ある。

| ファイル | 1 行の単位 | 対象 |
|---|---|---|
| `sources.csv` | 音声 | 試合・講評の両方 |
| `runs.csv` | 音声 × エンジン | 試合・講評の両方 |
| `stages.csv` | 音声 × エンジン | **試合のみ**（講評は話者 1 人で交代がない） |
| `truth.csv` | 音声 × エンジン × 帯 | **正解を作った帯のみ**（講評は作らない） |

引き継ぎ資料 §6.2 は精度列を `runs.csv` に置いていたが、`truth.csv` に分けた。
理由は 2026-08-27 の測定で分かった 2 点。

- **帯によって測れる指標が違う。**演説帯は冒頭 2 分に区間が 2 件しか入らず、
  区間単位の指標（相づち保持率・短発話再現・脱落の再現）の分母が立たない。
  1 行にまとめると、測っていない値と 0 の値が区別できない
- **帯によって取れる量が違う。**同じ「冒頭 2 分」でも ②否定質疑 37 区間に対し
  ⑥肯定質疑 18 区間。量を記録しないと、指標の差が「音声の差」か
  「サンプル量の差」か切り分けられない

---

## 帯の指定（`band_rule`）

```
HEnDA12:S02@0-120
└─────┘ └─┘ └───┘
 形式ID  ｜   帯の中の秒数（開始からの相対）
        セクション番号（1〜12）
```

**HEnDA の 12 セクションは順序が不変**（年度で変わるのは各セクションの持ち時間だけ）。
したがってセクション番号は年度をまたいで同じステージを指す。形式 ID に年度を
含める必要はない。持ち時間は `stages.csv` の `stage_seconds` に実測値を記録する。

| 番号 | セクション | 番号 | セクション |
|---|---|---|---|
| S01 | 肯定側 立論 | S07 | 肯定側 アタック |
| S02 | 否定側 質疑（立論への質疑） | S08 | 否定側 質疑（アタックへの質疑） |
| S03 | 否定側 立論 | S09 | 肯定側 ディフェンス |
| S04 | 肯定側 質疑（立論への質疑） | S10 | 否定側 ディフェンス |
| S05 | 否定側 アタック | S11 | 肯定側 総括 |
| S06 | 肯定側 質疑（アタックへの質疑） | S12 | 否定側 総括 |

### 標準の 3 帯（2026-08-28 決定）

```
HEnDA12:S02@0-120;HEnDA12:S08@0-120;HEnDA12:S01@0-120
```

| 帯 | 選んだ理由 |
|---|---|
| **S02** 否定質疑 | 難所が集中する。閾値を決めるための裾がここにある（2026-08-27 の測定で照合不能 8 件・被覆率 0.6 未満 6 件・最小 0.09） |
| **S08** 否定質疑 | 機能の違う質疑（アタックへの質疑）。S02 と同性質かは 10 本集めるまで分からない |
| **S01** 肯定立論 | 演説で最長。HEnDA ルール §3.1 が出典（著者名・肩書・媒体名・発行年月）の読み上げを義務づける帯で、固有名詞と数字の誤認識が最も痛い |

2 分 × 3 帯 = **6 分/試合**、10 本で 60 分。

**S01（演説帯）は `metrics_scope = text_only`。**冒頭 2 分に区間が 2 件しか入らないため、
区間単位の指標は測らず、文字単位（WER/CER・フィラー保持率）だけを埋める。

### 正解ファイルとの対応

正解ファイル（`test-audio\truth\verbatim.*.json`）は帯を内部に持つ。

```json
{ "audio_sha256": "...", "made_by": "human-verbatim",
  "band": { "name": "HEnDA12:S02", "start": 688.0, "end": 808.0 },
  "segments": [...], "missing": [...] }
```

`band.name` に**セクション番号を入れる**こと（従来は `a-setsumei` のような自由名だった）。
`band.start` / `band.end` はその試合での実時刻。`band_rule` は規則、`band.start/end` は
規則を当てはめた結果、という関係になる。

---

## 列の定義

### sources.csv — 音声の素性（エンジンに依存しない事実）

| 列 | 内容 |
|---|---|
| `source_id` | 音声の識別子。ファイル名の stem を使う（例 `debate-2018-final`） |
| `kind` | `match` / `judge` |
| `format` | 進行形式。HEnDA の全国・地区大会なら `HEnDA12` |
| `match_date` | 試合の実施日（YYYY-MM-DD）。年度から持ち時間の版が引ける |
| `tournament` | 大会名 |
| `round` | ラウンド（`final` / `semifinal` / `prelim-3` 等） |
| `duration_sec` | 音声の長さ（秒・小数 2 桁） |
| `sha256` | 音声の SHA-256。**正解ファイルとの紐付けの鍵** |
| `channels` | チャンネル数 |
| `channel_note` | 各チャンネルの中身（例 `ch0=ch1 有音 / ch2-5 全編無音`） |
| `ac1_gain_db` | `-ac 1` 後の平均レベルと、単独チャンネルとの差（例 `-28.1 (ch0比 +3.0)`） |
| `sample_rate` | サンプリングレート |
| `bit_depth` | ビット深度 |
| `codec` | 元のコーデック（例 `pcm_s16le`） |
| `note` | 自由記述 |

**`channel_note` と `ac1_gain_db` は毎回記録する。**録音構成は試合ごとに違いうる。
2026-08-27 の 6ch・無音 4 本の件があり、ここを飛ばすと精度の低い結果を
「会場の反響が悪い」と誤診する。

### runs.csv — 転写の実績（音声 × エンジン）

| 列 | 内容 |
|---|---|
| `source_id` | `sources.csv` への参照 |
| `engine` | `gemini-2.5-flash` / `faster-whisper-large-v3` 等 |
| `model` | エンジン内のモデル指定 |
| `run_date` | 実行日 |
| `params_note` | チャンク長・device・逐語フラグなど（例 `chunk=16min/verbatim/cluster_only`） |
| `elapsed_sec` | 所要秒数 |
| `api_calls` | API 呼び出し回数（ローカルは 0） |
| `tokens_in` / `tokens_out` | トークン数（ローカルは空欄） |
| `retry_count` | 通信・API エラーによる再試行回数 |
| `runaway_count` | **暴走ループの再生成回数。**講評は独話が 5 分続くので試合と条件が違う |
| `segments` | 生成された区間数 |
| `clusters` | 声のまとまりの種類数 |
| `text_len_median` / `text_len_p90` | 本文の長さ（文字） |
| `seg_sec_median` / `seg_sec_p90` | 区間の長さ（秒） |
| `uncovered_gaps` | 本文が付いていない時間帯の数（5 秒以上の空白） |
| `align_model` | 被覆率を測ったアライナ（例 `faster-whisper-small/cpu/int8`） |
| `coverage_median` / `coverage_p10` | 被覆率 |
| `unmatched` | 照合できなかった区間数 |
| `below_threshold` | `min_coverage` を下回った区間数 |
| `min_block` / `min_coverage` | **測定時の閾値。必ず記録する。**途中で変えると、どの行がどの値で測られたか分からなくなる |
| `notes` | 自由記述 |

### stages.csv — ステージ境界の検出（試合のみ）

| 列 | 内容 |
|---|---|
| `source_id` / `engine` | 参照 |
| `stages_detected` | 検出できたセクション数。**順序が不変なので期待値は 12** |
| `boundary_ok` | 境界が正しかった数 |
| `type_ok` / `side_ok` | 補完前に種別・サイドが取れた数 |
| `type_ok_after_rule` / `side_ok_after_rule` | 規則で補完したあとの数 |
| `start_cue` | **開始の合図の文言そのもの。**定型かどうかは 10 本集めて初めて言える |
| `start_cue_hits` / `ready_cue_hits` / `end_cue_hits` | 各合図の出現回数 |
| `attack_term` | アタックの呼称（`talk speech` / `attack` / その他） |
| `intro_section` | 開始前の自己紹介の有無と長さ（例 `yes/0:00-7:12`）。無いと氏名→サイドが取れない |
| `name_variants` | 紹介部と本編で綴りが違った氏名の件数 |
| `stage_seconds` | 各セクションの実測秒数を `;` 区切りで 12 個。**ルール表と照合せず実測を記録する**（持ち時間は年度で変わり、その年度のルールが手元にあるとは限らない） |
| `max_drift_sec` | 計時とのずれの最大値 |
| `drift_ref` | ずれを測った基準（例 `HEnDA2018進行表` / `none`）。基準が無ければ `none` にして `max_drift_sec` は空欄 |
| `notes` | 自由記述 |

**補完前と補完後を分ける。**規則で埋めた分が実力に見えてしまう。

### truth.csv — 精度（音声 × エンジン × 帯）

| 列 | 内容 |
|---|---|
| `source_id` / `engine` | 参照 |
| `band_id` | `HEnDA12:S02` |
| `band_rule` | `HEnDA12:S02@0-120` |
| `band_start_sec` / `band_end_sec` | 規則を当てはめた実時刻 |
| `truth_id` | 正解ファイル名（例 `verbatim.HEnDA12-S02.json`） |
| `truth_segments` / `truth_chars` / `truth_words` | **実際に取れた量。**指標の差が量の差でないことを示すために要る |
| `metrics_scope` | `full`（区間単位も測る） / `text_only`（文字単位のみ。演説帯） |
| `wer` / `cer` | 語誤り率 / 文字誤り率。**日本語の CER と英語の WER は並べて比較できない** |
| `filler_retention` / `filler_hit` / `filler_total` | フィラー保持率と、その分子・分母 |
| `backchannel_retention` / `backchannel_hit` / `backchannel_total` | 相づち保持率と分子・分母 |
| `short_utterance_rate` / `short_hit` / `short_total` | 短発話再現と分子・分母 |
| `missed_recovery` / `missed_hit` / `missed_total` | 脱落の再現と分子・分母 |
| `notes` | 自由記述 |

**率だけでなく分子・分母を持つ。**率だけだと分母が 2 の 50% と分母が 40 の 50% が
同じ値になる。`metrics_scope = text_only` の行では区間単位の 4 指標を**空欄**にする
（0 と書かない。測っていないことと 0 だったことは違う）。

---

## 使い方

```bash
python tools/measure/check_header.py C:\dev\01\test-audio\measure\runs.csv
```

見出しが雛形と一致するかを確かめる。引数を省略すると雛形どうしの自己検査になる。
`--init <出力先>` で雛形を実データの置き場にコピーする（既存ファイルは上書きしない）。
