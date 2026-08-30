# CLAUDE.md

## プロジェクト概要

whosaid-editor: 日本語会議音声の逐語反訳＋話者割当エディタ（Windows デスクトップ / Python + Tkinter）。
製品価値は転写速度ではなく「誰が言ったかの検証済み記録」。
AI は話者を A/B/C の記号でしか出さず、実名は人が音声を聴いて確定する。

転写は 2 経路（`EngineSpec` で選ぶ。設計書 `claude/claude_ローカル転写_設計書.md`）:

- **ローカル（既定）**: faster-whisper。端末内で完結し、API キーも要らない。
  **話者分離も端末内で動く**（`src/diarize.py` / sherpa-onnx + pyannote-segmentation-3.0
  + NeMo TitaNet。設計書 `claude/claude_話者分離_設計書.md`）。モデル 46MB は
  `build.spec` が同梱し、無ければビルドを止める。
  **クラスタは音声の全長で作る**ので `g:A` `g:B`（画面表示は `声A`）。
  チャンクをまたいで同じ人は同じ記号になる。
- **クラウド（明示選択）**: Gemini API（gemini-2.5-flash）。
  **クラスタはチャンク内で閉じる**ので `0:A` `1:A`（画面表示は `C1-A` `C2-A`）。
  C1-A と C2-A は別のまとまり。

**この 2 つでクラスタの粒度が違う**（`segments.py` の `GLOBAL_NAMESPACE`）。
割当の手数が変わる（実測 67 分・6 名: ローカル 6 回 / クラウドは
「チャンク数 × 人数」に近づく）ので、どちらの話をしているかを混ぜないこと。

## 絶対に守る設計原則

- `reviewed` の意味論を壊さない: ✓＝人が耳で聴いて確定 / △＝一括適用で埋めただけ。
  この区別が製品価値そのもの。自動処理が✓を立てることは決してない。
- 短い相づち（「はい」等）の自動削除・自動重複除去はしない。同意の意思表示が記録から消えるため。
- `speaker_id` は名簿の ID を指す。名簿の編集・並べ替えで ID を保つこと（振り直すと全員入れ替わる）。
- キャッシュキーは「音声指紋 + チャンク長 + 逐語フラグ（+ 名簿ハッシュ）」。
  どれかを欠くと古い転写の使い回し事故になる。
- `audio_fingerprint`（BLAKE2b 64bit・全量ハッシュ）のアルゴリズムは変更禁止。
  既存キャッシュと旧作業ファイルの `_is_same_audio` 互換が壊れる。
- スキーマ変更は一括で行う（v2→v3 は一回の移行にまとめる。散発的に足さない）。
- 録音内容は機微情報。ユーザーが明示した Gemini API 呼び出し以外に音声・本文を外部送信しない。

## 環境

- Windows / Python 3.12.10（CI と一致）/ リポジトリ: C:\dev\01\whosaid-editor
- 作業前に必ず `.venv` を有効化する。`src/transcribe.py` が google.genai を
  トップレベル import しており、素の Python では import 段階で落ちる。
- ffmpeg / ffplay は WinGet 導入済み（PATH にある）。

## テスト

- `pytest tests/` は命名規約の関係で GUI・結合テストを黙って skip する。
  全チェック（約 890 項目）は必ず個別実行で確認する:
  - `python tests\test_core.py`
  - `python tests\test_anchor.py`
  - `python tests\test_inspection.py`
  - `python tests\test_pipeline_integration.py`
  - `python tests\test_gui_smoke.py`
  - `python tests\test_listen_order.py`
  - `python tests\test_candidates.py`
  - `python tests\test_lang.py`
  - `python tests\test_cache_key.py`
  - `python tests\test_measure_header.py`
- 実モデル（faster-whisper）を使う確認は `python tests\test_align_integration.py`。
  CI 対象外で、SAPI 合成音声を作って align.py と local_asr.py の両方を見る。
- 新設テストも、個別実行で全チェックが走る構成にすること。
  複数の `run()` を持つファイルは、短絡評価（`run() or run2()`）にしない。

## 作業の進め方（必須）

- 着手前に実装計画を提示し、ユーザーの承認を得てから手を動かす。
- 変更は小さい単位に分割し、各単位でテストを通し、確認を得てから次へ進む。
  まとめて一気に変えない。
- ブランチ運用: `feature/xxx` → PR → `main`（non-squash マージ。コミット履歴を保つ）。
- 実装前に `claude/` の該当する `claude_*_設計書.md` を必ず読む。
  設計書と食い違う実装をしたくなったら、勝手に変えずユーザーに相談する。

## 既知の注意点

- Gemini のタイムスタンプはドリフトする既知バグがある（Google 未修正）。
  `redistribute_times()`（transcribe.py:512）が按分補正、
  `merge_consecutive()`（transcribe.py:603）が細切れ行の連結を担う。挙動を変える前に設計書を確認。
- 名称は **`WhosaidEditor` /「Whosaid 反訳エディタ」**（2026-08-21 に改名）。
  旧名 `GeminiTranscriber` は `config.LEGACY_APP_NAMES` にだけ残す
  ——**消すと旧版利用者の設定（API キー・名簿）が失われる。**
  `installer.iss` の `AppId` は**変えない**（変えると PC に 2 つ入る）。
  名前は APP_NAME / APP_TITLE / build.spec / installer.iss / build.bat /
  GitHub Actions / README が連動する。変えるときは一括で。
- 一部スクリプトは Git Bash 前提で PowerShell では動かない
  （`& "C:\Program Files\Git\bin\bash.exe"` で開いて実行する）。
