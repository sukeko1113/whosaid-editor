"""run_segment_pipeline の結合テスト(と、ffmpeg を実際に叩く audio の検証)。

Gemini API は呼ばず、transcribe_audio を差し替えて偽の応答を返す。
ffmpeg による分割・長さ取得・セグメント化・キャッシュ・割当の引き継ぎまでを通す。

    python3 tests/test_pipeline_integration.py

ffmpeg が無い環境ではスキップする。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows の既定コンソールは cp932 / cp1252 なので、日本語のラベルを print した
# 時点で UnicodeEncodeError になる(テストの中身とは無関係に落ちる)。
# 出力先のエンコーディングをここで固定しておく。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


from src import pipeline  # noqa: E402
from src.audio import extract_peaks  # noqa: E402
from src.segments import ENGINE_CLOUD, ENGINE_LOCAL, Project  # noqa: E402


# クラウド経路(従来の既定)。ローカル経路の検査は run_local() にある。
CLOUD = pipeline.EngineSpec(
    mode=ENGINE_CLOUD, model="gemini-2.5-flash", api_key="dummy")


FAKE_CHUNK_OUTPUT = """[00:00] 【A】 議事を始めます。
[00:20] 【B】 資料の確認をお願いします。
[00:45] 【A】 はい、こちらです。
[01:30] 【C】 一点よろしいですか。
"""


def make_tone(path: Path, seconds: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:a", "aac", "-b:a", "64k", str(path), "-loglevel", "error"],
        check=True,
    )


def make_tone_with_gap(path: Path) -> None:
    """1秒 音 → 1秒 無音 → 1秒 音 の WAV。波形の谷が見えるかの検証用。"""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-af", "volume=0:enable='between(t,1,2)'",
         "-c:a", "pcm_s16le", str(path), "-loglevel", "error"],
        check=True,
    )


def run() -> int:
    if not shutil.which("ffmpeg"):
        print("SKIPPED: ffmpeg が見つからないため、このテストは実行されていません。")
        print("         (ALL PASSED ではありません。CI では ffmpeg が入るので実行されます)")
        return 0

    failures = []

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        if not cond:
            failures.append(label)

    calls = {"n": 0}

    def fake_transcribe(client, audio_path, model, **kwargs):
        calls["n"] += 1
        assert kwargs.get("cluster_only") is True, "手動割当モードでは cluster_only=True のはず"
        assert not kwargs.get("roster"), "名簿を Gemini に渡してはいけない"
        return FAKE_CHUNK_OUTPUT

    class FakeClient:
        def __init__(self, api_key=None, http_options=None):
            # http_options を受け取らないと、_make_client() が渡す
            # タイムアウト設定で TypeError になる。
            # 併せて「タイムアウトが設定されているか」もここで検証する
            # (設定漏れは、実行してみるまで気づけない種類の不具合なので)。
            assert http_options is not None, "genai.Client にタイムアウトが設定されていない"
            self.http_options = http_options

    real_transcribe = pipeline.transcribe_audio
    real_genai = pipeline.genai
    pipeline.transcribe_audio = fake_transcribe

    class FakeGenai:
        Client = FakeClient

    pipeline.genai = FakeGenai  # type: ignore[assignment]

    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = tmp / "meeting.m4a"
            make_tone(audio, 150)          # 2分30秒 → 1分チャンクで 3 個

            # --- 波形の抽出(分割ダイアログ用) --------------------------
            wav = tmp / "tone.wav"
            make_tone_with_gap(wav)
            peaks = extract_peaks(wav, 0.0, 3.0, 30)
            check("要求した数のバケツが返る", len(peaks) == 30)
            check("各バケツは (最小, 最大)",
                  all(len(p) == 2 and p[0] <= p[1] for p in peaks))
            loud = max(abs(p[0]) for p in peaks[:8])        # 0.0〜0.8 秒(音あり)
            quiet = max(abs(p[0]) for p in peaks[12:18])    # 1.2〜1.8 秒(無音)
            check("音のある所は振幅が出る", loud > 1000)
            check("無音の谷が波形に出る", quiet * 4 < loud)
            # 範囲指定が効いていることを、無音側と音のある側の両方で見る
            # (無音の始まりはフィルタのフレーム境界ぶんずれるので内側を取る)
            part = extract_peaks(wav, 1.2, 1.8, 10)
            check("範囲を指定して切り出せる", len(part) == 10)
            check("無音の範囲を切り出すと静か", max(abs(p[1]) for p in part) * 4 < loud)
            tail = extract_peaks(wav, 2.2, 2.8, 10)
            check("音のある範囲を切り出すと振幅が出る", min(abs(p[1]) for p in tail) > 1000)
            check("範囲が逆なら空リスト", extract_peaks(wav, 2.0, 1.0, 10) == [])
            check("バケツ 0 なら空リスト", extract_peaks(wav, 0.0, 3.0, 0) == [])
            check("読めないファイルなら空リスト",
                  extract_peaks(tmp / "nope.wav", 0.0, 1.0, 10) == [])
            check("サンプルより多くのバケツを求めても落ちない",
                  len(extract_peaks(wav, 0.0, 0.01, 500)) == 500)

            logs: list[str] = []
            proj = pipeline.run_segment_pipeline(
                audio_path=audio,
                output_dir=tmp,
                engine=CLOUD,
                chunk_minutes=1,
                on_log=logs.append,
                on_progress=lambda c, t: None,
                is_cancelled=lambda: False,
                verbatim=False,
                roster="佐藤(理事長)\n田中\n鈴木",
            )

            assert proj is not None
            check("3 チャンクに分割された", calls["n"] == 3)
            check("区間が生成された", proj.total_count == 12)
            check("出席者が取り込まれた", [s.name for s in proj.speakers] == ["佐藤", "田中", "鈴木"])
            check("音声長を取得できた", 149 <= proj.duration <= 151)

            starts = [s.start for s in proj.segments]
            check("開始時刻が単調増加", starts == sorted(starts))
            # 開始位置は各チャンクの実測長を積み上げて決めるので、
            # 分割の端数(AAC のフレーム境界)ぶんだけ 60.0 からずれる
            check("チャンク 2 のオフセットが効いている",
                  abs(proj.segments[4].start - 60.0) < 0.5)
            check("クラスタがチャンクごとに分かれている",
                  proj.segments[0].cluster == "0:A" and proj.segments[4].cluster == "1:A")
            check("最終チャンクは音声末尾で終わる",
                  abs(proj.segments[-1].end - proj.duration) < 1.5)
            check("JSON が保存された",
                  Path(proj.json_path or "").exists()
                  and Path(proj.json_path or "").name == "meeting.speakers.json")

            # --- 割当してから再実行 → 引き継がれること -------------------
            sid = proj.speakers[0].id
            for seg in proj.segments[:5]:
                seg.speaker_id = sid
                seg.reviewed = True
            proj.save()

            calls["n"] = 0
            logs2: list[str] = []
            proj2 = pipeline.run_segment_pipeline(
                audio_path=audio,
                output_dir=tmp,
                engine=CLOUD,
                chunk_minutes=1,
                on_log=logs2.append,
                on_progress=lambda c, t: None,
                is_cancelled=lambda: False,
                verbatim=False,
                roster="佐藤(理事長)\n田中\n鈴木",
            )
            assert proj2 is not None
            check("キャッシュが効いて API を呼ばない", calls["n"] == 0)
            check("以前の割当が引き継がれた", proj2.assigned_count == 5)
            check("引き継ぎがログに出る", any("引き継ぎました" in m for m in logs2))
            check("引き継ぎ先が正しい話者",
                  all(s.speaker_id == sid for s in proj2.segments[:5]))

            # --- 人が足した相づちが、再実行で消えないこと -----------------
            # **対策前は黙って消えた**(相づちを足す設計書 §4・実測)。
            # 足した区間の orig_start は独自の値なので、照合に参加させると
            # どこにも当たらず捨てられるか、近くの区間を置き換えて消す。
            base = proj2.segments[0]
            inside = round(base.start + (base.end - base.start) / 2, 2)
            added = proj2.add_utterance(inside, inside + 0.6, "はいはい")
            before_texts = [s.text for s in proj2.segments]
            proj2.save()

            logs3: list[str] = []
            proj4 = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1, on_log=logs3.append,
                on_progress=lambda c, t: None, is_cancelled=lambda: False,
                verbatim=False, roster="佐藤(理事長)\n田中\n鈴木",
            )
            assert proj4 is not None
            check("足した相づちが再実行後も残る",
                  any(s.text == "はいはい" for s in proj4.segments))
            check("足した区間だとまだ分かる(edit_log が生きている)",
                  proj4.is_added_utterance(
                      next(s for s in proj4.segments if s.text == "はいはい")))
            check("他の区間を消していない",
                  all(t in [s.text for s in proj4.segments]
                      for t in before_texts))
            check("残したことをログに出す",
                  any("人が足した発話" in m for m in logs3))
            check("時間順の位置に入る",
                  [s.start for s in proj4.segments]
                  == sorted(s.start for s in proj4.segments))

            # 消してから再実行すると、もう戻らない
            proj4.remove_added_utterance(
                next(s.index for s in proj4.segments if s.text == "はいはい"))
            proj4.save()
            proj5 = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1, on_log=lambda m: None,
                on_progress=lambda c, t: None, is_cancelled=lambda: False,
                verbatim=False, roster="佐藤(理事長)\n田中\n鈴木",
            )
            check("消した相づちは再実行でも戻らない",
                  proj5 is not None
                  and not any(s.text == "はいはい" for s in proj5.segments))
            _ = added

            # --- 名簿を並べ替えて再実行しても人が入れ替わらないこと ---------
            # 話者 ID を振り直すと、確定済みの区間が別人を指してしまう
            proj3 = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False,
                roster="鈴木\n佐藤(理事長)\n田中",      # 並べ替え + 先頭に別人
            )
            assert proj3 is not None
            check("名簿を並べ替えても名前が入れ替わらない",
                  all(proj3.speaker_name(s.speaker_id) == "佐藤"
                      for s in proj3.segments[:5]))
            check("並べ替えが反映される",
                  [s.name for s in proj3.speakers] == ["鈴木", "佐藤", "田中"])
            check("話者 ID が重複しない",
                  len({s.id for s in proj3.speakers}) == len(proj3.speakers))

            # --- 名簿から消した人も、割当が残っていれば保持される -----------
            proj4 = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False,
                roster="田中",
            )
            assert proj4 is not None
            check("名簿から外れても割当済みの人は残る",
                  all(proj4.speaker_name(s.speaker_id) == "佐藤"
                      for s in proj4.segments[:5]))

            # --- 本文の手直しは再実行でも失われない -------------------------
            proj4.segments[7].text = "訂正した本文です"
            proj4.segments[7].text_edited = True
            proj4.save()
            proj5 = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="田中",
            )
            assert proj5 is not None
            check("手直しした本文が復元される",
                  proj5.segments[7].text == "訂正した本文です")

            # --- チャンク長を変えたらキャッシュを使い回さない ---------------
            calls["n"] = 0
            proj6 = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=2,    # 1分 → 2分
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="田中",
            )
            assert proj6 is not None
            check("チャンク長を変えたら転写し直す", calls["n"] == 2)
            check("音声全体が区間で覆われる",
                  abs(proj6.segments[-1].end - proj6.duration) < 1.5)

            # --- キャンセル ------------------------------------------------
            proj3 = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: True,
            )
            check("キャンセルで None が返る", proj3 is None)

            # --- 逐語モードは別キャッシュ ----------------------------------
            calls["n"] = 0
            pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, verbatim=True,
            )
            check("逐語モードは別キャッシュで再実行される", calls["n"] == 3)

            reloaded = Project.load(Path(proj.json_path or ""))
            check("保存済み JSON を読み直せる", reloaded.total_count == 12)
            check("音声の指紋が記録される", len(reloaded.audio_fingerprint) == 16)

        # ==============================================================
        # 同じファイル名のまま音声を差し替えたら、古い結果を再利用しない
        # ==============================================================
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = tmp / "meeting.m4a"
            make_tone(audio, 120)                 # 2分 → 1分チャンクで 2 個

            calls["n"] = 0
            proj_a = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤\n田中",
            )
            assert proj_a is not None
            fp_a = proj_a.audio_fingerprint
            sid = proj_a.speakers[0].id
            for seg in proj_a.segments:
                seg.speaker_id = sid
                seg.reviewed = True
            proj_a.save()
            check("1本目: 全区間を確定した", proj_a.assigned_count == proj_a.total_count)

            # 同じ名前のまま、中身の違う音声に差し替える(編集した想定)
            audio.unlink()
            make_tone(audio, 180)                 # 長さも中身も変わる
            calls["n"] = 0
            logs3: list[str] = []
            proj_b = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=logs3.append, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤\n田中",
            )
            assert proj_b is not None
            check("差し替えたら指紋が変わる", proj_b.audio_fingerprint != fp_a)
            # チャンク数は ffmpeg の切れ目次第なので、実際の個数と突き合わせる
            chunk_count = len({sg.chunk for sg in proj_b.segments})
            check("差し替えたら全チャンクを転写し直す(キャッシュを使わない)",
                  calls["n"] == chunk_count and chunk_count > 0)
            check("差し替えたら古い割当を引き継がない", proj_b.assigned_count == 0)
            check("差し替えを検知したログが出る",
                  any("音声の内容が変わっています" in m for m in logs3))
            check("出席者だけは引き継ぐ",
                  [s.name for s in proj_b.speakers] == ["佐藤", "田中"])
            check("古い作業ファイルを退避している",
                  any(p.name.endswith(".bak.json") for p in tmp.glob("*.bak.json")))
            check("新しい音声の長さで区間が作られる",
                  abs(proj_b.duration - 180) < 1.5)

            # 同じ音声で再実行すれば、これまでどおり引き継ぐ
            for seg in proj_b.segments[:3]:
                seg.speaker_id = proj_b.speakers[1].id
                seg.reviewed = True
            proj_b.save()
            calls["n"] = 0
            proj_c = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤\n田中",
            )
            assert proj_c is not None
            check("同じ音声ならキャッシュも割当も生きている",
                  calls["n"] == 0 and proj_c.assigned_count == 3)

            # 強制やり直しならキャッシュを使わない(割当は保持する)
            calls["n"] = 0
            proj_d = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤\n田中",
                force_retranscribe=True,
            )
            assert proj_d is not None
            check("強制やり直しは全チャンクを転写し直す",
                  calls["n"] == len({sg.chunk for sg in proj_d.segments}))
            check("強制やり直しでも割当は残る", proj_d.assigned_count == 3)

        # ==============================================================
        # 時刻の修正・分割・結合が、再実行しても保たれる
        #
        # 設計上いちばん壊れやすい所。照合の鍵を「ユーザーが直したあとの
        # start」にすると、直した区間ほど再実行のたびに迷子になる。
        # ==============================================================
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = tmp / "meeting.m4a"
            make_tone(audio, 120)
            proj_x = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤\n田中",
            )
            assert proj_x is not None
            sid = proj_x.speakers[0].id
            before_total = proj_x.total_count

            # (1) 時刻を 6 秒ずらす(実測で見たドリフトと同じ幅)
            moved_seg = proj_x.segments[1]
            moved_orig = (moved_seg.orig_start, moved_seg.orig_end)
            moved_seg.start += 6.0
            moved_seg.end += 6.0
            moved_seg.time_edited = True
            moved_seg.speaker_id = sid
            moved_seg.reviewed = True

            # (2) 分割して、空いた後半に落ちていた発言を書き足す
            s3 = proj_x.segments[3]
            head, tail = proj_x.split_segment(3, s3.start + s3.duration / 2, 3)
            split_orig = (head.orig_start, head.orig_end)
            tail.text = "落ちていた発言を書き足した"
            tail.text_edited = True

            # (3) 分割で 1 つ増えた後ろのほうで、2 区間を結合する
            merged = proj_x.merge_segments(6)
            merged_orig = (merged.orig_start, merged.orig_end)
            proj_x.save()

            logs_x: list[str] = []
            proj_y = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=logs_x.append, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤\n田中",
            )
            assert proj_y is not None
            check("再実行しても区間の数が変わらない",
                  proj_y.total_count == before_total)     # +1 分割 −1 結合
            check("index が 0..n-1 に振り直される",
                  [s.index for s in proj_y.segments] == list(range(proj_y.total_count)))

            moved_back = [s for s in proj_y.segments if s.orig_start == moved_orig[0]]
            check("直した時刻が再実行後も残る",
                  len(moved_back) == 1
                  and abs(moved_back[0].start - (moved_orig[0] + 6.0)) < 1e-6
                  and moved_back[0].time_edited is True)
            check("元の時刻(照合の鍵)は動いていない",
                  moved_back[0].orig_end == moved_orig[1])
            check("直した区間の割当も残る", moved_back[0].speaker_id == sid)

            family = [s for s in proj_y.segments
                      if (s.orig_start, s.orig_end) == split_orig]
            check("分割した 2 区間がそのまま戻る", len(family) == 2)
            check("書き足した本文が残る",
                  any(s.text == "落ちていた発言を書き足した" for s in family))
            check("分割の後半は擬似クラスタのまま", any(s.is_pseudo_cluster for s in family))

            merged_back = [s for s in proj_y.segments
                           if (s.orig_start, s.orig_end) == merged_orig]
            check("結合した区間が 2 つに戻らない", len(merged_back) == 1)
            check("取り込んだことがログに出る",
                  any("取り込みました" in m for m in logs_x))

        # ==============================================================
        # 指紋の無い古い作業ファイルは、継続時間で差し替えを判定する
        # ==============================================================
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = tmp / "meeting.m4a"
            make_tone(audio, 120)
            proj_e = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤",
            )
            assert proj_e is not None
            sid = proj_e.speakers[0].id
            for seg in proj_e.segments:
                seg.speaker_id = sid
            proj_e.audio_fingerprint = ""          # 旧形式を再現
            proj_e.save()

            audio.unlink()
            make_tone(audio, 180)
            proj_f = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤",
            )
            assert proj_f is not None
            check("旧形式でも長さの違いで差し替えを検知", proj_f.assigned_count == 0)

        # ==============================================================
        # API のエラーで中途半端な結果を作らない
        # ==============================================================
        from src.transcribe import FatalTranscriptionError

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = tmp / "meeting.m4a"
            make_tone(audio, 120)

            # 残高切れ: 1 チャンク目で即中断し、リトライもしない
            calls["n"] = 0

            def depleted(client, audio_path, model, **kwargs):
                calls["n"] += 1
                from src.transcribe import classify_api_error
                exc = Exception(
                    "429 RESOURCE_EXHAUSTED. {'error': {'message': "
                    "'Your prepayment credits are depleted.'}}")
                raise FatalTranscriptionError(classify_api_error(exc) or "")

            pipeline.transcribe_audio = depleted
            fatal_msg = ""
            try:
                pipeline.run_segment_pipeline(
                    audio_path=audio, output_dir=tmp, engine=CLOUD,
                    chunk_minutes=1,
                    on_log=lambda m: None, on_progress=lambda c, t: None,
                    is_cancelled=lambda: False, roster="佐藤",
                )
            except FatalTranscriptionError as e:
                fatal_msg = str(e)
            check("残高切れは中断して例外を投げる", "残高が尽きています" in fatal_msg)
            check("残高切れは1チャンク目で止まる(全部試さない)", calls["n"] == 1)
            check("残高切れでは作業ファイルを作らない",
                  not Project.default_json_path(tmp, audio).exists())

            # 全チャンク失敗(再試行対象のエラー): 割当画面を開かせない
            def always_fail(client, audio_path, model, **kwargs):
                raise RuntimeError("network unreachable")

            pipeline.transcribe_audio = always_fail
            err = ""
            try:
                pipeline.run_segment_pipeline(
                    audio_path=audio, output_dir=tmp, engine=CLOUD,
                    chunk_minutes=1,
                    on_log=lambda m: None, on_progress=lambda c, t: None,
                    is_cancelled=lambda: False, roster="佐藤",
                )
            except RuntimeError as e:
                err = str(e)
            check("全チャンク失敗なら例外を投げる", "すべてのチャンク" in err)
            check("失敗の原因が添えられる", "network unreachable" in err)

            # 一部だけ失敗なら、成功したぶんで作業を続けられる
            state = {"n": 0}

            def fail_first(client, audio_path, model, **kwargs):
                state["n"] += 1
                if state["n"] == 1:
                    raise RuntimeError("temporary glitch")
                return FAKE_CHUNK_OUTPUT

            pipeline.transcribe_audio = fail_first
            logs4: list[str] = []
            proj_g = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=logs4.append, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤",
            )
            check("一部失敗なら続行できる", proj_g is not None)
            check("失敗件数を知らせる", any("失敗しました" in m for m in logs4))

            # 失敗したチャンクはキャッシュされないので、再実行で取得し直す
            state["n"] = 99
            calls["n"] = 0
            pipeline.transcribe_audio = fake_transcribe
            proj_h = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=CLOUD,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤",
            )
            assert proj_h is not None
            check("失敗したチャンクだけ取り直す", calls["n"] == 1)
            check("再実行後は失敗の痕跡が残らない",
                  not any("文字起こし失敗" in s.text for s in proj_h.segments))

        pipeline.transcribe_audio = fake_transcribe
    finally:
        pipeline.transcribe_audio = real_transcribe
        pipeline.genai = real_genai

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


# ======================================================================
# ローカル経路(faster-whisper)。モデルは使わず、偽の転写器を差し込む。
# ======================================================================

LOCAL = pipeline.EngineSpec(mode=ENGINE_LOCAL, model="small", diarize=False)
LOCAL_DIAR = pipeline.EngineSpec(mode=ENGINE_LOCAL, model="small", diarize=True)


def _raise_unavailable(*a, **k):
    raise pipeline.diarize_mod.DiarizeUnavailable("モデルがありません(検査用)")


def run_broken_project_backup() -> int:
    """読めない作業ファイルを、上書きする前に退避するか(pipeline.py の except)。

    **同じ関数の上 2 分岐(音声が変わった・経路が変わった)は正常系で、どちらも
    退避している。異常系のここだけ退避が無かった。**このあと
    proj.save(json_path) が上書きするので、ログ 1 行だけ残して作業が消えていた。

    壊し方は `KeyError: 'index'` を選ぶ。**実態にいちばん近い壊れ方**で、区間
    1 つから index が欠けただけの、手で直せば救えるファイルがこの経路へ来る。
    JSON ごと壊れた場合より、失うものが大きい。
    """
    if not shutil.which("ffmpeg"):
        print("SKIPPED: ffmpeg が無いので壊れた作業ファイルの検査は"
              "実行されていません。")
        return 0

    failures = []

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        if not cond:
            failures.append(label)

    class FakeLocal:
        def __init__(self, model="small", model_dir=None, **kw):
            self.model, self.model_dir = model, model_dir
            self.device, self.compute_type = "cpu", "int8"
            self.prompt_ver = 0

        def ensure_available(self):
            pass

        def transcribe(self, audio_path, *, on_log=None, on_progress=None,
                       is_cancelled=None):
            return pipeline.local_asr.ChunkResult(
                utterances=[
                    pipeline.local_asr.Utterance(0.0, 3.5, "議事を始めます。"),
                    pipeline.local_asr.Utterance(4.0, 9.0, "資料の確認を。"),
                ],
                words=[], duration=60.0,
            )

    real_local = pipeline.local_asr.LocalTranscriber
    pipeline.local_asr.LocalTranscriber = FakeLocal        # type: ignore[misc]

    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = tmp / "meeting.m4a"
            make_tone(audio, 60)

            print("\n[壊れた作業ファイルの退避]")
            proj = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=LOCAL,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤\n鈴木",
            )
            assert proj is not None
            json_path = Path(proj.json_path)
            check("1 回目で作業ファイルができる", json_path.is_file())

            # **人が確定した印を立ててから壊す。**退避されなければ、この ✓ が
            # 消える。区間の数だけでなく「何が失われるか」を検査に出す。
            data = json.loads(json_path.read_text(encoding="utf-8"))
            data["segments"][0]["speaker_id"] = "sp01"
            data["segments"][0]["reviewed"] = True
            n_before = len(data["segments"])
            del data["segments"][-1]["index"]        # KeyError: 'index'
            json_path.write_text(json.dumps(data, ensure_ascii=False),
                                 encoding="utf-8")
            broken_bytes = json_path.read_bytes()

            # 壊れていることを検査自身で確かめる。**壊せていないまま
            # 「退避された」と読むのを防ぐ。**
            try:
                Project.load(json_path)
                check("壊したファイルが読めなくなっている", False)
            except Exception:
                check("壊したファイルが読めなくなっている", True)

            logs: list[str] = []
            proj2 = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=LOCAL,
                chunk_minutes=1,
                on_log=logs.append, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤\n鈴木",
            )
            assert proj2 is not None

            baks = sorted(tmp.glob("*.bak.json"))
            check("壊れた作業ファイルが .bak.json へ退避される", len(baks) == 1)
            if baks:
                check("退避したものが、壊れる前の中身そのもの",
                      baks[0].read_bytes() == broken_bytes)
                saved = json.loads(baks[0].read_text(encoding="utf-8"))
                check("人が確定した印(✓)が退避先に残っている",
                      saved["segments"][0].get("reviewed") is True
                      and saved["segments"][0].get("speaker_id") == "sp01")
                check("区間が欠けずに退避されている",
                      len(saved["segments"]) == n_before)
            check("退避したことをログに出す",
                  any(".bak.json" in m for m in logs))
            # 退避して終わりではない。作り直しは通常どおり進むこと
            check("作り直しは進む(新しい作業ファイルができる)",
                  json_path.is_file() and len(proj2.segments) > 0)
            check("作り直したファイルは読める",
                  Project.load(json_path) is not None)
    finally:
        pipeline.local_asr.LocalTranscriber = real_local   # type: ignore[misc]

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


def run_speaker_merge() -> int:
    """名簿の突き合わせ(設計書 §11.8)。**同じ人が 2 つの ID になるのを防ぐ。**

    実データで 8 組できた(2026-08-18)。文字起こし画面で名簿を「名前」と
    「企業・役職」に分けてから走らせると、名前だけの照合では別人と判断され、
    さらに「消えた人も保持する」規則で古い方も残っていた。
    """
    from src.segments import Speaker
    failures: list[str] = []

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        if not cond:
            failures.append(label)

    print(chr(10) + "[名簿の突き合わせ]")
    br = chr(10)

    old = [Speaker(id="sp01", name="三ツ林衆議院議員"),
           Speaker(id="sp02", name="山本学　文科省　高等教育局"),
           Speaker(id="sp03", name="西村香介")]
    got = pipeline._merge_speakers(
        list(old),
        br.join(["三ツ林(衆議院議員)", "山本学(文科省 高等教育局)", "西村香介"]))
    check("分けても人数が増えない", len(got) == 3)
    check("**同じ人の ID を保つ**",
          [sp.id for sp in got] == ["sp01", "sp02", "sp03"])
    check("名前が短くなる", got[0].name == "三ツ林" and got[1].name == "山本学")
    check("役職が別に入る", got[0].note == "衆議院議員")
    check("空白の違いは無視する", got[1].note.replace(" ", "") .replace(
        chr(12288), "") == "文科省高等教育局")

    # 名前だけ一致するほうを先に見る(役職を書き換えただけの人を取り違えない)
    old2 = [Speaker(id="sp01", name="佐藤", note="理事"),
            Speaker(id="sp02", name="佐藤理事")]
    got2 = pipeline._merge_speakers(
        list(old2), br.join(["佐藤(理事)", "佐藤理事"]))
    check("名前が一致するほうを先に当てる",
          [sp.id for sp in got2] == ["sp01", "sp02"])

    # 本当に別人なら、これまでどおり新しい ID を振る
    got3 = pipeline._merge_speakers(
        [Speaker(id="sp01", name="佐藤")], br.join(["佐藤", "田中"]))
    check("本当に新しい人には新しい ID", len(got3) == 2
          and got3[1].id != "sp01")

    # 名簿から消えた人は、割当が残っているかもしれないので保持する(従来どおり)
    got4 = pipeline._merge_speakers(
        [Speaker(id="sp01", name="佐藤"), Speaker(id="sp02", name="田中")],
        "佐藤")
    check("名簿から外しても消さない",
          [sp.id for sp in got4] == ["sp01", "sp02"])

    check("ID が重複しない", len({sp.id for sp in got}) == len(got))

    print(chr(10) + ('FAILED: ' + ', '.join(failures)
                     if failures else 'ALL PASSED'))
    return 1 if failures else 0


def run_local() -> int:
    if not shutil.which("ffmpeg"):
        print("SKIPPED: ffmpeg が無いのでローカル経路の検査は実行されていません。")
        return 0

    failures = []

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        if not cond:
            failures.append(label)

    calls = {"transcribe": 0, "client": 0}

    class FakeLocal:
        """LocalTranscriber の代わり。モデルもライブラリも要らない。"""

        def __init__(self, model="small", model_dir=None, **kw):
            self.model, self.model_dir = model, model_dir
            # 本物と同じだけの素性を持つ。**キャッシュキーに入るので
            # 欠けると、設定違いの転写が同じキーを共有する検査にならない。**
            self.device, self.compute_type = "cpu", "int8"
            self.prompt_ver = 0

        def ensure_available(self):
            pass

        def transcribe(self, audio_path, *, on_log=None, on_progress=None,
                       is_cancelled=None):
            calls["transcribe"] += 1
            # whisper は句点で区間を切る。長さは 1 分チャンクに収まる範囲。
            return pipeline.local_asr.ChunkResult(
                utterances=[
                    pipeline.local_asr.Utterance(0.0, 3.5, "議事を始めます。"),
                    pipeline.local_asr.Utterance(4.0, 9.0, "資料の確認をお願いします。"),
                    pipeline.local_asr.Utterance(20.0, 22.0, "はい。"),
                ],
                words=[pipeline.local_asr.Word("議事", 0.0, 0.8)],
                duration=60.0,
            )

    class FakeClient:
        def __init__(self, api_key=None, http_options=None):
            calls["client"] += 1

    class FakeGenai:
        Client = FakeClient

    real_local = pipeline.local_asr.LocalTranscriber
    real_genai = pipeline.genai
    pipeline.local_asr.LocalTranscriber = FakeLocal        # type: ignore[misc]
    pipeline.genai = FakeGenai                             # type: ignore[assignment]

    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = tmp / "meeting.m4a"
            make_tone(audio, 150)          # 2分30秒 → 1分チャンクで 3 個

            print("\n[ローカル経路]")
            proj = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=LOCAL,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤\n鈴木",
                verbatim=True,      # ローカルでは効かないはず
            )
            assert proj is not None

            # 鍵を渡していないのに動くこと自体が要件。クラウドの
            # クライアントを作ってしまうと、ローカルなのに鍵を求める形になる
            check("API キーを要求しない(クライアントを作らない)", calls["client"] == 0)
            check("チャンクの数だけ転写する", calls["transcribe"] == 3)
            check("区間が取り込まれる", len(proj.segments) == 9)
            check("処理経路が local で記録される",
                  proj.engine.get("mode") == ENGINE_LOCAL)
            check("モデルと量子化が記録される",
                  proj.engine.get("model") == "small"
                  and proj.engine.get("compute_type") == "int8")
            check("全区間が未判別",
                  all(s.is_pseudo_cluster for s in proj.segments))
            check("逐語モードは記録しない", proj.verbatim is False)
            # 2 チャンク目の先頭は 60 秒付近から始まる(オフセットが乗る)
            check("チャンクのオフセットが乗る", proj.segments[3].start >= 59.0)
            check("時刻が按分で作り直されていない",
                  proj.segments[0].end == 3.5 and proj.segments[1].start == 4.0)

            # --- キャッシュ ---------------------------------------------
            calls["transcribe"] = 0
            pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=LOCAL,
                chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤",
            )
            check("二度目はキャッシュから復元する", calls["transcribe"] == 0)

            calls["transcribe"] = 0
            pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=LOCAL,
                chunk_minutes=2,        # チャンク長を変えたら別物
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤",
            )
            check("チャンク長を変えると取り直す", calls["transcribe"] > 0)

            calls["transcribe"] = 0
            pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp,
                engine=pipeline.EngineSpec(mode=ENGINE_LOCAL, model="base"),
                chunk_minutes=1,        # モデルを変えたら別物
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤",
            )
            check("モデルを変えると取り直す", calls["transcribe"] > 0)

        # --- 部品が無いときは、重い処理に入る前に止まる -----------------
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = tmp / "meeting.m4a"
            make_tone(audio, 150)

            print("\n[部品が無いとき]")

            class Missing(FakeLocal):
                def ensure_available(self):
                    raise pipeline.local_asr.LocalAsrUnavailable(
                        "ローカル転写には faster-whisper が必要です。")

            split_calls = {"n": 0}
            real_split = pipeline.split_audio

            def counting_split(*a, **k):
                split_calls["n"] += 1
                return real_split(*a, **k)

            pipeline.local_asr.LocalTranscriber = Missing    # type: ignore[misc]
            pipeline.split_audio = counting_split            # type: ignore[assignment]
            try:
                raised = ""
                try:
                    pipeline.run_segment_pipeline(
                        audio_path=audio, output_dir=tmp, engine=LOCAL,
                        chunk_minutes=1,
                        on_log=lambda m: None, on_progress=lambda c, t: None,
                        is_cancelled=lambda: False, roster="佐藤",
                    )
                except pipeline.local_asr.LocalAsrUnavailable as e:
                    raised = str(e)
                check("部品が無ければ知らせる", "faster-whisper" in raised)
                # 370MB の音声だと分割だけで数分かかる。待たせてから
                # 「部品がありません」では、待った時間がまるごと無駄になる
                check("分割を始める前に止まる", split_calls["n"] == 0)
            finally:
                pipeline.local_asr.LocalTranscriber = FakeLocal  # type: ignore[misc]
                pipeline.split_audio = real_split                # type: ignore[assignment]

        # --- 話者分離を繋いだとき(話者分離設計書 §4・§8) -----------------
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = tmp / "meeting.m4a"
            make_tone(audio, 150)

            print("\n[話者分離を繋ぐ]")
            calls["transcribe"] = 0
            diar_calls = {"n": 0, "num_speakers": None}

            def fake_diarize(audio_path, *, num_speakers=6, **kw):
                diar_calls["n"] += 1
                diar_calls["num_speakers"] = num_speakers
                # 1 チャンク目は話者 0、2 チャンク目以降は話者 1 が話す想定
                return [
                    pipeline.diarize_mod.SpeakerTurn(0.0, 30.0, 0),
                    pipeline.diarize_mod.SpeakerTurn(30.0, 200.0, 1),
                ]

            real_diar = pipeline.diarize_mod.diarize
            real_ensure = pipeline.diarize_mod.ensure_available
            pipeline.diarize_mod.diarize = fake_diarize
            pipeline.diarize_mod.ensure_available = lambda *a, **k: None
            try:
                proj = pipeline.run_segment_pipeline(
                    audio_path=audio, output_dir=tmp, engine=LOCAL_DIAR,
                    chunk_minutes=1,
                    on_log=lambda m: None, on_progress=lambda c, t: None,
                    is_cancelled=lambda: False, roster="佐藤\n鈴木\n田中",
                )
                assert proj is not None
                check("話者分離を 1 回だけ呼ぶ(全長で 1 回)", diar_calls["n"] == 1)
                check("名簿の人数を上限として渡す",
                      diar_calls["num_speakers"] == 3)
                check("まとまりが全長の名前空間になる",
                      all(s.cluster.startswith("g:") for s in proj.segments))
                real = {s.cluster for s in proj.segments
                        if not s.is_pseudo_cluster}
                check("声のまとまりが付く", len(real) >= 1)
                check("処理経路に話者分離が残る",
                      "diarize" in proj.engine
                      and proj.engine["diarize"]["num_speakers"] == 3)
                check("確定はしない(✓ を立てない)",
                      not any(s.reviewed for s in proj.segments))
                check("話者は未確定のまま",
                      all(s.speaker_id is None for s in proj.segments))

                # --- 聴く順番(listen_order) --------------------------
                from src import listen_order as lo
                work = tmp / f".work_{audio.stem}"
                hp = lo.hints_path(work, proj.audio_fingerprint or "")
                hints = lo.load_hints(hp)
                check("聴く順番が sidecar に残る", hints is not None)
                check("本体の JSON には入れない",
                      "listen" not in json.loads(
                          Path(proj.json_path).read_text(encoding="utf-8")))
                if hints:
                    idx = {h.index for h in hints}
                    check("番号を振り直したあとの区間を指す",
                          idx == {s.index for s in proj.segments})
                    check("順番は点数の高い順",
                          [h.score for h in lo.listen_first(hints)]
                          == sorted((h.score for h in hints), reverse=True))

                # 話者分離が無ければ順番も出さない(手がかりが無いため)
                proj_nd = pipeline.run_segment_pipeline(
                    audio_path=audio, output_dir=tmp, engine=LOCAL,
                    chunk_minutes=1, force_retranscribe=True,
                    on_log=lambda m: None, on_progress=lambda c, t: None,
                    is_cancelled=lambda: False, roster="佐藤",
                )
                hp2 = lo.hints_path(
                    work, (proj_nd.audio_fingerprint if proj_nd else "") or "")
                check("話者分離が無ければ順番を作らない",
                      hp2 is None or not hp2.exists())

                # 部品が無くても転写は捨てない
                pipeline.diarize_mod.ensure_available = _raise_unavailable
                calls["transcribe"] = 0
                logs: list[str] = []
                proj2 = pipeline.run_segment_pipeline(
                    audio_path=audio, output_dir=tmp, engine=LOCAL_DIAR,
                    chunk_minutes=1, force_retranscribe=True,
                    on_log=logs.append, on_progress=lambda c, t: None,
                    is_cancelled=lambda: False, roster="佐藤",
                )
                check("話者分離が使えなくても転写は残る",
                      proj2 is not None and len(proj2.segments) > 0)
                check("使えない理由を知らせる",
                      any("話者分離は使えません" in m for m in logs))
            finally:
                pipeline.diarize_mod.diarize = real_diar
                pipeline.diarize_mod.ensure_available = real_ensure

        # --- 経路が変わったときの引き継ぎ(設計書 §8.1) ------------------
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            audio = tmp / "meeting.m4a"
            make_tone(audio, 150)

            print("\n[経路が変わったとき]")
            # まずクラウドで作って、人が 1 件確定した状態を作る
            def fake_cloud(client, audio_path, model, **kwargs):
                return FAKE_CHUNK_OUTPUT

            real_transcribe = pipeline.transcribe_audio
            pipeline.transcribe_audio = fake_cloud
            try:
                proj_c = pipeline.run_segment_pipeline(
                    audio_path=audio, output_dir=tmp, engine=CLOUD,
                    chunk_minutes=1,
                    on_log=lambda m: None, on_progress=lambda c, t: None,
                    is_cancelled=lambda: False, roster="佐藤",
                )
                assert proj_c is not None
                proj_c.segments[0].speaker_id = proj_c.speakers[0].id
                proj_c.segments[0].reviewed = True      # 聴いて確定した ✓
                proj_c.segments[1].text = "人が直した本文"
                proj_c.segments[1].text_edited = True
                proj_c.doc_revision = 3
                proj_c.save(proj_c.json_path)
            finally:
                pipeline.transcribe_audio = real_transcribe

            logs: list[str] = []
            proj_l = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, engine=LOCAL,
                chunk_minutes=1,
                on_log=logs.append, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤",
            )
            assert proj_l is not None
            # ✓ が別の区間へ移らないこと。ここが崩れると、人が一度も見ていない
            # 区間に「聴いて確定した」印が付く(製品価値そのものが壊れる)
            check("確定済みを引き継がない",
                  all(s.speaker_id is None for s in proj_l.segments))
            check("✓ が残らない", not any(s.reviewed for s in proj_l.segments))
            check("本文の手直しも引き継がない",
                  not any(s.text_edited for s in proj_l.segments))
            check("名簿は引き継ぐ",
                  [sp.name for sp in proj_l.speakers] == ["佐藤"])
            check("旧ファイルを退避する", bool(list(tmp.glob("*.bak.json"))))
            check("理由を知らせる",
                  any("引き継ぎません" in m for m in logs))
            check("版番号も引き継がない", proj_l.doc_revision == 0)
    finally:
        pipeline.local_asr.LocalTranscriber = real_local    # type: ignore[misc]
        pipeline.genai = real_genai

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    # 片方が落ちてももう片方を必ず走らせる(短絡すると検査が静かに減る)
    rc_cloud = run()
    rc_local = run_local()
    rc_merge = run_speaker_merge()
    rc_broken = run_broken_project_backup()
    sys.exit(rc_cloud or rc_local or rc_merge or rc_broken)
