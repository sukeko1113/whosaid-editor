"""run_segment_pipeline の結合テスト(と、ffmpeg を実際に叩く audio の検証)。

Gemini API は呼ばず、transcribe_audio を差し替えて偽の応答を返す。
ffmpeg による分割・長さ取得・セグメント化・キャッシュ・割当の引き継ぎまでを通す。

    python3 tests/test_pipeline_integration.py

ffmpeg が無い環境ではスキップする。
"""
from __future__ import annotations

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
from src.segments import Project  # noqa: E402


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
                api_key="dummy",
                model="gemini-2.5-flash",
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
                api_key="dummy",
                model="gemini-2.5-flash",
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

            # --- 名簿を並べ替えて再実行しても人が入れ替わらないこと ---------
            # 話者 ID を振り直すと、確定済みの区間が別人を指してしまう
            proj3 = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="田中",
            )
            assert proj5 is not None
            check("手直しした本文が復元される",
                  proj5.segments[7].text == "訂正した本文です")

            # --- チャンク長を変えたらキャッシュを使い回さない ---------------
            calls["n"] = 0
            proj6 = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=2,    # 1分 → 2分
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="田中",
            )
            assert proj6 is not None
            check("チャンク長を変えたら転写し直す", calls["n"] == 2)
            check("音声全体が区間で覆われる",
                  abs(proj6.segments[-1].end - proj6.duration) < 1.5)

            # --- キャンセル ------------------------------------------------
            proj3 = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: True,
            )
            check("キャンセルで None が返る", proj3 is None)

            # --- 逐語モードは別キャッシュ ----------------------------------
            calls["n"] = 0
            pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
                on_log=lambda m: None, on_progress=lambda c, t: None,
                is_cancelled=lambda: False, roster="佐藤\n田中",
            )
            assert proj_c is not None
            check("同じ音声ならキャッシュも割当も生きている",
                  calls["n"] == 0 and proj_c.assigned_count == 3)

            # 強制やり直しならキャッシュを使わない(割当は保持する)
            calls["n"] = 0
            proj_d = pipeline.run_segment_pipeline(
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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
                    audio_path=audio, output_dir=tmp, api_key="dummy",
                    model="gemini-2.5-flash", chunk_minutes=1,
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
                    audio_path=audio, output_dir=tmp, api_key="dummy",
                    model="gemini-2.5-flash", chunk_minutes=1,
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
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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
                audio_path=audio, output_dir=tmp, api_key="dummy",
                model="gemini-2.5-flash", chunk_minutes=1,
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


if __name__ == "__main__":
    sys.exit(run())
