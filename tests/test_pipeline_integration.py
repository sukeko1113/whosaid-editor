"""run_segment_pipeline の結合テスト。

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


def run() -> int:
    if not shutil.which("ffmpeg"):
        print("ffmpeg が無いためスキップします。")
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
        def __init__(self, api_key=None):
            pass

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
            check("チャンク 2 のオフセットが効いている", proj.segments[4].start == 60.0)
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
    finally:
        pipeline.transcribe_audio = real_transcribe
        pipeline.genai = real_genai

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
