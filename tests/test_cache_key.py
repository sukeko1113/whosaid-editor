"""転写キャッシュの鍵の検査。

    .venv\\Scripts\\python.exe tests\\test_cache_key.py

**同じ事故を 4 回起こしている。**「鍵に入れるべき要素を入れ忘れた」形:

    v2.0.1  ファイル名しか見ておらず、中身を変えた音声で古い転写を使った
    8-14    モデルフォルダを差し替えても同じ鍵だった
    8-27    言語を変えても同じ鍵だった
    8-28    モデル名と逐語プロンプトの版が、実経路の鍵に入っていなかった

規約では止まらないので、機械で縛る。この検査は 3 つのことを見る。

1. **分類漏れ** — EngineSpec に項目を足したら、鍵に入れるか除外するかを
   決めるまで落ちる。バックエンド抽象化でエンジン名を足す場面がこれ。
2. **実経路のファイル名** — 関数の戻り値ではなく、パイプラインが**実際に
   書き出したファイル名**を見る。8-27 は「変更した関数を直接呼んで検証し、
   その関数が呼ばれていない経路を見逃した」形だった。
3. **単一の出所** — パイプラインが書いた名前が _cache_suffix() の出力と
   一致する。自前組み立てが復活したら落ちる。

**守れないこと**(正直に書いておく):
- EngineSpec 以外の場所に新しい入力が増えた場合は捕まらない
- 「このプロンプト変更は転写を変えるので版を上げるべきだ」は人の判断。
  ここで担保できるのは VERBATIM_PROMPT_VER が鍵に入っていることまで
"""
from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src import pipeline                                    # noqa: E402
from src.pipeline import (                                  # noqa: E402
    CACHE_KEY_ENGINE_EXCLUDED,
    CACHE_KEY_ENGINE_FIELDS,
    EngineSpec,
    VERBATIM_PROMPT_VER,
    _cache_suffix,
)
from src.segments import ENGINE_CLOUD, ENGINE_LOCAL         # noqa: E402


FP = "abc123def456"
CS = 960

CLOUD = EngineSpec(mode=ENGINE_CLOUD, model="gemini-2.5-flash", api_key="k")


def key(engine=CLOUD, *, verbatim=True, chunk_seconds=CS, fingerprint=FP):
    return _cache_suffix(True, True, verbatim, "", chunk_seconds, fingerprint,
                         engine=engine, cluster_only=True)


# ======================================================================
# 1. 分類漏れ — EngineSpec に項目を足したら落ちる
# ======================================================================

def test_every_engine_field_is_classified():
    """**新しい項目を足したら、分類するまで落ちる。**

    これが 4 回目を止める本体。EngineSpec に項目を足しただけでは通らず、
    CACHE_KEY_ENGINE_FIELDS(鍵に入れる)か
    CACHE_KEY_ENGINE_EXCLUDED(入れない・理由つき)に入れる必要がある。
    """
    have = {f.name for f in dataclasses.fields(EngineSpec)}
    classified = set(CACHE_KEY_ENGINE_FIELDS) | set(CACHE_KEY_ENGINE_EXCLUDED)
    missing = have - classified
    assert not missing, (
        f"EngineSpec の項目 {sorted(missing)} が分類されていません。"
        "転写の中身が変わるなら CACHE_KEY_ENGINE_FIELDS に、"
        "変わらないなら理由を添えて CACHE_KEY_ENGINE_EXCLUDED に入れてください。"
        "**入れ忘れると、その項目を変えても古い転写が返ります。**")
    stale = classified - have
    assert not stale, f"EngineSpec に無い項目が分類に残っています: {sorted(stale)}"


def test_excluded_fields_have_a_reason():
    """除外は理由とセットにする。理由が無い除外は、ただの入れ忘れと区別できない。"""
    for name, why in CACHE_KEY_ENGINE_EXCLUDED.items():
        assert why and len(why) > 10, f"{name} の除外理由が書かれていません"


def test_keyed_fields_actually_change_the_key():
    """**鍵に入れると宣言した項目は、本当に鍵を変えること。**

    宣言だけして実装を忘れたら、分類は通るのに事故は起きる。
    """
    base = key()
    variants = {
        "mode": EngineSpec(mode=ENGINE_LOCAL, model="gemini-2.5-flash"),
        "model": EngineSpec(mode=ENGINE_CLOUD, model="gemini-2.5-pro",
                            api_key="k"),
        "model_dir": EngineSpec(mode=ENGINE_CLOUD, model="gemini-2.5-flash",
                                api_key="k", model_dir="C:\\\\models\\\\x"),
    }
    for name in CACHE_KEY_ENGINE_FIELDS:
        assert name in variants, f"{name} を変えた場合の検査がありません"
        assert key(variants[name]) != base, (
            f"{name} を変えても鍵が同じです({base})。"
            "**この項目を変えても古い転写が返ります。**")


def test_excluded_fields_do_not_change_the_key():
    """除外した項目は鍵を変えないこと(変えると無駄な取り直しが起きる)。"""
    base = key()
    assert key(EngineSpec(mode=ENGINE_CLOUD, model="gemini-2.5-flash",
                          api_key="別のキー")) == base, "api_key で鍵が変わっている"
    assert key(EngineSpec(mode=ENGINE_CLOUD, model="gemini-2.5-flash",
                          api_key="k", diarize=False)) == base
    assert key(EngineSpec(mode=ENGINE_CLOUD, model="gemini-2.5-flash",
                          api_key="k", num_speakers=9)) == base


# ======================================================================
# 鍵に入るその他の要素
# ======================================================================

def test_fingerprint_chunk_and_verbatim_change_the_key():
    base = key()
    assert key(fingerprint="ちがう指紋") != base, "音声の指紋"
    assert key(chunk_seconds=420) != base, "チャンク長"
    assert key(verbatim=False) != base, "逐語フラグ"


def test_verbatim_prompt_version_is_in_the_key():
    """**逐語プロンプトの版が鍵に入っていること。**

    2026-08-28 まで、実経路は `.vb`(版なし)を書いていた。版を上げても
    キャッシュが無効化されず、古い転写が返る状態だった。

    版を上げるべきかどうかは人の判断で、ここでは担保できない。
    担保できるのは「上げれば効く」ことまで。
    """
    assert f"vb{VERBATIM_PROMPT_VER}" in key()
    assert f"vb{VERBATIM_PROMPT_VER}" not in key(verbatim=False)


def test_no_fingerprint_is_still_usable():
    """指紋が取れなくても鍵は作れる(ffmpeg が無い等)。"""
    got = key(fingerprint="")
    assert got.endswith(".txt") and FP not in got


# ======================================================================
# 2 と 3. 実経路のファイル名 / 単一の出所
#
# **ここが要点。**関数を直接呼ぶ検査だけでは、その関数が呼ばれていない
# 経路を見逃す(2026-08-27 に実際に見逃した)。パイプラインを実際に走らせて、
# 書き出されたファイル名を見る。
# ======================================================================

FAKE_CHUNK = """[00:00] 【A】 議事を始めます。
[00:20] 【B】 資料の確認をお願いします。
"""


def _make_tone(path: Path, seconds: int = 70) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}",
         "-c:a", "aac", "-b:a", "64k", str(path), "-loglevel", "error"],
        check=True)


class _FakeClient:
    def __init__(self, api_key=None, http_options=None):
        pass


class _FakeGenai:
    Client = _FakeClient


def _run_and_list_cache(tmp: Path, engine: EngineSpec, *, verbatim=True,
                        chunk_minutes=1):
    """パイプラインを実際に走らせ、書き出されたキャッシュ名を返す。"""
    audio = tmp / "meeting.m4a"
    if not audio.exists():
        _make_tone(audio)

    real_tr, real_genai = pipeline.transcribe_audio, pipeline.genai
    pipeline.transcribe_audio = lambda *a, **k: FAKE_CHUNK
    pipeline.genai = _FakeGenai
    try:
        proj = pipeline.run_segment_pipeline(
            audio_path=audio, output_dir=tmp, engine=engine,
            chunk_minutes=chunk_minutes,
            on_log=lambda m: None, on_progress=lambda c, t: None,
            is_cancelled=lambda: False, verbatim=verbatim, roster="")
    finally:
        pipeline.transcribe_audio = real_tr
        pipeline.genai = real_genai
    assert proj is not None
    return sorted(p.name for p in
                  (tmp / ".work_meeting" / "transcripts").glob("*.txt"))


def test_written_file_names_come_from_the_shared_function():
    """**書き出された名前が _cache_suffix() の出力と一致すること。**

    自前組み立てが復活したら、ここで落ちる。
    """
    if not shutil.which("ffmpeg"):
        print("    (ffmpeg が無いので飛ばす)")
        return
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        names = _run_and_list_cache(tmp, CLOUD)
        assert names, "キャッシュが書き出されていない"
        fp = pipeline.audio_fingerprint(tmp / "meeting.m4a")
        want = _cache_suffix(True, True, True, "", 60, fp,
                             engine=CLOUD, cluster_only=True)
        for n in names:
            assert n.endswith(want), (
                f"書き出された名前が共有の関数と違います。\\n"
                f"  実物: {n}\\n  期待: chunk_XXXX{want}\\n"
                "**自前で組み立てていませんか。**")


def test_written_file_names_change_with_the_model():
    """**モデルを変えたら、実際のファイル名が変わること。**

    2026-08-28 まで gemini-2.5-flash と gemini-2.5-pro が同じ名前だった。
    同じ音声を複数のエンジンで測るという前提が成立していなかった。
    """
    if not shutil.which("ffmpeg"):
        return
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # _run_and_list_cache はフォルダ全体を返すので、2 回目は 1 回目のぶんも
        # 含む。「増えたかどうか」で見る。
        flash = set(_run_and_list_cache(tmp, CLOUD))
        after = set(_run_and_list_cache(
            tmp, EngineSpec(mode=ENGINE_CLOUD, model="gemini-2.5-pro",
                            api_key="k")))
        added = after - flash
        assert added, f"モデルを変えても新しい名前ができません: {sorted(flash)}"
        # 1 回目のぶんが消えていない = 取り違えずに並べて持てる
        assert flash <= after, sorted(after)
        assert len(added) == len(flash), sorted(after)
        assert all("gemini-2.5-flash" in n for n in flash)
        assert all("gemini-2.5-pro" in n for n in added)


def test_written_file_names_change_with_verbatim():
    if not shutil.which("ffmpeg"):
        return
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        vb = _run_and_list_cache(tmp, CLOUD, verbatim=True)
        plain = _run_and_list_cache(tmp, CLOUD, verbatim=False)
        assert set(vb) != set(plain), vb


def test_written_file_names_change_with_chunk_length( ):
    if not shutil.which("ffmpeg"):
        return
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        one = _run_and_list_cache(tmp, CLOUD, chunk_minutes=1)
        two = _run_and_list_cache(tmp, CLOUD, chunk_minutes=2)
        assert set(one) != set(two), one


def test_cache_is_reused_when_nothing_changed():
    """鍵を細かくしすぎて、同じ条件でも毎回転写し直すことがないこと。"""
    if not shutil.which("ffmpeg"):
        return
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return FAKE_CHUNK

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        audio = tmp / "meeting.m4a"
        _make_tone(audio)
        real_tr, real_genai = pipeline.transcribe_audio, pipeline.genai
        pipeline.transcribe_audio = counting
        pipeline.genai = _FakeGenai
        try:
            for _ in range(2):
                pipeline.run_segment_pipeline(
                    audio_path=audio, output_dir=tmp, engine=CLOUD,
                    chunk_minutes=1, on_log=lambda m: None,
                    on_progress=lambda c, t: None,
                    is_cancelled=lambda: False, verbatim=True, roster="")
        finally:
            pipeline.transcribe_audio = real_tr
            pipeline.genai = real_genai
        first = calls["n"]
        assert first > 0
        # 2 回目は 1 度も呼ばれない
        assert calls["n"] == first, "同じ条件なのに転写し直している"


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
