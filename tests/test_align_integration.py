"""align.py の動作確認(ローカル専用・CI 対象外)。

実際に faster-whisper を回すので、初回はモデルの取得(数百 MB)と CPU での
転写に時間がかかる。テスト音声は Windows の日本語 TTS で合成する(設計書 §12)。
既知のテキストを喋らせるので、拾えた内容が正しいかを機械的に確かめられる。

    .venv\\Scripts\\python.exe tests\\test_align_integration.py

pytest からは拾われない(検査は run() の中にあり、test_ で始まる関数が無い)。
モデルや音声合成が使えない環境では、その旨を出して該当分だけ飛ばす。
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 日本語のラベルを print した時点で落ちないようにする(既定は cp932)
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.audio import audio_fingerprint  # noqa: E402
from src.align import (  # noqa: E402
    ALIGN_VER,
    DEFAULT_MODEL,
    AlignUnavailable,
    Word,
    load_words,
    resolve_model,
    save_words,
    transcribe_words,
    words_cache_path,
)

# 合成する既知のテキスト。会議の言い回しに寄せてある。
KNOWN_TEXT = (
    "本日はお忙しい中お集まりいただきありがとうございます。"
    "それでは第一号議案について事務局から説明をお願いします。"
    "はい。お手元の資料の三ページをご覧ください。"
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "words_sapi_sample.json"

_STRIP = re.compile(r"[\s、。，．・？！?!「」『』（）()]")


def synth_wav(path: Path, text: str) -> bool:
    """Windows の日本語 TTS で音声を作る。日本語の声が無ければ False。"""
    script = path.with_suffix(".ps1")
    script.write_text(
        "Add-Type -AssemblyName System.Speech\n"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "$ja = $s.GetInstalledVoices() | "
        "Where-Object { $_.VoiceInfo.Culture.Name -eq 'ja-JP' } | Select-Object -First 1\n"
        "if (-not $ja) { exit 2 }\n"
        "$s.SelectVoice($ja.VoiceInfo.Name)\n"
        f"$s.SetOutputToWaveFile('{path}')\n"
        f"$s.Speak(@'\n{text}\n'@)\n"
        "$s.Dispose()\n",
        encoding="utf-8-sig",       # BOM が無いと PowerShell が日本語を壊す
    )
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(script)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except OSError:
        return False
    return res.returncode == 0 and path.exists() and path.stat().st_size > 1000


def flatten(words: list[Word]) -> str:
    return _STRIP.sub("", "".join(w.text for w in words))


def run() -> int:
    failures: list[str] = []

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok  ' if cond else 'NG  '} {label}")
        if not cond:
            failures.append(label)

    # ------------------------------------------------------------------
    # モデルもファイルも要らない検査(ここは必ず走る)
    # ------------------------------------------------------------------
    print("\n[キャッシュキーとモデル指定]")
    check("指紋が無ければキャッシュしない",
          words_cache_path("w", "", DEFAULT_MODEL) is None)
    p = words_cache_path("w", "abcd1234", "small")
    check("キーに指紋・モデル・実装版が入る",
          p is not None and p.name == f"words.abcd1234.small.a{ALIGN_VER}.json")
    check("モデルが違えば別のキャッシュ",
          words_cache_path("w", "abcd1234", "base") != p)

    check("モデル名はそのまま渡す", resolve_model("small") == "small")
    try:
        resolve_model("small", model_dir="C:/no/such/folder")
        check("無いフォルダを指すと知らせる", False)
    except AlignUnavailable as e:
        check("無いフォルダを指すと知らせる", "モデルフォルダ" in str(e))

    with tempfile.TemporaryDirectory() as d:
        empty = Path(d) / "model"
        empty.mkdir()
        try:
            resolve_model("small", model_dir=empty)
            check("中身が足りないフォルダも知らせる", False)
        except AlignUnavailable as e:
            check("中身が足りないフォルダも知らせる", "model.bin" in str(e))

    print("\n[キャッシュの読み書き]")
    with tempfile.TemporaryDirectory() as d:
        cache = words_cache_path(d, "ff00", "small")
        sample = [Word("あ", 0.0, 0.5), Word("い", 0.5, 1.0)]
        save_words(cache, sample, model="small", fingerprint="ff00", duration=1.0)
        check("書いて読める", load_words(cache) == sample)
        # 実装が変わったキャッシュは使わない(古い照合結果の使い回しを防ぐ)
        data = json.loads(cache.read_text(encoding="utf-8"))
        data["align_ver"] = ALIGN_VER + 1
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        check("実装版が違うキャッシュは読まない", load_words(cache) is None)
        cache.write_text("これは JSON ではない", encoding="utf-8")
        check("壊れたキャッシュは無かったことにする", load_words(cache) is None)
        check("無いファイルは None", load_words(Path(d) / "nope.json") is None)

    # ------------------------------------------------------------------
    # 実際に転写する検査(音声合成とモデルが要る)
    # ------------------------------------------------------------------
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        print("\n[転写] faster-whisper が入っていないので飛ばします。")
        print("    pip install faster-whisper")
        print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
        return 1 if failures else 0

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        wav = tmp / "sample.wav"
        print("\n[転写] 音声を合成しています...")
        if not synth_wav(wav, KNOWN_TEXT):
            print("  日本語の音声合成ができないので、転写の検査は飛ばします。")
            print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
            return 1 if failures else 0

        logs: list[str] = []
        seen: list[tuple[float, float]] = []
        print(f"  合成できました({wav.stat().st_size / 1024:.0f} KB)。"
              "転写します(初回はモデルの取得があります)...")
        t0 = time.time()
        words = transcribe_words(
            wav, work_dir=tmp / ".work_sample", model=DEFAULT_MODEL,
            on_log=logs.append, on_progress=lambda a, b: seen.append((a, b)),
        )
        took = time.time() - t0
        print(f"  {took:.1f} 秒かかりました。")

        check("単語が返る", bool(words))
        if not words:
            print(f"\n{'FAILED: ' + ', '.join(failures)}")
            return 1

        print("\n  --- 拾えた単語(先頭 20) ---")
        for w in words[:20]:
            print(f"    {w.start:6.2f} → {w.end:6.2f}  {w.text!r}")
        print(f"  --- 全 {len(words)} 語 / 本文: {flatten(words)[:60]}… ---\n")

        check("時刻が前から後ろへ並ぶ",
              all(a.start <= b.start for a, b in zip(words, words[1:])))
        check("開始が終了を追い越さない", all(w.start <= w.end for w in words))
        check("時刻が音声の中に収まる", words[0].start >= 0.0)
        check("単語に空白だけの物が混じらない", all(w.text.strip() for w in words))
        ratio = difflib.SequenceMatcher(
            None, flatten(words), _STRIP.sub("", KNOWN_TEXT), autojunk=False).ratio()
        print(f"  (喋らせた文との一致率 {ratio:.2f})")
        check("喋らせた内容とだいたい一致する", ratio > 0.5)
        check("進捗を知らせる", bool(seen) and seen[-1][0] <= seen[-1][1])

        cache = words_cache_path(tmp / ".work_sample",
                                 audio_fingerprint(wav), DEFAULT_MODEL)
        check("キャッシュが残る", cache is not None and cache.exists())

        logs.clear()
        again = transcribe_words(wav, work_dir=tmp / ".work_sample",
                                 model=DEFAULT_MODEL, on_log=logs.append)
        check("二度目はキャッシュから返す",
              again == words and any("キャッシュ" in m for m in logs))

        # 中断したら途中結果を残さない(半端なキャッシュを次に使わせない)
        cancel_dir = tmp / ".work_cancel"
        stopped = transcribe_words(wav, work_dir=cancel_dir, model=DEFAULT_MODEL,
                                   is_cancelled=lambda: True)
        check("中断すると None", stopped is None)
        check("中断分のキャッシュは残さない",
              not list(cancel_dir.rglob("words.*.json")))

        # anchor.py のテストで使うフィクスチャを作る(モデル無しで回せるように)
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps({
            "note": "SAPI で合成した既知テキストを faster-whisper に通した実物。"
                    "anchor.py のテストが実モデル無しで回せるように置いてある。",
            "model": DEFAULT_MODEL,
            "align_ver": ALIGN_VER,
            "spoken_text": KNOWN_TEXT,
            "words": [w.to_dict() for w in words],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  フィクスチャを書きました: {FIXTURE}")

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
