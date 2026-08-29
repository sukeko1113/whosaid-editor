"""実モデルを使う動作確認(ローカル専用・CI 対象外)。

align.py(時刻の物差し)と local_asr.py(本文を作るローカル転写)の両方を、
同じ合成音声で確かめる。同じ faster-whisper を使うので、実行環境の確認も
音声の合成も 1 回で済む。

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
import os
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
    find_bundled_model,
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
from src import diarize  # noqa: E402
from src.local_asr import LocalTranscriber  # noqa: E402
from src.segments import PSEUDO_UNKNOWN  # noqa: E402

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

    # 手動配置フォルダの差し替えは、モデル名が同じでも中身が別物。
    # フォルダをキーに含めないと、差し替え前の実測を使い回してしまう
    d1 = words_cache_path("w", "abcd1234", "small", model_dir="C:/models/a")
    d2 = words_cache_path("w", "abcd1234", "small", model_dir="C:/models/b")
    check("フォルダが違えば別のキャッシュ", d1 != d2 and d1 != p)
    check("同じフォルダなら同じキャッシュ",
          d1 == words_cache_path("w", "abcd1234", "small",
                                 model_dir="C:/models/a"))

    # HF のリポ ID には「/」が入る。そのままだとキャッシュのファイル名が
    # 下位フォルダに割れて、掃除や走査の前提が崩れる
    hf = words_cache_path("w", "abcd1234", "kotoba-tech/kotoba-whisper-v2.0-faster")
    check("リポ ID でもフォルダが割れない",
          hf is not None and hf.parent.name == "align"
          and "/" not in hf.name and "\\" not in hf.name)

    # **同梱していればそちらを使う。**名前を渡すと faster-whisper が
    # Hugging Face を見に行き、通信を遮断した環境で止まる(設計書 §9)。
    # 同梱を始めた 2026-08-22 に「名前をそのまま返す」ではなくなった。
    bundled = find_bundled_model("small")
    if bundled is not None:
        check("同梱していれば、その置き場を渡す",
              resolve_model("small") == str(bundled))
    else:
        check("同梱が無ければモデル名をそのまま渡す",
              resolve_model("small") == "small")
    check("同梱していないモデルは名前のまま",
          resolve_model("no-such-model-xyz") == "no-such-model-xyz")
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

        # --------------------------------------------------------------
        # ローカル転写(local_asr.py)。align.py と同じエンジンだが、
        # こちらは本文そのものを作る。偽のモデルでは確かめられない
        # ——「実物の segment が text を持っているか」——をここで見る。
        # --------------------------------------------------------------
        print("\n[ローカル転写] 同じ音声を本文として起こします...")
        t0 = time.time()
        local = LocalTranscriber(model=DEFAULT_MODEL)
        result = local.transcribe(wav)
        print(f"  {time.time() - t0:.1f} 秒かかりました。")

        check("区間が返る", result is not None and bool(result.utterances))
        if result and result.utterances:
            for u in result.utterances:
                print(f"    {u.rel_start:6.2f} → {u.rel_end:6.2f}  [{u.cluster}] {u.text}")
            body = _STRIP.sub("", "".join(u.text for u in result.utterances))
            ratio_l = difflib.SequenceMatcher(
                None, body, _STRIP.sub("", KNOWN_TEXT), autojunk=False).ratio()
            print(f"  (喋らせた文との一致率 {ratio_l:.2f})")
            check("喋らせた内容とだいたい一致する", ratio_l > 0.5)
            check("時刻が前から後ろへ並ぶ",
                  all(a.rel_start <= b.rel_start
                      for a, b in zip(result.utterances, result.utterances[1:])))
            check("負の長さの区間が無い",
                  all(u.rel_end >= u.rel_start for u in result.utterances))
            check("本文が空の区間が無い",
                  all(u.text.strip() for u in result.utterances))
            check("話者分離が入るまでは全区間が判別不能",
                  {u.cluster for u in result.utterances} == {PSEUDO_UNKNOWN})
            # 設計の要（§5.3）: 本文と単語時刻が同じ 1 回の転写から取れる。
            # ここが崩れると words キャッシュを兼ねる前提が成り立たない。
            check("単語時刻も同時に取れる", bool(result.words))
            # **align.py と一致する必要はない。**local_asr は句読点のための
            # 例文(initial_prompt)を渡すが align.py は渡さないので、転写が
            # 変わる。§5.3 が言うのは「本文と単語時刻が**同じ 1 回の転写**
            # から取れる」こと——つまり本文と単語が食い違わないこと。
            joined = _STRIP.sub("", flatten(result.words))
            check("単語をつなぐと本文になる(同じ 1 回の転写から取れている)",
                  bool(joined) and difflib.SequenceMatcher(
                      None, joined, body, autojunk=False).ratio() > 0.9)

        # --------------------------------------------------------------
        # 話者分離(diarize.py)。合成音声は 1 人が読み上げているだけなので、
        # 「誰が何人か」は測れない。ここで見るのは配線が通っているか——
        # モデルが見つかり、区間が返り、時刻が音声の中に収まり、キャッシュが
        # 効き、中断できること。品質は 67 分の実会議と正解ラベルで測る
        # (claude_正解ラベルの作り方_設計書.md §7)。
        # --------------------------------------------------------------
        print("\n[話者分離] 同じ音声を話者ごとに分けます...")
        try:
            diarize.ensure_available()
        except diarize.DiarizeUnavailable as e:
            print(f"  飛ばします: {str(e).splitlines()[0]}")
        else:
            t0 = time.time()
            turns = diarize.diarize(
                wav, num_speakers=2, work_dir=tmp / ".work_sample",
                on_log=lambda m: print(f"    {m}"))
            print(f"  {time.time() - t0:.1f} 秒かかりました。")
            check("話者区間が返る", bool(turns))
            if turns:
                check("時刻が前から後ろへ並ぶ",
                      all(a.start <= b.start for a, b in zip(turns, turns[1:])))
                check("開始が終了を追い越さない",
                      all(t.start <= t.end for t in turns))
                dur = float(len(turns) and max(t.end for t in turns))
                check("時刻が音声の中に収まる", dur <= 60.0)
                check("話者番号は 0 以上", all(t.speaker >= 0 for t in turns))

                cache = diarize.turns_cache_path(
                    tmp / ".work_sample", audio_fingerprint(wav), 2)
                check("キャッシュが残る", cache is not None and cache.exists())
                again = diarize.diarize(wav, num_speakers=2,
                                        work_dir=tmp / ".work_sample")
                check("二度目はキャッシュから返す", again == turns)
                # 話者数を変えれば別物として取り直す
                other = diarize.turns_cache_path(
                    tmp / ".work_sample", audio_fingerprint(wav), 3)
                check("話者数が違えば別のキャッシュ",
                      other is not None and not other.exists())

            cancel_dir = tmp / ".work_dcancel"
            stopped = diarize.diarize(wav, num_speakers=2, work_dir=cancel_dir,
                                      is_cancelled=lambda: True)
            check("中断すると None", stopped is None)
            check("中断分のキャッシュは残さない",
                  not list(cancel_dir.rglob("turns.*.json")))

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


def run_diarize_child() -> int:
    """**話者分離の間、画面が動けているか。**

    sherpa-onnx は計算中に GIL を握る。同じプロセスで回すと、画面を描く側が
    まったく動けない。実機で「途中で落ちたようです」と報告を受けた
    （67 分の音声で 11 分間「応答なし」・2026-08-22）。

    子プロセスへ出して直したが、**直ったことを数字で見張らないと元に戻る。**
    主スレッドが 20ms ごとに回れるかを数える（Tk のイベントループと同じ役）。
    """
    import threading
    import time

    failures: list[str] = []

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok  ' if cond else 'NG  '} {label}")
        if not cond:
            failures.append(label)

    print("\n[話者分離の間、画面が動けるか]")
    try:
        diarize.ensure_available()
    except diarize.DiarizeUnavailable as e:
        print(f"  -- 話者分離の部品が無いので飛ばします: {e}")
        return 0

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        wav = tmp / "sample.wav"
        if not synth_wav(wav, KNOWN_TEXT):
            print("  -- 音声を合成できないので飛ばします。")
            return 0

        def measure(inproc: bool):
            """(区間, 主スレッドが回れた割合, 最長の空白ミリ秒)"""
            if inproc:
                os.environ["WHOSAID_DIARIZE_INPROC"] = "1"
            else:
                os.environ.pop("WHOSAID_DIARIZE_INPROC", None)
            ticks: list[float] = []
            stop = threading.Event()
            got: dict = {}

            def work() -> None:
                try:
                    got["turns"] = diarize.diarize(wav, num_speakers=2)
                except Exception as e:                  # noqa: BLE001
                    got["err"] = e
                stop.set()

            t0 = time.perf_counter()
            th = threading.Thread(target=work, daemon=True)
            th.start()
            while not stop.is_set():
                ticks.append(time.perf_counter() - t0)
                time.sleep(0.02)
            th.join()
            el = time.perf_counter() - t0
            gaps = sorted(b - a for a, b in zip(ticks, ticks[1:]))
            ratio = len(ticks) / max(1, int(el / 0.02))
            return got.get("turns"), ratio, (gaps[-1] * 1000 if gaps else 0.0)

        turns_c, ratio_c, gap_c = measure(inproc=False)
        turns_i, ratio_i, gap_i = measure(inproc=True)
        print(f"  子プロセス: 主スレッド {ratio_c * 100:.0f}% / "
              f"最長の空白 {gap_c:.0f} ms")
        print(f"  同居      : 主スレッド {ratio_i * 100:.0f}% / "
              f"最長の空白 {gap_i:.0f} ms")

        check("子プロセスなら主スレッドが 8 割以上動ける", ratio_c >= 0.8)
        # 5 秒で Windows が「応答なし」を付ける。合成音声は数秒しかないので
        # 空白も小さいが、**同居との差が出ていること**が要点。
        check("子プロセスなら 1 秒以上止まらない", gap_c < 1000)
        check("同居より明らかに良い", ratio_c > ratio_i)
        check("結果は同じ",
              turns_c is not None and turns_i is not None
              and [t.to_dict() for t in turns_c] == [t.to_dict()
                                                     for t in turns_i])

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    # 片方が落ちてももう片方を必ず走らせる(短絡すると検査が静かに減る)
    rc_align = run()
    rc_child = run_diarize_child()
    sys.exit(rc_align or rc_child)
