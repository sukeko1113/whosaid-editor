"""配布版と同じ条件で、GPU の部品が読めるかだけを確かめる。

**開発環境で試しても意味がない。**`.venv` には `nvidia-cublas-cu12` が
入っており、シェルの PATH にも CUDA が通っていることがある。そのせいで
「直った」と 2 度言って 2 度とも外した(2026-08-22)。frozen で確かめる。

    python -m PyInstaller --noconfirm --clean --console --name gpucheck \
      --collect-all ctranslate2 --collect-all faster_whisper \
      --collect-all tokenizers --collect-all sherpa_onnx \
      --collect-all onnxruntime --collect-all av \
      --paths . --hidden-import src.gui --hidden-import src.local_asr \
      --distpath dist-check --workpath build-check --specpath build-check \
      tools/gpucheck.py
    dist-check/gpucheck/gpucheck.exe
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "--diarize" in sys.argv[1:]:
    from src.diarize import child_main
    sys.exit(child_main(sys.argv[sys.argv.index("--diarize") + 1:]))

print("frozen           :", getattr(sys, "frozen", False))
print("_MEIPASS         :", getattr(sys, "_MEIPASS", "(なし)"))
print("APPDATA          :", os.environ.get("APPDATA"))

# **アプリと同じ順で読み込む。**先に何かが CUDA を掴んでいると、
# あとから読ませても効かないことがある。
try:
    import src.gui                                   # noqa: F401
    print("src.gui 読み込み OK")
except Exception as e:
    print("src.gui 読み込み NG:", e)

try:
    from src import align, cuda_fetch, local_asr
    from src.config import config_dir
    print("config_dir()     :", config_dir())
    print("cuda_dir()       :", cuda_fetch.cuda_dir(),
          "存在:", cuda_fetch.cuda_dir().is_dir())
    print("部品はそろっているか:", cuda_fetch.is_available())
    print("large-v3 の場所  :", align.find_bundled_model("large-v3"))
    print("--- add_cuda_dll_path() ---")
    for line in align.add_cuda_dll_path():
        print("   ", line)
    print("cuda_available() :", align.cuda_available())
    print("pick_device()    :", align.pick_device())
    print("--- アプリと同じ入口(LocalTranscriber)---")
    t = local_asr.LocalTranscriber(model="large-v3", model_dir=None)
    print("  model  :", t.model)
    print("  device :", t.device, t.compute_type)
    print("  best   :", align.default_model(t.device))
    t._load(on_log=lambda m: print("  LOG:", m))
    print("  loaded :", t.loaded, "/", t.device, t.compute_type, t.model)
except Exception:
    traceback.print_exc()


# ======================================================================
# 話者分離が「親→子」で回るか（固めた状態で確かめる）
#
# **開発環境で動いても意味がない。**固めた実行ファイルでは自分自身を
# 呼び直す形になり、`__file__` も `sys.executable` も別物になる。
# この確認そのものが子役も兼ねる（--diarize を受ける）ので、
# 本体アプリと同じ経路を通る。
# ======================================================================
def _check_diarize():
    import threading
    import time
    from src import diarize

    wav = os.path.join(os.environ.get("TEMP", "."), "child60.wav")
    if not os.path.exists(wav):
        print("確認用の音声が無いので飛ばします:", wav)
        return

    def measure(inproc):
        if inproc:
            os.environ["WHOSAID_DIARIZE_INPROC"] = "1"
        else:
            os.environ.pop("WHOSAID_DIARIZE_INPROC", None)
        ticks, stop, got = [], threading.Event(), {}

        def work():
            try:
                got["t"] = diarize.diarize(wav, num_speakers=3)
            except Exception as e:
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
        return (got.get("t"), got.get("err"), el,
                len(ticks) / max(1, int(el / 0.02)),
                (gaps[-1] * 1000 if gaps else 0.0))

    for label, inproc in (("子プロセス", False), ("同居", True)):
        t, err, el, ratio, gap = measure(inproc)
        print(f"  [{label}] {el:.1f} 秒 / 主スレッド {ratio*100:.0f}% / "
              f"最長の空白 {gap:.0f} ms / 区間 {len(t or [])}"
              + (f" / エラー {err}" if err else ""))


if "--diarize-check" in sys.argv:
    print("--- 話者分離（親→子）---")
    _check_diarize()
