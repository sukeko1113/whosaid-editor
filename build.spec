# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec - WhosaidEditor

ビルド前提:
- ffmpeg/ffmpeg.exe を配置済み(Windows)
- ffmpeg/ffplay.exe も配置しておくと、話者割当画面の区間再生が高速になる
  (無い場合は winsound による簡易再生にフォールバックする)
- resources/icon.ico があれば自動でアイコン適用(なくてもOK)
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH)

# ローカル転写(faster-whisper)の同梱物。
#   - ctranslate2 / av は DLL を持つ。静的解析では拾われないので明示的に集める
#   - faster_whisper は assets/ に VAD の onnx を同梱している。いまは
#     vad_filter=False で使っていないが、入れておかないと有効にした瞬間に落ちる
whisper_binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("av")
whisper_datas = collect_data_files("faster_whisper")

# 話者分離(sherpa-onnx)の同梱物。onnxruntime の DLL を持つので明示的に集める。
# モデルは models/diarize に置いたものをそのまま入れる。実行時は
# diarize.py の model_dirs() が _MEIPASS/models/diarize を見る。
diarize_binaries = collect_dynamic_libs("sherpa_onnx")

# **モデルが無ければここで止める。**ffmpeg と同じ扱い。無いまま配ると、
# 利用者の端末で初めて「話者分離のモデルが見つかりません」と出る。
# models/ は .gitignore で除外してあるので、取得を忘れやすい。
# **名前は diarize.py の SEG_NAME / EMB_NAME から取る。**ここに直書きすると、
# 片方だけ直したときに「同梱したのに見つからない」になる。
# (設計書 §9 は int8 1.5MB と書いているが、実装が読むのは model.onnx 5.7MB。
#  同梱するのは実装が読むほう。§9 の記述は 9.1 に訂正を残した)
sys.path.insert(0, str(ROOT))
from src.diarize import EMB_NAME, SEG_NAME       # noqa: E402

diarize_datas = []
for rel in (SEG_NAME, EMB_NAME):
    src = ROOT / "models" / "diarize" / rel
    if not src.exists():
        raise SystemExit(
            f"話者分離のモデルが見つかりません: {src}\n"
            "README のモデル取得手順を参照してください。"
            "これが無いとローカル経路は全区間が『?』になります。"
        )
    # 実行時は _MEIPASS/models/diarize/<rel> を見る(diarize.py:89)ので、
    # 相対の階層をそのまま保つ
    diarize_datas.append((str(src), str(Path("models/diarize") / rel).replace("\\", "/").rsplit("/", 1)[0]))

# 転写のモデルの同梱。**これが無いと「通信を遮断したままでも動きます」が
# 嘘になる**——faster-whisper はモデル名を渡すと Hugging Face へ取りに行く
# ので、同梱していなければ新規インストール直後に通信が要る(設計書 §9)。
# 話者分離と同じく、無ければビルドを止める。
#
# **同梱するのは CPU 既定の small だけ。**GPU 向けの large-v3 は 3GB あり、
# GitHub Releases の 1 ファイル 2GB 制限に載らない。GPU を持つ利用者だけが
# 初回に取得する(設計書 §9.5.8 の案イ)。
from src.align import DEFAULT_MODEL                      # noqa: E402

asr_datas = []
_asr_src = ROOT / "models" / "asr" / DEFAULT_MODEL
if not (_asr_src / "model.bin").is_file():
    raise SystemExit(
        f"転写のモデルが見つかりません: {_asr_src}\n"
        f"    python tools\\fetch_asr_model.py {DEFAULT_MODEL}\n"
        "で取得してください。これが無いと、利用者の端末は初回に通信が要ります。"
    )
for _f in sorted(_asr_src.iterdir()):
    if _f.is_file():
        # 実行時は _MEIPASS/models/asr/<名前>/ を見る(align.asr_model_dirs)
        asr_datas.append((str(_f), f"models/asr/{DEFAULT_MODEL}"))

# ffmpeg / ffplay の同梱(Windows のみ)
binaries = []
for name in ("ffmpeg.exe", "ffplay.exe"):
    exe = ROOT / "ffmpeg" / name
    if exe.exists():
        binaries.append((str(exe), "."))
    elif name == "ffmpeg.exe":
        raise SystemExit("ffmpeg/ffmpeg.exe が見つかりません。README のビルド手順を参照してください。")
    else:
        print(f"[warn] ffmpeg/{name} が見つかりません。区間再生は簡易モードになります。")

# ライセンス表記。**TitaNet が CC-BY-4.0 なので、表示は配布の条件である。**
# 任意ではないので、無ければ止める。
credits = ROOT / "resources" / "CREDITS.txt"
if not credits.exists():
    raise SystemExit(
        f"ライセンス表記が見つかりません: {credits}\n"
        "同梱している TitaNet は CC-BY-4.0 で、表示が配布の条件です。"
    )
credits_datas = [(str(credits), ".")]

# アイコン(任意)
icon_path = ROOT / "resources" / "icon.ico"
icon_arg = str(icon_path) if icon_path.exists() else None


a = Analysis(
    [str(ROOT / "src" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries + whisper_binaries + diarize_binaries,
    datas=whisper_datas + diarize_datas + asr_datas + credits_datas,
    hiddenimports=[
        "google.genai",
        "google.auth",
        "docx",
        # align.py / local_asr.py は関数の中で import している(素の Python でも
        # 起動できるようにするため)。静的解析に頼らず明示する。
        "faster_whisper",
        "ctranslate2",
        "av",
        "tokenizers",
        # diarize.py も関数の中で import する
        "sherpa_onnx",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhosaidEditor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,        # GUI アプリ
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_arg,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WhosaidEditor",
)
