# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec - GeminiTranscriber

ビルド前提:
- ffmpeg/ffmpeg.exe を配置済み(Windows)
- ffmpeg/ffplay.exe も配置しておくと、話者割当画面の区間再生が高速になる
  (無い場合は winsound による簡易再生にフォールバックする)
- resources/icon.ico があれば自動でアイコン適用(なくてもOK)
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH)

# ローカル転写(faster-whisper)の同梱物。
#   - ctranslate2 / av は DLL を持つ。静的解析では拾われないので明示的に集める
#   - faster_whisper は assets/ に VAD の onnx を同梱している。いまは
#     vad_filter=False で使っていないが、入れておかないと有効にした瞬間に落ちる
whisper_binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("av")
whisper_datas = collect_data_files("faster_whisper")

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

# アイコン(任意)
icon_path = ROOT / "resources" / "icon.ico"
icon_arg = str(icon_path) if icon_path.exists() else None


a = Analysis(
    [str(ROOT / "src" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries + whisper_binaries,
    datas=whisper_datas,
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
    name="GeminiTranscriber",
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
    name="GeminiTranscriber",
)
