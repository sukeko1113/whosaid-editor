# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec - GeminiTranscriber

ビルド前提:
- ffmpeg/ffmpeg.exe を配置済み(Windows)
- ffmpeg/ffplay.exe も配置しておくと、話者割当画面の区間再生が高速になる
  (無い場合は winsound による簡易再生にフォールバックする)
- resources/icon.ico があれば自動でアイコン適用(なくてもOK)
"""
from pathlib import Path

ROOT = Path(SPECPATH)

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
    binaries=binaries,
    datas=[],
    hiddenimports=[
        "google.genai",
        "google.auth",
        "docx",
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
