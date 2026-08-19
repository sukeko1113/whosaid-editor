"""ユーザ設定の保存・読み込み (%APPDATA%\\GeminiTranscriber\\config.json)"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


APP_NAME = "GeminiTranscriber"

# アプリの版。installer.iss の MyAppVersion と揃えること(現状は手動同期。
# Day 60 のインストーラ作業で一元化を検討)。Word の検証要約に併記される。
APP_VERSION = "2.1.0"

# ライセンス表記のファイル名。**同梱している TitaNet は CC-BY-4.0 で、
# 表示が配布の条件**(claude/claude_話者分離_設計書.md §9)。
CREDITS_NAME = "CREDITS.txt"


def credits_path() -> Path:
    """ライセンス表記の置き場。凍結後は実行ファイルの隣に展開される。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / CREDITS_NAME
    return Path(__file__).resolve().parent.parent / "resources" / CREDITS_NAME


def credits_text() -> str:
    """表記の中身。**読めなくても落とさない**——表記が出ないほうが問題だが、
    それで転写そのものが止まるのは筋が違う。どこを見たかは残す。"""
    p = credits_path()
    try:
        return p.read_text(encoding="utf-8")
    except OSError as e:
        return (f"ライセンス表記を読めませんでした: {p}\n{e}\n\n"
                "同梱物には CC BY 4.0 の NVIDIA NeMo TitaNet-Small と、"
                "MIT の pyannote/segmentation-3.0 が含まれます。")


def config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    d = base / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> dict[str, Any]:
    p = config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(data: dict[str, Any]) -> None:
    p = config_path()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
