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
