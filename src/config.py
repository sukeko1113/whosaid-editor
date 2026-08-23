"""ユーザ設定の保存・読み込み (%APPDATA%\\WhosaidEditor\\config.json)"""
from __future__ import annotations

import base64
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional


APP_NAME = "WhosaidEditor"

# **旧名。設定を引き継ぐためだけに残す。**2026-08-21 に改名した
# （「Gemini 文字起こし」→「Whosaid 反訳エディタ」）。既定はローカルなので
# Gemini の名を冠するのは事実と違い、他社の商標でもある。
# **消さないこと**——消すと旧版からの利用者が API キーも名簿も失う。
LEGACY_APP_NAMES = ("GeminiTranscriber",)

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
    _migrate_legacy(base, d)
    return d


def _migrate_legacy(base: Path, dest: Path) -> None:
    """旧名のフォルダから設定を引き継ぐ（1 回だけ）。

    改名で `%APPDATA%` の場所が変わるため、そのままだと旧版の利用者が
    API キーも名簿も失う。**新しい側に config.json が無いときだけ**持ってくる。

    **旧フォルダは消さない。**旧版に戻したい人が残っているかもしれないし、
    消す理由もない（数 KB）。引き継いだことは印を置いて残す。
    """
    if (dest / "config.json").exists():
        return
    for old in LEGACY_APP_NAMES:
        src = base / old / "config.json"
        if not src.is_file():
            continue
        try:
            shutil.copy2(src, dest / "config.json")
            (dest / "migrated-from.txt").write_text(
                f"{old} から設定を引き継ぎました。\n"
                f"元: {src}\n"
                "旧フォルダはそのまま残してあります。\n",
                encoding="utf-8")
        except OSError:
            pass        # 引き継げなくても起動は止めない（設定が空になるだけ）
        return


def config_path() -> Path:
    return config_dir() / "config.json"


# ======================================================================
# API キーの保存
#
# **平文で置かない。**設定ファイルは %APPDATA% にあり、そのままだと
# 中を開けば読める。Windows の DPAPI で包むと、**そのパソコンの、その
# 利用者アカウントでしか復号できない**形になる(鍵の管理は OS に任せる)。
#
# 完全な防御ではない。同じアカウントで動く別のプログラムからは復号できる。
# それでも、設定ファイルを取り出して持ち去られたときに読めないことには
# 意味がある。**共有パソコンでは「保存しない」を選べるようにしてある。**
# ======================================================================

# 包んだ値の頭に付ける印。これが無ければ平文(旧版の設定)とみなす。
_ENC_PREFIX = "dpapi:"


def _dpapi(data: bytes, unprotect: bool) -> Optional[bytes]:
    """DPAPI で包む/ほどく。使えない環境では None(呼び出し側が平文に落ちる)。"""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD),
                    ("pbData", ctypes.POINTER(ctypes.c_char))]

    src = BLOB(len(data), ctypes.cast(ctypes.create_string_buffer(data),
                                      ctypes.POINTER(ctypes.c_char)))
    out = BLOB()
    fn = (ctypes.windll.crypt32.CryptUnprotectData if unprotect
          else ctypes.windll.crypt32.CryptProtectData)
    args = ([ctypes.byref(src), None, None, None, None, 0, ctypes.byref(out)]
            if not unprotect else
            [ctypes.byref(src), None, None, None, None, 0, ctypes.byref(out)])
    try:
        if not fn(*args):
            return None
        got = ctypes.string_at(out.pbData, out.cbData)
        ctypes.windll.kernel32.LocalFree(out.pbData)
        return got
    except Exception:
        return None


def protect_secret(value: str) -> str:
    """保存する形にする。包めなければ平文のまま返す(保存自体は成立させる)。"""
    if not value or value.startswith(_ENC_PREFIX):
        return value
    got = _dpapi(value.encode("utf-8"), unprotect=False)
    if got is None:
        return value
    return _ENC_PREFIX + base64.b64encode(got).decode("ascii")


def unprotect_secret(value: str) -> str:
    """読める形に戻す。**平文で入っていたらそのまま返す**(旧版の設定)。"""
    if not value or not value.startswith(_ENC_PREFIX):
        return value
    try:
        raw = base64.b64decode(value[len(_ENC_PREFIX):])
    except Exception:
        return ""
    got = _dpapi(raw, unprotect=True)
    return got.decode("utf-8", "replace") if got is not None else ""


def is_protected(value: str) -> bool:
    """包まれているか(画面の表示に使う)。"""
    return bool(value) and value.startswith(_ENC_PREFIX)


# 包んで保存する項目。増やすときはここに足す。
SECRET_KEYS = ("api_key",)


def load_config() -> dict[str, Any]:
    p = config_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    for k in SECRET_KEYS:
        if isinstance(data.get(k), str):
            data[k] = unprotect_secret(data[k])
    return data


def save_config(data: dict[str, Any]) -> None:
    """**秘密の項目は包んでから書く。**呼び出し側は平文のまま渡してよい。"""
    out = dict(data)
    for k in SECRET_KEYS:
        if isinstance(out.get(k), str):
            out[k] = protect_secret(out[k])
    p = config_path()
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
