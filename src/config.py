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
APP_VERSION = "2.2.0"

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

# **画面へ渡すためだけの印。**`_` で始まる項目は save_config が
# ファイルに書かない。設定として保存する値ではない。
UNREADABLE_MARK = "_unreadable"


def load_config() -> dict[str, Any]:
    p = config_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    for k in SECRET_KEYS:
        raw = data.get(k)
        if not isinstance(raw, str):
            continue
        plain = unprotect_secret(raw)
        # **「解けなかった」と「元から空」を分ける。**どちらも "" で
        # 表していたため、解けなかった鍵を**鍵と無関係な保存が空文字で
        # 上書きしていた**（2026-08-31 に再現）。上書き前なら、正しい PC・
        # 正しいアカウントへ戻れば復号できた。**上書き後は永久に失われる。**
        # 起きる場面: 設定を別 PC からコピーした / Windows のプロファイルを
        # 作り直した / バックアップから別機に戻した。どれも普通に起きる。
        if not plain and is_protected(raw):
            data.setdefault(UNREADABLE_MARK, []).append(k)
        data[k] = plain
    return data


def _read_raw() -> dict[str, Any]:
    """ファイルの中身を**包みを解かずに**読む。

    save_config が「渡されなかった鍵」を残すために要る。load_config を
    使うと解けない鍵が "" になって戻るので、残すべき値が取れない。
    """
    p = config_path()
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return d if isinstance(d, dict) else {}


def save_config(data: dict[str, Any]) -> list[str]:
    """**秘密の項目は包んでから書く。**呼び出し側は平文のまま渡してよい。

    **包めなかった項目は書かない。**戻り値はその項目名の並び。

    以前は包めないとき平文のまま書いていた。しかし PRIVACY.md も画面も
    「DPAPI で暗号化する」と断言しており、**成立しないことがある安全性を
    利用者に約束している**状態だった(2026-08-29 に判明)。書かなければ
    約束は破れない。鍵を入れ直す手間は増えるが、黙って平文で置くよりよい。

    秘密でない項目は書く。鍵が包めないからといって名簿や出力先まで
    失わせる理由が無い。

    **戻り値を見るのは呼び出し側の仕事。**画面は「保存できなかった」と
    伝えること——ここで黙ると、名前が変わっただけで同じ穴になる。

    **秘密の項目は「無い」と「空」で意味が違う（2026-08-31）。**

    | 渡された状態 | すること |
    | --- | --- |
    | 平文がある | 包んで書く。包めなければ落として戻り値に載せる |
    | **空文字** | **既存の値をそのまま残す。**渡すものが無いだけで、
      消してよいという意味ではない |
    | 項目が無い | 書かない（＝消す）。画面の「消す」は pop している |

    **空を「消してよい」と読んでいたのが穴だった。**DPAPI で解けない鍵は
    load_config が "" にして返すので、**出力先を変えただけの保存が、
    復旧できたはずの鍵を消していた**（再現済み）。
    """
    # `_` で始まる項目は画面へ渡すための印。ファイルには書かない
    out = {k: v for k, v in data.items() if not str(k).startswith("_")}
    prev = _read_raw()
    dropped: list[str] = []
    for k in SECRET_KEYS:
        if k not in out:
            continue                      # 無い＝消す
        v = out[k]
        if not isinstance(v, str) or not v:
            # 空＝渡すものが無い。**既存の値を残す**
            old = prev.get(k)
            if isinstance(old, str) and old:
                out[k] = old
            else:
                del out[k]
            continue
        wrapped = protect_secret(v)
        if is_protected(wrapped):
            out[k] = wrapped
        else:
            del out[k]
            dropped.append(k)
    p = config_path()
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return dropped
