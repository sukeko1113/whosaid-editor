"""ユーザ設定の保存・読み込み (%APPDATA%\\WhosaidEditor\\config.json)"""
from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


APP_NAME = "WhosaidEditor"

# **旧名。設定を引き継ぐためだけに残す。**2026-08-21 に改名した
# （「Gemini 文字起こし」→「Whosaid 反訳エディタ」）。既定はローカルなので
# Gemini の名を冠するのは事実と違い、他社の商標でもある。
# **消さないこと**——消すと旧版からの利用者が API キーも名簿も失う。
LEGACY_APP_NAMES = ("GeminiTranscriber",)

# **引き継ぎのときに包めず、落とした鍵の名前。**設定ファイルに書く
# （`_` 始まりではない＝次の起動でも読める）。画面はこれを見て
# 「旧版の鍵は引き継げませんでした。入れ直してください」と知らせ、
# 鍵が保存できたら消す。黙って落とすと「引き継いだのに鍵が無い」に
# 見える——落としたことを、落とした側が言う。
MIGRATE_DROPPED = "migrate_dropped"

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

    **複製ではなく、読んで鍵を包んでから書く（2026-09-03）。**旧版の設定は
    鍵が平文で入っている。ファイルごと複製すると、新しい側にも平文が
    入り、次に保存が走るまでそのまま残る。
    包めない機械では平文を写さず、落として `MIGRATE_DROPPED` に名前を残す
    ——保存時と同じ約束（平文では書かない）を引き継ぎでも守る。

    以前は shutil.copy2 で複製していたため、**更新時刻まで旧ファイルの
    ものになっていた。**更新時刻を根拠に「いつ書かれたか」を追うときに
    惑わせる（2026-08-31 の調査で引っかかった）。いまは書いた時刻になる。

    **旧フォルダの平文の鍵はそのまま残る。**ここでは触らない。消すかどうかは
    人が決める（画面が知らせる）。
    """
    if (dest / "config.json").exists():
        return
    for old in LEGACY_APP_NAMES:
        src = base / old / "config.json"
        if not src.is_file():
            continue
        try:
            data = json.loads(src.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return          # 読めないものは引き継げない（起動は止めない）
        if not isinstance(data, dict):
            return
        dropped: list[str] = []
        for k in SECRET_KEYS:
            v = data.get(k)
            if not isinstance(v, str) or not v:
                continue
            wrapped = protect_secret(v)
            if is_protected(wrapped):
                data[k] = wrapped
            else:
                del data[k]
                dropped.append(k)
        if dropped:
            data[MIGRATE_DROPPED] = dropped
        try:
            (dest / "config.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            note = (f"{old} から設定を引き継ぎました。\n"
                    f"元: {src}\n"
                    "旧フォルダはそのまま残してあります"
                    "（旧版の API キーは、そちらでは平文のままです）。\n")
            if dropped:
                note += ("API キーは、この機械では暗号化できないため"
                         "引き継いでいません。入れ直してください。\n")
            (dest / "migrated-from.txt").write_text(note, encoding="utf-8")
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


def key_protected_on_disk(key: str = "api_key") -> bool:
    """**いま設定ファイルに書かれている鍵が、包まれた形か。**ファイルを読み直す。

    画面が「平文の鍵」の注記を下ろす判定に使う（2026-09-03）。保存処理が
    例外を出さなかったことを根拠に下ろすと、**書いたつもりの場所と実際に
    書かれた場所が食い違う環境では、反映されていないのに注記が消える。**
    （同期ソフト・仮想化・環境変数の差し替えで起きる。同じパスを指して
    いるつもりで別のファイルを見ていた例が、開発中に実際にあった。）
    注記が消えないこと自体を検出に使うので、実物のファイルで確かめる。
    """
    raw = _read_raw().get(key)
    return isinstance(raw, str) and is_protected(raw)


# ======================================================================
# 設定の置き場と書き込みの診断（2026-09-03）
#
# **書いたつもりの場所と、実際に書かれた場所は食い違うことがある。**
# 環境変数 %APPDATA% の差し替え（測定用の窓で実際に起きた）、同期ソフト、
# 仮想化。開発中には、同じパスを指しているつもりで、人と道具が別の
# ファイルを見ていた例もあった（2026-09-03。気づいたのは人が更新時刻を
# 打って比べたから）。起動時に「標準か否か」と更新時刻を出し、保存の
# たびに読み返して一致を確かめる。読み返しは、環境変数が差し替わって
# いれば書き込みと同じ誤った場所を指して一致してしまうので、環境変数に
# 頼らない標準の場所の更新時刻を別に出す。
#
# **絶対パスは出さない**（既存方針）。「標準か否か」と時刻だけ。
# ======================================================================

def config_mtime() -> Optional[float]:
    """いま使う設定ファイルの更新時刻。無ければ None。"""
    try:
        return config_path().stat().st_mtime
    except OSError:
        return None


def standard_config_path() -> Optional[Path]:
    """**環境変数に頼らない**標準の置き場。Windows 以外は None。"""
    if sys.platform != "win32":
        return None
    return Path.home() / "AppData" / "Roaming" / APP_NAME / "config.json"


def _stamp(ts: Optional[float]) -> str:
    if ts is None:
        return "無し"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def is_standard_location() -> bool:
    """いま使う設定ファイルが標準の場所か。Windows 以外は常に True。"""
    std = standard_config_path()
    if std is None:
        return True
    return (os.path.normcase(str(config_path().resolve()))
            == os.path.normcase(str(std.resolve())))


def config_location_report() -> str:
    """起動時のログに出す 1 行。**絶対パスは出さない。**

    標準なら更新時刻だけ。標準でなければ、その回の設定が別の場所に
    読み書きされることと、標準の場所のファイルの更新時刻を出す。
    「更新時刻が動いたか」を人が見て確かめられるようにする。
    """
    when = _stamp(config_mtime())
    if is_standard_location():
        return f"設定の置き場: 標準（設定ファイルの更新: {when}）"
    std = standard_config_path()
    try:
        std_when = _stamp(std.stat().st_mtime) if std is not None else "無し"
    except OSError:
        std_when = "無し"
    return ("※ 設定の置き場が標準ではありません（環境変数 APPDATA が"
            "差し替わっています）。この回の設定は別の場所に読み書きされ、"
            "標準の場所の設定は変わりません。"
            f"いま使う設定の更新: {when} / 標準の場所の更新: {std_when}")


def verify_written(data: dict[str, Any], before_mtime: Optional[float]) -> list[str]:
    """**書いた直後に読み返して確かめる。**問題があれば、その文の並びを返す。

    - 更新時刻が動いていない（書いたはずなのに）
    - 読み返した内容が、書いたものと一致しない

    秘密の項目は包んで書くので比べない（key_protected_on_disk が別に見る）。
    `_` 始まりの印は書かないので比べない。**絶対パスは出さない。**
    """
    problems: list[str] = []
    now = config_mtime()
    if now is None:
        problems.append("設定ファイルが、書いたはずの場所にありません")
        return problems
    if before_mtime is not None and now <= before_mtime:
        problems.append("設定ファイルの更新時刻が動いていません")
    raw = _read_raw()
    for k, v in data.items():
        if str(k).startswith("_") or k in SECRET_KEYS:
            continue
        if json.loads(json.dumps(v)) != raw.get(k):
            problems.append(f"読み返した内容が、書いたものと一致しません（{k}）")
            break
    return problems


# 包んで保存する項目。増やすときはここに足す。
SECRET_KEYS = ("api_key",)

# **画面へ渡すためだけの印。**`_` で始まる項目は save_config が
# ファイルに書かない。設定として保存する値ではない。
UNREADABLE_MARK = "_unreadable"

# **包まれずに置かれている鍵の印。**旧版（v2.0.x 以前）が書いた設定や、
# 旧名フォルダから引き継いだ設定には、鍵が平文のまま入っている
# （2026-09-03 に、旧版の設定を写した端末で確認）。読み込みは旧版互換で
# そのまま通すので、**画面は暗号文と同じ顔で出ていた。**
#
# 印を付けるだけで、**値は書き換えない。**黙って包み直すと「いつから
# 平文だったか」を利用者が知れない。平文でディスクに置かれていた期間が
# あり、複製もされうるので、包み直しても「安全になった」ことにはならない
# ——知らせて、取り直しを勧めるところまでが仕事（解けない鍵を黙って
# 消さないのと同じ考え方）。
PLAINTEXT_MARK = "_plaintext"


def load_config() -> dict[str, Any]:
    p = config_path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    # **印はファイルから拾わない。**`_` 始まりの印は save_config が書かない
    # 約束だが、万一ファイルに紛れていると、下の setdefault(...).append が
    # 既存のリストに追記して、起動のたびに要素が増える。読むたびに
    # 実物から立て直す——約束に頼らず、ここで構造として保証する。
    data = {k: v for k, v in data.items() if not str(k).startswith("_")}
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
        # **包まれていない鍵は、解けた鍵と区別する。**旧版互換で読める
        # のは変えないが、平文で置かれている事実は画面に届ける。
        # 毎回の起動で実物のファイルを見て立てる。書いた先と読む先が
        # 食い違う環境では印が消えない——それ自体が食い違いの検出。
        elif plain and not is_protected(raw):
            data.setdefault(PLAINTEXT_MARK, []).append(k)
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
    | **空文字** | **既存の値を残す。**渡すものが無いだけで、
      消してよいという意味ではない。既存が平文なら包んでから残す |
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
                # **残すついでに包む。**既存が平文（旧版が書いたもの）なら、
                # そのまま写すと「保存を通ったのに平文のまま」になる
                # （2026-09-03 に一時フォルダで再現）。包まれた値や解けない
                # 暗号文は protect_secret が素通しするので、そのまま残る。
                # 包めなければ書かない——保存時と引き継ぎ時で約束を変えない。
                wrapped = protect_secret(old)
                if is_protected(wrapped):
                    out[k] = wrapped
                else:
                    del out[k]
                    dropped.append(k)
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
