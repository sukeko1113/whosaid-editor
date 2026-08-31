"""faster-whisper を呼び出して、単語ごとの時刻を取る薄いアダプタ。

自動点検(Step 1-3)は「Gemini の転写に付いた時刻」と「実音声の時刻」を突き合わせて
ずれを見つける。その実測側をここが担う。返すのは単語と時刻の並びだけで、
照合の理屈は anchor.py、提案の組み立ては inspect.py にある。

薄くしてあるのは差し替えを想定しているため。精度が足りなければアライナを
入れ替える(WhisperX など)が、そのときもこのモジュールの返り値の形は変えない。
features.extract を差し替え可能にしたのと同じ流儀。

faster_whisper は関数の中で import する。トップレベルに置くと、点検を使わない
利用者や CI でも import した時点で落ちる(transcribe.py が google.genai を
トップレベル import していて素の Python では動かない、あの状態を増やさない)。

音声は外に出さない。ここは完全にローカルで動く。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import lang
from .audio import audio_fingerprint


# 実装のバージョン。上げると words キャッシュを作り直す。
ALIGN_VER = 1

# **CPU での既定は small。**base は速いが精度が落ち、medium は CPU では重い(§9)。
DEFAULT_MODEL = "small"

# **GPU がある機械での既定は large-v3。**
#
# 根拠は**固有名詞**(2026-08-31 に測り直し・逐語正解 4 帯):
#   実際の利用で誤りとして気づいた 5 語(同窓会・文科省・耐震・新潟・建て替え)を
#   small は 2/5、large-v3 は 5/5 拾う。
#   ※ この 5 語は「誤っていた語」として選んだ標本で、固有名詞一般の証拠ではない。
#
# **文字全体の誤り率では差が出ない**(CER 27.3% → 27.8%)。5 語は正解 2,689 字中の
# 26 字(0.97%)しかないので、文字単位の指標では原理的に見えない。
#
# **2026-08-20 の「誤字 11.9% → 7.0%(約 4 割減)」は撤回した。**測り直すと
# 置換率は 7.0% → 8.1% で**向きが逆**だった。回収 11/34 → 17/34 も再現せず
# (1/33 → 4/33)。経緯は設計書 §9.5.1 の注記を見ること。
#
# CPU では large-v3 は実用外(実時間比が数倍)なので、既定を分ける。
GPU_DEFAULT_MODEL = "large-v3"

AVAILABLE_MODELS = ("base", "small", "medium", "large-v3")

# **CPU での既定。**int8 量子化で実用速度になる(torch は要らない)。
# GPU がある機械では pick_device() が cuda / int8_float16 を返す。
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# GPU で回すときの組み合わせ。**float16 ではなく int8_float16。**
# 実測(2026-08-20・GTX 1660 SUPER・逐語正解 4 帯)で、品質は変わらないのに
# 3.5 倍速い。large-v3 で 67 分の音声が 53.7 分 → 15.5 分になった。
# ※ この比較(fp16 と int8_float16)自体は測り直していない。**併記されていた
#    「誤字」の値は撤回した**(定義が復元できない。設計書 §9.5.1 の注記)。
#    なお 2026-08-31 に int8_float16 で測ると、67 分の音声の**転写だけ**に
#    17〜20 分かかった(2 回。うち 1 回は暴走の起こし直しを含む)。
#    **15.5 分は機械と条件で変わる。**同じ日の small は転写 4 分前後で、
#    話者分離まで含めた合計は small 13〜17 分 / large-v3 26〜32 分だった。
GPU_DEVICE = "cuda"
GPU_COMPUTE_TYPE = "int8_float16"

# faster-whisper に渡す言語。**言語で変わる**ので lang.py が持つ。
#
# **参照が 2 経路ある。**ここ(単語時刻の取得)と local_asr.py(本文の転写)。
# 片方だけ切り替えると、本文と物差しの言語が食い違う。どちらも
# lang.current().asr.whisper_language を見るようにしてある。
#
# 下の定数は互換のために残してある(日本語に固定)。**参照しないこと。**
LANGUAGE = lang.JA.asr.whisper_language


def whisper_language() -> str:
    """いま使う言語コード。**import 時ではなく呼ばれた時に決める。**"""
    return lang.current().asr.whisper_language


class AlignUnavailable(RuntimeError):
    """点検の下ごしらえができない(部品が無い・モデルが無い)。

    利用者に何をすれば動くのかを伝えるための例外。呼び出し側は
    そのままメッセージを見せてよい。
    """


def add_cuda_dll_path() -> list[str]:
    """pip で入れた CUDA の DLL を探索先に足す(あれば)。

    `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` は DLL を site-packages の
    中に置くので、**既定の探索先に入っていない。**足さないと
    `cublas64_12.dll is not found` で落ちる(実機で発生・2026-08-20)。

    無い環境では何もしない。同梱ビルドで別の場所に置く場合もここを直す。
    """
    if sys.platform != "win32":
        return []                   # Linux/macOS は既定の探索先で見つかる

    report: list[str] = []
    dirs: list[Path] = []
    # **利用者が取ってきたぶん。**配布物には同梱しない(NVIDIA の条件と
    # 本体の MIT が噛み合わないため。cuda_fetch の冒頭を見よ)。
    try:
        from .cuda_fetch import cuda_dir
        d = cuda_dir()
        if d.is_dir():
            dirs.append(d)
    except Exception:
        pass
    # 開発環境で pip 経由で入れているぶん
    try:
        import nvidia
        dirs.extend(p for root in getattr(nvidia, "__path__", [])
                    for p in Path(root).glob("*/bin") if p.is_dir())
    except ImportError:
        pass

    for d in dirs:
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(d))
        except OSError:
            pass

    # **フルパスで先に読み込む。**PATH も add_dll_directory も、凍結した
    # exe では CTranslate2 の DLL 読み込みに効かなかった(実機で
    # 「cublas64_12.dll is not found」・2026-08-22)。開発環境では動いていた
    # ——シェルの PATH に入っていたためで、仕組みが効いていたのではない。
    #
    # 一度プロセスに読み込ませてしまえば、あとから名前で LoadLibrary された
    # ときに既読のものが使われる。**依存の順に読む**(cublas は cublasLt を
    # 必要とする)。
    import ctypes
    for name in ("cublasLt64_12.dll", "cublas64_12.dll"):
        for d in dirs:
            f = d / name
            if f.is_file():
                try:
                    ctypes.WinDLL(str(f))
                    report.append(f"読み込み成功 {f}")
                except OSError as e:
                    # **黙って落とさない。**読めなかったことが分からないと、
                    # あとで「GPU が使えない」理由に辿り着けない。
                    report.append(f"読み込み失敗 {f} — {e}")
                break
        else:
            report.append(f"見つからない {name}(探した先: "
                          + " / ".join(str(d) for d in dirs) + ")")
    return report


def cuda_available() -> bool:
    """GPU で回せそうか。**これは目安であって保証ではない。**

    `get_cuda_device_count()` はドライバがあれば 1 を返すが、**cuBLAS の
    DLL が無くても 1 を返す**(実機で確認・2026-08-20)。実際に落ちるのは
    最初の推論のときなので、確定は `LocalTranscriber` 側の試し撃ちで行う。
    """
    try:
        import ctranslate2
    except ImportError:
        return False
    try:
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        return GPU_COMPUTE_TYPE in ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        return False


def pick_device(prefer_gpu: bool = True) -> tuple[str, str]:
    """(装置, 精度)を決める。GPU があれば cuda / int8_float16。

    **速いほうが正確でもある**という珍しい形になっている(上の定数の注記)。
    そのため「速度と品質のどちらを採るか」を利用者に選ばせる必要はない。
    """
    if prefer_gpu:
        add_cuda_dll_path()
        if cuda_available():
            return GPU_DEVICE, GPU_COMPUTE_TYPE
    return DEVICE, COMPUTE_TYPE


def default_model(device: Optional[str] = None) -> str:
    """その装置での既定モデル。**手元にあるものだけを既定にする。**

    GPU があれば large-v3 のほうが速くて正確だが（§9.5.1）、**同梱して
    いない**（3GB あり、GitHub Releases の 1 ファイル 2GB に載らない）。
    無いものを既定にすると、初回の転写でいきなり 3GB の取得が始まる。
    **断りも入れずに 3GB を落とすことはしない**ので、取得済みのときだけ
    既定にする。取得を勧めるのは画面側の仕事（一度だけ聞く）。

    装置を省くと `pick_device()` で判定する。**利用者が明示的に選んだ
    モデルは尊重する**(ここは「何も選ばなかったとき」の話)。
    """
    if device is None:
        device, _ = pick_device()
    if device == GPU_DEVICE and find_bundled_model(GPU_DEFAULT_MODEL):
        return GPU_DEFAULT_MODEL
    return DEFAULT_MODEL


def suggest_gpu_model() -> Optional[str]:
    """勧められるのに手元に無いモデル。無ければ None。

    画面が「取得しますか」と聞くために使う。**聞くのは一度だけ**で、
    断られたら二度と聞かない（その記録は設定側が持つ）。
    """
    device, _ = pick_device()
    if device != GPU_DEVICE:
        return None
    if find_bundled_model(GPU_DEFAULT_MODEL):
        return None
    return GPU_DEFAULT_MODEL


@dataclass
class Word:
    """アライナが返す 1 単語。これが anchor.py への唯一の入口。"""

    text: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Word":
        return cls(text=str(d.get("text", "")),
                   start=float(d.get("start", 0.0)),
                   end=float(d.get("end", 0.0)))


def asr_model_dirs(model: str) -> list[Path]:
    """同梱・手動配置のモデルを探す場所を、優先順に返す。

    話者分離(`diarize.model_dirs`)と同じ形にしてある。二通りに実装すると、
    片方だけ直したときに「同梱したのに見つからない」になる。

    **これが無いと「通信を遮断したままでも動きます」が嘘になる。**
    faster-whisper はモデル名を渡すと Hugging Face へ取りに行くので、
    同梱していなければ新規インストール直後は通信が要る(設計書 §9)。
    """
    out: list[Path] = []
    env = os.environ.get("WHOSAID_ASR_MODELS")
    if env:
        out.append(Path(env) / model)
    # **あとから取得したものはここ。**同梱先(_MEIPASS)は一時展開なので
    # 書けないし、実行ファイルの隣は Program Files で管理者権限が要る。
    out.append(downloaded_model_root() / model)
    if getattr(sys, "frozen", False):           # PyInstaller で固めた実行形式
        out.append(Path(getattr(sys, "_MEIPASS", ".")) / "models" / "asr" / model)
        out.append(Path(sys.executable).parent / "models" / "asr" / model)
    else:
        out.append(Path(__file__).resolve().parent.parent / "models" / "asr" / model)
    return out


def downloaded_model_root() -> Path:
    """あとから取得したモデルの置き場（利用者ごと・書き込める場所）。"""
    from .config import config_dir       # 循環を避けてここで読む
    return config_dir() / "models" / "asr"


def find_bundled_model(model: str) -> Optional[Path]:
    """同梱されているモデルのフォルダ。無ければ None。"""
    for d in asr_model_dirs(model):
        if (d / "model.bin").is_file():
            return d
    return None


def resolve_model(model: str = DEFAULT_MODEL,
                  model_dir: Optional[Path | str] = None) -> str:
    """faster-whisper に渡すモデルの指定を決める。

    手動で置いたフォルダが最優先(§9)。ローカル完結が要る現場(研究倫理審査・
    機密案件)ではオンライン取得そのものができないので、「別 PC で取った
    モデルフォルダを置けば動く」経路を必ず残す。

    次に**同梱したモデル**を見る。あればそれを使う——渡すのが名前だと
    faster-whisper が Hugging Face を見に行き、通信を遮断した環境で
    止まってしまう。

    どちらも無ければモデル名を返す。faster-whisper が既定の置き場に
    無ければ取りに行く(初回のみ)。
    """
    if model_dir:
        p = Path(model_dir)
        if not p.is_dir():
            raise AlignUnavailable(
                f"モデルフォルダが見つかりません: {p}\n"
                "別の PC で取得したモデルフォルダを指定するか、指定を外して"
                "自動取得に任せてください。"
            )
        if not (p / "model.bin").exists():
            raise AlignUnavailable(
                f"モデルフォルダの中身が足りません: {p}\n"
                "CTranslate2 形式のフォルダ(model.bin と config.json などが"
                "入ったもの)を指定してください。"
            )
        return str(p)
    bundled = find_bundled_model(model)
    if bundled is not None:
        return str(bundled)
    return model


def model_source(model: str = DEFAULT_MODEL,
                 model_dir: Optional[Path | str] = None) -> str:
    r"""モデルを**どこから読むか**を、画面に出せる短い言葉で返す。

    **絶対パスを出さないため**にある（2026-08-31）。ログの
    「モデルの場所: <絶対パス>」が、旧版から更新した利用者の画面で

        モデルの場所: C:\...\GeminiTranscriber\_internal\models\asr\small

    と出ていた。**「録音も本文も外へ出しません」の数行下に Gemini と出る**のは、
    この製品の主張と正面から衝突する。`installer.iss` の `AppId` は互換のため
    変えられない（変えると 1 台に 2 つ入る）ので、旧版から更新した人の
    インストール先は旧名のまま残る。**ログ側で出さないのが正しい。**

    **アプリの名前も出さない。**「設定フォルダ」で足りる。新名であっても、
    パスを出せば同じ形の問題を繰り返す余地が残る。

    **手動で指定したフォルダだけはパスを出す。**利用者が自分で入れた値なので、
    そのまま返して見せる意味がある（指定を間違えたときに気づける）。

    元の行は 2026-08-22 に**意図して**足された——「同じ実行ファイルなのに
    前は落ちた」を追う手掛かりが何も無かったため。**消さずに置き換える。**
    診断で効くのは絶対パスではなく「同梱か、あとから取得したか」の区別なので、
    手掛かりは残る。
    """
    if model_dir:
        return f"手動で指定したフォルダ: {Path(model_dir)}"
    found = find_bundled_model(model)
    if found is None:
        return "この PC には無いので、faster-whisper が取りに行きます"
    try:
        found.relative_to(downloaded_model_root())
        return "あとから取得したもの（設定フォルダ）"
    except ValueError:
        pass
    env = os.environ.get("WHOSAID_ASR_MODELS")
    if env:
        try:
            found.relative_to(Path(env))
            return f"環境変数 WHOSAID_ASR_MODELS の場所: {found}"
        except ValueError:
            pass
    return "同梱（実行ファイルに入っています）"


def model_tag(model: str, model_dir: Optional[Path | str] = None) -> str:
    """キャッシュのファイル名に入れる「モデルの素性」。

    モデルの素性には model_dir も含める。手動配置のフォルダを差し替えると
    モデル名が同じでも中身は別物で、名前だけをキーにすると差し替え前の
    実測を使い回してしまう。フォルダはパスの短縮ハッシュで区別する
    (パスが同じで中身だけ入れ替えた場合は拾えない。§7 に既知の限界として記載)。

    モデル名の区切り文字(HF のリポ ID に含まれる「/」等)は「-」に無害化する。
    そのままだとキャッシュのファイル名が下位フォルダに割れてしまう。

    ローカル転写のキャッシュ(local_asr.py)も同じ規則を使う。二重に実装すると
    片方だけ直したときに、別物の転写が同じキーを共有する事故になる。
    """
    tag = re.sub(r"[\\/:]+", "-", model)
    if model_dir:
        digest = hashlib.blake2b(
            str(Path(model_dir).resolve()).encode("utf-8"),
            digest_size=4).hexdigest()
        tag += f".d{digest}"
    return tag


def words_cache_path(work_dir: Path | str, fingerprint: str, model: str,
                     model_dir: Optional[Path | str] = None) -> Optional[Path]:
    """words キャッシュの置き場(§7)。指紋が無いときは None。

    キーは「音声の指紋 + モデルの素性 + 実装バージョン」。どれが欠けても
    別物の転写を使い回す事故になるので、指紋が取れなかった音声は
    そもそもキャッシュしない(毎回取り直すほうが安全)。
    """
    if not fingerprint:
        return None
    tag = model_tag(model, model_dir)
    return Path(work_dir) / "align" / f"words.{fingerprint}.{tag}.a{ALIGN_VER}.json"


def load_words(path: Optional[Path]) -> Optional[list[Word]]:
    """キャッシュを読む。無い・壊れている・別バージョンなら None。"""
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("align_ver", -1)) != ALIGN_VER:
            return None
        return [Word.from_dict(w) for w in data.get("words", [])]
    except Exception:
        return None         # 壊れたキャッシュは無かったことにして取り直す


def save_words(path: Optional[Path], words: list[Word], *,
               model: str, fingerprint: str, duration: float) -> None:
    """キャッシュを書く。書けなくても点検自体は続けられるので握りつぶす。"""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "align_ver": ALIGN_VER,
            "model": model,
            "fingerprint": fingerprint,
            "duration": duration,
            "words": [w.to_dict() for w in words],
        }, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def transcribe_words(
    audio_path: Path | str,
    *,
    work_dir: Optional[Path | str] = None,
    model: str = DEFAULT_MODEL,
    model_dir: Optional[Path | str] = None,
    fingerprint: Optional[str] = None,
    on_log=None,
    on_progress=None,
    is_cancelled=None,
    force: bool = False,
) -> Optional[list[Word]]:
    """音声を逐語で起こし、単語ごとの時刻を返す。

    重いのはここだけなので結果はキャッシュする(§7)。照合と提案づくりは
    軽いので毎回やり直す。redistribute_times を毎回計算し直すのと同じ考え方。

    work_dir: `.work_<音声名>` のパス。省略するとキャッシュしない。
    on_progress: (処理済み秒, 全体秒) で呼ぶ。
    is_cancelled: True を返したら中断し None を返す(途中結果は残さない)。
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise AlignUnavailable(f"音声ファイルが見つかりません: {audio_path}")

    if fingerprint is None:
        fingerprint = audio_fingerprint(audio_path)
    cache = (words_cache_path(work_dir, fingerprint, model, model_dir)
             if work_dir else None)

    if not force:
        cached = load_words(cache)
        if cached is not None:
            if on_log:
                on_log(f"実測の単語時刻をキャッシュから読みました({len(cached)} 語)。")
            return cached

    target = resolve_model(model, model_dir)
    try:
        from faster_whisper import WhisperModel      # 重いのでここで読む
    except ImportError as e:
        raise AlignUnavailable(
            "点検には faster-whisper が必要です。\n"
            "    pip install faster-whisper\n"
            "で導入してから、もう一度お試しください。"
        ) from e

    if on_log:
        on_log(f"実測用の転写を始めます(モデル {model} / CPU)。"
               "初回はモデルの取得に時間がかかります。")
    try:
        whisper = WhisperModel(target, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        raise AlignUnavailable(
            f"モデルを読み込めませんでした({model})。\n"
            "オンラインで取得できない環境では、別の PC で取得したモデル"
            "フォルダを指定してください。\n"
            f"--- 詳細 ---\n{e}"
        ) from e

    segments, info = whisper.transcribe(
        str(audio_path),
        language=whisper_language(),
        word_timestamps=True,       # これが目的。単語ごとの start/end が付く
        # VAD は使わない。無音を飛ばすと速くなるが、短い相づちを落とす
        # 恐れがある。落ちた区間は「照合不能」になって提案が出せなくなる。
        vad_filter=False,
    )
    total = float(getattr(info, "duration", 0.0) or 0.0)

    words: list[Word] = []
    for seg in segments:            # ここで初めて実際の転写が走る(遅延評価)
        if is_cancelled and is_cancelled():
            if on_log:
                on_log("実測用の転写を中止しました。")
            return None
        for w in (getattr(seg, "words", None) or []):
            text = (w.word or "").strip()
            if not text:
                continue
            words.append(Word(text=text, start=float(w.start), end=float(w.end)))
        if on_progress and total:
            on_progress(min(float(seg.end), total), total)

    if on_progress and total:
        on_progress(total, total)
    if on_log:
        on_log(f"実測の単語時刻を {len(words)} 語ぶん取りました。")
    save_words(cache, words, model=model, fingerprint=fingerprint, duration=total)
    return words
