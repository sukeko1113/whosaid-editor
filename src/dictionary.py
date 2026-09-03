"""固有名詞の聞き違いを蓄える辞書（設計書 §3）。

**辞書は「検索語のリスト」であって、適用エンジンではない。**辞書がやるのは
○×画面（assign_gui.ReplaceWordsDialog）に「この語を探せ」を渡すところまで。
どこを直すかは人が前後を読み、音声を聴いて決める。**自動適用はしない**——
「田中」→「田仲」が正しいのは特定の人物についてだけで、別の田中さんには誤り。

置き場はユーザーの設定ディレクトリ（プロジェクト単位ではなくグローバル）。
固有名詞は同じ組織の会議で繰り返すので、グローバルに持つほうが効く。
プロジェクト単位の無効化は、作業ファイルのスキーマを変えずに、**この
ファイル側に音声指紋で持つ**（`disabled_for`。設計書 §6 の 2）。

**個人情報。**辞書には人名が入る。このモジュールは中身をログに出さない。
呼び出し側も、辞書の項目をログ・不具合報告・Word の出力に混ぜないこと。

**効果は件数で測る**（§6 の 2）。項目ごとに「適用した数」「却下した数」を
持ち、適用 ÷（適用 ＋ 却下）が精度になる。作業ファイル側の記録
（edit_log の `origin: dictionary`）が一次で、ここは集計の写し。
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import config_dir

DICTIONARY_NAME = "dictionary.json"
FORMAT_VERSION = 1


class DictionaryUnwritable(RuntimeError):
    """辞書ファイルを上書きできない（読めなかった／一部が読めなかった）。

    **戻り値ではなく例外にしてある。**戻り値は呼び出し側が見なければ黙って
    消える（設定の保存で 9 か所中 7 か所が `except OSError: pass` だった件と
    同じ形）。例外は、握りつぶすコードを書かない限り画面まで上がる。
    呼び出し側（画面）はこれを捕まえて、**足したはずの項目が保存されていない**
    ことを人に伝えること。
    """

ORIGIN_MANUAL = "manual"                # 手入力
ORIGIN_REPLACE = "replace_history"      # 置換から登録


def dictionary_path() -> Path:
    """`%APPDATA%\\WhosaidEditor\\dictionary.json`（設定ファイルと同じフォルダ）。"""
    return config_dir() / DICTIONARY_NAME


@dataclass
class Entry:
    """辞書の 1 項目。**判定は持たない**——「この語は直す」は人が毎回決める。"""

    wrong: str                      # 誤変換の文字列（探す語）
    correct: str                    # 正しい文字列（直した後）
    id: str = ""                    # 項目の鍵。並べ替え・削除で変わらない
    enabled: bool = True
    added_at: str = ""
    origin: str = ORIGIN_MANUAL
    note: str = ""                  # 「○○課の田仲さん」など、同名別人の区別用
    # 探すときの条件。**直すときも同じ値を使う**（設計書 §2.3）
    ignore_case: bool = False
    whole_word: bool = False
    # 効果の集計（一次は作業ファイルの edit_log）
    applied: int = 0
    rejected: int = 0

    @property
    def options(self) -> dict[str, bool]:
        """find_text / replace_text に渡す条件（既定のときは空）。"""
        out: dict[str, bool] = {}
        if self.ignore_case:
            out["ignore_case"] = True
        if self.whole_word:
            out["whole_word"] = True
        return out

    @property
    def precision(self) -> Optional[float]:
        """適用 ÷（適用 ＋ 却下）。まだ一度も使っていなければ None。"""
        total = self.applied + self.rejected
        return (self.applied / total) if total else None


@dataclass
class Dictionary:
    entries: list[Entry] = field(default_factory=list)
    # この音声（audio_fingerprint）では辞書を使わない、の印
    disabled_for: set[str] = field(default_factory=set)
    path: Optional[Path] = None
    # 読めなかったとき（壊れた JSON）に立てる。黙って空にしたと見えないため
    load_error: str = ""
    # 1 件単位で読めなかった項目の数。**保存すると消える**ので、消してよいと
    # 人が決めるまで（save(force=True)）上書きしない
    skipped: int = 0

    # ------------------------------------------------------------------
    # 読み書き
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "Dictionary":
        """無ければ空。**壊れていても落とさない**が、その旨を `load_error` に残す。

        壊れたファイルは上書きしない（save が拒む）。辞書には人名が入るので、
        黙って空で上書きすると蓄えたものが消える。
        """
        p = Path(path) if path is not None else dictionary_path()
        d = cls(path=p)
        if not p.exists():
            return d
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            d.load_error = type(e).__name__        # 中身は出さない（パスも）
            return d
        if not isinstance(raw, dict):
            d.load_error = "not-an-object"
            return d
        for item in raw.get("entries") or []:
            if not isinstance(item, dict):
                d.skipped += 1
                continue
            try:
                d.entries.append(_entry_from_dict(item))
            except (TypeError, ValueError):
                d.skipped += 1                       # 1 件が壊れていても他は読む
        d.disabled_for = {str(x) for x in (raw.get("disabled_for") or []) if x}
        return d

    @property
    def can_save(self) -> bool:
        """そのまま保存してよいか。False なら save() は例外を投げる。"""
        return not self.load_error and self.skipped == 0

    def save(self, *, force: bool = False) -> Path:
        """書く。**読めなかったものがあれば拒む**（DictionaryUnwritable）。

        壊れていた項目は、保存すると消える。壊れていたのだから消えて構わない、
        と人が決めたときだけ `force=True` で上書きする。黙って消えるのと、
        消すと決めて消すのは違う。
        """
        if not force and self.load_error:
            raise DictionaryUnwritable(
                "辞書ファイルが読めなかったので上書きしません（蓄えたものが消えます）。")
        if not force and self.skipped:
            raise DictionaryUnwritable(
                f"辞書ファイルの {self.skipped} 件が読めなかったので上書きしません"
                "（保存するとその分が消えます）。")
        if self.path is None:
            self.path = dictionary_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "version": FORMAT_VERSION,
            "entries": [asdict(e) for e in self.entries],
            "disabled_for": sorted(self.disabled_for),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)
        return self.path

    # ------------------------------------------------------------------
    # 項目
    # ------------------------------------------------------------------
    def find(self, entry_id: str) -> Optional[Entry]:
        return next((e for e in self.entries if e.id == entry_id), None)

    def add(self, wrong: str, correct: str, *, origin: str = ORIGIN_MANUAL,
            note: str = "", ignore_case: bool = False,
            whole_word: bool = False) -> Entry:
        """項目を足す。**同じ語・同じ条件があれば足さず、それを返す**（二重登録しない）。"""
        wrong = (wrong or "").strip()
        correct = (correct or "").strip()
        if not wrong:
            raise ValueError("誤変換の語句が空です。")
        if wrong == correct:
            raise ValueError("誤変換と正しい語句が同じです。")
        for e in self.entries:
            if (e.wrong, e.correct, e.ignore_case, e.whole_word) == (
                    wrong, correct, ignore_case, whole_word):
                return e
        e = Entry(wrong=wrong, correct=correct, id=uuid.uuid4().hex[:12],
                  added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  origin=origin, note=note,
                  ignore_case=ignore_case, whole_word=whole_word)
        self.entries.append(e)
        return e

    def remove(self, entry_id: str) -> bool:
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.id != entry_id]
        return len(self.entries) < before

    def set_enabled(self, entry_id: str, enabled: bool) -> bool:
        e = self.find(entry_id)
        if e is None:
            return False
        e.enabled = bool(enabled)
        return True

    def record_outcome(self, entry_id: str, applied: int, rejected: int) -> bool:
        """○×画面で 1 項目を終えたときの集計。**一次の記録は edit_log 側。**"""
        e = self.find(entry_id)
        if e is None:
            return False
        e.applied += max(0, int(applied))
        e.rejected += max(0, int(rejected))
        return True

    # ------------------------------------------------------------------
    # 音声ごとの無効化（作業ファイルのスキーマは変えない）
    # ------------------------------------------------------------------
    def is_disabled_for(self, fingerprint: str) -> bool:
        return bool(fingerprint) and fingerprint in self.disabled_for

    def set_disabled_for(self, fingerprint: str, disabled: bool) -> None:
        if not fingerprint:
            return
        if disabled:
            self.disabled_for.add(fingerprint)
        else:
            self.disabled_for.discard(fingerprint)

    def active_entries(self, fingerprint: str = "") -> list[Entry]:
        """この音声で使う項目。無効の項目と、この音声で辞書を切っていれば空。"""
        if self.is_disabled_for(fingerprint):
            return []
        return [e for e in self.entries if e.enabled]


def _entry_from_dict(d: dict[str, Any]) -> Entry:
    wrong = str(d.get("wrong", "")).strip()
    correct = str(d.get("correct", "")).strip()
    if not wrong or wrong == correct:
        raise ValueError("bad entry")
    return Entry(
        wrong=wrong, correct=correct,
        id=str(d.get("id") or uuid.uuid4().hex[:12]),
        enabled=bool(d.get("enabled", True)),
        added_at=str(d.get("added_at", "")),
        origin=str(d.get("origin", ORIGIN_MANUAL)),
        note=str(d.get("note", "")),
        ignore_case=bool(d.get("ignore_case", False)),
        whole_word=bool(d.get("whole_word", False)),
        applied=int(d.get("applied", 0) or 0),
        rejected=int(d.get("rejected", 0) or 0),
    )
