"""話者割当エディタ(v2.0.0 の中心画面)。

やること:
    タイムラインを「発言区間」単位で前後に移動しながら、
    その区間の音声を聴いて、候補者リストから話者を選んで確定していく。

工夫している点:
    - 候補者リストは確定するたびに並べ替わる(suggest.SpeakerSuggester)
    - 「同じ声のまとまり(クラスタ)全体に適用」で一気に確定できる
    - キーボードだけで回せる(Space 再生 / 数字キーで確定 / 自動で次へ)
    - 作業状態は JSON に自動保存。閉じても続きから再開できる
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, Optional, Sequence

from .align import DEFAULT_MODEL, AlignUnavailable, transcribe_words
from .audio import audio_hashes, extract_peaks
from .config import load_config
from . import candidates as cand_mod
from . import listen_order
from .inspection import (
    Proposal,
    clip_to_neighbours,
    decided_history,
    inspect_times,
    load_proposals,
    merge_history,
    proposals_path,
    save_proposals,
    target_segment,
)
from .player import SegmentPlayer
from .segments import (
    INSERT_STYLE_INLINE,
    INSERT_STYLE_LINE,
    MIN_SEGMENT_SECONDS,
    PSEUDO_UNKNOWN,
    Project,
    SPECIAL_MULTI,
    SPECIAL_NOISE,
    SPECIAL_UNKNOWN,
    Segment,
    Speaker,
    audio_span,
    fmt_hms,
    fmt_hms_frac,
    has_inserted_utterances,
    inserted_marks,
    marks_for_segment,
    parse_hms,
    parse_roster,
    speaker_label,
    roster_to_text,
    segment_key,
    suggest_roster_rows,
    suggest_split,
    write_docx,
    write_text,
)
from .suggest import SpeakerSuggester, next_unassigned, next_unreviewed
from .transcribe import estimate_speech_seconds


SPEEDS = ["0.8x", "1.0x", "1.2x", "1.5x", "2.0x"]
# 小窓で使う速さ。**相づちは短くて速いので 0.5x が要る**
# (既定の下限 0.8x では聴き取れない、という実機の指摘・2026-08-18)。
DIALOG_SPEEDS = ["0.5x", "0.65x", "0.8x", "1.0x", "1.2x"]

# 1 区間の再生の長さ。
#
# 区間の終わりは「次の区間の始まり」で決めているため、
#   - 次の発言が遠いと、「うん。」の再生窓が数十秒になる(待たされる)
#   - 逆に相づちが重なると長い発言が切られる(こちらは区間側で解決済み)
# 再生は本文に見合う長さ + 余裕、とする。
PREVIEW_MAX_SECONDS = 60.0     # 上限。長い発言でも先頭だけで判別できる
PREVIEW_MIN_SECONDS = 8.0      # 本文が短いときに確保する長さ
PREVIEW_SLACK_SECONDS = 5.0    # Gemini の時刻推定がずれる分の余裕
PREVIEW_FLOOR_SECONDS = 1.5    # 区間自体が極端に短いときの最低再生時間

# 終了時刻を直したときに鳴らす、終わりの長さ(秒)。
# 日本語で 20 字程度の一句が入り、語尾が切れていないかを判断できる。
# 頭から鳴らすと長い区間では終わりに来るまで待たされるので、ここだけ聴かせる。
TAIL_PREVIEW_SECONDS = 3.0


def preview_length(text: str, duration: float) -> float:
    """この区間を何秒再生するか。

    詰まった箇所では 1 区間が 0.5 秒ということもある。そのまま鳴らしても
    聞き取れないので、区間より少しだけ長く再生する。
    """
    need = estimate_speech_seconds(text) + PREVIEW_SLACK_SECONDS
    need = max(PREVIEW_MIN_SECONDS, min(PREVIEW_MAX_SECONDS, need))
    if duration <= 0:
        return need
    return min(max(duration, PREVIEW_FLOOR_SECONDS), need)


def time_edit_base(seg: Segment, time_offset: float) -> tuple[float, float]:
    """時刻編集の入力欄に出す値(開始, 終了)。

    まだ直していない区間は、いま「ずれ補正込みで聴こえている位置」を初期値に
    する。そのまま確定すれば、聴こえたとおりの時刻がその区間に固定される。
    規約そのものは segments.audio_span に置いてある(再生・点検と同じ値を使う)。
    """
    return audio_span(seg, time_offset)


def clamp_times(
    start: float, end: float, duration: float, moved: str = "start"
) -> tuple[float, float]:
    """時刻編集の値を有効な範囲に収める(0.1 秒精度)。

    moved は今動かしたほうの端。行き過ぎたときに動かさなかった側を守る。
    隣の区間との重なりは直さない。同時発話は正当な記録なので、
    勝手に隣を詰めたり切ったりしてはいけない。
    """
    start = max(0.0, round(start, 1))
    end = max(0.0, round(end, 1))
    top = round(duration, 1) if duration and duration > 0 else None
    if top is not None:
        start = min(start, top)
        end = min(end, top)
    if end - start < MIN_SEGMENT_SECONDS:
        if moved == "start":
            start = round(end - MIN_SEGMENT_SECONDS, 1)
            if start < 0.0:
                start, end = 0.0, MIN_SEGMENT_SECONDS
        else:
            end = round(start + MIN_SEGMENT_SECONDS, 1)
            if top is not None and end > top:
                end = top
                start = max(0.0, round(end - MIN_SEGMENT_SECONDS, 1))
    return start, end


def move_edge(
    base_start: float, base_end: float, which: str, value: float, duration: float,
    shift_if_past: bool = False,
) -> tuple[float, float]:
    """時刻の片側を動かした結果の (開始, 終了) を返す。

    動かすのは片側だけで、区間の長さは変わる(境界の微調整)。もう一方に
    突き当たったらそこで止まる。

    shift_if_past は入力欄に時刻を直接打たれたときだけ True にする。
    もう一方を追い越す時刻を打たれたら、長さを保ったままそこへずらす。
    打った時刻を黙って手前に書き換えるより、そのまま置くほうが分かりやすい。
    ナッジボタンでは False。短い区間はボタン 1 回で追い越してしまうので、
    True にすると押すたびに区間ごとスライドして幅を調整できなくなる。

    区間ごとずらしたいときは shift_span を使う(画面の「区間ごと」ボタン)。
    """
    span = max(MIN_SEGMENT_SECONDS, base_end - base_start)
    if which == "start":
        start, end = value, base_end
        if shift_if_past and start > end - MIN_SEGMENT_SECONDS:
            end = start + span
    else:
        start, end = base_start, value
        if shift_if_past and end < start + MIN_SEGMENT_SECONDS:
            start = end - span
    return clamp_times(start, end, duration, moved=which)


def shift_span(
    base_start: float, base_end: float, delta: float, duration: float
) -> tuple[float, float]:
    """区間の長さを保ったまま前後にずらした (開始, 終了) を返す。

    時刻のずれを直す操作はこれがほとんど。相づちのような短い区間ほど
    長さより大きく動かす必要がある(1 秒の区間を 6 秒ずらす、など)ので、
    片側ずつ動かす操作とは分けてある。
    """
    span = max(MIN_SEGMENT_SECONDS, base_end - base_start)
    start = max(0.0, round(base_start + delta, 1))
    end = round(start + span, 1)
    if duration and duration > 0 and end > duration:
        end = round(duration, 1)
        start = max(0.0, round(end - span, 1))
    return start, end


def playback_window(
    seg: Segment, time_offset: float, back: float = 0.0, extend: float = 0.0
) -> tuple[float, float]:
    """この区間を音声のどこからどこまで鳴らすか(開始秒, 終了秒)。

    時刻を直した区間は、人が耳で合わせた実音声の時刻そのものなので
    ずれ補正を足さない。終わりも本文の長さから推測せず、確認した終了時刻を使う。
    """
    shift = 0.0 if seg.time_edited else time_offset
    if extend > 0 or seg.time_edited:
        preview_end = seg.end
    else:
        # 本文に見合う長さだけ再生する(「この先30秒▶」で続きを聴ける)
        preview_end = seg.start + preview_length(seg.text, seg.duration)
    return max(0.0, seg.start + shift - back), preview_end + shift + extend


def tail_window(
    seg: Segment, time_offset: float, seconds: float = TAIL_PREVIEW_SECONDS
) -> tuple[float, float]:
    """終わりの数秒だけを鳴らす範囲を返す。

    終了時刻を直したときの確認用。頭から鳴らすと、長い区間では終わりに
    たどり着くまで待たされる。確かめたいのは語尾が切れていないかなので、
    終わりだけを聴けば足りる。
    """
    start, end = audio_span(seg, time_offset)
    tail = min(seconds, max(MIN_SEGMENT_SECONDS, end - start))
    return max(0.0, end - tail), end


# 話者ごとの色(タイムライン帯・一覧の色分け用)
PALETTE = [
    "#3E7CB1", "#C1666B", "#4F9D69", "#B07B2F", "#7A5AA8",
    "#2E8B8B", "#C4622D", "#6A7B3C", "#A03E7C", "#4A6FA5",
]
COLOR_UNASSIGNED = "#DDDDDD"
COLOR_SPECIAL = "#9A9A9A"

QUICK_KEYS = "123456789"

# 候補欄の最低の高さ(px)。**画面が小さくても、ここは潰さない。**
# 潰れると候補が 1〜2 人しか出ず、選べなくなる。
CAND_MIN_HEIGHT = 190

# 一覧の絞り込み
FILTER_ALL = "all"
FILTER_UNASSIGNED = "unassigned"
FILTER_UNREVIEWED = "unreviewed"
FILTER_CANDIDATES = "candidates"

FILTER_LABELS = [
    ("すべて表示", FILTER_ALL),
    ("未確定のみ", FILTER_UNASSIGNED),
    ("未確認のみ", FILTER_UNREVIEWED),
    # 区間の中に別の声がある区間だけ(設計書 §10.3)。**判定ではない**——
    # 候補が無い区間にも取りこぼしはある。適合は 35/51 で 3 割は空振り。
    ("別の声あり", FILTER_CANDIDATES),
]


@dataclass
class RosterPlan:
    """出席者リストの変更内容(まだ適用していない下見)。

    「消してから確認する」と、確認で『いいえ』を選んでも割当が戻らない。
    先に計画を立てて、確認が済んでから apply() する。
    """

    speakers: list[Speaker]                 # 適用後の話者リスト
    added: list[str]
    removed: list[Speaker]
    affected_segments: int                  # 削除によって未確定に戻る区間数

    def apply(self, proj: Project) -> None:
        removed_ids = {sp.id for sp in self.removed}
        proj.speakers = self.speakers
        if removed_ids:
            # **外したことを記録に残す**(編集履歴設計書 §1.3)。
            proj.clear_speakers(removed_ids)


class RosterTable(ttk.Frame):
    """出席者を「名前」と「企業・役職」の 2 列で編集する表(設計書 §11.8)。

    **行が話者 ID を持ち回る。**名前の一致で引き継ぐと、名前を直しただけで
    別人が入って古い方が消えたと判断され、確定済みの割当が外れる。
    各行が元の ID を覚えているので、名前を直しても並べ替えても保たれる。

    文字起こし画面(名簿を作るとき)と割当画面(名簿を直すとき)の両方で使う。
    前者には話者 ID がまだ無いので、その場合は空文字のままでよい。
    """

    NAME_WIDTH = 16
    NOTE_WIDTH = 46

    def __init__(self, parent: tk.Misc, note_width: int = NOTE_WIDTH) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self._note_width = note_width
        self.rows: list[dict] = []

    def add_row(self, sid: str = "", name: str = "", note: str = "",
                focus: bool = False) -> dict:
        """1 人ぶんの行を足す。sid が空なら新しい人。"""
        frm = ttk.Frame(self)
        var_name = tk.StringVar(value=name)
        var_note = tk.StringVar(value=note)
        ent = ttk.Entry(frm, textvariable=var_name, width=self.NAME_WIDTH)
        ent.grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=var_note, width=self._note_width).grid(
            row=0, column=1, sticky="ew", padx=(6, 0))
        frm.columnconfigure(1, weight=1)
        row = {"sid": sid, "name": var_name, "note": var_note,
               "frame": frm, "entry": ent}
        ttk.Button(frm, text="×", width=3, command=lambda: self.del_row(row)).grid(
            row=0, column=2, padx=(6, 0))
        self.rows.append(row)
        self._relayout()
        if focus:
            ent.focus_set()
        return row

    def del_row(self, row: dict) -> None:
        if row not in self.rows:
            return
        self.rows.remove(row)
        row["frame"].destroy()
        if not self.rows:
            self.add_row()
        self._relayout()

    def clear(self) -> None:
        for row in list(self.rows):
            row["frame"].destroy()
        self.rows.clear()

    def set_speakers(self, speakers: Sequence[Speaker]) -> None:
        self.clear()
        for sp in speakers:
            self.add_row(sp.id, sp.name, sp.note)
        if not self.rows:
            self.add_row()

    def set_text(self, text: str, split: bool = True) -> None:
        """名簿テキストを流し込む。split=True なら分かれていない行を分ける。

        一覧をそのまま貼り込めるようにするためのもの。話者 ID は付かない。
        """
        rows = (suggest_roster_rows(text) if split
                else [(sp.name, sp.note) for sp in parse_roster(text)])
        self.clear()
        for name, note in rows:
            self.add_row("", name, note)
        if not self.rows:
            self.add_row()

    def _relayout(self) -> None:
        for i, row in enumerate(self.rows):
            row["frame"].grid(row=i, column=0, sticky="ew", pady=1)

    def auto_split(self) -> int:
        """名前の欄に肩書ごと入っている行を、名前と役職に分ける**提案**。

        **すでに役職が入っている行は触らない。**人が入れたものを
        推測で上書きしない。戻り値は分けた行数。
        """
        names = [r["name"].get().strip() for r in self.rows]
        done = 0
        for i, row in enumerate(self.rows):
            if row["note"].get().strip():
                continue
            others = [n for j, n in enumerate(names) if j != i and n]
            name, note = suggest_split(names[i], others)
            if note:
                row["name"].set(name)
                row["note"].set(note)
                done += 1
        return done

    def values(self) -> list[tuple[str, str, str]]:
        """(話者 ID, 名前, 企業・役職) の並び。名前が空の行は数えない。"""
        out: list[tuple[str, str, str]] = []
        for row in self.rows:
            name = row["name"].get().strip()
            if name:
                out.append((row["sid"], name, row["note"].get().strip()))
        return out

    def to_text(self) -> str:
        """1 行 1 人の「名前(役職)」形式。設定の保存と転写経路に渡す形。"""
        return roster_to_text(
            [Speaker(id=sid or f"sp{i + 1:02d}", name=name, note=note, order=i)
             for i, (sid, name, note) in enumerate(self.values())])

    def set_enabled(self, on: bool) -> None:
        state = "normal" if on else "disabled"
        for row in self.rows:
            for w in row["frame"].winfo_children():
                try:
                    w.configure(state=state)
                except tk.TclError:
                    pass


class RosterDialog(tk.Toplevel):
    """名簿を直す小窓。中身は RosterTable(設計書 §11.8)。"""

    def __init__(self, parent: tk.Misc, speakers: Sequence[Speaker]) -> None:
        super().__init__(parent)
        self.title("出席者(候補者リスト)")
        self.transient(parent)
        self.grab_set()
        # 画面より大きくしない(はみ出すと下のボタンが画面外に出る)
        scr_w, scr_h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(760, scr_w - 40)}x{min(560, scr_h - 90)}")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        self.result: Optional[list[tuple[str, str, str]]] = None

        ttk.Label(
            self,
            text=("名前と、企業・役職を分けて入れてください。" + chr(10)
                  + "本文の【 】に何を出すかは、出力のときに選べます"
                  + "(名前だけ / 役職も付ける)。出席者一覧には常に両方が載ります。"
                  + chr(10)
                  + "上から順に並ぶので、よく発言する人を上に置くと"
                  + "最初の候補順が良くなります。"),
            foreground="#555", justify="left",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        head = ttk.Frame(self)
        head.grid(row=1, column=0, sticky="w", padx=12)
        ttk.Label(head, text="名前", font=("", 9, "bold"),
                  width=RosterTable.NAME_WIDTH).grid(row=0, column=0, sticky="w")
        ttk.Label(head, text="企業・役職", font=("", 9, "bold")).grid(
            row=0, column=1, sticky="w", padx=(6, 0))

        outer = ttk.Frame(self)
        outer.grid(row=2, column=0, sticky="nsew", padx=12)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(outer, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=sb.set)
        self.table = RosterTable(self.canvas)
        item = self.canvas.create_window((0, 0), window=self.table, anchor="nw")
        self.table.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(item, width=e.width))
        # ホイールは「この表の上にいるとき」だけ拾う。bind_all を出しっぱなし
        # にすると、他の窓のホイールまで食う(主画面が同じ轍を踏んでいる)。
        self.canvas.bind(
            "<Enter>",
            lambda e: self.canvas.bind_all("<MouseWheel>", self._on_wheel))
        self.canvas.bind(
            "<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))
        self.table.set_speakers(speakers)

        tools = ttk.Frame(self)
        tools.grid(row=3, column=0, sticky="w", padx=12, pady=(8, 0))
        ttk.Button(tools, text="行を足す",
                   command=lambda: self.table.add_row(focus=True)).pack(side="left")
        ttk.Button(tools, text="名前と役職に自動で分ける",
                   command=self.auto_split).pack(side="left", padx=6)
        self.var_note = tk.StringVar(value="")
        ttk.Label(tools, textvariable=self.var_note, foreground="#555").pack(
            side="left", padx=6)

        btns = ttk.Frame(self)
        btns.grid(row=4, column=0, sticky="ew", padx=12, pady=12)
        ttk.Button(btns, text="OK", command=self._ok).pack(side="right")
        ttk.Button(btns, text="キャンセル", command=self._cancel).pack(
            side="right", padx=6)
        self.bind("<Escape>", lambda e: self._cancel())
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    @property
    def rows(self) -> list[dict]:
        return self.table.rows

    def add_row(self, *a, **k) -> dict:
        return self.table.add_row(*a, **k)

    def del_row(self, row: dict) -> None:
        self.table.del_row(row)

    def values(self) -> list[tuple[str, str, str]]:
        return self.table.values()

    def auto_split(self) -> int:
        done = self.table.auto_split()
        self.var_note.set(
            f"{done} 人を分けました。外れたものは直してください。" if done
            else "分けられる行がありませんでした。")
        return done

    def _on_wheel(self, event) -> None:
        try:
            self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        except tk.TclError:
            pass

    def _ok(self) -> None:
        if not self.values():
            messagebox.showwarning(
                "出席者", "少なくとも 1 人は必要です。", parent=self)
            return
        self.result = self.values()
        self._close()

    def _cancel(self) -> None:
        self.result = None
        self._close()

    def _close(self) -> None:
        try:
            self.canvas.unbind_all("<MouseWheel>")
        except tk.TclError:
            pass
        self.destroy()


def fmt_short_time(seconds: float) -> str:
    """候補の選択肢に出す短い時刻。1 時間未満なら M:SS、超えたら H:MM:SS。

    HH:MM:SS だと選択肢の幅に収まらず「00:00:29 声!」と切れた(実機で確認・
    2026-08-19)。**先頭の 00: は 1 時間の録音では常に同じで情報が無い。**
    ［時刻へ飛ぶ］が受け付ける書き方と同じなので、見て打ち直せる。
    """
    total = int(round(max(0.0, seconds)))
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def plan_roster_rows(
    proj: Project, rows: Sequence[tuple[str, str, str]]
) -> RosterPlan:
    """(話者 ID, 名前, 企業・役職) の並びから変更計画を作る(まだ何も変えない)。

    **ID は行が持ち回る。**名前の一致で引き継ぐと、「山本学　文科省…室長」を
    「山本学」に直しただけで別人が入って古い方が消えたと判断され、
    **確定済みの割当が外れる**(実データで起きうることを確認・2026-08-18)。
    表の各行が元の ID を覚えていれば、名前を直しても並べ替えても保たれる。

    新しい行は ID を空文字にする。渡されなかった ID は削除とみなす。
    """
    used = {sp.id for sp in proj.speakers}
    by_id = {sp.id: sp for sp in proj.speakers}
    new_list: list[Speaker] = []
    added: list[str] = []
    kept: set[str] = set()

    for sid, name, note in rows:
        name = (name or "").strip()
        note = (note or "").strip()
        if not name:
            continue
        if sid and sid in by_id and sid not in kept:
            kept.add(sid)
            new_list.append(Speaker(id=sid, name=name, note=note, order=0))
        else:
            i = 1
            while f"sp{i:02d}" in used:
                i += 1
            nid = f"sp{i:02d}"
            used.add(nid)
            new_list.append(Speaker(id=nid, name=name, note=note, order=0))
            added.append(name)

    removed = [sp for sp in proj.speakers if sp.id not in kept]
    for i, sp in enumerate(new_list):
        sp.order = i
    removed_ids = {sp.id for sp in removed}
    affected = sum(1 for s in proj.segments if s.speaker_id in removed_ids)
    return RosterPlan(speakers=new_list, added=added,
                      removed=removed, affected_segments=affected)


def plan_roster_text(proj: Project, text: str) -> RosterPlan:
    """出席者リストのテキストから変更計画を作る(この時点では何も変えない)。

    既存の話者 ID は名前一致で保持する。ID を振り直すと、確定済み区間が
    別人を指してしまうため。同姓の人が複数いても 1 人ずつ対応付ける。
    """
    wanted = parse_roster(text)

    remaining: dict[str, list[Speaker]] = {}
    for sp in proj.speakers:
        remaining.setdefault(sp.name, []).append(sp)

    used_ids = {sp.id for sp in proj.speakers}
    new_list: list[Speaker] = []
    added: list[str] = []

    for w in wanted:
        pool = remaining.get(w.name)
        if pool:
            src = pool.pop(0)
            new_list.append(Speaker(id=src.id, name=src.name, note=w.note, order=0))
        else:
            i = 1
            while f"sp{i:02d}" in used_ids:
                i += 1
            sid = f"sp{i:02d}"
            used_ids.add(sid)
            new_list.append(Speaker(id=sid, name=w.name, note=w.note, order=0))
            added.append(w.name)

    removed = [sp for pool in remaining.values() for sp in pool]
    for i, sp in enumerate(new_list):
        sp.order = i

    removed_ids = {sp.id for sp in removed}
    affected = sum(1 for s in proj.segments if s.speaker_id in removed_ids)
    return RosterPlan(speakers=new_list, added=added, removed=removed,
                      affected_segments=affected)


def cancel_pending_afters(widget: tk.Misc) -> int:
    """**この窓が予約した after() を全部取り消す。**戻り値は取り消した数。

    tkinter は after() のたびに Tcl 命令を登録し、窓を壊すときにその命令を
    消す。しかし**タイマーそのものは残る**ので、発火時に「invalid command
    name」の背景エラーになる。Tk はそれを画面に出さない（stderr だけ。凍結版
    では誰にも届かない）。実機では割当画面を閉じるたびに、4 秒後（自動保存）に
    1 回起きていた。検査では別の Tk が回っている間に発火し、「bgerror failed」
    が毎回 6〜8 行出ていた（2026-09-03 に特定。「以前からある」を根拠に
    無関係と判断しかけた）。

    窓ごとに登録した命令名（`_tclCommands`）と、予約済みタイマーの script を
    突き合わせて、**この窓のぶんだけ**取り消す。個別のタイマー ID を追う
    やり方は取りこぼす（8-24 に 1 つだけ取り消して、残りを見落とした）。
    """
    names = set(getattr(widget, "_tclCommands", None) or ())
    if not names:
        return 0
    cancelled = 0
    try:
        for tid in widget.tk.call("after", "info"):
            try:
                info = widget.tk.call("after", "info", tid)
            except tk.TclError:
                continue        # その間に発火した
            script = str(info[0]) if isinstance(info, (tuple, list)) and info else str(info)
            if script in names:
                widget.tk.call("after", "cancel", tid)
                cancelled += 1
    except tk.TclError:
        pass                    # インタプリタが既に無い
    return cancelled


class AssignWindow(tk.Toplevel):
    """話者割当エディタのウィンドウ"""

    def __init__(self, master: Optional[tk.Misc], project: Project) -> None:
        super().__init__(master)
        self.proj = project
        self.suggester = SpeakerSuggester(project)
        self.player = SegmentPlayer(on_finished=self._on_play_finished)
        self.current = 0
        self._undo: list[list[tuple[int, Optional[str], bool]]] = []
        self._dirty = False
        self._audio_declined = False
        # 点検(実測用の転写)は重いので別スレッドで回す。知らせは queue 経由で
        # 受ける(tkinter は本体スレッドからしか触れない)。
        self._inspect_thread: Optional[threading.Thread] = None
        self._inspect_cancel = threading.Event()
        self._inspect_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        # 直前の「まとめて適用」を丸ごと戻すための控え(時刻専用。話者の
        # _undo とは別系統 — あちらは時刻を持たない)
        self._time_undo: Optional[dict] = None
        self._candidates: list = []
        self._cand_widgets: list[ttk.Button] = []
        self._row_ids: list[str] = []

        self.title(f"話者の割当 - {Path(self.proj.audio_path).name}")
        # **既定で全部見えること。**窓を広げないと押せないボタンがあり、
        # 高さも足りずに下の［保存］が隠れていた(実機の指摘・2026-08-18)。
        #   幅 1320 … 右ペインのいちばん広い行(426px)が収まる
        #             (時刻と確定を 2 行ずつに分ける前は 571px だった)
        #   高さ 900 … 中身が 845px まであり、800 では下端が窓の縁に接する
        # ただし画面より大きくはしない(小さい画面では入るところまで)。
        want_w, want_h = 1320, 900
        scr_w = self.winfo_screenwidth()
        scr_h = self.winfo_screenheight()
        self.geometry(f"{min(want_w, scr_w - 40)}x{min(want_h, scr_h - 90)}")
        self.minsize(min(1180, scr_w - 40), min(700, scr_h - 90))

        self.var_speed = tk.StringVar(value="1.0x")
        self.var_autoplay = tk.BooleanVar(value=True)
        self.var_advance = tk.BooleanVar(value=True)
        # 一括適用は既定で ON。これが推奨の進め方で、これを使わないと
        # 90 分の会議で数百回の判断が必要になる。
        self.var_apply_cluster = tk.BooleanVar(value=True)
        # **既定は埋め込み。**足した発話は割り込んだ位置で親の本文に入るので、
        # 行としても出すと同じものが 2 か所に見える(実機の要望・2026-08-24)。
        self.var_added_rows = tk.BooleanVar(value=False)
        # 候補を見える位置へ送る予約(窓を閉じるときに取り消す)
        self._cand_scroll_after = None
        self.var_filter = tk.StringVar(value=FILTER_ALL)
        # 聴く順(取りこぼしを見つけやすい順)。**既定は時間順のまま**——
        # 並びが黙って変わると「上から順に聴いた」という作業の前提が崩れる。
        self.var_listen_order = tk.BooleanVar(value=False)
        # orig_start(丸め) → ListenHint。無ければ None(=並べ替えは出せない)
        self._listen_hints = self._load_listen_hints()
        # 候補の一覧(設計書 §10.3)。**検出器ではない**——候補が無い
        # 区間にも取りこぼしはある。却下は sidecar に残す(空振りが 3 割)。
        self._dismissed = cand_mod.load_dismissed(cand_mod.dismissed_path(
            self._work_dir(), project.audio_fingerprint or ""))
        self._voice_candidates: list = []
        self._current_voice_candidates: list = []
        self.var_cand = tk.StringVar(value="")
        self.var_status = tk.StringVar(value="")
        self.var_seginfo = tk.StringVar(value="")
        self.var_action = tk.StringVar(value="")
        self.var_backend = tk.StringVar(value="")
        self.var_offset = tk.DoubleVar(value=float(project.time_offset))
        self.var_start = tk.StringVar(value="")
        self.var_end = tk.StringVar(value="")

        self._build_ui()
        self._bind_keys()
        # 候補は turn から作る。話者分離を通していなければ空のまま
        # (呼び出し側で「使えない」と伝える)。
        self._voice_candidates = self._load_voice_candidates()
        # 声のまとまりが 1 つも無い作業ファイル(ローカル転写)では、一括適用は
        # 成り立たない。ON のまま残すと確定のたびに「一括適用できません」の
        # 警告が出る——1219 区間なら 1219 回になる。最初から外しておく。
        if not self.has_real_clusters():
            self.var_apply_cluster.set(False)
            self.chk_apply_cluster.configure(state="disabled")
            # **どうすればよいかまで書く。**「ありません」で終わると、
            # 打つ手が分からないまま 1 件ずつ確定することになる。
            self.lbl_cluster_note.configure(text=(
                f"※ このファイルには声のまとまりがありません。"
                f"{len(self.proj.segments):,} 区間を 1 件ずつ確定することに"
                "なります。転写のやり直しで［声のまとまりを端末内で分ける"
                "（話者分離）］を入れると、まとめて当てられます"
                "（転写はキャッシュが効くので、やり直しは短く済みます）。"))
        self.refresh_all()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(4000, self._autosave_tick)

        if not self.proj.speakers:
            self.after(300, self._first_run_hint)

    # ==================================================================
    # UI 構築
    # ==================================================================
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # --- 上部: 進捗 -------------------------------------------------
        top = ttk.Frame(self, padding=(10, 8, 10, 4))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="進捗:").grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(top, mode="determinate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Label(top, textvariable=self.var_status, width=34, anchor="w")\
            .grid(row=0, column=2, sticky="w")

        # --- タイムライン帯 --------------------------------------------
        tl = ttk.Frame(self, padding=(10, 0, 10, 4))
        tl.grid(row=1, column=0, sticky="ew")
        tl.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(tl, height=34, background="#F7F7F7",
                                highlightthickness=1, highlightbackground="#CCCCCC")
        self.canvas.grid(row=0, column=0, sticky="ew")
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda e: self._draw_timeline())

        # --- 本体: 左=一覧 / 右=作業 -----------------------------------
        body = ttk.Panedwindow(self, orient="horizontal")
        body.grid(row=2, column=0, sticky="nsew", padx=10, pady=4)

        left = ttk.Frame(body)
        body.add(left, weight=3)
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        # **1 行に詰めない。**「別の声あり」を足したぶんで 840px になり、
        # 左ペイン(775px)から［時刻へ飛ぶ］が押し出されて押せなくなった
        # (実機の指摘・2026-08-19)。絞り込みと道具で行を分ける。
        filt = ttk.Frame(left)
        filt.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        tools_row = ttk.Frame(left)
        tools_row.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        for label, value in FILTER_LABELS:
            ttk.Radiobutton(
                filt, text=label, value=value, variable=self.var_filter,
                command=self._on_filter_change, takefocus=False,
            ).pack(side="left", padx=(0, 8))
        # 聴く順。話者分離を通した作業ファイルでだけ選べる。
        # 「検出」とは名乗らない——順番が付かない区間にも取りこぼしはある。
        self.chk_listen = ttk.Checkbutton(
            tools_row, text="聴く順(取りこぼしを見つけやすい順)",
            variable=self.var_listen_order,
            command=self._on_listen_order_change, takefocus=False,
            state="normal" if self._listen_hints else "disabled")
        self.chk_listen.pack(side="left")

        # 時刻へ飛ぶ。**700 区間をスクロールで探すのは現実的でない**
        # (実機の指摘・2026-08-18)。逐語正解の道具には既にあった機能で、
        # 本体に無いのは抜けだった。
        jump = ttk.Frame(tools_row)
        jump.pack(side="right")
        ttk.Label(jump, text="時刻へ飛ぶ:").pack(side="left")
        self.var_jump = tk.StringVar()
        ent_jump = ttk.Entry(jump, textvariable=self.var_jump, width=11)
        ent_jump.pack(side="left", padx=4)
        ent_jump.bind("<Return>", lambda e: self.jump_to_time())
        ttk.Button(jump, text="→", width=3, takefocus=False,
                   command=self.jump_to_time).pack(side="left")

        cols = ("time", "cluster", "speaker", "text")
        # **複数選べるようにする。**Shift で並び、Ctrl で飛び飛び。同じ人が
        # 続けて話す帯をまとめて指定できないと、区間ごとに同じ操作を
        # 繰り返すことになる(実機の要望・2026-08-23)。
        self.tree = ttk.Treeview(left, columns=cols, show="headings",
                                 selectmode="extended")
        self.tree.heading("time", text="時刻")
        self.tree.heading("cluster", text="声")
        self.tree.heading("speaker", text="話者")
        self.tree.heading("text", text="発言")
        self.tree.column("time", width=78, anchor="w", stretch=False)
        self.tree.column("cluster", width=56, anchor="center", stretch=False)
        self.tree.column("speaker", width=120, anchor="w", stretch=False)
        self.tree.column("text", width=340, anchor="w")
        self.tree.grid(row=2, column=0, sticky="nsew")
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        sb.grid(row=2, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        # **背景の色と文字の色で、タグを分ける。**1 行に複数のタグが付くと
        # どれが効くかは Tk の実装次第で、確かめる手立てもない。同じ種類の
        # 色を 2 つ以上付けないようにして、当てにならない優先順に頼らない。
        self.tree.tag_configure("bg_unassigned", background="#FFF8E1")
        self.tree.tag_configure("bg_bulk", background="#F1F6FB")
        self.tree.tag_configure("fg_special", foreground="#8A8A8A")
        # 人が足した発話。**一覧で見分けられないと、あとから消す対象を
        # 探せない**(実機の指摘・2026-08-18)。薄い橙では埋もれて分からない
        # という指摘を受けたので(2026-08-22)、**背景・文字色・括弧の 3 つ**で
        # 示す。色だけでは印刷や色覚の条件で消える。
        self.tree.tag_configure("bg_added", background="#FFE0B2")
        self.tree.tag_configure("fg_added", foreground="#BF360C")

        right = ttk.Frame(body)
        body.add(right, weight=4)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        right.rowconfigure(4, weight=1)

        ttk.Label(right, textvariable=self.var_seginfo, font=("", 10, "bold"))\
            .grid(row=0, column=0, sticky="w", padx=6, pady=(0, 2))

        # --- 区間の時刻 --------------------------------------------------
        # Gemini の時刻推定は局所的にずれる(1 分の間に 0→6→2 秒と変動する)。
        # 全体一律のずれ補正では直せないので、区間ごとに耳で合わせられるようにする。
        frm_time = ttk.Frame(right)
        frm_time.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 2))
        # **開始と終了で行を分ける。**13 個を 1 行に詰めていたが、
        # 1280x720@125% で 要 713px / 幅 706px と足りず、確定の行を 2 行に
        # したあとはここが律速だった(実測 2026-08-29)。開始と終了は
        # 別のものなので、分ける線としても素直。
        #
        # **微調整ボタン 8 個をまとめて 1 行にはしない。**開始側と終了側の
        # ［−1］［+1］が隣り合い、取り違える。幅のために操作を危うくしない。
        #
        # 2 行目の頭には［時刻:］と同じ幅の空きを置き、［開始］［終了］の
        # 左端を揃える。
        self._time_rows: list = []
        for which, label in (("start", "開始"), ("end", "終了")):
            row_time = ttk.Frame(frm_time)
            row_time.pack(side="top", anchor="w", pady=(0, 2))
            self._time_rows.append(row_time)
            head = ttk.Label(row_time, text="時刻:" if which == "start" else "")
            head.pack(side="left", padx=(0, 8))
            if which == "end":
                # 空文字だと幅 0 になり揃わない。頭の文字と同じ幅を確保する。
                head.configure(width=len("時刻:"))
            ttk.Label(row_time, text=label).pack(side="left", padx=(6, 2))
            ent = ttk.Entry(
                row_time, width=11, justify="center",
                textvariable=self.var_start if which == "start" else self.var_end,
            )
            ent.pack(side="left", padx=(0, 2))
            # Enter は「この値でいい」という意思表示。ずれ補正込みの初期値を
            # そのまま押しても、その時刻でこの区間を固定する。
            ent.bind("<Return>", lambda e, w=which: self._commit_time(w, explicit=True))
            # **欄から離れただけでは書かない。**打ち間違えた時刻が、Enter を
            # 押していないのに記録へ入る事故が起きた(実機・2026-08-19。
            # ［時刻へ飛ぶ］のつもりで 00:35:09 を［開始］に打ち、区間が
            # 0〜35:25 の 2125 秒に膨らんだ)。**「検証済みの記録」を売る製品で、
            # 欄を通り過ぎただけで記録が変わるのは筋が悪い。**
            # 打ちかけの値は捨て、Enter か［この時刻で確認］を促す。
            ent.bind("<FocusOut>", lambda e, w=which: self._discard_time_edit(w))
            for text, delta in (("−1", -1.0), ("−0.1", -0.1), ("+0.1", +0.1), ("+1", +1.0)):
                ttk.Button(
                    row_time, text=text, width=5, takefocus=False,
                    command=lambda w=which, d=delta: self._nudge_time(w, d),
                ).pack(side="left", padx=1)
        # 機械が当てただけの時刻(✎△)を、聴いて確かめたら ✎ に上げる。
        # 「再生 → 合っていればこれを押す」で流せるようにボタンにしてある。
        # **「区間ごとスライド」の行に置く。**時刻の行に並べると 733px 必要に
        # なり、既定の窓幅では右端が切れた(実機の指摘・2026-08-18)。
        self.btn_confirm_time = ttk.Button(
            row_edit_time_actions := ttk.Frame(frm_time), text="この時刻で確認",
            takefocus=False, command=self.confirm_time)
        row_edit_time_actions.pack(side="top", anchor="w", pady=(3, 0))
        self.btn_confirm_time.pack(side="left")
        self.btn_revert_time = ttk.Button(
            row_edit_time_actions, text="元に戻す", takefocus=False,
            command=self.revert_time)
        self.btn_revert_time.pack(side="left", padx=(6, 0))
        ttk.Separator(row_edit_time_actions, orient="vertical").pack(
            side="left", fill="y", padx=10)
        ttk.Label(row_edit_time_actions, text="区間ごとスライド:").pack(
            side="left", padx=(0, 4))
        for _text, _delta in (("−1", -1.0), ("−0.1", -0.1),
                              ("+0.1", +0.1), ("+1", +1.0)):
            ttk.Button(row_edit_time_actions, text=_text, width=5,
                       takefocus=False,
                       command=lambda d=_delta: self._shift_time(d)).pack(
                side="left", padx=1)

        row_edit = ttk.Frame(frm_time)
        row_edit.pack(side="top", anchor="w", pady=(3, 0))

        # 1 区間に 2 人の発言が順に混ざることも、同じ発言が 2 行に割れることも
        # ある。どちらも人の目と耳でしか判断できないので、明示操作で直せるようにする。
        ttk.Button(row_edit, text="この区間を分割...", takefocus=False,
                   command=self.split_current).pack(side="left")
        ttk.Button(row_edit, text="前の区間と結合", takefocus=False,
                   command=self.merge_with_prev).pack(side="left", padx=6)
        ttk.Button(row_edit, text="次の区間と結合", takefocus=False,
                   command=self.merge_with_next).pack(side="left")
        # **区間を消すはこの行に置く。**下の行は候補の選択肢が入るので幅が
        # 足りない(1280x720 の画面で 508px 必要・幅 473px しかない)。
        # こちらの行は 292px しか使っておらず余裕がある。
        self.btn_del_added = ttk.Button(
            row_edit, text="この区間を消す", takefocus=False,
            command=self.remove_added, state="disabled")
        self.btn_del_added.pack(side="left", padx=(12, 0))

        # **もう 1 行に分ける。**1 行に詰めると既定の窓幅(1180)で右端が切れ、
        # 窓を広げないとボタンが押せなかった(実機の指摘・2026-08-18)。
        row_add = ttk.Frame(frm_time)
        row_add.pack(side="top", anchor="w", pady=(3, 0))
        # 聞こえたのに本文に無い発話を足す。**どのエンジンも会話途中の相づちを
        # 書けない**(7 系統の実測で、和集合でも 33 件中 12 件が拾えない)ので、
        # 機械が時刻と声を用意し、人が言葉を入れる。
        ttk.Button(row_add, text="＋この声を足す...", takefocus=False,
                   command=self.add_utterance).pack(side="left")
        # 候補の一覧(設計書 §10.3)。**行を増やさない。**右ペインは既に
        # 詰まっており、行を足すと下の保存ボタンが隠れる(実機の指摘)。
        # **幅の余裕も 21px しか無い**(要 571 / 幅 592)ので、説明文と候補は
        # 排他にする——同時には出さない。どちらか広いほうがこの行の幅になる。
        # **短くしてある。**元の説明文は 360px あり、右ペイン(520px)に
        # 収まっていなかった(2026-08-19 に実測)。
        self.lbl_cand = ttk.Label(
            row_add, foreground="#666",
            text="（本文に無い発話を、位置を指して足します）")
        self.lbl_cand.pack(side="left", padx=(10, 0))

        # **埋め込みにすると、足した発話に行が無くなる。**直す・消す入口を
        # ここに置かないと、届かなくなる(実機の要望・2026-08-24)。
        # 中身があるときだけ出す。行を占有し続けると右ペインが詰まる。
        self.row_inserted = ttk.Frame(frm_time)
        self._inserted_widgets: list = []

        self.frm_cand = ttk.Frame(row_add)
        ttk.Label(self.frm_cand, text="別の声:").pack(side="left", padx=(8, 3))
        self.cmb_cand = ttk.Combobox(self.frm_cand, textvariable=self.var_cand,
                                     width=11, state="disabled",
                                     takefocus=False)
        self.cmb_cand.pack(side="left")
        self.btn_cand_add = ttk.Button(
            self.frm_cand, text="ここから", takefocus=False,
            state="disabled", command=self.add_from_voice_candidate)
        self.btn_cand_add.pack(side="left", padx=(4, 0))
        self.btn_cand_skip = ttk.Button(
            self.frm_cand, text="×", width=3, takefocus=False,
            state="disabled", command=self.dismiss_voice_candidate)
        self.btn_cand_skip.pack(side="left", padx=(2, 0))
        self.lbl_cand_note = ttk.Label(self.frm_cand, foreground="#666",
                                       text="")
        self.lbl_cand_note.pack(side="left", padx=(8, 0))

        frm_text = ttk.LabelFrame(right, text="この区間の発言(編集できます)")
        frm_text.grid(row=2, column=0, sticky="nsew", padx=4, pady=2)
        frm_text.columnconfigure(0, weight=1)
        frm_text.rowconfigure(0, weight=1)
        self.txt_body = tk.Text(frm_text, height=6, wrap="word", font=("", 11))
        self.txt_body.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        sb2 = ttk.Scrollbar(frm_text, orient="vertical", command=self.txt_body.yview)
        sb2.grid(row=0, column=1, sticky="ns", pady=4)
        self.txt_body.configure(yscrollcommand=sb2.set)
        self.txt_body.bind("<FocusOut>", lambda e: self._commit_text())

        # --- 再生コントロール ------------------------------------------
        frm_play = ttk.LabelFrame(right, text="再生")
        frm_play.grid(row=3, column=0, sticky="ew", padx=4, pady=2)
        self.btn_play = ttk.Button(frm_play, text="▶ 再生 (Space)", command=self.toggle_play, width=16)
        self.btn_play.grid(row=0, column=0, padx=(6, 4), pady=6)
        ttk.Button(frm_play, text="◀ 5秒前から",
                   command=lambda: self.play_current(back=5.0, explicit=True))\
            .grid(row=0, column=1, padx=4, pady=6)
        ttk.Label(frm_play, text="速度:").grid(row=0, column=2, padx=(12, 2))
        ttk.Combobox(frm_play, values=SPEEDS, textvariable=self.var_speed,
                     state="readonly", width=6).grid(row=0, column=3)
        ttk.Button(frm_play, text="この先30秒▶",
                   command=lambda: self.play_current(extend=30.0, explicit=True))\
            .grid(row=0, column=4, padx=4, pady=6)
        ttk.Checkbutton(frm_play, text="移動したら自動再生", variable=self.var_autoplay,
                        takefocus=False).grid(row=0, column=5, padx=(12, 4))

        # --- ずれ補正 ---------------------------------------------------
        # 文字と音声がずれて聞こえるときに使う。Gemini の時刻推定は
        # 実音声と一致しないことがあり、こちらでは直せないため手動調整を置く。
        ttk.Label(frm_play, text="ずれ補正:").grid(row=1, column=0, sticky="e", padx=(6, 2))
        ttk.Spinbox(
            frm_play, from_=-10.0, to=10.0, increment=0.2, width=6,
            textvariable=self.var_offset, command=self._on_offset_change,
        ).grid(row=1, column=1, sticky="w")
        ttk.Button(frm_play, text="±0に戻す", command=lambda: self._set_offset(0.0))\
            .grid(row=1, column=2, padx=4)
        ttk.Label(
            frm_play,
            text="秒。音声が文字より遅れて聞こえるなら + 、早いなら −(Shift+← / Shift+→)",
            foreground="#666",
        ).grid(row=1, column=3, columnspan=3, sticky="w", padx=4)

        ttk.Label(frm_play, textvariable=self.var_backend, foreground="#888")\
            .grid(row=2, column=0, columnspan=6, sticky="w", padx=8, pady=(0, 4))

        # --- 候補者リスト ----------------------------------------------
        frm_cand = ttk.LabelFrame(right, text="話者を選ぶ(数字キーで即確定・可能性の高い順)")
        frm_cand.grid(row=4, column=0, sticky="nsew", padx=4, pady=2)
        frm_cand.columnconfigure(0, weight=1)
        frm_cand.rowconfigure(0, weight=1)
        # **候補は縦にスクロールできるようにする。**入れ物の高さが足りないと
        # 下の候補が黙って切れる。出席者 9 人で 6 人しか出ず、残りを選べ
        # なかった(実機の指摘・2026-08-24)。ボタン自体は 9 個とも作られて
        # いたので、**画面からは「そもそも居ない」ようにしか見えなかった。**
        cand_wrap = ttk.Frame(frm_cand)
        cand_wrap.grid(row=0, column=0, sticky="nsew", padx=6, pady=4)
        cand_wrap.columnconfigure(0, weight=1)
        cand_wrap.rowconfigure(0, weight=1)

        self.cand_canvas = tk.Canvas(cand_wrap, highlightthickness=0,
                                     height=CAND_MIN_HEIGHT)
        self.cand_canvas.grid(row=0, column=0, sticky="nsew")
        self.cand_scroll = ttk.Scrollbar(cand_wrap, orient="vertical",
                                         command=self.cand_canvas.yview)
        self.cand_canvas.configure(yscrollcommand=self._on_cand_scroll)

        self.cand_holder = ttk.Frame(self.cand_canvas)
        self._cand_window = self.cand_canvas.create_window(
            (0, 0), window=self.cand_holder, anchor="nw")
        self.cand_holder.columnconfigure(0, weight=1)
        self.cand_holder.bind(
            "<Configure>",
            lambda e: self.cand_canvas.configure(
                scrollregion=self.cand_canvas.bbox("all")))
        self.cand_canvas.bind(
            "<Configure>",
            lambda e: self.cand_canvas.itemconfigure(self._cand_window,
                                                    width=e.width))
        # ホイールは、この上にいるときだけ拾う(一覧のスクロールを食わない)
        self.cand_canvas.bind(
            "<Enter>",
            lambda e: self.cand_canvas.bind_all("<MouseWheel>", self._cand_wheel))
        self.cand_canvas.bind(
            "<Leave>", lambda e: self.cand_canvas.unbind_all("<MouseWheel>"))

        # **特別な選択肢は名簿と別枠に、横一列で置く。**名簿に入れると、
        # 出席者が多いときに下から押し出されて画面から切れる(9 人で
        # 「発言なし・雑音」が消えた・実機の指摘 2026-08-19)。
        # 別の行にすれば、伸び縮みするのは名簿側だけになる。
        self.special_holder = ttk.Frame(frm_cand)
        self.special_holder.grid(row=1, column=0, sticky="ew", padx=6,
                                 pady=(0, 4))

        # 直前の操作の結果。区間を移動しても消えないように専用の行にする
        # (一括適用が何区間に効いたのかが分からないと、事故に気づけない)
        # **row=2 を opts と共有していた**ので、あとから作る opts が上に
        # 重なってこの文字列が見えていなかった(実測 y=256..275 と y=250..275)。
        # 行を分けた。
        ttk.Label(frm_cand, textvariable=self.var_action, foreground="#1B5E20",
                  wraplength=560).grid(row=2, column=0, sticky="w", padx=8, pady=(2, 0))

        # **確定まわりは 2 行に分ける。**6 つを 1 行に詰めていたが、
        # 1920x1080 でも余りが +5px しかなく、1280x720@125% では 40px
        # 足りずに右端の［取り消し］が押し出されていた(実測 2026-08-29)。
        # 文字列を短くしても仕切りを動かしても、次に何か足せばまた壊れる
        # ——§10.3.6 で 3 度目を数えているのは、この行が常に限界だから。
        # **構造で余裕を作る。**切り替え(チェック)と操作(ボタン)は
        # 種類が違うので、分ける線としても素直。
        opts = ttk.Frame(frm_cand)
        opts.grid(row=3, column=0, sticky="ew", padx=6, pady=(2, 0))
        self.chk_apply_cluster = ttk.Checkbutton(
            opts, text="同じ声のまとまり全体に適用 (A)",
            variable=self.var_apply_cluster, takefocus=False)
        self.chk_apply_cluster.pack(side="left")
        ttk.Checkbutton(opts, text="確定したら次へ", variable=self.var_advance,
                        takefocus=False).pack(side="left", padx=12)
        ttk.Checkbutton(opts, text="足した発言も行で出す",
                        variable=self.var_added_rows, takefocus=False,
                        command=self._on_added_rows_toggled).pack(side="left")

        # **キー表示は削らない。**(U) (D) (Ctrl+Z) は操作を覚えるための
        # 手がかりで、幅のために落とすと可用性が下がる。
        opts_btn = ttk.Frame(frm_cand)
        opts_btn.grid(row=4, column=0, sticky="ew", padx=6, pady=(2, 6))
        self.btn_unknown = ttk.Button(
            opts_btn, text="不明 (U)", command=lambda: self.assign(SPECIAL_UNKNOWN))
        self.btn_unknown.pack(side="right")
        self.btn_unassign = ttk.Button(opts_btn, text="未確定に戻す (D)",
                                       command=self.unassign)
        self.btn_unassign.pack(side="right", padx=6)
        self.btn_undo = ttk.Button(opts_btn, text="取り消し (Ctrl+Z)",
                                   command=self.undo)
        self.btn_undo.pack(side="right")

        # **注記はチェックボックスの文字列に埋めない。**埋めたら 353px になり、
        # 右端の[取り消し]を押し出した(要 734px / 幅 713px。2026-08-21 に実測)。
        # §10.3.6・GPU の注記と同じ種類の失敗。**別の行に置く。**
        self.lbl_cluster_note = ttk.Label(frm_cand, foreground="#B26500",
                                          wraplength=700, justify="left", text="")
        self.lbl_cluster_note.grid(row=5, column=0, sticky="w",
                                   padx=8, pady=(0, 4))

        # --- 下部: ボタン ----------------------------------------------
        # **下の帯は 2 行に分ける。**10 個を 1 行に並べていたが、全部を
        # 数えると 100% で 1015px・125% で 1233px・150% で 1486px 要る。
        # 1024x768(帯の幅 984px)では 100% でも［元音声と照合］が、125% では
        # **［保存］が消えていた**(CI が実測・2026-08-29)。幅の検査は通って
        # いた——消えた部品は数に入らないので要求幅が小さく出るため。
        #
        # **3 行には分けない。**高さの余裕が 1280x720@150% で 0 行、
        # 1024x768@150% と 1366x768@150% で 1 行しかなく、2 行足すと
        # 上の「縮まない行」が切れる(実測)。
        #
        # 分ける線は編集系(左)と出力系(右)。出力は作業の終わりに押すもので、
        # 種類が違う。
        bottom = self.bottom_edit = ttk.Frame(self, padding=(10, 4, 10, 0))
        bottom.grid(row=3, column=0, sticky="ew")
        ttk.Button(bottom, text="出席者を編集...", command=self.edit_roster).pack(side="left")
        ttk.Button(bottom, text="残作業を一覧...", command=self.show_remaining).pack(side="left", padx=6)
        ttk.Button(bottom, text="語句をまとめて直す...",
                   command=self.replace_words).pack(side="left", padx=(0, 6))
        ttk.Button(bottom, text="話者をまとめて置き換える...",
                   command=self.replace_speaker).pack(side="left", padx=(0, 6))
        ttk.Button(bottom, text="このまとまりを未確定に戻す", command=self.unassign_cluster)\
            .pack(side="left")
        self.btn_inspect = ttk.Button(bottom, text="時刻を点検...",
                                      command=self.run_inspection)
        self.btn_inspect.pack(side="left", padx=6)
        # 一括適用は百件級になりうる。Ctrl+Z は話者専用、[元に戻す]は
        # 区間 1 つずつなので、まとめて戻す手段を別に置く
        self.btn_undo_bulk = ttk.Button(bottom, text="一括適用を取り消す",
                                        command=self.undo_bulk_times,
                                        state="disabled")
        self.btn_undo_bulk.pack(side="left")

        # 出力系。**［保存］はここ。**画面の右下という定位置を変えない。
        self.bottom_out = ttk.Frame(self, padding=(10, 0, 10, 10))
        self.bottom_out.grid(row=4, column=0, sticky="ew")
        ttk.Button(self.bottom_out, text="Word で出力...",
                   command=self.export_docx).pack(side="right")
        self.btn_save = ttk.Button(self.bottom_out, text="保存", command=self.save)
        self.btn_save.pack(side="right", padx=6)
        # 「この書面はこの録音から作った」の検算(SHA-256 の再計算と突き合わせ)
        ttk.Button(self.bottom_out, text="元音声と照合",
                   command=self.verify_source_audio).pack(side="right", padx=(0, 6))

        self.var_backend.set({
            "ffplay": "再生: ffplay(同梱)",
            "winsound": "再生: winsound(ffplay 未同梱のため簡易再生)",
            "none": "⚠ 再生できるプログラムが見つかりません(ffplay を同梱してください)",
        }.get(self.player.backend, ""))

    # ------------------------------------------------------------------
    def _bind_keys(self) -> None:
        self.bind("<space>", self._key_space)
        self.bind("<Return>", lambda e: self._key_enter())
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-s>", lambda e: self.save())
        self.bind("<Shift-Left>", lambda e: self._guarded(lambda: self._nudge_offset(-0.2)))
        self.bind("<Shift-Right>", lambda e: self._guarded(lambda: self._nudge_offset(+0.2)))
        self.bind("<Tab>", lambda e: self._guarded_break(self._goto_next_target))
        self.bind("<Shift-Tab>",
                  lambda e: self._guarded_break(lambda: self._goto_next_target(forward=False)))
        # X11 では Shift+Tab が ISO_Left_Tab として届く。
        # ただしこのキーシムを知らない Tk(Windows の一部のバージョン)では
        # bind した時点で TclError になり、割当画面が開かなくなる。
        # Shift+Tab 自体は上の <Shift-Tab> で拾えているので、
        # ここは「使えるなら足す」程度の扱いにする。
        try:
            self.bind("<ISO_Left_Tab>",
                      lambda e: self._guarded_break(lambda: self._goto_next_target(forward=False)))
        except tk.TclError:
            pass
        for ch in QUICK_KEYS:
            self.bind(ch, self._key_digit)
        for key, fn in (
            ("u", lambda: self.assign(SPECIAL_UNKNOWN)),
            ("d", self.unassign),
            ("a", self._toggle_cluster_mode),
            ("j", lambda: self.move(1)),
            ("k", lambda: self.move(-1)),
        ):
            self.bind(key, lambda e, f=fn: self._guarded(f))
            self.bind(key.upper(), lambda e, f=fn: self._guarded(f))

    def _typing(self) -> bool:
        """本文編集中はショートカットを無効にする"""
        return isinstance(self.focus_get(), (tk.Text, tk.Entry, ttk.Entry))

    def _guarded(self, fn) -> Optional[str]:
        if self._typing():
            return None
        fn()
        return "break"

    def _guarded_break(self, fn) -> Optional[str]:
        """Tab のように、本文編集中は既定動作(フォーカス移動)を残したいもの用。"""
        if self._typing():
            return None
        fn()
        return "break"

    def _on_offset_change(self) -> None:
        """ずれ補正の値が変わったら保存し、その場で聴き直せるようにする。"""
        try:
            value = float(self.var_offset.get())
        except (tk.TclError, ValueError):
            return
        value = max(-10.0, min(10.0, round(value, 2)))
        if abs(value - self.proj.time_offset) < 1e-6:
            return
        self.proj.time_offset = value
        self._dirty = True
        sign = "遅らせて" if value > 0 else "早めて"
        self._set_action(
            f"ずれ補正を {value:+.1f} 秒にしました"
            + (f"(音声を{sign}再生します)" if value else "(補正なし)")
        )
        self.show_current()
        if self.var_autoplay.get():
            self.play_current()

    def _set_offset(self, value: float) -> None:
        self.var_offset.set(round(value, 2))
        self._on_offset_change()

    def _nudge_offset(self, delta: float) -> None:
        self._set_offset(self.proj.time_offset + delta)

    def _set_action(self, message: str) -> None:
        """直前の操作の結果を、区間を移動しても消えない場所に表示する。"""
        self.var_action.set(message)

    def _key_space(self, event) -> Optional[str]:
        if self._typing():
            return None
        self.toggle_play()
        return "break"

    def _key_enter(self) -> Optional[str]:
        if self._typing():
            return None
        if self._candidates:
            self.assign(self._candidates[0].speaker.id)
        return "break"

    def _key_digit(self, event) -> Optional[str]:
        if self._typing():
            return None
        i = QUICK_KEYS.index(event.char)
        if i < len(self._candidates):
            self.assign(self._candidates[i].speaker.id)
        return "break"

    def has_real_clusters(self) -> bool:
        """一括適用の対象になる「声のまとまり」が 1 つでもあるか。

        ローカル転写には声を聞き分ける者がいないので、全区間が擬似クラスタ
        (?)になる。そのときは一括適用という機能自体が成り立たない。
        """
        return any(not s.is_pseudo_cluster for s in self.proj.segments)

    def _toggle_cluster_mode(self) -> None:
        if not self.has_real_clusters():
            self._set_action(
                "この作業ファイルには声のまとまりがありません"
                "(ローカル転写では作られません)。一括適用は使えません。")
            return
        self.var_apply_cluster.set(not self.var_apply_cluster.get())

    def _first_run_hint(self) -> None:
        messagebox.showinfo(
            "出席者を登録してください",
            "候補者リストが空です。\n\n"
            "「出席者を編集...」から、この会議の出席者を 1 行 1 人で入力してください。\n"
            "例:\n  佐藤(理事長)\n  田中(事務局長)\n  鈴木\n\n"
            "登録後は、区間を聴きながら数字キーで話者を確定していきます。",
            parent=self,
        )
        self.edit_roster()

    # ==================================================================
    # 表示更新
    # ==================================================================
    def refresh_all(self) -> None:
        self.suggester.refresh()
        self.reload_tree()
        self.update_status()
        self.show_current()

    def update_status(self) -> None:
        total = self.proj.total_count
        done = self.proj.assigned_count
        heard = self.proj.reviewed_count
        bulk = self.proj.unreviewed_count
        pct = (done / total * 100) if total else 0
        self.progress.configure(maximum=max(1, total), value=done)
        text = f"確定 {done}/{total} ({pct:.0f}%)  聴いて確定 {heard}"
        if bulk:
            text += f" / まとめて適用 {bulk}"
        self.var_status.set(text)

    # -------------------------------------------------- 絞り込みと移動
    def _match_filter(self, seg) -> bool:
        mode = self.var_filter.get()
        if mode == FILTER_UNASSIGNED:
            return not seg.speaker_id
        if mode == FILTER_UNREVIEWED:
            return not (seg.speaker_id and seg.reviewed)
        if mode == FILTER_CANDIDATES:
            # **待ち行列なので、済んだものだけの区間は出さない。**
            # 候補そのものは残っており、すべて表示にすれば「済」付きで見える。
            items = self._voice_candidates_for(seg)
            if not items:
                return False
            done = cand_mod.done_keys(
                items, [s for s in self.proj.segments
                        if self.proj.is_added_utterance(s)])
            return len(done) < len(items)
        return True

    # --- 候補の一覧(設計書 §10.3)------------------------------------
    def _load_voice_candidates(self) -> list:
        """区間の中の「別の声」を集める。話者分離が無ければ空。

        **これは検出器ではない。**適合 35/51・再現 31/34 で、候補が無い区間
        にも取りこぼしはある。判定は出さず、聴く場所を絞るためだけに使う。
        """
        try:
            turns = self._load_turns()
            if not turns:
                return []
            got = cand_mod.find_candidates(self.proj.segments, turns)
            return cand_mod.drop_dismissed(got, self._dismissed)
        except Exception:
            return []

    def _dismissed_path(self):
        return cand_mod.dismissed_path(
            self._work_dir(), self.proj.audio_fingerprint or "")

    def _voice_candidates_for(self, seg) -> list:
        return cand_mod.for_segment(self._voice_candidates, seg)

    def _refresh_voice_candidates(self, keep: bool = False) -> None:
        """候補を作り直して画面に反映する。keep=True なら選択位置を保つ。"""
        if not keep:
            self._voice_candidates = self._load_voice_candidates()
        self._show_voice_candidates()

    def _show_voice_candidates(self) -> None:
        """いまの区間の候補を、足す行の右側に出す(高さを増やさない)。"""
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        items = self._voice_candidates_for(seg)
        # **もう足した位置は「済」と印を付ける。隠さない。**隠すと
        # 「候補が無い＝やることが無い」と読まれ、この道具の性格
        # (印が無い＝安全ではない)と食い違う。済は後ろに回す。
        done = cand_mod.done_keys(
            items, [s for s in self.proj.segments
                    if self.proj.is_added_utterance(s)])
        items = sorted(items, key=lambda c: (c.key in done, c.at))
        self._current_voice_candidates = items
        if not items:
            self.frm_cand.pack_forget()             # 幅を空ける(排他)
            self.lbl_cand.pack(side="left", padx=(10, 0))
            self.cmb_cand.configure(values=[], state="disabled")
            self.var_cand.set("")
            self.btn_cand_add.configure(state="disabled")
            self.btn_cand_skip.configure(state="disabled")
            return
        self.lbl_cand.pack_forget()
        self.frm_cand.pack(side="left")
        labels = [("済" if c.key in done else "")
                  + f"{fmt_short_time(c.at)} {self._voice_label(c.speaker)}"
                  for c in items]
        self.cmb_cand.configure(values=labels, state="readonly")
        self.cmb_cand.current(0)
        self.btn_cand_add.configure(state="normal")
        self.btn_cand_skip.configure(state="normal")
        # **言い切らない。**3 割は空振りで、候補が無い区間にも脱落はある。
        # 件数は選択肢の一覧で分かる。**「空振りあり」は落とさない**——
        # 3 割は外れるので、書かないと「候補＝脱落」と読まれる。
        left = len(items) - len(done)
        self.lbl_cand_note.configure(
            text="空振りあり" if left == len(items) else f"残{left}・空振りあり")

    def _selected_voice_candidate(self):
        items = getattr(self, "_current_voice_candidates", [])
        i = self.cmb_cand.current()
        return items[i] if 0 <= i < len(items) else None

    def add_from_voice_candidate(self) -> None:
        """選んだ候補の位置に、カーソルを立てて小窓を開く(設計書 §10.3.3)。

        **話者は入れない。**候補が持っているのは turn の声(声B)であって
        名簿の誰かではない。機械が選んだ話者のまま押されると ✓ が立ち、
        「✓＝人が聴いて決めた」という意味が壊れる(CLAUDE.md)。
        どの声だったかは案内文で伝えるだけにする。
        """
        c = self._selected_voice_candidate()
        if c is None:
            return
        seg = self.proj.segments[self.current]
        self.add_utterance(
            initial_cut=self._cut_for_time(seg, c.at),
            initial_note=f"候補: {fmt_short_time(c.at)} に"
                         f"{self._voice_label(c.speaker)}が聞こえます。"
                         "位置は目安なので、聴いて直してください。")

    @staticmethod
    def _cut_for_time(seg: Segment, at: float) -> int:
        """時刻 → 本文の文字位置(文字数按分)。**目安であって実測ではない。**

        小窓が位置 → 時刻を出すのと同じ按分の逆。区間の中で話す速さが
        一定という粗い仮定なので、人が聴いて直す前提(設計書 §5.0.5)。
        """
        text = seg.text or ""
        span = max(1e-6, seg.end - seg.start)
        cut = int(round(len(text) * (at - seg.start) / span))
        return max(0, min(len(text), cut))

    def dismiss_voice_candidate(self) -> None:
        """× — この候補は違った、と記録する。**次からは出てこない。**

        空振りが 3 割あるので、捨てた判断が残らないと毎回出てきて使えない。
        """
        c = self._selected_voice_candidate()
        if c is None:
            return
        self._dismissed.append(c.key)
        cand_mod.save_dismissed(self._dismissed_path(), self._dismissed)
        self._voice_candidates = cand_mod.drop_dismissed(self._voice_candidates, [c.key])
        self._show_voice_candidates()
        if self.var_filter.get() == FILTER_CANDIDATES:
            self.reload_tree()
        self._set_action(
            f"{fmt_hms(c.at)} の候補を外しました(次からは出ません)。")

    def _load_listen_hints(self):
        """聴く順の sidecar を読む。無ければ None(並べ替えは出せない)。

        **これは検出器ではない。**再現率は約 4 割で、順番が付かない区間にも
        取りこぼしはある。判定は出さず、並び順としてだけ使う。
        """
        try:
            hints = listen_order.load_hints(listen_order.hints_path(
                self._work_dir(), self.proj.audio_fingerprint or ""))
            if not hints:
                return None
            return listen_order.match(hints, self.proj.segments) or None
        except Exception:
            return None

    def _refresh_inserted(self) -> None:
        """「どの区間に、どの追加発話が入るか」を数え直す。

        **Word の出力とまったく同じ計算を使う**(segments.inserted_marks)。
        別々に実装すると「画面ではここ、出力では別のところ」になる。
        """
        self._marks_by_parent, placed = inserted_marks(self.proj)
        self._inlined = {s.index for s in self.proj.segments
                         if self.proj.is_added_utterance(s) and id(s) in placed}

    def inserted_in(self, seg) -> list:
        """その区間に重なって入っている発話。[(何文字目, 区間), ...]"""
        return marks_for_segment(getattr(self, "_marks_by_parent", {}), seg)

    def _visible_indexes(self) -> list[int]:
        self._refresh_inserted()
        # **足した発言に行を作らない。**割り込んだ位置で親の本文に埋め込む
        # ので、行が増えると同じものが 2 か所に出る。行として見たいときは
        # 下のチェックで戻せる(実機の要望・2026-08-24)。
        hide = (not self.var_added_rows.get()) and self._inlined
        if self.var_filter.get() == FILTER_ALL:
            out = [s.index for s in self.proj.segments]
        else:
            out = [s.index for s in self.proj.segments if self._match_filter(s)]
        if hide:
            out = [i for i in out if i not in self._inlined]
        if self.var_listen_order.get() and self._listen_hints:
            # 点数の高い順、同点は時間順。順番が付かない区間(分割で増えた側
            # など)は末尾に時間順で置く——**外すと「安全」に見えてしまう。**
            hints = self._listen_hints
            out.sort(key=lambda i: (
                -(hints[i].score if i in hints else -1),
                self.proj.segments[i].start))
        return out

    def _remaining_count(self) -> int:
        """今の絞り込み基準で、まだ手を付けていない区間の数。"""
        if self.var_filter.get() == FILTER_UNREVIEWED:
            return self.proj.total_count - self.proj.reviewed_count
        return self.proj.total_count - self.proj.assigned_count

    def _next_target(self, from_index: int, forward: bool = True) -> Optional[int]:
        """次に処理すべき区間。絞り込みが『未確認のみ』なら未確認を、
        それ以外なら未確定を探す。"""
        if self.var_filter.get() == FILTER_UNREVIEWED:
            return next_unreviewed(self.proj, from_index, forward)
        return next_unassigned(self.proj, from_index, forward)

    def _on_filter_change(self) -> None:
        self.reload_tree()
        vis = self._visible_indexes()
        if vis and self.current not in vis:
            self.goto(vis[0])
        label = dict((v, k) for k, v in FILTER_LABELS).get(self.var_filter.get(), "")
        if self.var_filter.get() != FILTER_CANDIDATES:
            self._set_action(f"表示: {label}({len(vis)} 区間)")
            return
        # **言い切らない。**適合 35/51 で 3 割は空振り、再現 31/34 なので
        # 候補の無い区間にも取りこぼしはある。「ここだけ見れば済む」とは
        # 読ませない(listen_order と同じ線)。
        if not self._voice_candidates:
            self._set_action(
                "候補を出せません。話者分離を通した作業ファイルでだけ使えます。")
            return
        self._set_action(
            f"別の声がある区間 {len(vis)}/{self.proj.total_count} を出しました"
            f"(候補 {len(self._voice_candidates)} 箇所)。"
            "3 割ほどは空振りです。候補の無い区間にも取りこぼしはあります。")

    def _on_listen_order_change(self) -> None:
        self.reload_tree()
        if not self.var_listen_order.get():
            self._set_action("時間順に戻しました。")
            return
        hints = self._listen_hints or {}
        high = sum(1 for h in hints.values() if h.is_high)
        # **言い切らない。**実測の再現率は約 4 割で、順番が付かない区間や
        # 下位の区間にも取りこぼしはある。「上位だけ聴けば済む」とは
        # 読ませない。
        self._set_action(
            f"聴く順に並べました(手がかりの強い区間 {high}/{len(hints)})。"
            "上位から聴くと取りこぼしを見つけやすくなります。"
            "下位や順番の無い区間にも取りこぼしはあります。")

    def reload_tree(self) -> None:
        sel = self.current
        self.tree.delete(*self.tree.get_children())
        self._row_ids = []
        for idx in self._visible_indexes():
            seg = self.proj.segments[idx]
            self._row_ids.append(self._insert_row(seg))
        self._draw_timeline()
        self._select_index(sel, scroll=True)

    def _row_values(self, seg) -> tuple[tuple, tuple]:
        name = self.proj.speaker_name(seg.speaker_id)
        mark = ""
        bg = fg = ""
        if not seg.speaker_id:
            bg = "bg_unassigned"
        else:
            if seg.speaker_id in (SPECIAL_UNKNOWN, SPECIAL_MULTI, SPECIAL_NOISE):
                fg = "fg_special"
            if not seg.reviewed:
                bg = "bg_bulk"
                mark = "△"      # まとめて適用しただけ(自分の耳では未確認)
            else:
                mark = "✓"
        # 時刻をどこまで人が確かめたかも見えるようにする(話者の ✓/△ と同じ)
        time_mark = ""
        if seg.time_edited:
            time_mark = "✎ " if seg.time_reviewed else "✎△"
        time_cell = time_mark + fmt_hms(seg.start)
        # 人が足した発話は本文の頭に印を付ける。色だけだと印刷や
        # 色覚の条件で消えるので、文字でも分かるようにする。
        body = self._body_with_inserts(seg)
        if self.proj.is_added_utterance(seg):
            # **足した発話は背景・文字色・括弧の 3 つで示す。**
            # 「＋」だけでは一覧に埋もれて分からない(実機の指摘・2026-08-22)。
            bg, fg = "bg_added", "fg_added"
            body = f"＋（{body}）"
        elif not self.var_added_rows.get() and self.inserted_in(seg):
            # 重なりを埋め込んだ行にも印を付ける。**括弧が主、色は従。**
            # 色だけでは印刷や色覚の条件で消える。
            bg = "bg_added"
        values = (time_cell, seg.cluster_label,
                  f"{mark}{name}" if name else "—", body)
        return values, tuple(t for t in (bg, fg) if t)

    def _refresh_inserted_row(self, seg) -> None:
        """「この発言に重なって入っているもの」を右ペインに出す。

        埋め込み表示のときは一覧に行が無いので、**ここが唯一の入口**になる。
        """
        for w in self._inserted_widgets:
            w.destroy()
        self._inserted_widgets = []

        marks = [] if self.var_added_rows.get() else self.inserted_in(seg)
        if not marks:
            self.row_inserted.pack_forget()
            return
        self.row_inserted.pack(side="top", anchor="w", pady=(3, 0))

        lbl = ttk.Label(self.row_inserted, text="重なって入っている発言:",
                        foreground="#BF360C")
        lbl.pack(side="left")
        self._inserted_widgets.append(lbl)
        for _cut, add in marks:
            who = self.proj.speaker_name(add.speaker_id) or "？"
            box = ttk.Frame(self.row_inserted)
            box.pack(side="left", padx=(8, 0))
            self._inserted_widgets.append(box)
            ttk.Label(box, text=f"（{who}：{add.preview(16)}）"
                                f" {fmt_hms_frac(add.start)}").pack(side="left")
            ttk.Button(box, text="直す", width=5, takefocus=False,
                       command=lambda i=add.index: self._goto_inserted(i)
                       ).pack(side="left", padx=(4, 0))
            ttk.Button(box, text="消す", width=5, takefocus=False,
                       command=lambda i=add.index: self._remove_inserted(i)
                       ).pack(side="left", padx=(2, 0))

    def _goto_inserted(self, index: int) -> None:
        """重なって入っている発話へ移る。**行が無くても届くようにする。**"""
        if not (0 <= index < len(self.proj.segments)):
            return
        if not self.var_added_rows.get():
            # 行を出さない設定のままでは選べないので、その場で行表示へ倒す
            self.var_added_rows.set(True)
            self.reload_tree()
        self.goto(index)
        self._set_action(
            "重なって入っている発言に移りました。"
            "本文と話者を直せます（［足した発言も行で出す］を外すと、"
            "また埋め込みに戻ります）。")

    def _remove_inserted(self, index: int) -> None:
        """重なって入っている発話を消す。"""
        if not (0 <= index < len(self.proj.segments)):
            return
        here = self.current
        self.current = index
        try:
            self.remove_added()
        finally:
            self.current = min(here, len(self.proj.segments) - 1)
        self.reload_tree()
        self._select_index(self.current, scroll=True)
        self.show_current()

    def _on_added_rows_toggled(self) -> None:
        """行で出すかを切り替える。いま見ている区間は見失わない。"""
        here = self.current
        self.reload_tree()
        self._select_index(here, scroll=True)
        self.show_current()

    def _body_with_inserts(self, seg) -> str:
        """一覧に出す本文。**重なって入った発話を、その位置に埋め込む。**

        Word の「埋め込む形」と同じ書き方(BTSJ 3.2.3)。一覧だけ別の見せ方に
        すると、画面と出力が食い違う。
        """
        marks = self.inserted_in(seg) if not self.var_added_rows.get() else []
        if not marks:
            return seg.preview(70)
        text = seg.text or ""
        out, pos = [], 0
        for cut, add in marks:
            cut = max(0, min(int(cut), len(text)))
            out.append(text[pos:cut])
            who = self.proj.speaker_name(add.speaker_id) or "？"
            out.append(f"（{who}：{(add.text or '').strip()}）")
            pos = cut
        out.append(text[pos:])
        joined = "".join(out)
        return joined if len(joined) <= 70 else joined[:69] + "…"

    def _insert_row(self, seg) -> str:
        values, tags = self._row_values(seg)
        return self.tree.insert("", "end", iid=f"s{seg.index}", values=values, tags=tags)

    def _update_row(self, index: int) -> None:
        iid = f"s{index}"
        if not self.tree.exists(iid):
            return
        values, tags = self._row_values(self.proj.segments[index])
        self.tree.item(iid, values=values, tags=tags)

    def _speaker_color(self, speaker_id: Optional[str]) -> str:
        if not speaker_id:
            return COLOR_UNASSIGNED
        if speaker_id in (SPECIAL_UNKNOWN, SPECIAL_MULTI, SPECIAL_NOISE):
            return COLOR_SPECIAL
        for i, sp in enumerate(self.proj.speakers):
            if sp.id == speaker_id:
                return PALETTE[i % len(PALETTE)]
        return COLOR_SPECIAL

    def _draw_timeline(self) -> None:
        c = self.canvas
        c.delete("all")
        segs = self.proj.segments
        if not segs:
            return
        w = max(1, c.winfo_width())
        h = int(c.winfo_height())
        total = max(1e-6, self.proj.duration or segs[-1].end)
        for seg in segs:
            x0 = seg.start / total * w
            x1 = max(x0 + 1.0, seg.end / total * w)
            c.create_rectangle(x0, 4, x1, h - 10, fill=self._speaker_color(seg.speaker_id),
                               outline="", tags="seg")
        cur = segs[min(self.current, len(segs) - 1)]
        x = cur.start / total * w
        c.create_line(x, 0, x, h, fill="#D32F2F", width=2)
        c.create_text(4, h - 5, anchor="sw", text="0:00", fill="#888", font=("", 7))
        c.create_text(w - 4, h - 5, anchor="se", text=fmt_hms(total), fill="#888", font=("", 7))

    def _on_canvas_click(self, event) -> None:
        segs = self.proj.segments
        if not segs:
            return
        w = max(1, self.canvas.winfo_width())
        total = max(1e-6, self.proj.duration or segs[-1].end)
        t = event.x / w * total
        best = min(range(len(segs)), key=lambda i: abs(segs[i].start - t))
        self.goto(best)

    # ==================================================================
    # 現在区間
    # ==================================================================
    def _select_index(self, index: int, scroll: bool = False) -> None:
        if not self.proj.segments:
            return
        index = max(0, min(index, len(self.proj.segments) - 1))
        self.current = index
        iid = f"s{index}"
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            if scroll:
                self.tree.see(iid)

    def selected_indices(self) -> list[int]:
        """いま選ばれている区間の番号（画面の並び順）。

        複数選択に対応したので、割当は**選ばれている全部**に効く。
        1 つだけ選んでいるときは、これまでどおり 1 件だけ。
        """
        out: list[int] = []
        for iid in self.tree.selection():
            try:
                i = int(iid[1:])
            except ValueError:
                continue
            if 0 <= i < len(self.proj.segments):
                out.append(i)
        return sorted(out)

    def _on_tree_select(self, event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        # **右ペインに出すのは「いま focus のある行」。**複数選んでいる
        # ときに先頭へ飛ぶと、聴きながら選べない。
        focused = self.tree.focus() or sel[0]
        try:
            index = int(focused[1:])
        except ValueError:
            return
        if index == self.current:
            return
        self._commit_text()
        self.current = index
        self.show_current()
        if self.var_autoplay.get():
            self.play_current()

    def goto(self, index: int) -> None:
        self._commit_text()
        self._select_index(index, scroll=True)
        self.show_current()
        if self.var_autoplay.get():
            self.play_current()

    def goto_key(self, key: tuple[float, float]) -> Optional[int]:
        """区間の鍵（segment_key）で指した区間へ飛ぶ。飛べたら区間番号。

        語句をまとめて直す画面（ReplaceWordsDialog）が、ヒットの行を選んだ
        ときに呼ぶ。**番号ではなく鍵で指す**——番号は分割・結合で振り直る。
        絞り込みで隠れていれば「すべて表示」に戻してから飛ぶ（jump_to_time
        と同じ）。飛んだあと再生するかは「移動したら自動再生」に従う（goto）。
        """
        want = (round(float(key[0]), 3), round(float(key[1]), 3))
        seg = next((s for s in self.proj.segments if segment_key(s) == want), None)
        if seg is None:
            return None
        if seg.index not in self._visible_indexes():
            self.var_filter.set(FILTER_ALL)
            self.var_listen_order.set(False)
            self.reload_tree()
            self._set_action("絞り込みを「すべて表示」に戻して飛びました。")
        self.goto(seg.index)
        return seg.index

    def jump_to_time(self) -> None:
        """入力された時刻に一番近い区間へ飛ぶ。

        「25:02」「1502」「00:25:02.8」のどれでも受ける(parse_hms が吸収する)。
        **絞り込みで隠れていても飛ぶ**——探しに来た区間が出ていないと意味が
        ないので、必要なら絞り込みを「すべて表示」へ戻す。
        """
        raw = self.var_jump.get().strip()
        if not raw or not self.proj.segments:
            return
        try:
            at = parse_hms(raw)
        except Exception:
            self._set_action(
                f"時刻が読めません: {raw}"
                "（00:25:02 / 25:02 / 1502 のどれかで入れてください）")
            return
        # その時刻を含む区間があればそれ、無ければ一番近いもの
        hit = next((s for s in self.proj.segments if s.start <= at < s.end), None)
        if hit is None:
            hit = min(self.proj.segments, key=lambda s: abs(s.start - at))
        if hit.index not in self._visible_indexes():
            self.var_filter.set(FILTER_ALL)
            self.var_listen_order.set(False)
            self.reload_tree()
            self._set_action("絞り込みを「すべて表示」に戻して飛びました。")
        self.goto(hit.index)
        self._set_action(
            f"{fmt_hms_frac(at)} → {fmt_hms(hit.start)} の区間へ飛びました。")

    def move(self, delta: int) -> None:
        """一覧上で delta 件ぶん移動する。

        現在位置が絞り込みで隠れている場合は、まず進行方向の最寄りに
        「吸い付く」だけにする(そこから更に delta 進めると 1 件飛ばしになる)。
        """
        vis = self._visible_indexes()
        if not vis:
            return
        if self.current in vis:
            pos = vis.index(self.current)
            target = vis[max(0, min(pos + delta, len(vis) - 1))]
        elif delta >= 0:
            later = [i for i in vis if i > self.current]
            target = later[0] if later else vis[-1]
        else:
            earlier = [i for i in vis if i < self.current]
            target = earlier[-1] if earlier else vis[0]
        self.goto(target)

    def _goto_next_target(self, forward: bool = True) -> None:
        """次に手を付けるべき区間へ。絞り込みの基準に従う。"""
        nxt = self._next_target(self.current, forward)
        if nxt is None:
            self.bell()
            direction = "この先" if forward else "ここより前"
            remaining = self._remaining_count()
            if remaining:
                self._set_action(f"{direction}に対象はありません(全体ではあと {remaining} 区間)。")
            else:
                self._set_action("残りの区間はありません。")
        else:
            self.goto(nxt)

    def show_current(self) -> None:
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        self._update_seginfo()
        self._show_times()
        self.txt_body.delete("1.0", "end")
        self.txt_body.insert("1.0", seg.text)
        self._body_index = self.current   # この欄がどの区間のものか（§_commit_text）
        self._refresh_inserted_row(seg)  # 重なって入っている発話(§10.3.7)
        self._rebuild_candidates()      # 話者の候補(suggest.py)
        self._show_voice_candidates()   # 声の候補(candidates.py・§10.3)
        self._draw_timeline()
        # **人が明示的に消すのは許す。**禁じているのは自動削除で、音声を
        # 聴いて機械の重複だと判断した人が消すのは別のこと(実機の指摘・
        # 2026-08-22)。転写が出した区間を消したときは中身を履歴に残す。
        if hasattr(self, "btn_del_added"):
            self.btn_del_added.configure(state="normal")

    def _update_seginfo(self) -> None:
        """区間ヘッダ(右ペイン最上段)を書き直す。"""
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        pos = self.current + 1
        state = "未確定"
        if seg.speaker_id:
            state = "確認済み" if seg.reviewed else "△まとめて適用(未確認)"
        # 時刻を直した区間は、確認した終了時刻まで丸ごと鳴らす(先頭だけにしない)
        plen = preview_length(seg.text, seg.duration)
        long_note = (
            "" if seg.time_edited
            else (f"(再生は先頭{plen:.0f}秒)" if plen < seg.duration - 0.5 else "")
        )
        self.var_seginfo.set(
            f"区間 {pos}/{len(self.proj.segments)}   "
            f"[{fmt_hms(seg.start)} → {fmt_hms(seg.end)}]  {seg.duration:.0f}秒{long_note}   "
            f"{seg.cluster_label}({self.suggester.cluster_summary(seg.cluster)})   "
            f"{state}"
            + ("" if not seg.time_edited
               else "   ✎時刻を修正済み" if seg.time_reviewed
               else "   ✎△時刻は推定(未確認)")
            + (f"   ずれ補正 {self.proj.time_offset:+.1f}秒"
               if self.proj.time_offset and not seg.time_edited else "")
        )

    # ==================================================================
    # 区間の時刻を直す
    # ==================================================================
    def _show_times(self) -> None:
        """時刻の入力欄を、いまの区間の値に合わせる。"""
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        start, end = time_edit_base(seg, self.proj.time_offset)
        self.var_start.set(fmt_hms_frac(start))
        self.var_end.set(fmt_hms_frac(end))
        self._time_index = self.current   # この欄がどの区間のものか（§_commit_time）
        self.btn_revert_time.state(["!disabled"] if seg.time_edited else ["disabled"])
        # 確認済みの区間で押しても意味がない(それ以外は押せる。まだ直して
        # いない区間でも「聴いてこの時刻で合っている」と言えるため)
        done = seg.time_edited and seg.time_reviewed
        self.btn_confirm_time.state(["disabled"] if done else ["!disabled"])

    def _discard_time_edit(self, which: str) -> None:
        """欄から離れたときの後始末。**書かずに表示だけ元へ戻す。**

        打ちかけの値をそのまま残すと、次に Enter を押したときに別の区間へ
        入る。捨てたことを黙らない（打った本人は入ったつもりでいる）。
        """
        if not self.proj.segments:
            return
        var = self.var_start if which == "start" else self.var_end
        typed = var.get()
        self._show_times()
        if typed.strip() != var.get().strip():
            self._set_action(
                "時刻は変えていません。変えるには Enter か"
                "［この時刻で確認］を押してください。")

    def _commit_time(self, which: str, explicit: bool = False) -> None:
        """入力欄の値を区間に反映する。読めない値は元に戻して知らせる。

        **この欄がどの区間のものかを確かめてから書く。**確かめずに書くと、
        区間を移った直後に前の区間の値が入る（本文欄で同じ穴を塞いだのに
        時刻欄に残っていた・2026-08-19）。
        """
        if not self.proj.segments:
            return
        loaded = getattr(self, "_time_index", None)
        if loaded is not None and loaded != self.current:
            self._show_times()
            return
        var = self.var_start if which == "start" else self.var_end
        try:
            value = parse_hms(var.get())
        except ValueError:
            self._show_times()
            self._set_action("時刻の形式が読めませんでした(例 00:43:51.5)。元に戻します。")
            return
        # 打たれた時刻はそのまま置く(反対側を追い越すなら区間ごとそこへ移す)
        self._apply_time(which, value, explicit=explicit, shift_if_past=True)

    def _nudge_time(self, which: str, delta: float) -> None:
        """(−1)(−0.1)(+0.1)(+1) ボタン。入力欄に打ちかけの値があればそこから動かす。"""
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        var = self.var_start if which == "start" else self.var_end
        try:
            current = parse_hms(var.get())
        except ValueError:
            base = time_edit_base(seg, self.proj.time_offset)
            current = base[0] if which == "start" else base[1]
        self._apply_time(which, current + delta, explicit=True)

    def _shift_time(self, delta: float) -> None:
        """区間ごと前後にずらす(長さは変えない)。ずれを直す主な操作。"""
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        base_start, base_end = time_edit_base(seg, self.proj.time_offset)
        start, end = shift_span(base_start, base_end, delta, self.proj.duration)
        self._commit_times(start, end, base_start, base_end, explicit=True)

    def _apply_time(self, which: str, value: float, explicit: bool = False,
                    shift_if_past: bool = False) -> None:
        seg = self.proj.segments[self.current]
        base_start, base_end = time_edit_base(seg, self.proj.time_offset)
        start, end = move_edge(base_start, base_end, which, value,
                               self.proj.duration, shift_if_past=shift_if_past)
        # 反対側を追い越して区間ごと動いたときは、頭から聴かせる
        moved = which
        if shift_if_past and abs(start - base_start) > 1e-6 \
                and abs(end - base_end) > 1e-6:
            moved = "both"
        self._commit_times(start, end, base_start, base_end, explicit, moved)

    def _commit_times(self, start: float, end: float, base_start: float,
                      base_end: float, explicit: bool,
                      moved: str = "both") -> None:
        seg = self.proj.segments[self.current]
        changed = abs(start - base_start) > 1e-6 or abs(end - base_end) > 1e-6
        if not changed and (seg.time_edited or not explicit):
            # 入力欄を通り過ぎただけ。勝手に「修正済み」にはしない。
            # (未編集の区間で Enter を押したときだけは、ずれ補正込みの値を確定する)
            self._show_times()
            return

        # **他の区間を丸ごとまたぐ長さになるなら、黙って通さない。**
        # 打ち間違いで 15.8 秒の区間が 2125 秒になり、290 区間をまたいだ
        # (実機・2026-08-19)。形式としては正しい時刻なので、書式の検査では
        # 止まらない。**長さの筋が通っているかは、ここでしか見られない。**
        if not self._confirm_wide_span(seg, start, end):
            self._show_times()
            return

        self._write_times(seg, start, end, reviewed=True)
        tail_only = moved == "end"
        self._set_action(
            f"区間 {seg.index + 1} の時刻を {fmt_hms_frac(seg.start)} → "
            f"{fmt_hms_frac(seg.end)} にしました。"
            + (f"(終わりの{TAIL_PREVIEW_SECONDS:.0f}秒を鳴らします)"
               if tail_only and self.var_autoplay.get() else "")
        )
        if self.var_autoplay.get():
            if tail_only:
                self.play_tail()
            else:
                self.play_current()

    # またぐ区間がこの数以上なら確認する。1 つ重なるのは相づちなどで普通に
    # 起きるが、2 つ以上を丸ごと飲み込むのは打ち間違いの疑いが濃い。
    WIDE_SPAN_SEGMENTS = 2
    # 短い区間の微調整で毎回聞かれないよう、長さの下限も置く
    WIDE_SPAN_SECONDS = 60.0

    def _confirm_wide_span(self, seg: Segment, start: float,
                           end: float) -> bool:
        """他の区間をまたぐ長さなら確認する。続けてよければ True。"""
        if end - start < self.WIDE_SPAN_SECONDS:
            return True
        covered = [s for s in self.proj.segments
                   if s.index != seg.index and start < s.start < end]
        if len(covered) < self.WIDE_SPAN_SEGMENTS:
            return True
        br = chr(10)
        return bool(messagebox.askyesno(
            "確認",
            f"この区間の長さが {end - start:.0f} 秒になり、"
            f"ほかの {len(covered)} 区間を丸ごとまたぎます。{br}{br}"
            f"{fmt_hms_frac(start)} → {fmt_hms_frac(end)}{br}{br}"
            "打ち間違いではありませんか?",
            parent=self, default="no"))

    def play_tail(self) -> None:
        """いまの区間の終わりだけを鳴らす。

        終了時刻を直したときの確認用。確かめたいのは語尾が切れていないかで、
        長い区間を頭から鳴らすと、そこへ来るまで待たされる。
        """
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        start, end = tail_window(seg, self.proj.time_offset)
        self.play_span(start, end, pre_roll=0.0)

    def _write_times(self, seg: Segment, start: float, end: float, *,
                     reviewed: bool, refresh: bool = True,
                     log: bool = True) -> None:
        """区間に時刻を書く、ただ 1 つの経路。

        画面で直したときも、点検の提案を当てたときも必ずここを通す。
        経路を分けると、どちらかで ✎/✎△ の意味が食い違っても気づけない。

        reviewed: その時刻を人が耳で確かめたか。画面で決めた時刻は True、
        点検の提案をまとめて当てただけなら False(✎△ のまま残す)。
        refresh: 画面を描き直すか。まとめて適用では 1 件ごとに描き直すと
        百件級で待たされるので、呼び出し側が最後に 1 回だけ描き直す。
        """
        # **必ずデータ層を通す**(編集履歴設計書 §1.3)。値が同じでも
        # ✎△ → ✎ は「確認した」という変化なので記録される。
        # `_log=False` はまとめて適用のとき——呼び出し側が 1 件にまとめる。
        self.proj.edit_time(seg.index, start, end, reviewed, _log=log)
        self._dirty = True
        if not refresh:
            return
        if seg.index == self.current:
            self._show_times()
            self._update_seginfo()
        self._update_row(seg.index)
        self._draw_timeline()

    def confirm_time(self) -> None:
        """いま出ている時刻を「自分の耳で確かめた」ものとして確定する。

        点検が当てただけの時刻(✎△)を ✎ に上げるための操作。時刻の値は
        変えない(合っているから押している)。再生と組み合わせて
        「聴く → 合っていればキー一発」で流せるようにしてある。
        """
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        if seg.time_edited and seg.time_reviewed:
            self._set_action("この区間の時刻はすでに確認済みです。")
            return
        start, end = audio_span(seg, self.proj.time_offset)
        self._write_times(seg, start, end, reviewed=True)
        self._set_action(
            f"区間 {seg.index + 1} の時刻を確認済みにしました"
            f"(✎ {fmt_hms_frac(seg.start)} → {fmt_hms_frac(seg.end)})。"
        )

    def plan_proposals(
        self, proposals: Iterable[Proposal],
    ) -> tuple[list[tuple[Segment, float, float, Proposal]], list[Proposal]]:
        """提案の行き先を先に全部解く(適用順序で結果が変わらないようにする)。

        1 件ずつ隣に合わせて切り詰めながら書くと、まとめて適用の結果が
        適用順序に依存する。ドリフト帯を前から順に当てると、まだ動いて
        いない隣の古い位置に切り詰められて、後の提案ほど潰れる
        (+7 秒帯の再現実験では昇順で 4 件中 3 件が適用不可になった)。

        そこで隣の位置は「同じバッチに提案があればその提案値、なければ
        いまの位置」として先に全件を解き、書き込みはそのあと一括で行う。

        戻り値: (書き込む計画, 当てられなかった提案)。
        """
        resolved: dict[int, tuple[Segment, Proposal]] = {}
        failed: list[Proposal] = []
        for p in proposals:
            seg = target_segment(self.proj, p)
            # 人が耳で確定した時刻は上書きしない(§3.4)。提案の生成後に
            # ユーザーが確定した場合もここで止まる。
            if seg is None or seg.time_reviewed or seg.index in resolved:
                failed.append(p)
                continue
            resolved[seg.index] = (seg, p)

        def dest_end(i: int) -> Optional[float]:
            """区間 i の「行き先の終了」。バッチ内なら提案値、外ならいまの値。"""
            if i < 0:
                return None
            if i in resolved:
                seg, p = resolved[i]
                return float(p.payload.get("end", seg.end))
            return audio_span(self.proj.segments[i], self.proj.time_offset)[1]

        def dest_start(i: int) -> Optional[float]:
            if i >= len(self.proj.segments):
                return None
            if i in resolved:
                seg, p = resolved[i]
                return float(p.payload.get("start", seg.start))
            return audio_span(self.proj.segments[i], self.proj.time_offset)[0]

        planned: list[tuple[Segment, float, float, Proposal]] = []
        for i in sorted(resolved):
            seg, p = resolved[i]
            start = float(p.payload.get("start", seg.start))
            end = float(p.payload.get("end", seg.end))
            clipped = clip_to_neighbours(start, end,
                                         dest_end(i - 1), dest_start(i + 1))
            if clipped is None:
                failed.append(p)    # 切り詰めると潰れる。当てないほうがいい
                continue
            start, end = clamp_times(*clipped, self.proj.duration, moved="end")
            planned.append((seg, start, end, p))
        return planned, failed

    def apply_proposal(self, proposal: Proposal, *, reviewed: bool) -> bool:
        """点検の提案を 1 件当てる。当てられなければ False を返す。

        書き込みは画面の時刻編集と同じ _write_times を通す。
        """
        planned, _ = self.plan_proposals([proposal])
        if not planned:
            return False
        seg, start, end, _p = planned[0]
        self._write_times(seg, start, end, reviewed=reviewed)
        return True

    def apply_proposals_bulk(
        self, proposals: Iterable[Proposal],
    ) -> tuple[list[Proposal], list[Proposal]]:
        """提案をまとめて当てる(すべて ✎△。人の耳の確認は後から)。

        戻り値: (当てられた提案, 当てられなかった提案)。status もここで更新する。
        当てる前の時刻を控えておき、まとめて元に戻せるようにする。
        """
        planned, failed = self.plan_proposals(proposals)
        if planned:
            # 当てる前の状態を控える。百件を 1 件ずつ戻すのは現実的でない
            self._time_undo = {
                "total": len(self.proj.segments),
                "items": [(seg.index, seg.start, seg.end,
                           seg.time_edited, seg.time_reviewed)
                          for seg, _s, _e, _p in planned],
            }
        # **1 回の判断は 1 件の記録**(編集履歴設計書 §1.1)。百件を
        # 1 件ずつ残すと「何回判断したか」が件数の山に埋もれる。
        self.proj.apply_times_to(
            [(seg.index, start, end) for seg, start, end, _p in planned],
            reviewed=False)
        for seg, start, end, p in planned:
            self._write_times(seg, start, end, reviewed=False, refresh=False,
                              log=False)
            p.status = "accepted"
        if planned:
            self.reload_tree()
            self._show_times()
            self._update_seginfo()
            self._sync_time_undo_button()
        return [p for _, _, _, p in planned], failed

    def undo_bulk_times(self) -> None:
        """直前の「まとめて適用」を丸ごと元に戻す。

        Ctrl+Z は話者専用(スナップショットに時刻を持たない)、[元に戻す]は
        区間 1 つずつなので、百件級の一括適用を戻す手段が別に要る。
        """
        snap = self._time_undo
        if not snap:
            self._set_action("取り消せる一括適用がありません。")
            return
        if snap["total"] != len(self.proj.segments):
            # 分割・結合で区間の並びが変わっている。index で戻すと別の区間を
            # 壊すので、戻さずに知らせる
            self._time_undo = None
            self._sync_time_undo_button()
            messagebox.showinfo(
                "取り消せません",
                "一括適用のあとで区間を分割または結合したため、"
                "まとめて元に戻すことはできません。\n"
                "区間ごとの[元に戻す]をお使いください。",
                parent=self,
            )
            return
        # **戻したことも履歴に残す**(編集履歴設計書 §1.4)。
        self.proj.restore_times(snap["items"])
        self._dirty = True
        self._time_undo = None
        self.reload_tree()
        self._show_times()
        self._update_seginfo()
        self._sync_time_undo_button()
        self._set_action(f"{len(snap['items'])} 区間の一括適用を元に戻しました。")

    def _sync_time_undo_button(self) -> None:
        state = ["!disabled"] if self._time_undo else ["disabled"]
        self.btn_undo_bulk.state(state)

    def revert_time(self) -> None:
        """パイプラインが出した元の時刻に戻す(以後はまたずれ補正が効く)。"""
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        if not self.proj.revert_time(self.current):
            self._set_action("この区間の時刻は直されていません。")
            return
        self._dirty = True
        self._show_times()
        self._update_seginfo()
        self._update_row(seg.index)
        self._draw_timeline()
        self._set_action(
            f"区間 {seg.index + 1} の時刻を元に戻しました"
            f"({fmt_hms_frac(seg.start)} → {fmt_hms_frac(seg.end)})。"
        )
        if self.var_autoplay.get():
            self.play_current()

    # ==================================================================
    # 区間を分ける・束ねる
    # ==================================================================
    def _remap_undo_for_split(self, index: int) -> None:
        """index の区間が 2 つになった。それより後ろを指す取り消しを 1 つずらす。"""
        for snapshot in self._undo:
            for i, (idx, sid, reviewed) in enumerate(snapshot):
                if idx > index:
                    snapshot[i] = (idx + 1, sid, reviewed)

    def _remap_undo_for_merge(self, index: int) -> None:
        """index と index+1 が 1 つになった。消えた側を指す取り消しは捨てる。

        結合先へ付け替えると、1 つのスナップショットに同じ index が 2 つ並び、
        どちらが復元されるかが並び順で決まってしまう。しかも取り消しで結合区間
        全体が後側の話者になり、✓ まで付く。それより「消えた区間ぶんは戻せない」
        ほうが安全(結合をやり直したければ分割し直せる)。
        """
        kept: list[list[tuple[int, Optional[str], bool]]] = []
        for snapshot in self._undo:
            fixed = [
                (idx - 1 if idx > index + 1 else idx, sid, reviewed)
                for idx, sid, reviewed in snapshot
                if idx != index + 1
            ]
            if fixed:       # 空になった世代を残すと Ctrl+Z が 1 回不発になる
                kept.append(fixed)
        self._undo = kept

    def split_current(self) -> None:
        """分割ダイアログを開き、決まった位置で現在の区間を 2 つに分ける。"""
        if not self.proj.segments:
            return
        self._commit_text()
        seg = self.proj.segments[self.current]
        if seg.duration < MIN_SEGMENT_SECONDS * 2:
            messagebox.showinfo(
                "分割できません",
                "この区間は短すぎて 2 つに分けられません。",
                parent=self,
            )
            return
        self.player.stop()
        dlg = SplitDialog(self, seg)
        self.wait_window(dlg)
        if dlg.result is None:
            return
        self.apply_split(*dlg.result)

    def apply_split(self, boundary: float, cut: int) -> None:
        """分割を反映して画面を作り直す(ダイアログ抜きでも呼べる)。"""
        index = self.current
        head, tail = self.proj.split_segment(index, boundary, cut)
        self._remap_undo_for_split(index)
        self._dirty = True
        self.suggester.refresh()
        # 後半は話者が未確定で聴き直しが要るので、そちらへ移す
        self.current = index + 1
        self.reload_tree()
        self.show_current()
        self.update_status()
        self._set_action(
            f"区間を {fmt_hms_frac(head.start)}〜{fmt_hms_frac(head.end)} と "
            f"{fmt_hms_frac(tail.start)}〜{fmt_hms_frac(tail.end)} に分けました。"
            "後半は話者を確定してください(分割の取り消しは「前の区間と結合」)。"
        )

    # ---------------------------------------------------- 相づちを足す
    def _load_turns(self) -> list:
        """話者分離の turn をキャッシュから読む。無ければ空。

        作業ファイルは turn を持たない(クラスタだけ)ので、
        `.work_<名前>/diarize/` から読む。消えていれば手入力に落ちる。
        """
        try:
            from . import diarize as dz
            n = int((self.proj.engine or {}).get("diarize", {})
                    .get("num_speakers") or dz.DEFAULT_NUM_SPEAKERS)
            turns = dz.load_turns(dz.turns_cache_path(
                self._work_dir(), self.proj.audio_fingerprint or "", n))
            return list(turns or [])
        except Exception:
            return []

    def _voice_letter(self, speaker: int) -> str:
        """話者番号 → A/B/C…。**本体の表示と同じ対応にする**
        (`_speaker_letters` は出現順に振るので、同じ turn 一式から作る)。"""
        if not hasattr(self, "_letters_cache"):
            from .segments import _speaker_letters
            self._letters_cache = _speaker_letters(self._load_turns())
        return self._letters_cache.get(speaker, "?")

    def _voice_label(self, speaker: int) -> str:
        return f"声{self._voice_letter(speaker)}"

    def _added_for(self, seg: Segment) -> list[tuple[int, dict]]:
        """この区間に足した発話を、小窓に渡す形で集める。

        戻り値は (区間の index, 小窓の項目)。index は入れ替えのときに消す先。
        「この区間のもの」は**時刻が区間の範囲に入るもの**で決める。
        """
        text = seg.text or ""
        span = max(1e-6, seg.end - seg.start)
        out: list[tuple[int, dict]] = []
        for s in self.proj.segments:
            if s is seg or not self.proj.is_added_utterance(s):
                continue
            if not (seg.start - 0.01 <= s.start <= seg.end + 0.01):
                continue
            cut = int(round(len(text) * (s.start - seg.start) / span))
            out.append((s.index, {
                "cut": max(0, min(len(text), cut)),
                "text": s.text, "at": s.start, "end": s.end,
                "cluster": s.cluster, "sid": s.speaker_id,
            }))
        out.sort(key=lambda x: (x[1]["cut"], x[1]["at"]))
        return out

    def _ask_utterance(self, seg: Segment, turns: list,
                       existing: Optional[list[dict]] = None,
                       initial_cut: Optional[int] = None,
                       initial_note: str = ""):
        """小窓を開いて [(開始, 終了, 本文, まとまり, 話者), …] を返す。

        **押されたのが［…にする］なら applied=True。**やめたときと
        「全部消して確定」を区別するために要る。

        **画面を出す処理なので、検査では必ず差し替える。**差し替え忘れると
        応答待ちで止まる(GUI 検査で実際に止めた前例がある)。
        """
        dlg = AddUtteranceDialog(self, seg, turns, existing,
                                 initial_cut=initial_cut,
                                 initial_note=initial_note)
        self.wait_window(dlg)
        return (dlg.result, dlg.applied)

    def add_utterance(self, initial_cut: Optional[int] = None,
                      initial_note: str = "") -> None:
        """聞こえたのに本文に無い発話を足す。

        initial_cut を渡すと、その文字位置にカーソルを立てて開く
        (候補の一覧から呼ぶとき・設計書 §10.3.3)。
        """
        if not self.proj.segments:
            return
        self._commit_text()
        seg = self.proj.segments[self.current]
        # いまの区間の前後 3 秒に重なる turn を候補に出す(聴きどころと同じ窓)
        turns = [t for t in self._load_turns()
                 if t.end > seg.start - listen_order.WINDOW_SECONDS
                 and t.start < seg.end + listen_order.WINDOW_SECONDS]
        turns.sort(key=lambda t: (t.start, t.end))
        prev = self._added_for(seg)
        got, applied = self._ask_utterance(
            seg, turns, [item for _i, item in prev],
            initial_cut=initial_cut, initial_note=initial_note)
        if not applied:
            return
        # **開いたときにあったものは、いったん全部消してから作り直す。**
        # 位置も本文も話者も変えられるので、差分を追うより作り直すほうが
        # 確実で、edit_log にも経緯が残る。index の大きい順に消す。
        # **消せなかったら黙って進まない。**ここで握りつぶしていたため、
        # 「消したのに消えない」が起きても誰にも見えず、消し残しが作業
        # ファイルに溜まっていた(実データで発生・2026-08-20)。
        failed: list[str] = []
        for index in sorted((i for i, _it in prev), reverse=True):
            try:
                self.proj.remove_added_utterance(index)
            except ValueError as e:
                seg_bad = self.proj.segments[index] if (
                    0 <= index < len(self.proj.segments)) else None
                failed.append(
                    f"{fmt_short_time(seg_bad.start)}「{seg_bad.text}」: {e}"
                    if seg_bad else str(e))
        if failed:
            messagebox.showerror(
                "消せなかったものがあります",
                "次の発話を消せませんでした。ほかは指示どおりに直しています。\n\n"
                + "\n".join(failed), parent=self)
        # **区間は割らない。**割ると、直すときに元の本文を復元できなくなる。
        # 代わりに「本文のどこに割り込んだか」(cut)を残し、Word に出すときだけ
        # 差し込む(設計書 §5.0.5)。
        first = None
        for it in got:
            added = self.proj.add_utterance(
                float(it.get("at", it.get("start", 0.0))), float(it["end"]),
                it["text"], it.get("cluster") or "",
                cut=it.get("cut"), parent_orig=seg.orig_start,
                parent_start=seg.start)
            if it.get("sid"):
                # **いま聴いた直後に人が選んだので ✓。**機械が立てる経路ではない。
                self.proj.set_added_speaker(added.index, it["sid"])
            self._remap_undo_for_split(added.index)
            if first is None:
                first = added
        self._dirty = True
        self.suggester.refresh()
        if first is not None:
            self.current = first.index
        else:
            # 全部消した場合。元の区間へ戻す(見失わないように)
            back = next((s.index for s in self.proj.segments
                         if abs(s.start - seg.start) < 0.01
                         and s.text == seg.text), None)
            self.current = back if back is not None else min(
                self.current, len(self.proj.segments) - 1)
        self.reload_tree()
        self.show_current()
        self.update_status()
        if got:
            self._set_action(
                f"この区間の足した発話を {len(got)} 件にしました。"
                "一覧では「＋」の付いた行です。"
                "直すときは同じ［＋この声を足す...］から。")
        else:
            self._set_action("この区間に足した発話を全部消しました。")

    def replace_words(self) -> None:
        """転写の聞き違いを、前後を見ながらまとめて直す(設計書 §16.3)。

        **置換ではなく○×。**「資格」のように、同じ語でも直してよい箇所と
        そうでない箇所が混ざる。判定は機械が出さない。
        """
        if not self.proj.segments:
            return
        self._commit_text()
        dlg = ReplaceWordsDialog(self)
        self.wait_window(dlg)
        if not dlg.result:
            return
        before, after, targets = dlg.result
        # 探したときと同じ一致の条件で直す（§2.3。ずれると別の箇所が直る）
        n = self.proj.replace_text(before, after, targets,
                                   **getattr(dlg, "options", {}))
        if not n:
            self._set_action("直すところがありませんでした"
                             "(本文が変わっていた可能性があります)。")
            return
        self._dirty = True
        self.suggester.refresh()
        self.reload_tree()
        self.show_current()
        self.update_status()
        self._set_action(
            f"「{before}」を「{after}」に {n} 箇所直しました。"
            "編集の履歴には 1 件として残ります。"
            "「聴いて確定」の印は付いていません。")

    def replace_speaker(self) -> None:
        """途中退席した人の発言を、まとめて別の人に付け替える。

        **すべて △(まとめて適用)になる。**確かめたのは「その人は居なかった」
        ことであって、1 区間ずつの声ではない(CLAUDE.md の ✓/△ の意味論)。
        """
        if not self.proj.segments:
            return
        if not self.proj.speakers:
            messagebox.showinfo(
                "出席者がいません",
                "先に［出席者を編集...］で名簿を入れてください。", parent=self)
            return
        self._commit_text()
        dlg = ReplaceSpeakerDialog(self)
        self.wait_window(dlg)
        if not dlg.result:
            return
        before, after, keys = dlg.result
        n = self.proj.replace_speaker(before, after, keys)
        if not n:
            self._set_action("置き換えるところがありませんでした。")
            return
        self._dirty = True
        self.suggester.refresh()
        self.reload_tree()
        self.show_current()
        self.update_status()
        self._set_action(
            f"{self.proj.speaker_name(before)} の {n} 区間を "
            f"{self.proj.speaker_name(after)} にしました。"
            "すべて △(まとめて適用)です。編集の履歴には 1 件として残ります。")

    def remove_added(self) -> None:
        """人が足した区間を消す。**それ以外は消せない。**"""
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        if self.proj.is_added_utterance(seg):
            title, extra = "足した発話を消す", ""
        else:
            # **転写が出した区間。**取り消しにくい操作なので、何が起きるかを
            # はっきり書いてから消す。
            title = "転写した区間を消す"
            extra = ("\n\nこれは転写が出した区間です。"
                     "同じ音声が二重に書かれたときのために消せるように"
                     "してあります。\n"
                     "聴いて確かめたときだけ消してください"
                     "(消したことと中身は編集履歴に残ります)。")
        if not messagebox.askyesno(
                title,
                f"{fmt_hms_frac(seg.start)}「{seg.preview(30)}」を消します。"
                f"よろしいですか。{extra}", parent=self):
            return
        index = seg.index
        self.proj.remove_added_utterance(index)
        self._remap_undo_for_merge(index)
        self._dirty = True
        self.suggester.refresh()
        self.current = max(0, min(index, len(self.proj.segments) - 1))
        self.reload_tree()
        self.show_current()
        self.update_status()
        self._set_action("足した発話を消しました。")

    def merge_with_prev(self) -> None:
        self._merge_at(self.current - 1)

    def merge_with_next(self) -> None:
        self._merge_at(self.current)

    def _merge_at(self, index: int) -> None:
        """index と index+1 を結合する。話者が食い違うときは確認する。"""
        if not self.proj.segments:
            return
        if index < 0 or index + 1 >= len(self.proj.segments):
            self.bell()
            self._set_action("結合できる隣の区間がありません。")
            return
        self._commit_text()
        a, b = self.proj.segments[index], self.proj.segments[index + 1]
        if a.speaker_id and b.speaker_id and a.speaker_id != b.speaker_id:
            if not messagebox.askokcancel(
                "話者が違います",
                f"「{self.proj.speaker_name(a.speaker_id)}」と"
                f"「{self.proj.speaker_name(b.speaker_id)}」の区間を結合します。\n\n"
                "どちらが正しいか決められないため、結合後は未確定に戻ります。\n"
                "続けますか?",
                parent=self,
            ):
                return
        self.player.stop()
        self.apply_merge(index)

    def apply_merge(self, index: int) -> None:
        """結合を反映して画面を作り直す(確認ダイアログ抜きでも呼べる)。"""
        merged = self.proj.merge_segments(index)
        self._remap_undo_for_merge(index)
        self._dirty = True
        self.suggester.refresh()
        self.current = index
        self.reload_tree()
        self.show_current()
        self.update_status()
        self._set_action(
            f"2 つの区間を {fmt_hms_frac(merged.start)}〜{fmt_hms_frac(merged.end)} に"
            "まとめました(やり直すには「この区間を分割」)。"
        )

    def _commit_text(self) -> None:
        """本文欄の中身を、**その欄に読み込んだ区間へ**書き戻す。

        `current` に無条件で書くと、`current` を直接動かした直後に呼ばれた
        ときに**別の区間へ前の本文を上書きする**(検査で実際に起きた)。
        いまの経路では goto() が必ず本文欄を更新するので実害は無かったが、
        踏める余地は塞いでおく。
        """
        if not self.proj.segments:
            return
        # 本文欄がどの区間のものか分からない/食い違うなら書かない
        loaded = getattr(self, "_body_index", None)
        if loaded is None or not (0 <= loaded < len(self.proj.segments)):
            return
        if loaded != self.current:
            return
        new = self.txt_body.get("1.0", "end").strip()
        seg = self.proj.segments[loaded]
        if self.proj.edit_text(loaded, new):
            self._dirty = True
            self._update_row(seg.index)

    # ==================================================================
    # 候補者リスト
    # ==================================================================
    def _on_cand_scroll(self, first: str, last: str) -> None:
        """**要るときだけスクロールバーを出す。**常に出すと候補が横に狭まる。"""
        self.cand_scroll.set(first, last)
        need = not (float(first) <= 0.0 and float(last) >= 1.0)
        if need and not self.cand_scroll.winfo_ismapped():
            self.cand_scroll.grid(row=0, column=1, sticky="ns")
        elif not need and self.cand_scroll.winfo_ismapped():
            self.cand_scroll.grid_remove()

    def _cand_wheel(self, event) -> None:
        self.cand_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _scroll_candidate_into_view(self, widget) -> None:
        """選ばれている候補が隠れていたら、見える位置まで送る。"""
        self._cand_scroll_after = None
        try:
            self.cand_canvas.update_idletasks()
            top = widget.winfo_y()
            bottom = top + widget.winfo_height()
            view_h = self.cand_canvas.winfo_height()
            total = max(1, self.cand_holder.winfo_height())
            cur = self.cand_canvas.canvasy(0)
            if top < cur:
                self.cand_canvas.yview_moveto(top / total)
            elif bottom > cur + view_h:
                self.cand_canvas.yview_moveto(max(0.0, (bottom - view_h) / total))
        except Exception:
            pass        # 描画前などは何もしない(選べなくなるほうが困る)

    def _rebuild_candidates(self) -> None:
        for w in self._cand_widgets:
            w.destroy()
        self._cand_widgets = []

        if not self.proj.speakers:
            lbl = ttk.Button(self.cand_holder, text="出席者が未登録です — クリックして登録",
                             command=self.edit_roster)
            lbl.grid(row=0, column=0, sticky="ew", pady=2)
            self._cand_widgets.append(lbl)
            self._candidates = []
            return

        self._candidates = self.suggester.rank(self.current)
        seg = self.proj.segments[self.current]
        for i, cand in enumerate(self._candidates):
            key = QUICK_KEYS[i] if i < len(QUICK_KEYS) else " "
            mark = "●" if seg.speaker_id == cand.speaker.id else "　"
            reason = cand.reason_text
            text = f"{mark} [{key}] {cand.speaker.display}"
            if reason:
                text += f"    ← {reason}"
            btn = ttk.Button(
                self.cand_holder, text=text,
                command=lambda sid=cand.speaker.id: self.assign(sid),
            )
            btn.grid(row=i, column=0, sticky="ew", pady=1)
            self._cand_widgets.append(btn)
            if seg.speaker_id == cand.speaker.id:
                # **予約は 1 つだけ持ち、窓を閉じるときに取り消す。**
                # 置きっぱなしにすると、閉じたあとに発火して Tk が
                # 「invalid command name」を投げる(検査が不安定になった)。
                if self._cand_scroll_after is not None:
                    try:
                        self.after_cancel(self._cand_scroll_after)
                    except Exception:
                        pass
                self._cand_scroll_after = self.after_idle(
                    self._scroll_candidate_into_view, btn)

        # **特別な選択肢は別枠に横一列。**名簿に混ぜると出席者が多いときに
        # 下から切れる(実機の指摘 2026-08-19)。
        self.special_buttons = {}
        for sid, label in ((SPECIAL_UNKNOWN, "発言者不明"),
                           (SPECIAL_MULTI, "複数人が同時"),
                           (SPECIAL_NOISE, "発言なし・雑音")):
            mark = "●" if seg.speaker_id == sid else "　"
            btn = ttk.Button(self.special_holder, text=f"{mark} {label}",
                             command=lambda s=sid: self.assign(s))
            btn.pack(side="left", expand=True, fill="x", padx=1)
            self._cand_widgets.append(btn)
            self.special_buttons[sid] = btn

    # ==================================================================
    # 割当
    # ==================================================================
    def assign(self, speaker_id: Optional[str]) -> None:
        """現在の区間(設定によっては同じクラスタ全体)に話者を割り当てる。

        speaker_id=None なら未確定に戻す。
        一括適用したぶんは reviewed=False のままにする。「確定はしたが
        自分の耳では聴いていない」区間をあとから見直せるようにするため。
        """
        if not self.proj.segments:
            return
        self._commit_text()
        seg = self.proj.segments[self.current]

        # **複数選んでいるなら、そちらを優先する。**明示的に選んだ範囲より
        # 「同じまとまり全体」が勝つと、意図しない区間まで変わる。
        picked = self.selected_indices()
        if len(picked) > 1:
            targets = [self.proj.segments[i] for i in picked]
            self._assign_many(targets, speaker_id)
            return

        bulk = self.var_apply_cluster.get() and speaker_id is not None
        if bulk and seg.is_pseudo_cluster:
            messagebox.showwarning(
                "一括適用できません",
                f"{seg.cluster_label} は「判別できなかった/複数人が重なった」区間の寄せ集めで、"
                "同じ声のまとまりではありません。\n\nこの区間だけを確定します。",
                parent=self,
            )
            bulk = False

        targets = [seg]
        if bulk:
            targets = [
                s for s in self.proj.cluster_segments(seg.cluster)
                if s.index == seg.index or not s.speaker_id
            ]

        snapshot = [(s.index, s.speaker_id, s.reviewed) for s in targets]
        self._undo.append(snapshot)
        del self._undo[:-200]

        # **必ずデータ層を通す。**ここで seg.speaker_id を直接書き換えると
        # 編集履歴を素通りする(編集履歴設計書 §1.3)。
        # 自分で聴いた区間だけ「確認済み」。まとめて埋めた分は未確認扱い。
        self.proj.apply_speaker_to(
            [s.index for s in targets], speaker_id, heard_index=seg.index)
        self._dirty = True

        self.suggester.refresh()
        for s in targets:
            self._update_row(s.index)
        self.update_status()
        self._draw_timeline()

        name = self.proj.speaker_name(speaker_id) or "未確定"
        if len(targets) > 1:
            self._set_action(
                f"「{name}」を {len(targets)} 区間にまとめて適用しました"
                f"(うち {len(targets) - 1} 区間は未確認)。取り消しは Ctrl+Z。"
            )
        else:
            self._set_action(f"[{fmt_hms(seg.start)}] を「{name}」に確定しました。")

        self._after_change()

    def _assign_many(self, targets, speaker_id: Optional[str]) -> None:
        """選んだ複数の区間に、まとめて話者を当てる。

        **全部 △（一括適用で埋めただけ）になる。**Shift+クリックで 20 区間を
        一度に選べる操作は、聴かずに選ぶことを容易にする。ここで ✓ を立てると
        「✓＝人が耳で聴いて確定」が崩れる（CLAUDE.md）。聴いて確定したい
        区間は、1 つずつ選べばこれまでどおり ✓ になる。

        取り消しは 1 回で全部戻る（Ctrl+Z）。編集履歴にも 1 件の判断として
        残る——50 区間変えても、人がした判断は 1 回だから。
        """
        snapshot = [(s.index, s.speaker_id, s.reviewed) for s in targets]
        self._undo.append(snapshot)
        del self._undo[:-200]

        # heard_index を渡さない = 全部 △。**ここが要点。**
        self.proj.apply_speaker_to([s.index for s in targets], speaker_id)
        self._dirty = True

        self.suggester.refresh()
        for s in targets:
            self._update_row(s.index)
        self.update_status()
        self._draw_timeline()
        # **選択は保ったまま右ペインを描き直す。**goto() を使うと選択が
        # 1 件に戻ってしまい、続けて別の人へ変えられない。
        self.show_current()

        if speaker_id:
            name = self.proj.speaker_name(speaker_id) or "未確定"
            note = ("すべて未確認（△）です。聴いて確かめた区間は、"
                    "1 つずつ選んで確定してください")
            self._set_action(
                f"選んだ {len(targets)} 区間を「{name}」にしました"
                f"（{note}）。取り消しは Ctrl+Z。")
        else:
            self._set_action(
                f"選んだ {len(targets)} 区間を未確定に戻しました。"
                "取り消しは Ctrl+Z。")

    def unassign(self) -> None:
        """未確定に戻す。一括適用を間違えたときの復旧に使う。"""
        self.assign(None)

    def unassign_cluster(self) -> None:
        """現在のクラスタの割当をまとめて未確定に戻す。"""
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        targets = [s for s in self.proj.cluster_segments(seg.cluster) if s.speaker_id]
        if not targets:
            self._set_action("このまとまりに確定済みの区間はありません。")
            return
        if not messagebox.askyesno(
            "確認",
            f"{seg.cluster_label} の {len(targets)} 区間をすべて未確定に戻します。よろしいですか?",
            parent=self,
        ):
            return
        self._undo.append([(s.index, s.speaker_id, s.reviewed) for s in targets])
        del self._undo[:-200]
        self.proj.apply_speaker_to([s.index for s in targets], None)
        self._dirty = True
        self.suggester.refresh()
        self.update_status()
        self._set_action(f"{seg.cluster_label} の {len(targets)} 区間を未確定に戻しました。")
        self.reload_tree()
        self.show_current()

    def _after_change(self) -> None:
        """割当後の移動・再描画。"""
        if self.var_advance.get():
            nxt = self._next_target(self.current)
            if nxt is not None:
                if self.var_filter.get() != FILTER_ALL:
                    self.reload_tree()
                self.goto(nxt)
                return
            # 前方に対象が無い。ファイル全体で本当に終わっているかを確認する
            if self._remaining_count() == 0:
                self.player.stop()
                self._rebuild_candidates()
                messagebox.showinfo(
                    "完了",
                    "すべての区間の話者が確定しました。\nWord で出力できます。",
                    parent=self,
                )
            else:
                back = self._next_target(self.current, forward=False)
                if back is not None:
                    self._set_action("ここから先は対象がありません。前方の残りへ戻りました。")
                    if self.var_filter.get() != FILTER_ALL:
                        self.reload_tree()
                    self.goto(back)
                    return
        if self.var_filter.get() != FILTER_ALL:
            self.reload_tree()
        else:
            self._rebuild_candidates()

    def undo(self) -> None:
        if not self._undo:
            self.bell()
            self._set_action("取り消せる操作がありません。")
            return
        snapshot = self._undo.pop()
        # **取り消したことも履歴に残す。**消すと「当てたが戻した」経緯が
        # 読めなくなる(編集履歴設計書 §1.4)。
        self.proj.restore_assignments(snapshot)
        self._dirty = True
        self.suggester.refresh()
        if self.var_filter.get() != FILTER_ALL:
            self.reload_tree()
        else:
            for index, _, _ in snapshot:
                self._update_row(index)
        self.update_status()
        self._set_action(f"{len(snapshot)} 区間の操作を取り消しました。")
        self.goto(snapshot[0][0])

    # ==================================================================
    # 再生
    # ==================================================================
    def _speed(self) -> float:
        try:
            return float(self.var_speed.get().rstrip("x"))
        except ValueError:
            return 1.0

    def toggle_play(self) -> None:
        if self.player.is_playing():
            self.player.stop()
            self.btn_play.configure(text="▶ 再生 (Space)")
        else:
            self.play_current(explicit=True)

    def play_current(self, back: float = 0.0, extend: float = 0.0,
                     explicit: bool = False) -> None:
        """現在区間を再生する。

        back: 開始を何秒さかのぼるか(文脈を聴きたいとき)
        extend: 終了を何秒延ばすか(区切りが早すぎたと感じたとき)
        """
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        # 再生範囲。時刻を直した区間にはずれ補正を足さない(playback_window)。
        play_start, play_end = playback_window(
            seg, self.proj.time_offset, back=back, extend=extend
        )

        # 短い区間で先読みを固定 0.4 秒にすると、再生窓の大半が前の発言に
        # なってしまう(1 秒の区間なら 4 割)。区間の長さに応じて縮める。
        pre_roll = min(0.4, max(0.05, seg.duration * 0.25))
        self.play_span(play_start, play_end, pre_roll=pre_roll, explicit=explicit)

    def play_span(self, start: float, end: float, *, pre_roll: float = 0.4,
                  explicit: bool = False) -> None:
        """音声の任意の範囲を鳴らす。

        区間の再生も、点検の提案を「その時刻で聴いてみる」のもここを通る。
        音声が見つからないときの扱いを 1 か所にまとめておく。
        """
        audio = Path(self.proj.audio_path)
        if not audio.exists():
            # 移動のたびに警告を出し続けないよう、断られたら自動再生を切る
            if self._audio_declined and not explicit:
                return
            if not self._relocate_audio():
                self._audio_declined = True
                self.var_autoplay.set(False)
                self.var_seginfo.set(
                    "音声が見つからないため自動再生を止めました。"
                    "「▶ 再生」を押すと場所を指定し直せます。"
                )
                return
            self._audio_declined = False
            audio = Path(self.proj.audio_path)
        try:
            self.player.play(
                audio,
                start=max(0.0, start),
                end=end,
                speed=self._speed(),
                pre_roll=pre_roll,
            )
            self.btn_play.configure(text="■ 停止 (Space)")
        except Exception as e:
            messagebox.showerror("再生エラー", str(e), parent=self)

    def _on_play_finished(self) -> None:
        try:
            self.after(0, lambda: self.btn_play.configure(text="▶ 再生 (Space)"))
        except Exception:
            pass

    def _relocate_audio(self) -> bool:
        messagebox.showwarning(
            "音声が見つかりません",
            f"元の音声ファイルが見つかりません:\n{self.proj.audio_path}\n\n"
            "場所を指定してください。",
            parent=self,
        )
        path = filedialog.askopenfilename(
            title="音声ファイルを選択",
            filetypes=[("音声ファイル", "*.m4a *.mp3 *.wav *.aac *.flac *.ogg *.mp4"), ("すべて", "*.*")],
            parent=self,
        )
        if not path:
            return False
        self.proj.audio_path = path
        self._dirty = True
        return True

    # ==================================================================
    # 出席者・保存・出力
    # ==================================================================
    def edit_roster(self) -> None:
        """出席者を「名前」と「企業・役職」の 2 列で編集する(設計書 §11.8)。"""
        dlg = RosterDialog(self, self.proj.speakers)
        self.wait_window(dlg)
        if dlg.result is None:
            return
        plan = plan_roster_rows(self.proj, dlg.result)
        if plan.removed:
            names = "、".join(sp.display for sp in plan.removed)
            br = chr(10)
            detail = (
                f"次の出席者が削除されます:{br}{br}{names}{br}{br}"
                + (f"この人たちに割り当てていた {plan.affected_segments} 区間は"
                   f"未確定に戻ります。{br}" if plan.affected_segments else "")
                + "続けますか?"
            )
            if not messagebox.askyesno("確認", detail, parent=self):
                return          # ここまで何も変更していない
        plan.apply(self.proj)
        self._dirty = True
        self.refresh_all()

    def show_remaining(self) -> None:
        """残作業の内訳。未確定と「まとめて適用しただけ(未確認)」を分けて出す。"""
        unassigned = [s for s in self.proj.segments if not s.speaker_id]
        unreviewed = [s for s in self.proj.segments if s.speaker_id and not s.reviewed]

        if not unassigned and not unreviewed:
            messagebox.showinfo(
                "残作業", "すべての区間を聴いて確定済みです。", parent=self)
            return

        def breakdown(segs) -> list[str]:
            counts: dict[str, int] = {}
            for s in segs:
                counts[s.cluster_label] = counts.get(s.cluster_label, 0) + 1
            return [f"    {k}: {v} 区間"
                    for k, v in sorted(counts.items(), key=lambda x: -x[1])]

        lines: list[str] = []
        if unassigned:
            lines += [f"■ 未確定(話者が入っていない): {len(unassigned)} 区間"]
            lines += breakdown(unassigned)
            lines += ["", "  「同じ声のまとまり全体に適用」で、まとまり単位に一気に確定できます。", ""]
        if unreviewed:
            lines += [f"■ 未確認(まとめて適用しただけ): {len(unreviewed)} 区間"]
            lines += breakdown(unreviewed)
            lines += ["",
                      "  左上の「未確認のみ」に切り替えると、この区間だけを Tab で",
                      "  たどって聴き直せます。正しければ同じ話者をもう一度選べば",
                      "  「確認済み」になります。"]
        messagebox.showinfo("残作業の内訳", "\n".join(lines), parent=self)

    def save(self) -> None:
        self._commit_text()
        try:
            path = self.proj.save()
            self._dirty = False
            self.var_seginfo.set(f"保存しました: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("保存エラー", str(e), parent=self)

    def _autosave_tick(self) -> None:
        if not self.winfo_exists():
            return
        # **未来の版は保存できない。**試みても必ず失敗するので、4 秒ごとに
        # 書き込みを叩き続けない。開くときに知らせてある(gui.py)。
        if self.proj.is_from_newer_schema():
            try:
                self.after(4000, self._autosave_tick)
            except tk.TclError:
                pass
            return
        if self._dirty and self.proj.json_path:
            try:
                self.proj.save()
                self._dirty = False
            except Exception:
                pass
        try:
            self.after(4000, self._autosave_tick)
        except tk.TclError:
            pass    # ウィンドウが閉じられた

    def ensure_source_sha(self) -> bool:
        """元音声の SHA-256 が未記録なら計算して記録する。

        古い作業ファイル(v4 以前)には無いので、Word 出力の前にここで埋める。
        音声が無い・読めないときは False(出力自体は続けてよい。検証要約に
        SHA-256 の行が出ないだけ)。
        """
        if self.proj.source_sha256:
            return True
        audio = Path(self.proj.audio_path)
        if not audio.exists():
            return False
        self._set_action("元音声の SHA-256 を計算しています...")
        self.update_idletasks()
        _fp, sha = audio_hashes(audio)
        if not sha:
            self._set_action("元音声を読めなかったため、SHA-256 は未記録のままです。")
            return False
        self.proj.source_sha256 = sha
        self._dirty = True
        self._set_action(f"元音声の SHA-256 を記録しました({sha[:16]}…)。")
        return True

    def verify_source_audio(self) -> None:
        """いまの元音声を読み直して、記録した SHA-256 と一致するか確かめる。

        「この書面はこの録音から作った」の検算。第三者は Get-FileHash 等で
        同じ値を計算できる。
        """
        audio = Path(self.proj.audio_path)
        if not audio.exists():
            messagebox.showinfo("照合できません",
                                f"元音声が見つかりません:\n{audio}", parent=self)
            return
        if not self.proj.source_sha256:
            # 未記録なら、照合ではなく初回の記録として扱う
            if self.ensure_source_sha():
                self.save()
                messagebox.showinfo(
                    "記録しました",
                    "元音声の SHA-256 を記録しました。\n\n"
                    f"{self.proj.source_sha256}\n\n"
                    "次回からは、この値と一致するかを照合できます。",
                    parent=self)
            return
        self._set_action("元音声を読み直して照合しています...")
        self.update_idletasks()
        _fp, sha = audio_hashes(audio)
        if sha == self.proj.source_sha256:
            self._set_action("元音声と一致しました。")
            messagebox.showinfo(
                "一致しました",
                "元音声の SHA-256 は記録と一致しています。\n\n"
                f"{sha}",
                parent=self)
        else:
            self._set_action("元音声が記録と一致しません。")
            messagebox.showwarning(
                "一致しません",
                "元音声の SHA-256 が記録と違います。ファイルが差し替わって"
                "いるか、内容が変わっています。\n\n"
                f"記録: {self.proj.source_sha256}\n"
                f"現在: {sha or '(読めませんでした)'}",
                parent=self)

    def export_docx(self) -> None:
        self._commit_text()
        # 検証要約に載せる SHA-256 が未記録なら、ここで埋めてから保存する
        self.ensure_source_sha()
        self.save()

        dlg = tk.Toplevel(self)
        dlg.title("出力の設定")
        dlg.transient(self)
        dlg.grab_set()
        dlg.columnconfigure(1, weight=1)

        warn = []
        unassigned = self.proj.total_count - self.proj.assigned_count
        if unassigned:
            warn.append(f"未確定 {unassigned} 区間は【発言者不明】(赤字)になります。")
        if self.proj.unreviewed_count:
            warn.append(
                f"まとめて適用しただけで未確認の区間が {self.proj.unreviewed_count} あります。")
        if warn:
            ttk.Label(dlg, text="\n".join(warn), foreground="#B71C1C", justify="left")\
                .grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 6))

        ttk.Label(dlg, text="表題:").grid(row=1, column=0, sticky="w", padx=(12, 4), pady=4)
        var_title = tk.StringVar(value=Path(self.proj.audio_path).stem)
        ent = ttk.Entry(dlg, textvariable=var_title, width=42)
        ent.grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=4)
        ent.focus_set()

        var_ts = tk.BooleanVar(value=True)
        var_merge = tk.BooleanVar(value=True)
        var_attend = tk.BooleanVar(value=True)
        var_noise = tk.BooleanVar(value=True)
        var_txt = tk.BooleanVar(value=False)
        # 本文の【 】に役職も入れるか(設計書 §11.8)。出席者一覧は常に両方。
        var_role = tk.BooleanVar(value=False)
        for i, (text, var) in enumerate((
            ("段落の先頭に時刻を入れる", var_ts),
            ("同じ話者の連続発言をまとめる", var_merge),
            ("冒頭に出席者一覧を入れる", var_attend),
            ("「発言なし・雑音」と印を付けた区間を省く", var_noise),
            ("本文の【 】に企業・役職も入れる(出席者一覧には常に入ります)", var_role),
            ("同じ内容のテキストファイル(.txt)も出す", var_txt),
        )):
            ttk.Checkbutton(dlg, text=text, variable=var)\
                .grid(row=2 + i, column=0, columnspan=2, sticky="w", padx=12, pady=1)

        # 人が足した相づちの書き方(設計書 §11)。データは 1 つのまま、出し方
        # だけを選ぶ。同じ作業ファイルから (ア) で反訳書、(イ) で QDA 用を
        # 続けて出せる。差し込みが無いときは選びようが無いので出さない。
        self.var_insert_style = tk.StringVar(value=INSERT_STYLE_LINE)
        row = 8
        if has_inserted_utterances(self.proj):
            box = ttk.LabelFrame(dlg, text="足した相づちの入れ方")
            box.grid(row=row, column=0, columnspan=2, sticky="ew",
                     padx=12, pady=(8, 2))
            for text, value in (
                ("行を分ける（別の行に出し、続く行を ,, でつなぐ）",
                 INSERT_STYLE_LINE),
                ("行に埋め込む（本文の中に (氏名：はい) の形で入れる）",
                 INSERT_STYLE_INLINE),
            ):
                ttk.Radiobutton(box, text=text, value=value,
                                variable=self.var_insert_style).pack(
                                    anchor="w", padx=8, pady=1)
            ttk.Label(
                box,
                text="どちらも日本語会話研究の標準（BTSJ）に沿った書き方です。"
                     "読み方の説明は冒頭に入ります。",
                foreground="#555555", wraplength=430, justify="left").pack(
                    anchor="w", padx=8, pady=(2, 6))
            row += 1

        result: dict[str, object] = {}

        def ok() -> None:
            result["go"] = True
            dlg.destroy()

        btns = ttk.Frame(dlg)
        btns.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=12)
        ttk.Button(btns, text="保存先を選ぶ...", command=ok).pack(side="right")
        ttk.Button(btns, text="キャンセル", command=dlg.destroy).pack(side="right", padx=6)
        dlg.bind("<Return>", lambda e: ok())
        dlg.bind("<Escape>", lambda e: dlg.destroy())
        self.wait_window(dlg)
        if not result.get("go"):
            return

        default_dir = str(Path(self.proj.json_path or self.proj.audio_path).parent)
        path = filedialog.asksaveasfilename(
            title="Word ファイルの保存先",
            initialdir=default_dir,
            initialfile=f"{var_title.get() or Path(self.proj.audio_path).stem}.docx",
            defaultextension=".docx",
            filetypes=[("Word 文書", "*.docx")],
            parent=self,
        )
        if not path:
            return
        try:
            # 版はこの出力で 1 つ進める(出力できたときだけ確定して保存する)
            next_revision = self.proj.doc_revision + 1
            out = write_docx(
                self.proj, path,
                title=var_title.get() or None,
                with_timestamps=var_ts.get(),
                merge_consecutive=var_merge.get(),
                include_attendees=var_attend.get(),
                drop_noise=var_noise.get(),
                revision=next_revision,
                insert_style=self.var_insert_style.get(),
                with_role=var_role.get(),
            )
            if var_txt.get():
                write_text(self.proj, Path(path).with_suffix(".txt"),
                           merge_consecutive=var_merge.get(),
                           drop_noise=var_noise.get(),
                           insert_style=self.var_insert_style.get(),
                           with_role=var_role.get())
        except Exception:
            messagebox.showerror("出力エラー", traceback.format_exc(), parent=self)
            return
        self.proj.doc_revision = next_revision
        self._dirty = True
        self.save()
        if messagebox.askyesno("出力完了", f"{out}\n\nファイルを開きますか?", parent=self):
            _open_path(out)

    # ==================================================================
    # 時刻の点検(実測と突き合わせて提案を出す)
    # ==================================================================
    def run_inspection(self) -> None:
        """点検を始める。実測用の転写は重いので別スレッドで回す。

        音声を外へ送らない。ここは最初から最後までこの PC の中で終わる。
        """
        if self._inspect_thread and self._inspect_thread.is_alive():
            self._cancel_inspection()
            return
        if not self.proj.segments:
            return
        self._commit_text()

        audio = Path(self.proj.audio_path)
        if not audio.exists() and not self._relocate_audio():
            return
        audio = Path(self.proj.audio_path)

        cfg = load_config()
        self._inspect_cancel.clear()
        self.btn_inspect.configure(text="点検を中止")
        self._set_action("実測用の転写を始めます(この PC の中だけで処理します)...")
        self._inspect_thread = threading.Thread(
            target=self._inspect_worker,
            args=(audio, self._work_dir(),
                  str(cfg.get("align_model") or DEFAULT_MODEL),
                  cfg.get("align_model_dir") or None),
            daemon=True,
        )
        self._inspect_thread.start()
        self.after(200, self._drain_inspect)

    def _work_dir(self) -> Path:
        """作業ディレクトリ(.work_<音声名>)。パイプラインと同じ置き方にする。"""
        base = Path(self.proj.json_path or self.proj.audio_path).parent
        return base / f".work_{Path(self.proj.audio_path).stem}"

    def _cancel_inspection(self) -> None:
        self._inspect_cancel.set()
        self._set_action("点検の中止を伝えました。区切りのいい所で止まります。")

    def _inspect_worker(self, audio: Path, work_dir: Path,
                        model: str, model_dir) -> None:
        """別スレッド。重い転写だけをここで済ませ、結果を本体へ渡す。

        照合と提案づくりは本体側でやる(0.2 秒ほどで終わるうえ、区間を
        読むので、作業中のデータを別スレッドから触らずに済む)。
        """
        def post(kind: str, data) -> None:
            self._inspect_queue.put((kind, data))

        try:
            words = transcribe_words(
                audio, work_dir=work_dir, model=model, model_dir=model_dir,
                on_log=lambda m: post("log", m),
                on_progress=lambda a, b: post("progress", (a, b)),
                is_cancelled=self._inspect_cancel.is_set,
            )
        except AlignUnavailable as e:
            post("unavailable", str(e))
            return
        except Exception:
            post("fatal", traceback.format_exc())
            return
        post("cancelled" if words is None else "words", words)

    def _drain_inspect(self) -> None:
        """別スレッドからの知らせを本体で受ける(tkinter は本体からしか触れない)。"""
        running = True
        try:
            while True:
                kind, data = self._inspect_queue.get_nowait()
                if kind == "log":
                    self._set_action(str(data))
                elif kind == "progress":
                    done, total = data
                    if total:
                        self._set_action(
                            f"実測用の転写 {done / total:.0%} "
                            f"({fmt_hms(done)} / {fmt_hms(total)})...")
                elif kind == "words":
                    running = False
                    self._finish_inspection(data)
                elif kind == "cancelled":
                    running = False
                    self._end_inspection("点検を中止しました。")
                elif kind == "unavailable":
                    running = False
                    self._end_inspection("点検を始められませんでした。")
                    messagebox.showinfo("点検を始められません", str(data), parent=self)
                elif kind == "fatal":
                    running = False
                    self._end_inspection("点検が失敗しました。")
                    messagebox.showerror("点検エラー", str(data), parent=self)
        except queue.Empty:
            pass
        except tk.TclError:
            return              # 画面が閉じられた
        if running:
            self.after(200, self._drain_inspect)

    def _end_inspection(self, message: str) -> None:
        self.btn_inspect.configure(text="時刻を点検...")
        self._set_action(message)

    def _finish_inspection(self, words) -> None:
        """実測が揃った。照合して提案を作り、一覧を出す。"""
        result = inspect_times(self.proj, words)
        path = proposals_path(self._work_dir(), self.proj.audio_fingerprint)
        history = load_proposals(path)
        fresh = merge_history(result.proposals, history)

        summary = (f"{result.checked} 区間を点検: 提案 {len(fresh)} 件"
                   f"(確認済み {result.reviewed} / 照合できず {result.unmatched} / "
                   f"根拠が弱い {result.weak} / ずれ小 {result.close_enough})")
        self._end_inspection(summary)
        if not fresh:
            messagebox.showinfo(
                "点検が終わりました",
                summary + "\n\n直したほうがよさそうな区間は見つかりませんでした。",
                parent=self,
            )
            return
        self._open_proposals(fresh, path, decided_history(history))

    def _proposal_rows(self, proposals) -> list[ProposalRow]:
        """提案を一覧の行にする。ここでしか画面用の言い回しを作らない。"""
        rows: list[ProposalRow] = []
        for p in proposals:
            seg = target_segment(self.proj, p)
            if seg is None:
                continue
            now_start, _ = audio_span(seg, self.proj.time_offset)
            new_start = float(p.payload.get("start", now_start))
            rows.append(ProposalRow(
                key=p.id,
                kind="時刻",
                target=f"区間 {seg.index + 1}",
                now=fmt_hms_frac(now_start),
                measured=fmt_hms_frac(new_start),
                delta=f"{new_start - now_start:+.1f}秒",
                evidence=p.evidence,
                confidence=("高" if p.confidence >= 0.9
                            else "中" if p.confidence >= 0.75 else "低"),
                text=seg.preview(60),
            ))
        return rows

    def decide_proposal(self, proposal: Proposal, decision: str) -> None:
        """提案 1 件の始末をつける。

        decision: "accept"(聴いて承認 → ✎) / "bulk"(まとめて適用 → ✎△)
                  / "reject"(却下)
        当てられなかった提案は pending のまま残す。隣が動いた後なら
        当てられたかもしれず、再点検で出直せるようにするため。却下として
        記録すると、同じ根拠の正当な提案が二度と出なくなる。
        再提示を抑止するのは、人が明示的に却下したものだけ。
        """
        if decision == "reject":
            proposal.status = "rejected"
            return
        if self.apply_proposal(proposal, reviewed=(decision == "accept")):
            proposal.status = "accepted"

    def _open_proposals(self, proposals, path: Optional[Path],
                        history: Iterable[Proposal] = ()) -> None:
        by_id = {p.id: p for p in proposals}

        def play(key: str, *, proposed: bool = True) -> None:
            """その行の箇所を鳴らす。

            既定は「提案した時刻」のほう。確かめたいのは提案が合っているか
            なので、いまのずれた時刻で鳴らしても判断できない。
            """
            p = by_id[key]
            seg = target_segment(self.proj, p)
            if seg is None:
                return
            self.goto(seg.index)            # 本文と話者を画面に出す
            if not proposed:
                self.play_current(explicit=True)
                return
            start = float(p.payload.get("start", seg.start))
            end = float(p.payload.get("end", seg.end))
            self.play_span(start, end, explicit=True,
                           pre_roll=min(0.4, max(0.05, (end - start) * 0.25)))

        def bulk(keys) -> None:
            ok, failed = self.apply_proposals_bulk([by_id[k] for k in keys])
            # 当てられなかった分は pending のまま(隣が動けば次は当たるかもしれない)
            msg = (f"{len(ok)} 件をまとめて当てました(✎△)。"
                   "聴いて確かめると ✎ になります。")
            if failed:
                msg += f" {len(failed)} 件は当てられませんでした。"
            self._set_action(msg)

        dlg = ProposalDialog(
            self, self._proposal_rows(proposals),
            on_play=play,
            on_play_now=lambda k: play(k, proposed=False),
            on_accept=lambda k: self.decide_proposal(by_id[k], "accept"),
            on_bulk=bulk,
            on_reject=lambda k: self.decide_proposal(by_id[k], "reject"),
        )
        self.wait_window(dlg)
        # 判断を残す。却下したものを再点検で出し直さないため(§6.1)。
        # 過去の判断済み(history)も一緒に書き戻す。今回の提案だけで
        # 上書きすると、点検を 2 回挟んだだけで却下の記録が消える。
        save_proposals(path, [*proposals, *history])
        self.update_status()

    def destroy(self) -> None:
        """**閉じる前に、予約した処理を全部取り消す。**

        置きっぱなしにすると、窓が消えたあとに発火して Tk が
        「invalid command name」を投げる。検査が不安定になり、実機でも
        閉じるたびに背景でエラーが出る(2026-08-24)。

        8-24 は候補スクロールのタイマーだけ取り消していたが、**自動保存の
        `_autosave_tick`（4 秒ごと）が残っていた**（2026-09-03 に特定）。
        個別に追うと取りこぼすので、この窓が予約したものを全部取り消す。
        """
        cancel_pending_afters(self)
        self._cand_scroll_after = None
        super().destroy()

    def _on_close(self) -> None:
        self._inspect_cancel.set()
        self._commit_text()
        if self._dirty and self.proj.json_path:
            try:
                self.proj.save()
            except Exception:
                pass
        self.player.close()
        self.destroy()


class SplitDialog(tk.Toplevel):
    """区間を 2 つに分ける位置を決めるダイアログ。

    決めることは 2 つ:
      - 本文のどこで切るか(Text のカーソル位置)
      - 音声のどこで切るか(境界時刻)

    境界は「波形の谷を見る」「聴いて確かめる」「0.1 秒ずつ動かす」のどれでも
    合わせられる。発言の切れ目は無音の谷として波形に出るので、まず目で当たりを
    付けて、耳で詰めるのが速い。ffmpeg が無い環境では波形が出ないが、
    入力と試聴だけで作業できる。
    """

    WAVE_MARGIN = 2.0        # 波形に映す前後の余白(秒)
    WAVE_BUCKETS = 700       # 取り込む解像度。描画時に幅へ引き伸ばす
    WAVE_HEIGHT = 96
    LISTEN_SECONDS = 1.5     # 境界の前後を試聴する長さ

    def __init__(self, master: "AssignWindow", seg: Segment) -> None:
        super().__init__(master)
        self.win = master
        self.seg = seg
        self.result: Optional[tuple[float, int]] = None
        self._mark_origin: Optional[float] = None    # 「再生してマーク」の起点
        self._mark_speed = 1.0
        self._mark_timer: Optional[str] = None

        self.lo = seg.start + MIN_SEGMENT_SECONDS
        self.hi = seg.end - MIN_SEGMENT_SECONDS
        self.view_start = max(0.0, seg.start - self.WAVE_MARGIN)
        self.view_end = seg.end + self.WAVE_MARGIN
        self.boundary = round((seg.start + seg.end) / 2, 1)

        self.title("区間を分割")
        self.transient(master)
        self.resizable(True, False)
        self.var_bound = tk.StringVar(value="")
        self.var_hint = tk.StringVar(value="")

        self._build(seg)
        self._load_wave()
        # 本文のカーソル位置から境界の当たりを付ける(文字数の比で按分)
        self._estimate_from_cursor()
        self.grab_set()
        self.txt.focus_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda e: self._cancel())

    # ------------------------------------------------------------------
    def _build(self, seg: Segment) -> None:
        self.columnconfigure(0, weight=1)

        ttk.Label(
            self,
            text=f"[{fmt_hms_frac(seg.start)} → {fmt_hms_frac(seg.end)}]  "
                 "本文はカーソルの位置で切れます。",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        frm_text = ttk.LabelFrame(self, text="本文(切りたい位置をクリックしてカーソルを置く)")
        frm_text.grid(row=1, column=0, sticky="nsew", padx=12, pady=2)
        frm_text.columnconfigure(0, weight=1)
        self.txt = tk.Text(frm_text, height=5, wrap="word", font=("", 11))
        self.txt.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.txt.insert("1.0", seg.text)
        # カーソルの初期位置は本文の真ん中。ここから按分して境界の初期値を出す
        self.txt.mark_set("insert", f"1.0 + {len(seg.text) // 2} chars")

        frm_wave = ttk.LabelFrame(self, text="波形(クリックで境界を移動・谷が発言の切れ目)")
        frm_wave.grid(row=2, column=0, sticky="ew", padx=12, pady=4)
        frm_wave.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(frm_wave, height=self.WAVE_HEIGHT, bg="#FAFAFA",
                                highlightthickness=1, highlightbackground="#DDD")
        self.canvas.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_click)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        frm_b = ttk.Frame(self)
        frm_b.grid(row=3, column=0, sticky="ew", padx=12, pady=2)
        ttk.Label(frm_b, text="境界:").pack(side="left", padx=(0, 4))
        ent = ttk.Entry(frm_b, textvariable=self.var_bound, width=11, justify="center")
        ent.pack(side="left")
        ent.bind("<Return>", lambda e: self._commit_bound())
        ent.bind("<FocusOut>", lambda e: self._commit_bound())
        for text, delta in (("−0.1", -0.1), ("+0.1", +0.1)):
            ttk.Button(frm_b, text=text, width=5, takefocus=False,
                       command=lambda d=delta: self._set_boundary(self.boundary + d))\
                .pack(side="left", padx=1)
        self.btn_mark = ttk.Button(frm_b, text="▶ 再生してマーク", takefocus=False,
                                   command=self._toggle_mark)
        self.btn_mark.pack(side="left", padx=(12, 4))
        ttk.Button(frm_b, text="境界の前を聴く", takefocus=False,
                   command=lambda: self._listen(before=True)).pack(side="left", padx=2)
        ttk.Button(frm_b, text="境界の後を聴く", takefocus=False,
                   command=lambda: self._listen(before=False)).pack(side="left", padx=2)

        ttk.Label(self, textvariable=self.var_hint, foreground="#666", wraplength=760)\
            .grid(row=4, column=0, sticky="w", padx=12, pady=(2, 0))

        frm_ok = ttk.Frame(self)
        frm_ok.grid(row=5, column=0, sticky="e", padx=12, pady=(6, 10))
        ttk.Button(frm_ok, text="この位置で分割", command=self._ok).pack(side="left")
        ttk.Button(frm_ok, text="キャンセル", command=self._cancel).pack(side="left", padx=6)

    # ------------------------------------------------------------------
    def _load_wave(self) -> None:
        audio = Path(self.win.proj.audio_path)
        self._peaks: list[tuple[int, int]] = []
        if audio.exists():
            self._peaks = extract_peaks(
                audio, self.view_start, self.view_end, self.WAVE_BUCKETS
            )
        if not self._peaks:
            self.var_hint.set(
                "波形は表示できません(音声または ffmpeg が見つかりません)。"
                "時刻の入力と試聴で合わせてください。"
            )

    def _cursor_pos(self) -> int:
        return len(self.txt.get("1.0", "insert"))

    def _estimate_from_cursor(self) -> None:
        """本文のどこにカーソルがあるかで、境界の初期値を按分して決める。"""
        text = self.seg.text
        pos = self._cursor_pos()
        ratio = (pos / len(text)) if text else 0.5
        self._set_boundary(self.seg.start + self.seg.duration * ratio)

    def _time_to_x(self, t: float) -> float:
        w = max(1, self.canvas.winfo_width())
        return (t - self.view_start) / max(1e-6, self.view_end - self.view_start) * w

    def _x_to_time(self, x: float) -> float:
        w = max(1, self.canvas.winfo_width())
        return self.view_start + x / w * (self.view_end - self.view_start)

    def _set_boundary(self, value: float) -> None:
        self.boundary = min(max(round(float(value), 1), self.lo), self.hi)
        self.var_bound.set(fmt_hms_frac(self.boundary))
        self._redraw()

    def _commit_bound(self) -> None:
        try:
            self._set_boundary(parse_hms(self.var_bound.get()))
        except ValueError:
            self.var_bound.set(fmt_hms_frac(self.boundary))
            self.var_hint.set("時刻の形式が読めませんでした(例 00:43:51.5)。")

    def _on_canvas_click(self, event) -> None:
        self._set_boundary(self._x_to_time(event.x))

    def _redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        w = max(1, c.winfo_width())
        h = self.WAVE_HEIGHT
        mid = h / 2

        # 親区間の外(前後の余白)は沈めて、どこが今の区間かを分かるようにする
        for t0, t1 in ((self.view_start, self.seg.start), (self.seg.end, self.view_end)):
            c.create_rectangle(self._time_to_x(t0), 0, self._time_to_x(t1), h,
                               fill="#EEEEEE", width=0)
        if self._peaks:
            n = len(self._peaks)
            for i, (lo, hi) in enumerate(self._peaks):
                x = i * w / n
                c.create_line(x, mid - hi / 32768 * (mid - 2),
                              x, mid - lo / 32768 * (mid - 2), fill="#5B8DB8")
        else:
            c.create_text(w / 2, mid, text="(波形なし)", fill="#AAA")

        x = self._time_to_x(self.boundary)
        c.create_line(x, 0, x, h, fill="#C1666B", width=2)

    # ------------------------------------------------------------------
    def _audio(self) -> Optional[Path]:
        audio = Path(self.win.proj.audio_path)
        if not audio.exists():
            self.var_hint.set("音声が見つからないので試聴できません。")
            return None
        return audio

    def _play(self, start: float, end: float) -> None:
        audio = self._audio()
        if audio is None:
            return
        try:
            self.win.player.play(audio, start=max(0.0, start), end=end,
                                 speed=self.win._speed(), pre_roll=0.0)
        except Exception as e:
            self.var_hint.set(f"再生できませんでした: {e}")

    def _listen(self, before: bool) -> None:
        self._stop_mark()
        if before:
            self._play(max(self.seg.start, self.boundary - self.LISTEN_SECONDS), self.boundary)
        else:
            self._play(self.boundary, min(self.seg.end, self.boundary + self.LISTEN_SECONDS))

    def _toggle_mark(self) -> None:
        """1 回目で区間を頭から再生し、2 回目に押した時点を境界にする。

        ffplay は再生位置を教えてくれないので、経過時間から推定する。
        起動の遅れで 0.2〜0.3 秒ずれる粗いマークなので、そのあと波形と
        試聴で詰める前提。
        """
        if self._mark_origin is not None:
            elapsed = (time.monotonic() - self._mark_origin) * self._mark_speed
            self._set_boundary(self.seg.start + elapsed)
            self._stop_mark()
            self.var_hint.set(
                "押した位置を境界にしました。0.2〜0.3 秒ほどずれるので、"
                "波形・ナッジ・試聴で詰めてください。"
            )
            return
        if self._audio() is None:
            return
        self._mark_speed = self.win._speed()
        self._play(self.seg.start, self.seg.end)
        self._mark_origin = time.monotonic()
        self.btn_mark.configure(text="■ ここで区切る")
        self.var_hint.set("再生中です。切れ目だと思った所で「ここで区切る」を押してください。")
        # 再生が終わったらボタンを元に戻す(押しても意味がなくなるため)
        wait = int((self.seg.duration / max(0.1, self._mark_speed)) * 1000) + 300
        self._mark_timer = self.after(wait, self._stop_mark)

    def _stop_mark(self) -> None:
        if self._mark_timer is not None:
            try:
                self.after_cancel(self._mark_timer)
            except tk.TclError:
                pass
            self._mark_timer = None
        if self._mark_origin is not None:
            self._mark_origin = None
            self.btn_mark.configure(text="▶ 再生してマーク")

    # ------------------------------------------------------------------
    def _ok(self) -> None:
        self.result = (self.boundary, self._cursor_pos())
        self._close()

    def _cancel(self) -> None:
        self.result = None
        self._close()

    def _close(self) -> None:
        self._stop_mark()
        self.win.player.stop()
        self.grab_release()
        self.destroy()


class AddUtteranceDialog(tk.Toplevel):
    """聞こえたのに本文に無い発話を足す小窓（相づちを足す設計書 §5）。

    **本文の中をクリックして位置を決め、①②③…と積んでいく。**
    積んだ印は本文の中に出るので、**どこに何を入れたかが一目で分かる**。
    1 回の小窓で何件でも入れられる（2026-08-18・ユーザーの提案）。

    初版は「turn の時刻を選んで 1 件入れる」形で、実機で 2 度たて続けに
    行き詰まった——時刻は入力ではなく結果であること（§5.0）、1 件ずつでは
    区間に 3 つある相づちを入れられないこと。この版で両方を直している。

    時刻はクリック位置から文字数按分で見積もる。**目安であり実測ではない。**
    """

    MARKS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

    def __init__(self, win: "AssignWindow", seg: Segment,
                 turns: list, existing: Optional[list[dict]] = None,
                 initial_cut: Optional[int] = None,
                 initial_note: str = "") -> None:
        super().__init__(win)
        self.win = win
        self.seg = seg
        self._turns = sorted(turns, key=lambda t: (t.start, t.end))
        # 積んだ発話。{"cut": 文字位置, "text": 本文, "cluster": …, "sid": …}
        # **既に足したものは最初から積んだ状態にする。**空から始めると、
        # 位置を直したいときに入れ直すしかない(実機の指摘・2026-08-18)。
        self._items: list[dict] = list(existing or [])
        self._map: list[int] = []      # 表示上の位置 → 本文の文字位置
        self._cut = 0                  # いま選んでいる挿入位置
        self._follow_job = None
        self._follow_span = None
        self.result: list[tuple[float, float, str, str, Optional[str]]] = []
        self.applied = False          # ［…にする］を押したか(やめる と区別)
        self._had_existing = bool(self._items)
        # 候補から開いたときの初期位置(設計書 §10.3.3)。
        # **話者は入れない。**機械が選んだ話者のまま押されると ✓ が立つ。
        # ✓ は人が決めたという意味なので、そこは譲れない(CLAUDE.md)。
        self._initial_cut = initial_cut
        self._initial_note = initial_note
        self.title("聞こえた発話を足す"
                   + (f"（いま {len(self._items)} 件）" if self._items else ""))
        self.transient(win)
        self.resizable(False, False)

        ttk.Label(self, wraplength=620, justify="left", foreground="#1B5E20",
                  text="① 全部聴く → ② 聞こえた場所を本文でクリック → "
                       "③ 言葉を打って［この位置に足す］。何件でも積めます。"
                       "既に足したものは①②…として出るので、"
                       "［直す］［消す］で直せます。最後に［まとめて入れる］。").grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 4))

        # --- 本文（クリックで位置を決める。積んだ印がここに出る）---------
        box = ttk.LabelFrame(self, text="この区間の本文（入れたい場所をクリック）")
        box.grid(row=1, column=0, sticky="ew", padx=12, pady=(2, 2))
        box.columnconfigure(0, weight=1)
        self.txt = tk.Text(box, height=5, wrap="word", font=("", 12))
        self.txt.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.txt.bind("<Key>", self._on_key)
        self.txt.bind("<ButtonRelease-1>", self._on_move)
        self.txt.bind("<KeyRelease>", self._on_move)
        self.txt.tag_configure("here", background="#FFE082")
        self.txt.tag_configure("playing", background="#B3E5FC")
        self.txt.tag_configure("mark", foreground="#D84315",
                               font=("", 12, "bold"))

        # 候補から開いたときの案内(設計書 §10.3.3)。ふだんは空。
        self.var_from_cand = tk.StringVar(value="")
        ttk.Label(box, textvariable=self.var_from_cand, foreground="#1B5E20",
                  wraplength=600, justify="left").grid(
            row=1, column=0, sticky="w", padx=8)
        self.var_where = tk.StringVar()
        ttk.Label(box, textvariable=self.var_where, foreground="#666",
                  wraplength=600, justify="left").grid(
            row=2, column=0, sticky="w", padx=8, pady=(0, 6))

        # --- 聴く ---------------------------------------------------------
        play = ttk.LabelFrame(self, text="聴く")
        play.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 2))
        ttk.Button(play, text="▶ 全部聴く", takefocus=False,
                   command=self._play_all).pack(side="left", padx=(8, 4), pady=6)
        ttk.Button(play, text="▶ この辺だけ", takefocus=False,
                   command=self._play).pack(side="left", padx=4)
        self.btn_pause = ttk.Button(play, text="⏸ 一時停止", takefocus=False,
                                    command=self._toggle_pause)
        self.btn_pause.pack(side="left", padx=4)
        ttk.Label(play, text="速さ:").pack(side="left", padx=(12, 2))
        self.var_speed = tk.StringVar(value="0.8x")
        ttk.Combobox(play, values=DIALOG_SPEEDS, textvariable=self.var_speed,
                     state="readonly", width=7).pack(side="left")
        ttk.Label(play, text="前後:").pack(side="left", padx=(12, 2))
        self.var_around = tk.StringVar(value="1.5")
        ttk.Combobox(play, values=["0.8", "1.5", "3.0"], width=5,
                     textvariable=self.var_around,
                     state="readonly").pack(side="left")
        ttk.Label(play, text="秒").pack(side="left", padx=(2, 8))

        # --- 1 件を組み立てて積む -----------------------------------------
        one = ttk.LabelFrame(self, text="ここに入る発話")
        one.grid(row=3, column=0, sticky="ew", padx=12, pady=(6, 2))
        one.columnconfigure(1, weight=1)
        ttk.Label(one, text="聞こえた言葉:").grid(row=0, column=0, sticky="w",
                                                  padx=(8, 4), pady=(6, 2))
        self.var_text = tk.StringVar()
        ent = ttk.Entry(one, textvariable=self.var_text, width=44)
        ent.grid(row=0, column=1, sticky="ew", pady=(6, 2))
        ent.bind("<Return>", lambda e: self._add_item())

        ttk.Label(one, text="誰が:").grid(row=1, column=0, sticky="w",
                                          padx=(8, 4), pady=(0, 6))
        wrow = ttk.Frame(one)
        wrow.grid(row=1, column=1, sticky="w", pady=(0, 6))
        names = [("（決めない）", "")] + [(sp.name, sp.id)
                                          for sp in win.proj.speakers]
        self._sp_ids = [v for _, v in names]
        self.cmb_sp = ttk.Combobox(wrow, values=[n for n, _ in names],
                                   state="readonly", width=26)
        self.cmb_sp.current(0)
        self.cmb_sp.pack(side="left")
        ttk.Label(wrow, foreground="#666",
                  text="選ぶと ✓（聴いて確定）").pack(side="left", padx=6)
        ttk.Button(wrow, text="＋ この位置に足す",
                   command=self._add_item).pack(side="left", padx=(12, 0))

        # --- 積んだもの ---------------------------------------------------
        lst = ttk.LabelFrame(self, text="入れる発話（本文の印と対応します）")
        lst.grid(row=4, column=0, sticky="ew", padx=12, pady=(4, 2))
        lst.columnconfigure(0, weight=1)
        self.lb = tk.Listbox(lst, height=4, activestyle="dotbox")
        self.lb.grid(row=0, column=0, sticky="ew", padx=(8, 4), pady=6)
        side = ttk.Frame(lst)
        side.grid(row=0, column=1, sticky="n", padx=(0, 8), pady=6)
        ttk.Button(side, text="直す", width=6,
                   command=self._edit_item).pack()
        ttk.Button(side, text="消す", width=6,
                   command=self._del_item).pack(pady=(4, 0))

        # --- 時刻（目安）---------------------------------------------------
        tm = ttk.Frame(self)
        tm.grid(row=5, column=0, sticky="w", padx=12, pady=(2, 2))
        self.var_at = tk.StringVar()
        ttk.Label(tm, textvariable=self.var_at,
                  foreground="#666").pack(side="left")
        if self._turns:
            ttk.Label(tm, text="／ 声の候補に合わせる:").pack(side="left",
                                                              padx=(10, 4))
            self.cmb_turn = ttk.Combobox(
                tm, state="readonly", width=26,
                values=["（合わせない）"] + [
                    f"{fmt_hms_frac(t.start)}〜{fmt_hms_frac(t.end)}  "
                    f"{win._voice_label(t.speaker)}" for t in self._turns])
            self.cmb_turn.current(0)
            self.cmb_turn.pack(side="left")
            self.cmb_turn.bind("<<ComboboxSelected>>", self._sync)
        else:
            self.cmb_turn = None
            ttk.Label(tm, foreground="#B26500",
                      text="／ 話者分離の結果がありません").pack(side="left",
                                                                padx=(10, 0))

        btns = ttk.Frame(self)
        btns.grid(row=6, column=0, sticky="e", padx=12, pady=(8, 12))
        self.btn_ok = ttk.Button(btns, text="まとめて入れる", command=self._ok)
        self.btn_ok.pack(side="left")
        ttk.Button(btns, text="やめる",
                   command=self._cancel).pack(side="left", padx=8)
        self.bind("<Escape>", lambda e: self._cancel())

        self._render()
        # 候補から開いたときは、その位置にカーソルを立てる(設計書 §10.3.3)。
        # **位置は文字数按分の目安**なので、聴いて直す前提。
        if self._initial_cut is not None:
            self._cut = max(0, min(len(self.seg.text or ""),
                                   int(self._initial_cut)))
            try:
                self.txt.mark_set(
                    "insert", f"1.0 + {self._cut_to_disp(self._cut)} chars")
            except tk.TclError:
                pass
            self._show_cut()
        if self._initial_note:
            self.var_from_cand.set(self._initial_note)
        ent.focus_set()
        self.grab_set()

    # ------------------------------------------------------------------
    # 表示（本文＋積んだ印）
    # ------------------------------------------------------------------
    def _mark(self, i: int) -> str:
        return self.MARKS[i] if i < len(self.MARKS) else f"({i + 1})"

    def _render(self) -> None:
        """本文に積んだ印を差し込んで表示し直す。

        **印は表示だけのもので、区間の本文には入れない。**言われていない
        文字を記録に混ぜないため（逐語の原則）。
        """
        text = self.seg.text or ""
        # 位置ごとに、その位置へ入る印を集める（積んだ順を保つ）
        at_pos: dict[int, list[int]] = {}
        for i, it in enumerate(self._items):
            at_pos.setdefault(int(it["cut"]), []).append(i)

        parts: list[str] = []
        self._map = []
        marks: list[tuple[int, int]] = []      # 表示上の (開始, 終わり)
        for pos in range(len(text) + 1):
            for i in at_pos.get(pos, []):
                m = self._mark(i)
                marks.append((len("".join(parts)), len("".join(parts)) + len(m)))
                parts.append(m)
                self._map.extend([pos] * len(m))
            if pos < len(text):
                parts.append(text[pos])
                self._map.append(pos)
        self._map.append(len(text))

        disp = "".join(parts)
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", disp)
        for a, b in marks:
            self.txt.tag_add("mark", f"1.0 + {a} chars", f"1.0 + {b} chars")

        self.lb.delete(0, "end")
        if self._items:
            for i, it in enumerate(self._items):
                who = self.win.proj.speaker_name(it["sid"]) if it["sid"] else "未定"
                self.lb.insert("end",
                               f"{self._mark(i)} 「{it['text']}」  {who}"
                               f"  {fmt_hms_frac(it['at'])}")
        else:
            self.lb.insert("end", "（まだありません。位置を選んで［＋ この位置に足す］）")
        self.btn_ok.configure(
            text=f"この {len(self._items)} 件にする"
            if self._items else "まとめて入れる")
        self._show_cut()

    def _disp_to_cut(self, disp_index: int) -> int:
        if not self._map:
            return 0
        return self._map[max(0, min(disp_index, len(self._map) - 1))]

    def _cut_to_disp(self, cut: int) -> int:
        for d, c in enumerate(self._map):
            if c >= cut:
                return d
        return len(self._map) - 1

    # ------------------------------------------------------------------
    def _on_key(self, event):
        """本文は変えさせない。移動キーだけ通す（位置を選ぶため）。"""
        if event.keysym in ("Left", "Right", "Up", "Down", "Home", "End",
                            "Prior", "Next"):
            return None
        return "break"

    def _on_move(self, _e=None) -> None:
        try:
            d = len(self.txt.get("1.0", "insert"))
        except tk.TclError:
            d = 0
        self._cut = self._disp_to_cut(d)
        self._show_cut()
        if self.cmb_turn is not None and self.cmb_turn.current() > 0:
            self.cmb_turn.current(0)      # 手で動かしたら候補合わせを解除

    def _show_cut(self) -> None:
        self.txt.tag_remove("here", "1.0", "end")
        text = self.seg.text or ""
        d = self._cut_to_disp(self._cut)
        lo, hi = max(0, d - 1), min(len(self.txt.get("1.0", "end-1c")), d + 1)
        if hi > lo:
            self.txt.tag_add("here", f"1.0 + {lo} chars", f"1.0 + {hi} chars")
        if text:
            self.var_where.set(
                f"ここに入ります → …{text[max(0, self._cut - 12):self._cut]}"
                f"【ここ】{text[self._cut:self._cut + 12]}…")
        else:
            self.var_where.set("この区間には本文がありません。")
        self.var_at.set(f"推定の時刻: {fmt_hms_frac(self._estimate())}（目安）")

    def _estimate(self) -> float:
        """クリック位置から時刻を見積もる（文字数按分）。目安であり実測ではない。"""
        text = self.seg.text or ""
        span = max(0.0, self.seg.end - self.seg.start)
        if not text or span <= 0:
            return round((self.seg.start + self.seg.end) / 2, 2)
        return round(self.seg.start + span * (self._cut / len(text)), 2)

    def _sync(self, _e=None) -> None:
        i = self.cmb_turn.current() - 1 if self.cmb_turn else -1
        if 0 <= i < len(self._turns):
            t = self._turns[i]
            self.var_at.set(f"推定の時刻: {fmt_hms_frac(t.start)}"
                            f"（{self.win._voice_label(t.speaker)} の候補）")

    def _span(self) -> tuple[float, float]:
        i = self.cmb_turn.current() - 1 if self.cmb_turn else -1
        if 0 <= i < len(self._turns):
            t = self._turns[i]
            return (float(t.start), float(t.end))
        at = self._estimate()
        return (at, at + 0.8)

    def _cluster(self) -> str:
        i = self.cmb_turn.current() - 1 if self.cmb_turn else -1
        if 0 <= i < len(self._turns):
            return f"g:{self.win._voice_letter(self._turns[i].speaker)}"
        return f"{self.seg.chunk}:{PSEUDO_UNKNOWN}"

    # ------------------------------------------------------------------
    # 積む・直す・消す
    # ------------------------------------------------------------------
    def _add_item(self) -> None:
        text = self.var_text.get().strip()
        if not text:
            messagebox.showwarning("言葉が空です",
                                   "聞こえた言葉を入れてください。", parent=self)
            return
        a, b = self._span()
        self._items.append({
            "cut": self._cut, "text": text, "at": a, "end": b,
            "cluster": self._cluster(),
            "sid": self._sp_ids[self.cmb_sp.current()] or None,
        })
        # 本文の位置の順に並べ替える（印の番号が本文の並びと合うように）
        self._items.sort(key=lambda x: (x["cut"], x["at"]))
        self.var_text.set("")
        self._render()

    def _selected(self) -> int:
        sel = self.lb.curselection()
        if not self._items or not sel or sel[0] >= len(self._items):
            return -1
        return sel[0]

    def _edit_item(self) -> None:
        i = self._selected()
        if i < 0:
            return
        it = self._items[i]
        # 選んだものを組み立て欄へ戻して、積み直してもらう
        self.var_text.set(it["text"])
        try:
            self.cmb_sp.current(self._sp_ids.index(it["sid"] or ""))
        except ValueError:
            self.cmb_sp.current(0)
        self._cut = int(it["cut"])
        del self._items[i]
        self._render()

    def _del_item(self) -> None:
        i = self._selected()
        if i < 0:
            return
        del self._items[i]
        self._render()

    # ------------------------------------------------------------------
    # 聴く
    # ------------------------------------------------------------------
    def _speed(self) -> float:
        try:
            return float(self.var_speed.get().rstrip("x"))
        except ValueError:
            return 1.0

    def _play_all(self) -> None:
        """区間を頭から終わりまで鳴らす。**相づちの場所を探すための再生。**"""
        self.win.player.play(self.win.proj.audio_path,
                             self.seg.start, self.seg.end,
                             speed=self._speed(), pre_roll=0.0, post_roll=0.0)
        self.btn_pause.configure(text="⏸ 一時停止")
        self._follow(self.seg.start, self.seg.end)

    def _play(self) -> None:
        a, b = self._span()
        try:
            around = float(self.var_around.get())
        except ValueError:
            around = 1.5
        lo, hi = max(0.0, a - around), b + around
        self.win.player.play(self.win.proj.audio_path, lo, hi,
                             speed=self._speed(), pre_roll=0.0, post_roll=0.0)
        self.btn_pause.configure(text="⏸ 一時停止")
        self._follow(lo, hi)

    def _follow(self, a: float, b: float) -> None:
        self._stop_follow()
        self._follow_span = (a, b)
        self._tick()

    def _stop_follow(self) -> None:
        if self._follow_job is not None:
            try:
                self.after_cancel(self._follow_job)
            except tk.TclError:
                pass
        self._follow_job = None
        self.txt.tag_remove("playing", "1.0", "end")

    def _tick(self) -> None:
        p = self.win.player
        cur = getattr(p, "_cur", None)
        text = self.seg.text or ""
        span = max(1e-6, self.seg.end - self.seg.start)
        if cur is None or not p.is_playing() or not text:
            self._stop_follow()
            return
        import time as _t
        played = (_t.monotonic() - cur["since"]) * float(cur["speed"] or 1.0)
        at = cur["ss"] + played
        pos = max(0, min(len(text) - 1,
                         int(len(text) * (at - self.seg.start) / span)))
        d = self._cut_to_disp(pos)
        self.txt.tag_remove("playing", "1.0", "end")
        self.txt.tag_add("playing", f"1.0 + {d} chars", f"1.0 + {d + 1} chars")
        self.txt.see(f"1.0 + {d} chars")
        self._follow_job = self.after(120, self._tick)

    def _toggle_pause(self) -> None:
        p = self.win.player
        if p.is_paused():
            p.resume()
            self.btn_pause.configure(text="⏸ 一時停止")
            if self._follow_span:
                self._follow(*self._follow_span)
        elif p.pause():
            self.btn_pause.configure(text="▶ 続きから")
            if self._follow_job is not None:
                try:
                    self.after_cancel(self._follow_job)
                except tk.TclError:
                    pass
                self._follow_job = None   # 印は残す（どこで止めたか見える）

    # ------------------------------------------------------------------
    def _ok(self) -> None:
        # 打ちかけの 1 件があれば拾う（打って［まとめて入れる］を押す人がいる）
        if self.var_text.get().strip():
            self._add_item()
        if not self._items and not self._had_existing:
            messagebox.showwarning(
                "入れるものがありません",
                "位置を選んで言葉を打ち、［＋ この位置に足す］で積んでください。",
                parent=self)
            return
        # **空にして押した場合は「全部消す」。**開いたときに既にあったなら、
        # それを消したいという意思なので通す。
        # **cut(本文の位置)も返す。**これが無いと、どこで区間を割るかが
        # 決まらず、Word の並びが「長い発言 → 相づち」のままになる。
        self.result = [dict(it) for it in self._items]
        self.applied = True
        self._close()

    def _cancel(self) -> None:
        self.result = []
        self.applied = False
        self._close()

    def _close(self) -> None:
        self._stop_follow()
        self.win.player.stop()
        self.grab_release()
        self.destroy()


@dataclass
class ProposalRow:
    """点検の提案を一覧に 1 行出すための、表示用の値。

    点検側(inspect.py)が持つ提案そのものではなく、そこから作った文字列だけを
    持つ。画面が提案の作り方を知らずに済むので、照合の実装を差し替えても
    ここは変えなくてよい(align.py を薄いアダプタに隔離するのと同じ理由)。
    """

    key: str            # 呼び出し側が提案を特定するための鍵
    kind: str           # 種別。"時刻" / "分割"
    target: str         # 対象区間の見出し(例 "区間 12")
    now: str            # いまの時刻
    measured: str       # 実測した時刻
    delta: str          # ずれ
    evidence: str       # なぜそう言えるのか(照合の根拠)
    confidence: str     # どれくらい確からしいか
    text: str           # 発言のプレビュー


class ReplaceSpeakerDialog(tk.Toplevel):
    """話者をまとめて置き換える。**途中退席のための画面。**

    「32:17 に吉沢さんが帰ったので、それ以降の吉沢忠一は全部西村香介」
    ——実データで 108 区間あった（2026-08-20）。1 つずつ付け直すのは
    現実的でない。

    `ReplaceWordsDialog` と同じ形にしてある。**機械が判定せず、人が
    1 区間ずつ○×する。**退席の直前・直後は本人の発言が混ざるので
    （「ちょっと中座させてもらいます」は本人）、一律に変えると壊れる。

    **すべて △（まとめて適用）になる。**✓ が付いていた区間も △ に落ちる。
    確かめたのは「その人はもう居なかった」という事実であって、区間ごとの
    声ではない。件数を画面に出して、そうなることを先に伝える。
    """

    MARK_ON, MARK_OFF = "○", "×"

    def __init__(self, master: "AssignWindow") -> None:
        super().__init__(master)
        self.win = master
        self.rows: list = []                 # list[Segment]
        self.marks: list[bool] = []
        self.result: Optional[tuple[Optional[str], Optional[str], list]] = None

        self.title("話者をまとめて置き換える")
        self.transient(master)
        self._names = [(sp.name, sp.id) for sp in master.proj.speakers]
        self.var_after_time = tk.StringVar()
        self.var_status = tk.StringVar(value="置き換える人を選んで［探す］。")
        self._build()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda e: self._close())

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        ttk.Label(
            self, wraplength=720,
            text="途中で退席した人の発言を、まとめて別の人に付け替えます。"
                 "退席の直前は本人の発言なので（「中座させてもらいます」など）、"
                 "前後を読んで、変えない行は × にしてください。",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 6))

        top = ttk.Frame(self)
        top.grid(row=1, column=0, columnspan=2, sticky="w", padx=12)
        ttk.Label(top, text="いまの話者:").pack(side="left")
        self.cmb_before = ttk.Combobox(top, state="readonly", width=22,
                                       values=[n for n, _ in self._names])
        self.cmb_before.pack(side="left", padx=(4, 10))
        ttk.Label(top, text="→  こちらにする:").pack(side="left")
        self.cmb_after = ttk.Combobox(top, state="readonly", width=22,
                                      values=[n for n, _ in self._names])
        self.cmb_after.pack(side="left", padx=(4, 14))
        ttk.Label(top, text="この時刻より後だけ:").pack(side="left")
        self.ent_time = ttk.Entry(top, textvariable=self.var_after_time, width=10)
        self.ent_time.pack(side="left", padx=(4, 2))
        ttk.Label(top, text="(32:17 のように。空なら全部)",
                  foreground="#666").pack(side="left", padx=(0, 10))
        ttk.Button(top, text="探す", command=self.search).pack(side="left")
        self.ent_time.bind("<Return>", lambda e: self.search())

        cols = ("mark", "at", "seen", "text")
        self.tree = ttk.Treeview(self, columns=cols, show="headings",
                                 height=14, selectmode="browse")
        for key, label, width, anchor in (
                ("mark", "変える", 52, "center"),
                ("at", "時刻", 78, "center"),
                ("seen", "印", 40, "center"),
                ("text", "発言", 600, "w")):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor=anchor)
        self.tree.grid(row=2, column=0, sticky="nsew", padx=(12, 0), pady=(8, 0))
        sb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        sb.grid(row=2, column=1, sticky="ns", padx=(0, 12), pady=(8, 0))
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<Button-1>", self._on_click)
        self.tree.bind("<space>", lambda e: (self._toggle_selected(), "break"))

        ttk.Label(self, textvariable=self.var_status, foreground="#666",
                  wraplength=720)\
            .grid(row=3, column=0, sticky="w", padx=12, pady=(6, 0))

        btns = ttk.Frame(self)
        btns.grid(row=4, column=0, columnspan=2, sticky="ew", padx=12, pady=10)
        ttk.Button(btns, text="全部に○", command=lambda: self._mark_all(True))\
            .pack(side="left")
        ttk.Button(btns, text="全部に×", command=lambda: self._mark_all(False))\
            .pack(side="left", padx=6)
        ttk.Button(btns, text="やめる", command=self._close).pack(side="right")
        self.btn_ok = ttk.Button(btns, text="置き換える", command=self._ok,
                                 state="disabled")
        self.btn_ok.pack(side="right", padx=6)

    # ------------------------------------------------------------------
    def _id_of(self, cmb: ttk.Combobox) -> Optional[str]:
        i = cmb.current()
        return self._names[i][1] if 0 <= i < len(self._names) else None

    def search(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.rows, self.marks = [], []
        sid = self._id_of(self.cmb_before)
        if sid is None:
            self.var_status.set("いまの話者を選んでください。")
            self._refresh_ok()
            return
        raw = self.var_after_time.get().strip()
        floor = None
        if raw:
            floor = parse_hms(raw)
            if floor is None:
                messagebox.showwarning(
                    "時刻を読み取れません",
                    "32:17 や 1:04:30 のように入れてください。", parent=self)
                return
        self.rows = [s for s in self.win.proj.segments
                     if s.speaker_id == sid
                     and (floor is None or s.start > floor)]
        self.marks = [True] * len(self.rows)
        for i, s in enumerate(self.rows):
            self.tree.insert("", "end", iid=str(i), values=(
                self.MARK_ON, fmt_short_time(s.start),
                "✓" if s.reviewed else "△", (s.text or "")[:80]))
        if not self.rows:
            self.var_status.set("その条件に当てはまる区間はありません。")
        self._refresh_ok()

    # ------------------------------------------------------------------
    def _on_click(self, event) -> None:
        if self.tree.identify_region(event.x, event.y) != "cell":
            return
        row = self.tree.identify_row(event.y)
        if row and self.tree.identify_column(event.x) == "#1":
            self._toggle(int(row))

    def _toggle_selected(self) -> None:
        sel = self.tree.selection()
        if sel:
            self._toggle(int(sel[0]))

    def _toggle(self, i: int) -> None:
        if not (0 <= i < len(self.marks)):
            return
        self.marks[i] = not self.marks[i]
        self.tree.set(str(i), "mark",
                      self.MARK_ON if self.marks[i] else self.MARK_OFF)
        self._refresh_ok()

    def _mark_all(self, on: bool) -> None:
        for i in range(len(self.marks)):
            self.marks[i] = on
            self.tree.set(str(i), "mark", self.MARK_ON if on else self.MARK_OFF)
        self._refresh_ok()

    def _refresh_ok(self) -> None:
        n = sum(self.marks)
        self.btn_ok.configure(
            text=f"この {n} 区間を置き換える" if n else "置き換える",
            state="normal" if n else "disabled")
        if not self.rows:
            return
        heard = sum(1 for s, m in zip(self.rows, self.marks) if m and s.reviewed)
        msg = (f"{len(self.rows)} 区間のうち {n} 区間を置き換えます。"
               "置き換えた区間はすべて △(まとめて適用)になります"
               "——確かめたのは「その人は居なかった」ことであって、"
               "1 区間ずつの声ではないためです。")
        if heard:
            msg += f" うち {heard} 区間は今 ✓(聴いて確定)で、△ に戻ります。"
        self.var_status.set(msg)

    # ------------------------------------------------------------------
    def _ok(self) -> None:
        before, after = self._id_of(self.cmb_before), self._id_of(self.cmb_after)
        if after is None:
            messagebox.showwarning(
                "付け替える先が未選択です",
                "「こちらにする」で人を選んでください。", parent=self)
            return
        if before == after:
            messagebox.showwarning(
                "同じ人です", "違う人を選んでください。", parent=self)
            return
        chosen = [segment_key(s) for s, m in zip(self.rows, self.marks) if m]
        if not chosen:
            return
        self.result = (before, after, chosen)
        self._close()

    def _close(self) -> None:
        self.grab_release()
        self.destroy()


class ReplaceWordsDialog(tk.Toplevel):
    """語句をまとめて直す。**置換ではなく、1 箇所ずつ○×する画面。**

    無条件の一括置換は本文を壊す。実データで「資格」は 10 回出るが、
    そのうち 1 回は本物の資格(防災士の資格)で、直してはいけない
    (設計書 §16.3)。**だから前後の文脈を必ず出す。**時刻と語だけを
    並べても、どれが誤りかは判断できない。

    既定は全部○。外すものだけ外してもらう(誤りは大半が同じ聞き違いなので、
    1 つずつ付けさせると 14 箇所で 14 回押させることになる)。
    """

    MARK_ON, MARK_OFF = "○", "×"

    def __init__(self, master: "AssignWindow") -> None:
        super().__init__(master)
        self.win = master
        self.hits: list = []                 # list[TextHit]。行の並びと対応
        self.marks: list[bool] = []
        self.result: Optional[tuple[str, str, list]] = None

        self.title("語句をまとめて直す")
        self.transient(master)
        self.var_before = tk.StringVar()
        self.var_after = tk.StringVar()
        self.var_status = tk.StringVar(value="直す前の語句を入れて［探す］。")
        # 「n / N」。行を選ぶたびに更新する（設計書 §2.2）
        self.var_pos = tk.StringVar(value="")
        # 一致の条件（設計書 §2.2・§2.3）。**探すときと直すときで同じ値を使う**
        # ——search() が self.options に写し、呼び出し側が replace_text に渡す。
        # 既定は部分一致・区別する（日本語には語境界が無く、実物の str.find と同じ）
        self.var_whole = tk.BooleanVar(value=False)
        self.var_nocase = tk.BooleanVar(value=False)
        self.options: dict[str, bool] = {}
        self._build()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda e: self._close())
        self.ent_before.focus_set()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)      # 行: 0 説明 / 1 入力 / 2 条件 / 3 一覧 / 4 状態 / 5 ボタン

        ttk.Label(
            self, wraplength=720,
            text="転写が固有名詞を取り違えたときに、まとめて直します。"
                 "同じ語でも、直してよい箇所とそうでない箇所があります"
                 "(「資格」は本物の資格のこともあります)。"
                 "前後を読んで、直さない行は × にしてください。",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 6))

        top = ttk.Frame(self)
        top.grid(row=1, column=0, columnspan=2, sticky="w", padx=12)
        ttk.Label(top, text="直す前:").pack(side="left")
        self.ent_before = ttk.Entry(top, textvariable=self.var_before, width=18)
        self.ent_before.pack(side="left", padx=(4, 10))
        ttk.Label(top, text="→  直した後:").pack(side="left")
        self.ent_after = ttk.Entry(top, textvariable=self.var_after, width=18)
        self.ent_after.pack(side="left", padx=(4, 10))
        ttk.Button(top, text="探す", command=self.search).pack(side="left")
        self.ent_before.bind("<Return>", lambda e: self.search())

        # 一致の条件。切り替えたら探し直す（○×は付け直しになる）
        opts = ttk.Frame(self)
        opts.grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(4, 0))
        self.chk_whole = ttk.Checkbutton(
            opts, text="完全一致（前後が英数字でない）", variable=self.var_whole,
            command=self._on_option_changed)
        self.chk_whole.pack(side="left")
        self.chk_nocase = ttk.Checkbutton(
            opts, text="英大文字小文字を区別しない", variable=self.var_nocase,
            command=self._on_option_changed)
        self.chk_nocase.pack(side="left", padx=(10, 0))
        ttk.Label(opts, foreground="#888",
                  text="※ 完全一致は英語向けです。日本語には語境界が無いので効きません。")\
            .pack(side="left", padx=(10, 0))

        cols = ("mark", "at", "text")
        self.tree = ttk.Treeview(self, columns=cols, show="headings",
                                 height=12, selectmode="browse")
        for key, label, width, anchor in (
                ("mark", "直す", 44, "center"),
                ("at", "時刻", 78, "center"),
                ("text", "前後", 620, "w")):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, anchor=anchor)
        self.tree.grid(row=3, column=0, sticky="nsew", padx=(12, 0), pady=(8, 0))
        sb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        sb.grid(row=3, column=1, sticky="ns", padx=(0, 12), pady=(8, 0))
        self.tree.configure(yscrollcommand=sb.set)
        # クリックでも Space でも切り替えられるように(片方だけだと迷う)
        self.tree.bind("<Button-1>", self._on_click)
        self.tree.bind("<space>", lambda e: (self._toggle_selected(), "break"))
        # **行を選ぶと本体がその区間に飛ぶ**（設計書 §2.2）。ここで音声を
        # 聴けるのが、Word の検索置換には無いこの画面の値打ち。↓↑ は
        # Treeview がそのまま動かし、Enter は次の行へ
        self.tree.bind("<<TreeviewSelect>>", self._on_row_selected)
        self.tree.bind("<Return>", lambda e: (self._move_row(1), "break"))

        ttk.Label(self, textvariable=self.var_status, foreground="#666",
                  wraplength=720)\
            .grid(row=4, column=0, sticky="w", padx=12, pady=(6, 0))

        btns = ttk.Frame(self)
        btns.grid(row=5, column=0, columnspan=2, sticky="ew", padx=12, pady=10)
        ttk.Button(btns, text="全部に○", command=lambda: self._mark_all(True))\
            .pack(side="left")
        ttk.Button(btns, text="全部に×", command=lambda: self._mark_all(False))\
            .pack(side="left", padx=6)
        self.btn_listen = ttk.Button(btns, text="▶ 聴く", command=self.listen,
                                     state="disabled")
        self.btn_listen.pack(side="left", padx=(12, 0))
        ttk.Label(btns, textvariable=self.var_pos, foreground="#666")\
            .pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="やめる", command=self._close).pack(side="right")
        self.btn_ok = ttk.Button(btns, text="直す", command=self._ok,
                                 state="disabled")
        self.btn_ok.pack(side="right", padx=6)

    # ------------------------------------------------------------------
    def search(self) -> None:
        term = self.var_before.get().strip()
        self.tree.delete(*self.tree.get_children())
        self.hits, self.marks = [], []
        if not term:
            self.var_status.set("直す前の語句を入れてください。")
            self._refresh_ok()
            return
        # **直すときにも同じ値を渡す**（呼び出し側が self.options を使う）。
        # 片方にしか効かないと、一覧に出た箇所と直る箇所がずれる（§2.3）
        self.options = {}
        if self.var_nocase.get():
            self.options["ignore_case"] = True
        if self.var_whole.get():
            self.options["whole_word"] = True
        self.hits = self.win.proj.find_text(term, **self.options)
        self.marks = [True] * len(self.hits)
        for i, h in enumerate(self.hits):
            self.tree.insert("", "end", iid=str(i), values=(
                self.MARK_ON, fmt_short_time(h.at), self._line(h)))
        if not self.hits:
            self.var_status.set(f"「{term}」は本文にありません。")
        self._refresh_ok()
        self._update_pos()
        if self.hits:
            # 最初のヒットを選ぶ → 本体がその区間へ飛ぶ（自動再生の設定に従う）
            self.tree.selection_set("0")
            self.tree.focus("0")
            self.tree.focus_set()

    # ------------------------------------------------------------------
    # 行の選択 → 本体をその区間へ（設計書 §2.2）
    # ------------------------------------------------------------------
    def _on_option_changed(self) -> None:
        """条件を切り替えたら、語句が入っていれば探し直す。"""
        if self.var_before.get().strip():
            self.search()

    def _selected_row(self) -> Optional[int]:
        sel = self.tree.selection()
        if not sel:
            return None
        try:
            i = int(sel[0])
        except ValueError:
            return None
        return i if 0 <= i < len(self.hits) else None

    def _update_pos(self) -> None:
        i = self._selected_row()
        n = len(self.hits)
        self.var_pos.set(f"{(i + 1) if i is not None else 0} / {n}")
        self.btn_listen.configure(state="normal" if i is not None else "disabled")

    def _on_row_selected(self, event=None) -> None:
        """選んだ行の区間へ本体を飛ばす。○×の付け替えは本文を変えない。"""
        self._update_pos()
        i = self._selected_row()
        if i is None:
            return
        self.win.goto_key(self.hits[i].key)

    def _move_row(self, delta: int) -> None:
        """Enter で次の行へ（↓↑ は Treeview に任せる）。"""
        if not self.hits:
            return
        i = self._selected_row()
        j = 0 if i is None else max(0, min(i + delta, len(self.hits) - 1))
        self.tree.selection_set(str(j))
        self.tree.focus(str(j))
        self.tree.see(str(j))

    def listen(self) -> None:
        """選んでいる行の区間を鳴らす。自動再生が切れていても、押せば鳴る。"""
        i = self._selected_row()
        if i is None:
            return
        if self.win.goto_key(self.hits[i].key) is None:
            self.var_status.set("その区間が見つかりません（本文が変わった可能性があります）。")
            return
        self.win.play_current(explicit=True)

    def _line(self, hit) -> str:
        return ("…" if hit.head else "") + hit.before \
            + f"【{hit.term}】" + hit.after + ("…" if hit.tail else "")

    # ------------------------------------------------------------------
    def _on_click(self, event) -> None:
        if self.tree.identify_region(event.x, event.y) != "cell":
            return
        row = self.tree.identify_row(event.y)
        if row and self.tree.identify_column(event.x) == "#1":
            self._toggle(int(row))

    def _toggle_selected(self) -> None:
        sel = self.tree.selection()
        if sel:
            self._toggle(int(sel[0]))

    def _toggle(self, i: int) -> None:
        if not (0 <= i < len(self.marks)):
            return
        self.marks[i] = not self.marks[i]
        self.tree.set(str(i), "mark",
                      self.MARK_ON if self.marks[i] else self.MARK_OFF)
        self._refresh_ok()

    def _mark_all(self, on: bool) -> None:
        for i in range(len(self.marks)):
            self.marks[i] = on
            self.tree.set(str(i), "mark", self.MARK_ON if on else self.MARK_OFF)
        self._refresh_ok()

    def _refresh_ok(self) -> None:
        n = sum(self.marks)
        self.btn_ok.configure(text=f"この {n} 箇所を直す" if n else "直す",
                              state="normal" if n else "disabled")
        if self.hits:
            self.var_status.set(
                f"{len(self.hits)} 箇所のうち {n} 箇所を直します。"
                "直したところに「聴いて確定」の印は付きません"
                "(音声を聴いたわけではないため)。")

    # ------------------------------------------------------------------
    def _ok(self) -> None:
        before = self.var_before.get().strip()
        after = self.var_after.get().strip()
        chosen = [h.target for h, m in zip(self.hits, self.marks) if m]
        if not chosen:
            return
        if before == after:
            messagebox.showwarning(
                "同じ語句です", "「直した後」を変えてください。", parent=self)
            return
        if not after and not messagebox.askyesno(
                "語句を消します",
                f"「{before}」を {len(chosen)} 箇所で消します。よろしいですか。",
                parent=self):
            return
        self.result = (before, after, chosen)
        self._close()

    def _close(self) -> None:
        self.grab_release()
        self.destroy()


class ProposalDialog(tk.Toplevel):
    """点検が出した提案の一覧。承認するか却下するかを決めてもらう。

    ここは見せて選ばせるだけで、本体データには触らない。決まったぶんだけ
    呼び出し側の関数を呼び、適用は既存の時刻編集の経路にやらせる
    (点検専用の書き込み経路は作らない)。

    決め方は話者の ✓/△ と同じ形にしてある:
      - [聴いて承認]   選んだ行を再生して確かめたうえで採る → ✎
      - [まとめて適用] 残りを聴かずに当てる → ✎△(あとで聴いて ✎ に上げる)
      - [却下]         その行は採らない

    行は 1 つずつ選ぶ(選ぶと再生する)。まとめて承認できるようにすると、
    聴かずに ✎ を付けられてしまい、✎ の意味が壊れる。

    ※ 提案を作る側(align.py / anchor.py / inspect.py)と、この画面を開く
      ボタンは Step 1-3a で入れる。いまは表示と決定の受け口だけ。
    """

    COLUMNS = (
        ("kind", "種別", 60),
        ("target", "対象", 80),
        ("now", "いまの時刻", 100),
        ("measured", "実測の時刻", 100),
        ("delta", "ずれ", 70),
        ("evidence", "根拠", 190),
        ("confidence", "信頼度", 70),
        ("text", "発言", 300),
    )

    def __init__(
        self,
        master: "AssignWindow",
        rows: Iterable[ProposalRow],
        *,
        on_play=None,       # (key) 行を選んだとき。提案した時刻で再生する
        on_play_now=None,   # (key) 比べるために、いまの時刻でも再生する
        on_accept=None,     # (key) 聴いて承認した
        on_bulk=None,       # (keys) 残りをまとめて適用した
        on_reject=None,     # (key) 却下した
    ) -> None:
        super().__init__(master)
        self.win = master
        self.on_play = on_play
        self.on_play_now = on_play_now
        self.on_accept = on_accept
        self.on_bulk = on_bulk
        self.on_reject = on_reject
        self.rows: dict[str, ProposalRow] = {r.key: r for r in rows}
        # 何をどう決めたかの記録。呼び出し側が後から見返せるようにする
        self.decisions: list[tuple[str, tuple[str, ...]]] = []

        self.title("点検の提案")
        self.transient(master)
        self.var_status = tk.StringVar(value="")
        self._build()
        self._update_status()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda e: self._close())

    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        ttk.Label(
            self,
            text="行を選ぶとその箇所を再生します。聴いて合っていれば[聴いて承認]。"
                 "[まとめて適用]した区間は ✎△(未確認)のままなので、あとから"
                 "聴いて確かめられます。",
            wraplength=760,
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 6))

        keys = tuple(key for key, _, _ in self.COLUMNS)
        self.tree = ttk.Treeview(self, columns=keys, show="headings",
                                 height=14, selectmode="browse")
        for key, label, width in self.COLUMNS:
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width,
                             anchor="w" if key in ("evidence", "text") else "center")
        for row in self.rows.values():
            self._insert(row)
        self.tree.grid(row=1, column=0, sticky="nsew", padx=(12, 0))
        sb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        sb.grid(row=1, column=1, sticky="ns", padx=(0, 12))
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        ttk.Label(self, textvariable=self.var_status, foreground="#666")\
            .grid(row=2, column=0, sticky="w", padx=12, pady=(4, 0))

        btns = ttk.Frame(self)
        btns.grid(row=3, column=0, columnspan=2, sticky="e", padx=12, pady=10)
        # 行を選ぶと提案した時刻で鳴る。合っているかは、いまの時刻と
        # 聴き比べると分かりやすい
        ttk.Button(btns, text="▶ 提案の時刻", command=self._play_again)\
            .pack(side="left")
        ttk.Button(btns, text="▶ いまの時刻", command=self._play_now)\
            .pack(side="left", padx=(4, 18))
        ttk.Button(btns, text="聴いて承認", command=self._accept).pack(side="left")
        ttk.Button(btns, text="却下", command=self._reject).pack(side="left", padx=6)
        ttk.Button(btns, text="残りをまとめて適用", command=self._bulk)\
            .pack(side="left", padx=(18, 6))
        ttk.Button(btns, text="閉じる", command=self._close).pack(side="left")

    def _insert(self, row: ProposalRow) -> None:
        self.tree.insert("", "end", iid=row.key, values=(
            row.kind, row.target, row.now, row.measured,
            row.delta, row.evidence, row.confidence, row.text,
        ))

    def _selected_key(self) -> Optional[str]:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _update_status(self) -> None:
        self.var_status.set(f"残り {len(self.rows)} 件")

    def _on_select(self, _event=None) -> None:
        self._play_again()

    def _play_again(self) -> None:
        key = self._selected_key()
        if key and self.on_play:
            self.on_play(key)

    def _play_now(self) -> None:
        """比べるために、いまの(直す前の)時刻でも鳴らせるようにする。"""
        key = self._selected_key()
        if key and self.on_play_now:
            self.on_play_now(key)

    # ------------------------------------------------------------------
    def _accept(self) -> None:
        key = self._selected_key()
        if key is None:
            self.var_status.set("承認する行を選んでください。")
            return
        if self.on_accept:
            self.on_accept(key)
        self._done("accept", (key,))

    def _reject(self) -> None:
        key = self._selected_key()
        if key is None:
            self.var_status.set("却下する行を選んでください。")
            return
        if self.on_reject:
            self.on_reject(key)
        self._done("reject", (key,))

    def _bulk(self) -> None:
        keys = tuple(self.rows)
        if not keys:
            return
        ok = messagebox.askyesno(
            "残りをまとめて適用",
            f"残り {len(keys)} 件を聴かずに適用します。\n"
            "適用した区間は ✎△(未確認)になります。あとから聴いて"
            "確かめると ✎ に変わります。\n\nよろしいですか?",
            parent=self,
        )
        if not ok:
            return
        if self.on_bulk:
            self.on_bulk(keys)
        self._done("bulk", keys)

    def _done(self, kind: str, keys: tuple[str, ...]) -> None:
        """決まった行を一覧から外す(同じ提案を二度決めさせない)。"""
        self.decisions.append((kind, keys))
        for key in keys:
            self.rows.pop(key, None)
            if self.tree.exists(key):
                self.tree.delete(key)
        self._update_status()

    def _close(self) -> None:
        self.win.player.stop()
        self.grab_release()
        self.destroy()


def _open_path(path: Path | str) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)])
        else:
            subprocess.run(["xdg-open", str(path)])
    except Exception:
        pass


def open_assign_window(master: Optional[tk.Misc], project: Project) -> AssignWindow:
    win = AssignWindow(master, project)
    win.focus_set()
    return win
