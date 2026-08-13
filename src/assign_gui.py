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
from typing import Iterable, Optional

from .align import DEFAULT_MODEL, AlignUnavailable, transcribe_words
from .audio import extract_peaks
from .config import load_config
from .inspection import (
    Proposal,
    clip_to_neighbours,
    inspect_times,
    load_proposals,
    merge_history,
    proposals_path,
    save_proposals,
    target_segment,
)
from .player import SegmentPlayer
from .segments import (
    MIN_SEGMENT_SECONDS,
    Project,
    SPECIAL_MULTI,
    SPECIAL_NOISE,
    SPECIAL_UNKNOWN,
    Segment,
    Speaker,
    audio_span,
    fmt_hms,
    fmt_hms_frac,
    parse_hms,
    parse_roster,
    roster_to_text,
    write_docx,
    write_text,
)
from .suggest import SpeakerSuggester, next_unassigned, next_unreviewed
from .transcribe import estimate_speech_seconds


SPEEDS = ["0.8x", "1.0x", "1.2x", "1.5x", "2.0x"]

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

# 一覧の絞り込み
FILTER_ALL = "all"
FILTER_UNASSIGNED = "unassigned"
FILTER_UNREVIEWED = "unreviewed"

FILTER_LABELS = [
    ("すべて表示", FILTER_ALL),
    ("未確定のみ", FILTER_UNASSIGNED),
    ("未確認のみ(一括適用したぶんを含む)", FILTER_UNREVIEWED),
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
            for seg in proj.segments:
                if seg.speaker_id in removed_ids:
                    seg.speaker_id = None
                    seg.reviewed = False


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
        self._candidates: list = []
        self._cand_widgets: list[ttk.Button] = []
        self._row_ids: list[str] = []

        self.title(f"話者の割当 - {Path(self.proj.audio_path).name}")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.var_speed = tk.StringVar(value="1.0x")
        self.var_autoplay = tk.BooleanVar(value=True)
        self.var_advance = tk.BooleanVar(value=True)
        # 一括適用は既定で ON。これが推奨の進め方で、これを使わないと
        # 90 分の会議で数百回の判断が必要になる。
        self.var_apply_cluster = tk.BooleanVar(value=True)
        self.var_filter = tk.StringVar(value=FILTER_ALL)
        self.var_status = tk.StringVar(value="")
        self.var_seginfo = tk.StringVar(value="")
        self.var_action = tk.StringVar(value="")
        self.var_backend = tk.StringVar(value="")
        self.var_offset = tk.DoubleVar(value=float(project.time_offset))
        self.var_start = tk.StringVar(value="")
        self.var_end = tk.StringVar(value="")

        self._build_ui()
        self._bind_keys()
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
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        filt = ttk.Frame(left)
        filt.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        for label, value in FILTER_LABELS:
            ttk.Radiobutton(
                filt, text=label, value=value, variable=self.var_filter,
                command=self._on_filter_change, takefocus=False,
            ).pack(side="left", padx=(0, 8))

        cols = ("time", "cluster", "speaker", "text")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("time", text="時刻")
        self.tree.heading("cluster", text="声")
        self.tree.heading("speaker", text="話者")
        self.tree.heading("text", text="発言")
        self.tree.column("time", width=78, anchor="w", stretch=False)
        self.tree.column("cluster", width=56, anchor="center", stretch=False)
        self.tree.column("speaker", width=120, anchor="w", stretch=False)
        self.tree.column("text", width=340, anchor="w")
        self.tree.grid(row=1, column=0, sticky="nsew")
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        sb.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.tag_configure("unassigned", background="#FFF8E1")
        self.tree.tag_configure("bulk", background="#F1F6FB")
        self.tree.tag_configure("special", foreground="#8A8A8A")

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
        row_time = ttk.Frame(frm_time)
        row_time.pack(side="top", anchor="w")
        ttk.Label(row_time, text="時刻:").pack(side="left", padx=(0, 8))
        for which, label in (("start", "開始"), ("end", "終了")):
            ttk.Label(row_time, text=label).pack(side="left", padx=(6, 2))
            ent = ttk.Entry(
                row_time, width=11, justify="center",
                textvariable=self.var_start if which == "start" else self.var_end,
            )
            ent.pack(side="left", padx=(0, 2))
            # Enter は「この値でいい」という意思表示。ずれ補正込みの初期値を
            # そのまま押しても、その時刻でこの区間を固定する。
            ent.bind("<Return>", lambda e, w=which: self._commit_time(w, explicit=True))
            ent.bind("<FocusOut>", lambda e, w=which: self._commit_time(w))
            for text, delta in (("−1", -1.0), ("−0.1", -0.1), ("+0.1", +0.1), ("+1", +1.0)):
                ttk.Button(
                    row_time, text=text, width=5, takefocus=False,
                    command=lambda w=which, d=delta: self._nudge_time(w, d),
                ).pack(side="left", padx=1)
        # 機械が当てただけの時刻(✎△)を、聴いて確かめたら ✎ に上げる。
        # 「再生 → 合っていればこれを押す」で流せるようにボタンにしてある。
        self.btn_confirm_time = ttk.Button(
            row_time, text="この時刻で確認", takefocus=False,
            command=self.confirm_time,
        )
        self.btn_confirm_time.pack(side="left", padx=(12, 0))
        self.btn_revert_time = ttk.Button(
            row_time, text="元に戻す", takefocus=False, command=self.revert_time,
        )
        self.btn_revert_time.pack(side="left", padx=(6, 0))

        row_edit = ttk.Frame(frm_time)
        row_edit.pack(side="top", anchor="w", pady=(3, 0))
        # 時刻のずれは区間ごと動かして直すことがほとんど。上の開始・終了は
        # 片側だけ動かす(長さを変える)ので、区間ごとの移動は別に用意する。
        ttk.Label(row_edit, text="区間ごとスライド:").pack(side="left", padx=(0, 4))
        for text, delta in (("−1", -1.0), ("−0.1", -0.1), ("+0.1", +0.1), ("+1", +1.0)):
            ttk.Button(row_edit, text=text, width=5, takefocus=False,
                       command=lambda d=delta: self._shift_time(d)).pack(side="left", padx=1)
        ttk.Separator(row_edit, orient="vertical").pack(side="left", fill="y", padx=10)

        # 1 区間に 2 人の発言が順に混ざることも、同じ発言が 2 行に割れることも
        # ある。どちらも人の目と耳でしか判断できないので、明示操作で直せるようにする。
        ttk.Button(row_edit, text="この区間を分割...", takefocus=False,
                   command=self.split_current).pack(side="left")
        ttk.Button(row_edit, text="前の区間と結合", takefocus=False,
                   command=self.merge_with_prev).pack(side="left", padx=6)
        ttk.Button(row_edit, text="次の区間と結合", takefocus=False,
                   command=self.merge_with_next).pack(side="left")

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
        self.cand_holder = ttk.Frame(frm_cand)
        self.cand_holder.grid(row=0, column=0, sticky="nsew", padx=6, pady=4)
        self.cand_holder.columnconfigure(0, weight=1)

        # 直前の操作の結果。区間を移動しても消えないように専用の行にする
        # (一括適用が何区間に効いたのかが分からないと、事故に気づけない)
        ttk.Label(frm_cand, textvariable=self.var_action, foreground="#1B5E20",
                  wraplength=560).grid(row=1, column=0, sticky="w", padx=8, pady=(2, 0))

        opts = ttk.Frame(frm_cand)
        opts.grid(row=2, column=0, sticky="ew", padx=6, pady=(2, 6))
        ttk.Checkbutton(opts, text="同じ声のまとまり全体に適用 (A)",
                        variable=self.var_apply_cluster, takefocus=False).pack(side="left")
        ttk.Checkbutton(opts, text="確定したら次へ", variable=self.var_advance,
                        takefocus=False).pack(side="left", padx=12)
        ttk.Button(opts, text="不明 (U)", command=lambda: self.assign(SPECIAL_UNKNOWN))\
            .pack(side="right")
        ttk.Button(opts, text="未確定に戻す (D)", command=self.unassign).pack(side="right", padx=6)
        ttk.Button(opts, text="取り消し (Ctrl+Z)", command=self.undo).pack(side="right")

        # --- 下部: ボタン ----------------------------------------------
        bottom = ttk.Frame(self, padding=(10, 4, 10, 10))
        bottom.grid(row=3, column=0, sticky="ew")
        ttk.Button(bottom, text="出席者を編集...", command=self.edit_roster).pack(side="left")
        ttk.Button(bottom, text="残作業を一覧...", command=self.show_remaining).pack(side="left", padx=6)
        ttk.Button(bottom, text="このまとまりを未確定に戻す", command=self.unassign_cluster)\
            .pack(side="left")
        self.btn_inspect = ttk.Button(bottom, text="時刻を点検...",
                                      command=self.run_inspection)
        self.btn_inspect.pack(side="left", padx=6)
        ttk.Button(bottom, text="Word で出力...", command=self.export_docx).pack(side="right")
        ttk.Button(bottom, text="保存", command=self.save).pack(side="right", padx=6)

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

    def _toggle_cluster_mode(self) -> None:
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
        return True

    def _visible_indexes(self) -> list[int]:
        if self.var_filter.get() == FILTER_ALL:
            return [s.index for s in self.proj.segments]
        return [s.index for s in self.proj.segments if self._match_filter(s)]

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
        self._set_action(f"表示: {label}({len(vis)} 区間)")

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
        tags: list[str] = []
        if not seg.speaker_id:
            tags.append("unassigned")
        else:
            if seg.speaker_id in (SPECIAL_UNKNOWN, SPECIAL_MULTI, SPECIAL_NOISE):
                tags.append("special")
            if not seg.reviewed:
                tags.append("bulk")
                mark = "△"      # まとめて適用しただけ(自分の耳では未確認)
            else:
                mark = "✓"
        # 時刻をどこまで人が確かめたかも見えるようにする(話者の ✓/△ と同じ)
        time_mark = ""
        if seg.time_edited:
            time_mark = "✎ " if seg.time_reviewed else "✎△"
        time_cell = time_mark + fmt_hms(seg.start)
        values = (time_cell, seg.cluster_label,
                  f"{mark}{name}" if name else "—", seg.preview(70))
        return values, tuple(tags)

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

    def _on_tree_select(self, event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        try:
            index = int(sel[0][1:])
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
        self._rebuild_candidates()
        self._draw_timeline()

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
        self.btn_revert_time.state(["!disabled"] if seg.time_edited else ["disabled"])
        # 確認済みの区間で押しても意味がない(それ以外は押せる。まだ直して
        # いない区間でも「聴いてこの時刻で合っている」と言えるため)
        done = seg.time_edited and seg.time_reviewed
        self.btn_confirm_time.state(["disabled"] if done else ["!disabled"])

    def _commit_time(self, which: str, explicit: bool = False) -> None:
        """入力欄の値を区間に反映する。読めない値は元に戻して知らせる。"""
        if not self.proj.segments:
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
                     reviewed: bool, refresh: bool = True) -> None:
        """区間に時刻を書く、ただ 1 つの経路。

        画面で直したときも、点検の提案を当てたときも必ずここを通す。
        経路を分けると、どちらかで ✎/✎△ の意味が食い違っても気づけない。

        reviewed: その時刻を人が耳で確かめたか。画面で決めた時刻は True、
        点検の提案をまとめて当てただけなら False(✎△ のまま残す)。
        refresh: 画面を描き直すか。まとめて適用では 1 件ごとに描き直すと
        百件級で待たされるので、呼び出し側が最後に 1 回だけ描き直す。
        """
        seg.start, seg.end = start, end
        seg.time_edited = True          # 以後この区間にずれ補正を足さない
        seg.time_reviewed = reviewed
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
        """
        planned, failed = self.plan_proposals(proposals)
        for seg, start, end, p in planned:
            self._write_times(seg, start, end, reviewed=False, refresh=False)
            p.status = "accepted"
        if planned:
            self.reload_tree()
            self._show_times()
            self._update_seginfo()
        return [p for _, _, _, p in planned], failed

    def revert_time(self) -> None:
        """パイプラインが出した元の時刻に戻す(以後はまたずれ補正が効く)。"""
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        if not seg.time_edited:
            self._set_action("この区間の時刻は直されていません。")
            return
        seg.start = float(seg.orig_start)
        seg.end = float(seg.orig_end)
        seg.time_edited = False
        seg.time_reviewed = False
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
        if not self.proj.segments:
            return
        new = self.txt_body.get("1.0", "end").strip()
        seg = self.proj.segments[self.current]
        if new != seg.text:
            seg.text = new
            seg.text_edited = True      # 再実行時に上書きされないよう印を付ける
            self._dirty = True
            self._update_row(seg.index)

    # ==================================================================
    # 候補者リスト
    # ==================================================================
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

        row = len(self._candidates)
        for sid, label in ((SPECIAL_UNKNOWN, "発言者不明"),
                           (SPECIAL_MULTI, "複数人が同時"),
                           (SPECIAL_NOISE, "発言なし・雑音")):
            mark = "●" if seg.speaker_id == sid else "　"
            btn = ttk.Button(self.cand_holder, text=f"{mark} {label}",
                             command=lambda s=sid: self.assign(s))
            btn.grid(row=row, column=0, sticky="ew", pady=1)
            self._cand_widgets.append(btn)
            row += 1

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

        for s in targets:
            s.speaker_id = speaker_id
            # 自分で聴いた区間だけ「確認済み」。まとめて埋めた分は未確認扱い。
            s.reviewed = bool(speaker_id) and s.index == seg.index
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
        for s in targets:
            s.speaker_id = None
            s.reviewed = False
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
        for index, sid, reviewed in snapshot:
            seg = self.proj.segments[index]
            seg.speaker_id = sid
            seg.reviewed = reviewed
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
        dlg = tk.Toplevel(self)
        dlg.title("出席者(候補者リスト)")
        dlg.geometry("480x420")
        dlg.transient(self)
        dlg.grab_set()
        dlg.columnconfigure(0, weight=1)
        dlg.rowconfigure(1, weight=1)

        ttk.Label(
            dlg,
            text="1 行 1 人。「名前(役職)」の形式も使えます。\n"
                 "上から順に並ぶので、よく発言する人を上に置くと最初の候補順が良くなります。",
            foreground="#555", justify="left",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))

        txt = tk.Text(dlg, wrap="none", font=("", 11))
        txt.grid(row=1, column=0, sticky="nsew", padx=10)
        txt.insert("1.0", roster_to_text(self.proj.speakers))
        txt.focus_set()

        btns = ttk.Frame(dlg)
        btns.grid(row=2, column=0, sticky="ew", padx=10, pady=10)

        def ok() -> None:
            plan = plan_roster_text(self.proj, txt.get("1.0", "end"))
            if plan.removed:
                names = "、".join(sp.display for sp in plan.removed)
                detail = (
                    f"次の出席者が削除されます:\n\n{names}\n\n"
                    + (f"この人たちに割り当てていた {plan.affected_segments} 区間は未確定に戻ります。\n"
                       if plan.affected_segments else "")
                    + "続けますか?"
                )
                if not messagebox.askyesno("確認", detail, parent=dlg):
                    return          # ここまで何も変更していないので、そのまま編集を続けられる
            plan.apply(self.proj)
            self._dirty = True
            dlg.destroy()
            self.refresh_all()

        ttk.Button(btns, text="OK", command=ok).pack(side="right")
        ttk.Button(btns, text="キャンセル", command=dlg.destroy).pack(side="right", padx=6)
        dlg.bind("<Escape>", lambda e: dlg.destroy())

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

    def export_docx(self) -> None:
        self._commit_text()
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
        for i, (text, var) in enumerate((
            ("段落の先頭に時刻を入れる", var_ts),
            ("同じ話者の連続発言をまとめる", var_merge),
            ("冒頭に出席者一覧を入れる", var_attend),
            ("「発言なし・雑音」と印を付けた区間を省く", var_noise),
            ("同じ内容のテキストファイル(.txt)も出す", var_txt),
        )):
            ttk.Checkbutton(dlg, text=text, variable=var)\
                .grid(row=2 + i, column=0, columnspan=2, sticky="w", padx=12, pady=1)

        result: dict[str, object] = {}

        def ok() -> None:
            result["go"] = True
            dlg.destroy()

        btns = ttk.Frame(dlg)
        btns.grid(row=8, column=0, columnspan=2, sticky="ew", padx=12, pady=12)
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
            out = write_docx(
                self.proj, path,
                title=var_title.get() or None,
                with_timestamps=var_ts.get(),
                merge_consecutive=var_merge.get(),
                include_attendees=var_attend.get(),
                drop_noise=var_noise.get(),
            )
            if var_txt.get():
                write_text(self.proj, Path(path).with_suffix(".txt"),
                           merge_consecutive=var_merge.get(), drop_noise=var_noise.get())
        except Exception:
            messagebox.showerror("出力エラー", traceback.format_exc(), parent=self)
            return
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
        fresh = merge_history(result.proposals, load_proposals(path))

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
        self._open_proposals(fresh, path)

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
        当てられなかった提案は却下として残す。次の点検で同じものを
        出し直しても、また当てられないため。
        """
        if decision == "reject":
            proposal.status = "rejected"
            return
        applied = self.apply_proposal(proposal, reviewed=(decision == "accept"))
        proposal.status = "accepted" if applied else "rejected"

    def _open_proposals(self, proposals, path: Optional[Path]) -> None:
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
            for p in failed:
                p.status = "rejected"       # 当てられない提案は繰り返さない
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
        # 判断を残す。却下したものを再点検で出し直さないため(§6.1)
        save_proposals(path, proposals)
        self.update_status()

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
