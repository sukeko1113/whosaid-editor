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
import subprocess
import sys
import tkinter as tk
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from .player import SegmentPlayer
from .segments import (
    Project,
    SPECIAL_MULTI,
    SPECIAL_NOISE,
    SPECIAL_UNKNOWN,
    Speaker,
    fmt_hms,
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
        right.rowconfigure(1, weight=1)
        right.rowconfigure(3, weight=1)

        ttk.Label(right, textvariable=self.var_seginfo, font=("", 10, "bold"))\
            .grid(row=0, column=0, sticky="w", padx=6, pady=(0, 2))

        frm_text = ttk.LabelFrame(right, text="この区間の発言(編集できます)")
        frm_text.grid(row=1, column=0, sticky="nsew", padx=4, pady=2)
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
        frm_play.grid(row=2, column=0, sticky="ew", padx=4, pady=2)
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
        frm_cand.grid(row=3, column=0, sticky="nsew", padx=4, pady=2)
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
        values = (fmt_hms(seg.start), seg.cluster_label,
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
        pos = self.current + 1
        state = "未確定"
        if seg.speaker_id:
            state = "確認済み" if seg.reviewed else "△まとめて適用(未確認)"
        plen = preview_length(seg.text, seg.duration)
        long_note = (f"(再生は先頭{plen:.0f}秒)" if plen < seg.duration - 0.5 else "")
        self.var_seginfo.set(
            f"区間 {pos}/{len(self.proj.segments)}   "
            f"[{fmt_hms(seg.start)} → {fmt_hms(seg.end)}]  {seg.duration:.0f}秒{long_note}   "
            f"{seg.cluster_label}({self.suggester.cluster_summary(seg.cluster)})   "
            f"{state}"
            + (f"   ずれ補正 {self.proj.time_offset:+.1f}秒" if self.proj.time_offset else "")
        )
        self.txt_body.delete("1.0", "end")
        self.txt_body.insert("1.0", seg.text)
        self._rebuild_candidates()
        self._draw_timeline()

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
        # 本文に見合う長さだけ再生する(「この先30秒▶」で続きを聴ける)。
        preview_end = seg.end
        if extend <= 0:
            preview_end = seg.start + preview_length(seg.text, seg.duration)

        # 短い区間で先読みを固定 0.4 秒にすると、再生窓の大半が前の発言に
        # なってしまう(1 秒の区間なら 4 割)。区間の長さに応じて縮める。
        pre_roll = min(0.4, max(0.05, seg.duration * 0.25))

        # ずれ補正: Gemini の時刻推定が実音声とずれている場合の手動調整。
        shift = self.proj.time_offset
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
                start=max(0.0, seg.start + shift - back),
                end=preview_end + shift + extend,
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

    def _on_close(self) -> None:
        self._commit_text()
        if self._dirty and self.proj.json_path:
            try:
                self.proj.save()
            except Exception:
                pass
        self.player.close()
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
