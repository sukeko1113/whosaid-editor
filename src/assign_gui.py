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
)
from .suggest import SpeakerSuggester, next_unassigned


SPEEDS = ["0.8x", "1.0x", "1.2x", "1.5x", "2.0x"]

# 話者ごとの色(タイムライン帯・一覧の色分け用)
PALETTE = [
    "#3E7CB1", "#C1666B", "#4F9D69", "#B07B2F", "#7A5AA8",
    "#2E8B8B", "#C4622D", "#6A7B3C", "#A03E7C", "#4A6FA5",
]
COLOR_UNASSIGNED = "#DDDDDD"
COLOR_SPECIAL = "#9A9A9A"

QUICK_KEYS = "123456789"


def _apply_roster_text(proj: Project, text: str) -> tuple[int, int, list[str]]:
    """出席者リストのテキストを Project に反映する。

    既存の話者 ID は名前一致で保持するので、既に確定した区間の割当は壊れない。
    戻り値: (追加数, 削除数, 削除された話者名)
    """
    wanted = parse_roster(text)
    by_name = {sp.name: sp for sp in proj.speakers}

    new_list: list[Speaker] = []
    added = 0
    for i, w in enumerate(wanted):
        existing = by_name.pop(w.name, None)
        if existing:
            existing.note = w.note
            existing.order = i
            new_list.append(existing)
        else:
            sid = _fresh_id(proj, new_list)
            new_list.append(Speaker(id=sid, name=w.name, note=w.note, order=i))
            added += 1

    removed_names = [sp.name for sp in by_name.values()]
    removed_ids = {sp.id for sp in by_name.values()}
    proj.speakers = new_list
    if removed_ids:
        for seg in proj.segments:
            if seg.speaker_id in removed_ids:
                seg.speaker_id = None
                seg.reviewed = False
    return added, len(removed_names), removed_names


def _fresh_id(proj: Project, pending: list[Speaker]) -> str:
    used = {sp.id for sp in proj.speakers} | {sp.id for sp in pending}
    i = 1
    while f"sp{i:02d}" in used:
        i += 1
    return f"sp{i:02d}"


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
        self.var_apply_cluster = tk.BooleanVar(value=False)
        self.var_only_unassigned = tk.BooleanVar(value=False)
        self.var_status = tk.StringVar(value="")
        self.var_seginfo = tk.StringVar(value="")
        self.var_backend = tk.StringVar(value="")

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
        ttk.Checkbutton(filt, text="未確定のみ表示", variable=self.var_only_unassigned,
                        command=self.reload_tree).pack(side="left")
        ttk.Label(filt, text="  (↑↓ で移動 / Tab で次の未確定へ)", foreground="#777")\
            .pack(side="left")

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
        ttk.Checkbutton(frm_play, text="移動したら自動再生", variable=self.var_autoplay)\
            .grid(row=0, column=4, padx=(12, 4))
        ttk.Label(frm_play, textvariable=self.var_backend, foreground="#888")\
            .grid(row=1, column=0, columnspan=5, sticky="w", padx=8, pady=(0, 4))

        # --- 候補者リスト ----------------------------------------------
        frm_cand = ttk.LabelFrame(right, text="話者を選ぶ(数字キーで即確定・可能性の高い順)")
        frm_cand.grid(row=3, column=0, sticky="nsew", padx=4, pady=2)
        frm_cand.columnconfigure(0, weight=1)
        frm_cand.rowconfigure(0, weight=1)
        self.cand_holder = ttk.Frame(frm_cand)
        self.cand_holder.grid(row=0, column=0, sticky="nsew", padx=6, pady=4)
        self.cand_holder.columnconfigure(0, weight=1)

        opts = ttk.Frame(frm_cand)
        opts.grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
        ttk.Checkbutton(opts, text="同じ声のまとまり全体に適用 (A)",
                        variable=self.var_apply_cluster).pack(side="left")
        ttk.Checkbutton(opts, text="確定したら次の未確定へ", variable=self.var_advance)\
            .pack(side="left", padx=12)
        ttk.Button(opts, text="不明にする (U)", command=lambda: self.assign(SPECIAL_UNKNOWN))\
            .pack(side="right")
        ttk.Button(opts, text="取り消し (Ctrl+Z)", command=self.undo).pack(side="right", padx=6)

        # --- 下部: ボタン ----------------------------------------------
        bottom = ttk.Frame(self, padding=(10, 4, 10, 10))
        bottom.grid(row=3, column=0, sticky="ew")
        ttk.Button(bottom, text="出席者を編集...", command=self.edit_roster).pack(side="left")
        ttk.Button(bottom, text="未確定を一覧...", command=self.show_remaining).pack(side="left", padx=6)
        ttk.Button(bottom, text="Word で出力", command=self.export_docx).pack(side="right")
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
        self.bind("<Tab>", lambda e: self._goto_next_unassigned())
        self.bind("<Shift-Tab>", lambda e: self._goto_next_unassigned(forward=False))
        # X11 では Shift+Tab は ISO_Left_Tab として届く
        self.bind("<ISO_Left_Tab>", lambda e: self._goto_next_unassigned(forward=False))
        for ch in QUICK_KEYS:
            self.bind(ch, self._key_digit)
        self.bind("u", lambda e: self._guarded(lambda: self.assign(SPECIAL_UNKNOWN)))
        self.bind("U", lambda e: self._guarded(lambda: self.assign(SPECIAL_UNKNOWN)))
        self.bind("a", lambda e: self._guarded(self._toggle_cluster_mode))
        self.bind("A", lambda e: self._guarded(self._toggle_cluster_mode))
        self.bind("j", lambda e: self._guarded(lambda: self.move(1)))
        self.bind("k", lambda e: self._guarded(lambda: self.move(-1)))

    def _typing(self) -> bool:
        """本文編集中はショートカットを無効にする"""
        return isinstance(self.focus_get(), (tk.Text, tk.Entry, ttk.Entry))

    def _guarded(self, fn) -> Optional[str]:
        if self._typing():
            return None
        fn()
        return "break"

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
        done, total = self.proj.assigned_count, self.proj.total_count
        pct = (done / total * 100) if total else 0
        self.progress.configure(maximum=max(1, total), value=done)
        self.var_status.set(f"{done} / {total} 区間 確定 ({pct:.0f}%)")

    def _visible_indexes(self) -> list[int]:
        if self.var_only_unassigned.get():
            return [s.index for s in self.proj.segments if not s.speaker_id]
        return [s.index for s in self.proj.segments]

    def reload_tree(self) -> None:
        sel = self.current
        self.tree.delete(*self.tree.get_children())
        self._row_ids = []
        for idx in self._visible_indexes():
            seg = self.proj.segments[idx]
            self._row_ids.append(self._insert_row(seg))
        self._draw_timeline()
        self._select_index(sel, scroll=True)

    def _insert_row(self, seg) -> str:
        name = self.proj.speaker_name(seg.speaker_id)
        tags = []
        if not seg.speaker_id:
            tags.append("unassigned")
        elif seg.speaker_id in (SPECIAL_UNKNOWN, SPECIAL_MULTI, SPECIAL_NOISE):
            tags.append("special")
        return self.tree.insert(
            "", "end", iid=f"s{seg.index}",
            values=(fmt_hms(seg.start), seg.cluster_label, name or "—", seg.preview(70)),
            tags=tuple(tags),
        )

    def _update_row(self, index: int) -> None:
        iid = f"s{index}"
        if not self.tree.exists(iid):
            return
        seg = self.proj.segments[index]
        name = self.proj.speaker_name(seg.speaker_id)
        tags = []
        if not seg.speaker_id:
            tags.append("unassigned")
        elif seg.speaker_id in (SPECIAL_UNKNOWN, SPECIAL_MULTI, SPECIAL_NOISE):
            tags.append("special")
        self.tree.item(
            iid,
            values=(fmt_hms(seg.start), seg.cluster_label, name or "—", seg.preview(70)),
            tags=tuple(tags),
        )

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
        vis = self._visible_indexes()
        if not vis:
            return
        if self.current in vis:
            pos = vis.index(self.current)
        else:
            pos = min(range(len(vis)), key=lambda i: abs(vis[i] - self.current))
        self.goto(vis[max(0, min(pos + delta, len(vis) - 1))])

    def _goto_next_unassigned(self, forward: bool = True) -> str:
        nxt = next_unassigned(self.proj, self.current, forward)
        if nxt is None:
            self.bell()
            self.var_seginfo.set(self.var_seginfo.get() + "   ← これ以降に未確定はありません")
        else:
            self.goto(nxt)
        return "break"

    def show_current(self) -> None:
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
        pos = self.current + 1
        self.var_seginfo.set(
            f"区間 {pos}/{len(self.proj.segments)}   "
            f"[{fmt_hms(seg.start)} → {fmt_hms(seg.end)}]  {seg.duration:.0f}秒   "
            f"声のまとまり {seg.cluster_label}  "
            f"({self.suggester.cluster_summary(seg.cluster)})"
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
    def assign(self, speaker_id: str) -> None:
        if not self.proj.segments:
            return
        self._commit_text()
        seg = self.proj.segments[self.current]

        targets = [seg]
        if self.var_apply_cluster.get():
            targets = [
                s for s in self.proj.cluster_segments(seg.cluster)
                if s.index == seg.index or not s.speaker_id
            ]

        snapshot = [(s.index, s.speaker_id, s.reviewed) for s in targets]
        self._undo.append(snapshot)
        del self._undo[:-200]

        for s in targets:
            s.speaker_id = speaker_id
            s.reviewed = True
        self._dirty = True

        self.suggester.refresh()
        for s in targets:
            self._update_row(s.index)
        self.update_status()
        self._draw_timeline()

        if len(targets) > 1:
            name = self.proj.speaker_name(speaker_id)
            self.var_seginfo.set(f"「{name}」を {len(targets)} 区間にまとめて適用しました。")

        if self.var_advance.get():
            nxt = next_unassigned(self.proj, self.current, True)
            if nxt is not None:
                if self.var_only_unassigned.get():
                    self.reload_tree()
                self.goto(nxt)
                return
            self.player.stop()
            self._rebuild_candidates()
            messagebox.showinfo("完了", "未確定の区間がなくなりました。\nWord で出力できます。", parent=self)
        if self.var_only_unassigned.get():
            self.reload_tree()
        else:
            self._rebuild_candidates()

    def undo(self) -> None:
        if not self._undo:
            self.bell()
            return
        snapshot = self._undo.pop()
        for index, sid, reviewed in snapshot:
            seg = self.proj.segments[index]
            seg.speaker_id = sid
            seg.reviewed = reviewed
        self._dirty = True
        self.suggester.refresh()
        if self.var_only_unassigned.get():
            self.reload_tree()
        else:
            for index, _, _ in snapshot:
                self._update_row(index)
        self.update_status()
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

    def play_current(self, back: float = 0.0, explicit: bool = False) -> None:
        if not self.proj.segments:
            return
        seg = self.proj.segments[self.current]
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
                start=max(0.0, seg.start - back),
                end=seg.end,
                speed=self._speed(),
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
            text = txt.get("1.0", "end")
            added, removed, names = _apply_roster_text(self.proj, text)
            if removed:
                if not messagebox.askyesno(
                    "確認",
                    "次の出席者が削除されます。その人に割り当てていた区間は未確定に戻ります。\n\n"
                    + "、".join(names),
                    parent=dlg,
                ):
                    return
            self._dirty = True
            dlg.destroy()
            self.refresh_all()

        ttk.Button(btns, text="OK", command=ok).pack(side="right")
        ttk.Button(btns, text="キャンセル", command=dlg.destroy).pack(side="right", padx=6)
        dlg.bind("<Escape>", lambda e: dlg.destroy())

    def show_remaining(self) -> None:
        remaining = [s for s in self.proj.segments if not s.speaker_id]
        by_cluster: dict[str, int] = {}
        for s in remaining:
            by_cluster[s.cluster_label] = by_cluster.get(s.cluster_label, 0) + 1
        if not remaining:
            messagebox.showinfo("未確定", "未確定の区間はありません。", parent=self)
            return
        lines = [f"未確定 {len(remaining)} 区間", ""]
        lines += [f"  {k}: {v} 区間" for k, v in sorted(by_cluster.items(), key=lambda x: -x[1])]
        lines += ["", "「同じ声のまとまり全体に適用」を使うと、まとまり単位で一気に確定できます。"]
        messagebox.showinfo("未確定の内訳", "\n".join(lines), parent=self)

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
        unassigned = self.proj.total_count - self.proj.assigned_count
        if unassigned:
            if not messagebox.askyesno(
                "未確定があります",
                f"未確定の区間が {unassigned} 件あります。\n"
                "未確定は【発言者不明】として出力されます。続けますか?",
                parent=self,
            ):
                return
        default_dir = str(Path(self.proj.json_path or self.proj.audio_path).parent)
        path = filedialog.asksaveasfilename(
            title="Word ファイルの保存先",
            initialdir=default_dir,
            initialfile=f"{Path(self.proj.audio_path).stem}.docx",
            defaultextension=".docx",
            filetypes=[("Word 文書", "*.docx")],
            parent=self,
        )
        if not path:
            return
        try:
            out = write_docx(self.proj, path)
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
