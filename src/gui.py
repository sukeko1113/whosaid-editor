"""Tkinter による GUI"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .assign_gui import RosterTable, open_assign_window
from .audio import audio_fingerprint
from .config import credits_text, load_config, save_config
from typing import Optional

from .align import AVAILABLE_MODELS as LOCAL_MODELS, DEFAULT_MODEL as DEFAULT_LOCAL_MODEL
from .diarize import DEFAULT_NUM_SPEAKERS
from .pipeline import DEFAULT_CLOUD_MODEL, EngineSpec, run_pipeline, run_segment_pipeline
from .transcribe import FatalTranscriptionError
from .segments import ENGINE_CLOUD, ENGINE_LOCAL, Project


APP_TITLE = "Gemini 文字起こし"
MODELS = [DEFAULT_CLOUD_MODEL, "gemini-2.5-pro"]

MODE_MANUAL = "manual"
MODE_AUTO = "auto"

ENGINE_LOCAL_DESC = (
    "録音も本文もこの PC から出ません(ネットワークを遮断したままでも動きます)。API キーは不要。\n"
    "※ 声のまとまり(A/B/C…)は作られず、話者は 1 件ずつ確定します。逐語モードも指定できず、"
    "相づち・言い淀みが脱落することがあります。"
)
ENGINE_CLOUD_DESC = (
    "音声を Gemini へ送ります。声のまとまりが付いて割当は速くなりますが、録音が外部へ出ます。"
)
ENGINE_LOCAL_NOTE_AUTO = "※ ローカルでは使えません(名簿から実名を推定する仕組みがないため)。"

MODE_MANUAL_DESC = (
    "AI は声質で発言者を A/B/C… に分けるだけ。実名は、区間ごとに音声を聴いて"
    "候補リストから選んで確定します(確定するたびに候補の並び順を学習)。"
)
MODE_AUTO_DESC = (
    "AI が名簿を見て実名まで推定します(v1 の方式)。速いが誤りが多く、"
    "後から直すのが大変です。"
)
ROSTER_HINT = (
    "名前と、企業・役職を分けて入れてください(例: 佐藤 / 理事長)。"
    "本文の【 】に何を出すかは、出力のときに選べます。"
    "よく発言する人を上に置くと、初期の候補順が良くなります。"
)


def _ask_speaker_count(parent) -> Optional[int]:
    """出席者が空のとき、話者分離に渡す人数を尋ねる。

    キャンセルなら None。話者分離設計書 §7 のとおり、**開始を押した時点で**
    聞く——転写が終わってからでは長い会議で 20 分以上あとになり、席を外して
    いればそこで処理が止まる。

    この数は正解である必要がない。上限を与えて過剰分割を防ぐのが目的で、
    自動に任せると 246 人に割れることを実測している。
    """
    dlg = tk.Toplevel(parent)
    dlg.title("出席者リストが空です")
    dlg.transient(parent)
    dlg.resizable(False, False)
    result: dict[str, Optional[int]] = {"value": None}

    ttk.Label(
        dlg, wraplength=420, justify="left",
        text="出席者を入力しておくと、割当画面ですぐ候補から選べます。\n"
             "(あとから割当画面で追加することもできます)",
    ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(14, 8))

    ttk.Label(
        dlg, wraplength=420, justify="left", foreground="#555",
        text="この録音では何人が話しますか？\n"
             "声のまとまりを分けるときの目安に使います。"
             "多めでも構いません(正確でなくても動きます)。",
    ).grid(row=1, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 6))

    var = tk.IntVar(value=DEFAULT_NUM_SPEAKERS)
    row = ttk.Frame(dlg)
    row.grid(row=2, column=0, columnspan=2, sticky="w", padx=14, pady=(0, 12))
    spin = ttk.Spinbox(row, from_=1, to=30, textvariable=var, width=6)
    spin.pack(side="left")
    ttk.Label(row, text=" 人").pack(side="left")

    btns = ttk.Frame(dlg)
    btns.grid(row=3, column=0, columnspan=2, sticky="e", padx=14, pady=(0, 14))

    def ok() -> None:
        try:
            result["value"] = max(1, int(var.get()))
        except Exception:
            result["value"] = DEFAULT_NUM_SPEAKERS
        dlg.destroy()

    ttk.Button(btns, text="この人数で進める", command=ok).pack(side="left")
    ttk.Button(btns, text="やめる", command=dlg.destroy).pack(side="left", padx=8)
    dlg.bind("<Return>", lambda e: ok())
    dlg.bind("<Escape>", lambda e: dlg.destroy())

    spin.focus_set()
    dlg.grab_set()
    parent.wait_window(dlg)
    return result["value"]


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        # 画面より高い窓を出すと、下端(進捗とログ)がタスクバーの裏へ回って
        # 見えなくなる。中身はスクロールできるので、まず画面に収める。
        height = max(560, min(900, self.winfo_screenheight() - 120))
        self.geometry(f"800x{height}")
        self.minsize(640, 480)

        self.cfg = load_config()
        self.msg_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._cancel_flag = threading.Event()
        self._worker: threading.Thread | None = None
        self._assign_win = None

        self._build_ui()
        self._populate_from_config()
        self._update_mode_state()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_queue)

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}

        # 中身は縦に長い(設定 + 進捗 + ログ)。画面の高さが足りない機械では
        # 下のログが画面外へ押し出されて、処理の様子が見えなくなる。
        # 説明を削って詰める手もあるが、この製品では「何が起きるか」を
        # 先に伝えることのほうが大事なので、入れ物側をスクロールさせる。
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        canvas = tk.Canvas(self, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        vsb.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=vsb.set)
        body = ttk.Frame(canvas)
        body_id = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(body_id, width=e.width))
        # ホイールは「主画面の上にいるとき」だけ拾う。bind_all を出しっぱなしに
        # すると、割当画面(別ウィンドウ)のホイールまでこちらが食う。
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", self._on_wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        self._canvas = canvas
        body.columnconfigure(0, weight=1)

        # === 入力ファイル ===
        frm_in = ttk.LabelFrame(body, text="音声ファイル")
        frm_in.grid(row=0, column=0, sticky="ew", **pad)
        frm_in.columnconfigure(0, weight=1)
        self.var_input = tk.StringVar()
        ttk.Entry(frm_in, textvariable=self.var_input).grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        ttk.Button(frm_in, text="参照...", command=self._pick_input).grid(row=0, column=1, padx=(0, 6), pady=6)

        # === 出力フォルダ ===
        frm_out = ttk.LabelFrame(body, text="出力フォルダ")
        frm_out.grid(row=1, column=0, sticky="ew", **pad)
        frm_out.columnconfigure(0, weight=1)
        self.var_output = tk.StringVar()
        self.var_use_input_dir = tk.BooleanVar(value=True)
        self.entry_output = ttk.Entry(frm_out, textvariable=self.var_output)
        self.entry_output.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        self.btn_pick_out = ttk.Button(frm_out, text="参照...", command=self._pick_output)
        self.btn_pick_out.grid(row=0, column=1, padx=(0, 6), pady=6)
        ttk.Checkbutton(
            frm_out,
            text="音声ファイルと同じフォルダに出力する",
            variable=self.var_use_input_dir,
            command=self._on_toggle_use_input_dir,
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6))

        # === 処理経路(どこで転写するか) ===
        # 既定はローカル。クラウドは明示的に選んだときだけ使う。
        # 録音を外へ出す判断を、既定に紛れ込ませない。
        frm_engine = ttk.LabelFrame(body, text="処理経路")
        frm_engine.grid(row=2, column=0, sticky="ew", **pad)
        frm_engine.columnconfigure(0, weight=1)
        self.var_engine = tk.StringVar(value=ENGINE_LOCAL)
        ttk.Radiobutton(
            frm_engine, text="この PC で処理する(ローカル)",
            value=ENGINE_LOCAL, variable=self.var_engine,
            command=self._update_engine_state,
        ).grid(row=0, column=0, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(frm_engine, text=ENGINE_LOCAL_DESC, foreground="#666", wraplength=700)\
            .grid(row=1, column=0, sticky="w", padx=28, pady=(0, 4))
        ttk.Radiobutton(
            frm_engine, text="クラウド(Gemini)へ送って処理する",
            value=ENGINE_CLOUD, variable=self.var_engine,
            command=self._update_engine_state,
        ).grid(row=2, column=0, sticky="w", padx=6)
        ttk.Label(frm_engine, text=ENGINE_CLOUD_DESC, foreground="#666", wraplength=700)\
            .grid(row=3, column=0, sticky="w", padx=28, pady=(0, 6))

        # === 話者の決め方(v2.0.0) ===
        frm_mode = ttk.LabelFrame(body, text="話者の決め方")
        frm_mode.grid(row=3, column=0, sticky="ew", **pad)
        frm_mode.columnconfigure(0, weight=1)
        self.var_mode = tk.StringVar(value=MODE_MANUAL)
        ttk.Radiobutton(
            frm_mode, text="区間ごとに聴いて割り当てる(推奨)",
            value=MODE_MANUAL, variable=self.var_mode, command=self._update_mode_state,
        ).grid(row=0, column=0, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(frm_mode, text=MODE_MANUAL_DESC, foreground="#666", wraplength=700)\
            .grid(row=1, column=0, sticky="w", padx=28, pady=(0, 4))
        self.rdo_mode_auto = ttk.Radiobutton(
            frm_mode, text="AI にすべて任せる(従来方式)",
            value=MODE_AUTO, variable=self.var_mode, command=self._update_mode_state,
        )
        self.rdo_mode_auto.grid(row=2, column=0, sticky="w", padx=6)
        ttk.Label(frm_mode, text=MODE_AUTO_DESC, foreground="#666", wraplength=700)\
            .grid(row=3, column=0, sticky="w", padx=28, pady=(0, 0))
        # ローカルのときだけ、灰色にした理由をその場に出す。
        # 遠くの欄に書いても、押せない理由には結び付かない。
        self.lbl_auto_note = ttk.Label(
            frm_mode, text="", foreground="#888", wraplength=700)
        self.lbl_auto_note.grid(row=4, column=0, sticky="w", padx=28, pady=(0, 6))
        ttk.Button(
            frm_mode, text="保存済みの割当作業を開く...", command=self._open_saved_project,
        ).grid(row=5, column=0, sticky="w", padx=6, pady=(0, 8))

        # === 詳細設定 ===
        frm_adv = ttk.LabelFrame(body, text="詳細設定")
        frm_adv.grid(row=4, column=0, sticky="ew", **pad)
        frm_adv.columnconfigure(1, weight=1)

        self.lbl_api = ttk.Label(frm_adv, text="API キー:")
        self.lbl_api.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.var_api = tk.StringVar()
        self.entry_api = ttk.Entry(frm_adv, textvariable=self.var_api, show="●")
        self.entry_api.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        self.btn_api_show = ttk.Button(frm_adv, text="表示", command=self._toggle_api_visibility)
        self.btn_api_show.grid(row=0, column=2, padx=(0, 4), pady=4)
        self.btn_api_save = ttk.Button(frm_adv, text="保存", command=self._save_api_key)
        self.btn_api_save.grid(row=0, column=3, padx=(0, 6), pady=4)

        ttk.Label(frm_adv, text="モデル:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        # クラウドで選んでいた逐語モードの控え(ローカルの間は外しておく)
        self._verbatim_for_cloud = False
        self.var_model = tk.StringVar(value=DEFAULT_LOCAL_MODEL)
        # 経路ごとに選んだモデルを覚えておく。切り替えて戻ったときに
        # 選び直させない(クラウドの gemini とローカルの small は別物)。
        self._model_by_engine = {
            ENGINE_CLOUD: MODELS[0],
            ENGINE_LOCAL: DEFAULT_LOCAL_MODEL,
        }
        self.cmb_model = ttk.Combobox(
            frm_adv, values=list(LOCAL_MODELS), textvariable=self.var_model,
            state="readonly")
        self.cmb_model.grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=4)
        self.cmb_model.bind("<MouseWheel>", self._on_wheel_over_value)

        ttk.Label(frm_adv, text="チャンク長(分):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        # 長いほど声のまとまりが減り、割当の手数も減る。既定を 15 分にしている。
        self.var_chunk = tk.IntVar(value=15)
        self.spin_chunk = ttk.Spinbox(
            frm_adv, from_=1, to=30, textvariable=self.var_chunk, width=6)
        self.spin_chunk.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        self.spin_chunk.bind("<MouseWheel>", self._on_wheel_over_value)
        ttk.Label(frm_adv, text="(長いほど割当の手数が減りますが、転写の失敗率は上がります)",
                  foreground="#666").grid(row=2, column=2, columnspan=2, sticky="w", padx=6, pady=4)

        # タイムスタンプ・チェックボックス
        self.var_timestamps = tk.BooleanVar(value=False)
        self.chk_timestamps = ttk.Checkbutton(
            frm_adv,
            text="タイムスタンプを付ける(段落ごとに [時:分:秒] を挿入)",
            variable=self.var_timestamps,
        )
        self.chk_timestamps.grid(row=3, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 2))

        # 話者識別チェックボックス
        self.var_diarization = tk.BooleanVar(value=False)
        self.chk_diarization = ttk.Checkbutton(
            frm_adv,
            text="話者を識別する(話者切替時にタイムスタンプを挿入)",
            variable=self.var_diarization,
            command=self._update_diarization_state,
        )
        self.chk_diarization.grid(row=4, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 2))

        self.lbl_diar_note = ttk.Label(
            frm_adv,
            text="※ 名簿なしの場合、話者ラベルはチャンクごとに識別されるため、境界をまたぐと同一人物が別ラベルになる場合があります",
            foreground="#888",
            wraplength=700,
        )
        self.lbl_diar_note.grid(row=5, column=0, columnspan=4, sticky="w", padx=24, pady=(0, 2))

        # 逐語モード・チェックボックス(v1.3.0)
        self.var_verbatim = tk.BooleanVar(value=False)
        self.chk_verbatim = ttk.Checkbutton(
            frm_adv,
            text="逐語モード(「えー」等のフィラー・言い直しを残し、整文しない。反訳・記録用)",
            variable=self.var_verbatim,
        )
        self.chk_verbatim.grid(row=6, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 2))

        # 話者分離(ローカル経路のみ)
        self.var_diarize_local = tk.BooleanVar(value=True)
        self.chk_diarize = ttk.Checkbutton(
            frm_adv,
            text="声のまとまりを端末内で分ける(話者分離。ローカル処理のみ)",
            variable=self.var_diarize_local,
            command=self._update_diarize_note,
        )
        self.chk_diarize.grid(row=7, column=0, columnspan=4, sticky="w",
                              padx=6, pady=(0, 0))
        self.lbl_diarize_note = ttk.Label(
            frm_adv, text="", foreground="#888", wraplength=700)
        self.lbl_diarize_note.grid(row=8, column=0, columnspan=4, sticky="w",
                                   padx=24, pady=(0, 2))

        # やり直しチェックボックス(v2.0.1)
        # 通常は音声の指紋で自動判定するが、手動で強制できる逃げ道も用意する。
        self.var_force = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm_adv,
            text="キャッシュを使わず最初からやり直す(結果がおかしいときに)",
            variable=self.var_force,
        ).grid(row=9, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 6))

        # === 出席者(候補者リスト) ===
        self.frm_roster = ttk.LabelFrame(body, text="出席者(候補者リスト)")
        self.frm_roster.grid(row=5, column=0, sticky="ew", **pad)
        self.frm_roster.columnconfigure(0, weight=1)
        ttk.Label(self.frm_roster, text=ROSTER_HINT, foreground="#666", wraplength=700)\
            .grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 0))
        head_roster = ttk.Frame(self.frm_roster)
        head_roster.grid(row=1, column=0, sticky="w", padx=6, pady=(4, 0))
        ttk.Label(head_roster, text="名前", font=("", 9, "bold"),
                  width=RosterTable.NAME_WIDTH).grid(row=0, column=0, sticky="w")
        ttk.Label(head_roster, text="企業・役職", font=("", 9, "bold"))            .grid(row=0, column=1, sticky="w", padx=(6, 0))
        # 主画面はそれ自体がスクロールするので、表は内側にスクロールを持たない
        self.tbl_roster = RosterTable(self.frm_roster, note_width=40)
        self.tbl_roster.grid(row=2, column=0, columnspan=2, sticky="ew",
                             padx=6, pady=(2, 4))
        self.tbl_roster.add_row()
        tools_roster = ttk.Frame(self.frm_roster)
        tools_roster.grid(row=3, column=0, sticky="w", padx=6, pady=(0, 6))
        ttk.Button(tools_roster, text="行を足す",
                   command=lambda: self.tbl_roster.add_row(focus=True))            .pack(side="left")
        ttk.Button(tools_roster, text="一覧をまとめて貼り付け...",
                   command=self._paste_roster).pack(side="left", padx=6)
        ttk.Button(tools_roster, text="名前と役職に自動で分ける",
                   command=self._split_roster).pack(side="left")
        self.var_roster_note = tk.StringVar(value="")
        ttk.Label(tools_roster, textvariable=self.var_roster_note,
                  foreground="#555").pack(side="left", padx=6)

        # === 操作ボタン ===
        frm_btn = ttk.Frame(body)
        frm_btn.grid(row=6, column=0, sticky="ew", **pad)
        self.btn_start = ttk.Button(frm_btn, text="文字起こし開始", command=self._start)
        self.btn_start.pack(side="left")
        self.btn_cancel = ttk.Button(frm_btn, text="キャンセル", command=self._cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=8)
        self.btn_open_out = ttk.Button(frm_btn, text="出力フォルダを開く", command=self._open_output_dir)
        self.btn_open_out.pack(side="right")
        # 同梱している TitaNet は CC-BY-4.0 で、表示が配布の条件
        # (claude/claude_話者分離_設計書.md §9)。画面から辿れないと表示にならない。
        ttk.Button(frm_btn, text="使っている部品と表示",
                   command=self._show_credits).pack(side="right", padx=8)

        # === 進捗 ===
        frm_prog = ttk.Frame(body)
        frm_prog.grid(row=7, column=0, sticky="ew", **pad)
        frm_prog.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(frm_prog, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew")
        self.var_status = tk.StringVar(value="待機中")
        ttk.Label(frm_prog, textvariable=self.var_status, width=14, anchor="e")\
            .grid(row=0, column=1, padx=(8, 0))

        # === ログ ===
        frm_log = ttk.LabelFrame(body, text="ログ")
        frm_log.grid(row=8, column=0, sticky="nsew", **pad)
        frm_log.columnconfigure(0, weight=1)
        frm_log.rowconfigure(0, weight=1)
        body.rowconfigure(8, weight=1)
        self.txt_log = tk.Text(frm_log, height=10, wrap="word", state="disabled")
        self.txt_log.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        sb = ttk.Scrollbar(frm_log, orient="vertical", command=self.txt_log.yview)
        sb.grid(row=0, column=1, sticky="ns", pady=6)
        self.txt_log.configure(yscrollcommand=sb.set)

    def _populate_from_config(self) -> None:
        if api := self.cfg.get("api_key"):
            self.var_api.set(api)
        # 経路ごとにモデルを覚える。旧来の "model" はクラウドのモデル。
        if model := self.cfg.get("model"):
            if model in MODELS:
                self._model_by_engine[ENGINE_CLOUD] = str(model)
        if lmodel := self.cfg.get("local_model"):
            if lmodel in LOCAL_MODELS:
                self._model_by_engine[ENGINE_LOCAL] = str(lmodel)
        if self.cfg.get("engine") in (ENGINE_LOCAL, ENGINE_CLOUD):
            self.var_engine.set(str(self.cfg["engine"]))
        if "diarize" in self.cfg:
            self.var_diarize_local.set(bool(self.cfg.get("diarize")))
        if chunk := self.cfg.get("chunk_minutes"):
            try:
                self.var_chunk.set(int(chunk))
            except Exception:
                pass
        if self.cfg.get("mode") in (MODE_MANUAL, MODE_AUTO):
            self.var_mode.set(str(self.cfg["mode"]))
        if "with_timestamps" in self.cfg:
            self.var_timestamps.set(bool(self.cfg.get("with_timestamps")))
        if "with_diarization" in self.cfg:
            self.var_diarization.set(bool(self.cfg.get("with_diarization")))
        if "verbatim" in self.cfg:
            self.var_verbatim.set(bool(self.cfg.get("verbatim")))
        if roster := self.cfg.get("roster"):
            # 保存済みは「名前(役職)」形式。分かれていない古い名簿だけ分ける
            self.tbl_roster.set_text(str(roster))
        if last_in := self.cfg.get("last_input"):
            if Path(last_in).exists():
                self.var_input.set(last_in)
        self._on_toggle_use_input_dir()
        # 経路を反映してからモデル欄と有効/無効を整える(mode の状態もここで揃う)
        self._update_engine_state()

    # 自分でスクロールする欄。この上ではホイールを窓に回さない
    # (両方動くと、行を 1 つ送るつもりが画面ごと飛ぶ)。
    _WHEEL_OWN_SCROLL = (tk.Text, tk.Listbox)

    def _on_wheel(self, event) -> None:
        if isinstance(event.widget, self._WHEEL_OWN_SCROLL):
            return
        self._canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_wheel_over_value(self, event) -> str:
        """値の欄(モデル・チャンク長)の上では、値を変えずに画面を送る。

        Spinbox と Combobox は既定でホイールに反応して値が変わる。窓が
        スクロールするようになると設定欄の上でホイールを回す機会が増え、
        気づかないうちに値が変わる。チャンク長はキャッシュキーに入るので、
        変わると転写がまるごとやり直しになる。
        """
        self._canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"      # 既定の「値を変える」動作へ渡さない

    def _scroll_to_log(self) -> None:
        """ログが見えるところまで送る(処理中に見たいのはそこだけ)。"""
        self.update_idletasks()
        self._canvas.yview_moveto(1.0)

    def _update_diarize_note(self) -> None:
        """話者分離の設定に応じて、何が起きるかをその場に出す。"""
        if self.var_engine.get() != ENGINE_LOCAL:
            self.lbl_diarize_note.configure(
                text="※ クラウドでは AI が声のまとまりを付けるため、この設定は"
                     "使いません。")
        elif self.var_diarize_local.get():
            self.lbl_diarize_note.configure(
                text="※ 転写のあとに実行します(音声の長さの 2 割ほど時間が"
                     "増えます)。まとまりは会議全体で 1 つになるので、"
                     "割当をまとめて行えます。")
        else:
            self.lbl_diarize_note.configure(
                text="※ 切ると全区間が未判別になり、割当は 1 件ずつになります。")

    def _update_engine_state(self) -> None:
        """処理経路の切替に応じて、要らない設定を触れなくする。

        ローカルでは API キーも逐語モードも意味を持たない。触れる状態で
        残すと「指定したのに効かない」という誤解になる(逐語を書式の
        選択肢として見せてしまうのは、事業計画 4.4 で撤回した過大表示と
        同じ形)。
        """
        local = self.var_engine.get() == ENGINE_LOCAL
        other = ENGINE_CLOUD if local else ENGINE_LOCAL

        # いま表示しているモデルを、切り替える前の経路の側に覚えておく
        if self.var_model.get():
            self._model_by_engine[other] = self.var_model.get()

        # 逐語モードは灰色にするだけでなく、チェックも外す。入ったまま
        # 灰色にすると「有効なのに触れない」と読めてしまい、実際には
        # 効いていないことが伝わらない。クラウドへ戻したら元に戻す。
        if local:
            if self.chk_verbatim.instate(("!disabled",)):
                self._verbatim_for_cloud = bool(self.var_verbatim.get())
            self.var_verbatim.set(False)
        else:
            self.var_verbatim.set(self._verbatim_for_cloud)

        engine = ENGINE_LOCAL if local else ENGINE_CLOUD
        values = list(LOCAL_MODELS) if local else list(MODELS)
        self.cmb_model.configure(values=values)
        remembered = self._model_by_engine.get(engine, values[0])
        self.var_model.set(remembered if remembered in values else values[0])

        state = "disabled" if local else "normal"
        for w in (self.lbl_api, self.entry_api, self.btn_api_show, self.btn_api_save):
            w.configure(state=state)
        self.chk_verbatim.configure(state=state)
        # 話者分離はローカル経路だけの設定(クラウドは Gemini が A/B/C を出す)
        self.chk_diarize.configure(state="normal" if local else "disabled")
        self._update_diarize_note()

        if local:
            # 従来方式(名簿から実名を推定させる)はクラウド専用。
            # 選べるまま残すと、開始してから使えないと知ることになる。
            if self.var_mode.get() == MODE_AUTO:
                self.var_mode.set(MODE_MANUAL)
            self.rdo_mode_auto.configure(state="disabled")
            self.lbl_auto_note.configure(text=ENGINE_LOCAL_NOTE_AUTO)
        else:
            self.rdo_mode_auto.configure(state="normal")
            self.lbl_auto_note.configure(text="")
        self._update_mode_state()

    def _update_mode_state(self) -> None:
        """モード切替に応じてチェックボックスと名簿欄の有効/無効を整える。

        手動割当モードでは、タイムスタンプも話者識別も必ず ON になるため
        個別のチェックボックスは意味を持たない(無効化して誤解を防ぐ)。
        """
        if self.var_mode.get() == MODE_MANUAL:
            self.var_timestamps.set(True)
            self.var_diarization.set(True)
            self.chk_timestamps.configure(state="disabled")
            self.chk_diarization.configure(state="disabled")
            if self.var_engine.get() == ENGINE_LOCAL:
                # 中身は「処理経路」の説明で言い切っているので、ここでは繰り返さない
                self.lbl_diar_note.configure(
                    text="※ ローカルでは声のまとまりを作らないため、全区間が未判別になります。"
                )
            else:
                self.lbl_diar_note.configure(
                    text="※ この方式では名簿を AI に渡しません。声質だけで区切った区間を、"
                         "あとの割当画面で 1 区間ずつ確定します。"
                )
            self.tbl_roster.set_enabled(True)
            self.btn_start.configure(text="文字起こし → 割当画面へ")
        else:
            self.chk_diarization.configure(state="normal")
            self.lbl_diar_note.configure(
                text="※ 名簿なしの場合、話者ラベルはチャンクごとに識別されるため、"
                     "境界をまたぐと同一人物が別ラベルになる場合があります"
            )
            self._update_diarization_state()
            self.btn_start.configure(text="文字起こし開始")

    def _update_diarization_state(self) -> None:
        """(従来方式のみ)話者識別 ON: タイムスタンプは強制 ON。名簿欄を有効化。"""
        if self.var_mode.get() == MODE_MANUAL:
            return
        if self.var_diarization.get():
            self.var_timestamps.set(True)
            self.chk_timestamps.configure(state="disabled")
            self.tbl_roster.set_enabled(True)
        else:
            self.chk_timestamps.configure(state="normal")
            self.tbl_roster.set_enabled(False)

    # ------------------------------------------------------------------
    def _open_saved_project(self) -> None:
        """既存の <音声名>.speakers.json を開いて割当作業を再開する。"""
        initial = self.var_output.get() or os.path.dirname(self.var_input.get() or "")
        path = filedialog.askopenfilename(
            title="割当作業ファイルを選択",
            initialdir=initial or None,
            filetypes=[("割当作業ファイル", "*.speakers.json"), ("JSON", "*.json"), ("すべて", "*.*")],
        )
        if not path:
            return
        try:
            proj = Project.load(path)
        except Exception as e:
            messagebox.showerror("読み込みエラー", f"作業ファイルを読めませんでした。\n\n{e}")
            return
        if not self._warn_if_audio_changed(proj):
            return
        self._open_assign(proj)

    def _warn_if_audio_changed(self, proj: Project) -> bool:
        """作業ファイルが作られたときの音声と、いまの音声が別物なら警告する。

        音声を録り直した・編集したのに同じ名前で保存した場合、区間の時刻が
        ずれているので、聴いている音とテキストが食い違ったまま作業が進む。
        """
        audio = Path(proj.audio_path)
        if not proj.audio_fingerprint or not audio.exists():
            return True    # 判定材料がないので、そのまま開く
        current = audio_fingerprint(audio)
        if not current or current == proj.audio_fingerprint:
            return True
        return messagebox.askyesno(
            "音声が変わっています",
            f"この作業ファイルが作られたときと、音声ファイルの内容が変わっています。\n"
            f"{audio.name}\n\n"
            "区間の時刻がずれている可能性が高く、聴いている音と表示されている\n"
            "テキストが食い違ったまま作業を進めることになります。\n\n"
            "メイン画面から文字起こしをやり直すことをおすすめします。\n"
            "それでもこのまま開きますか?",
        )

    def _open_assign(self, proj: Project) -> None:
        try:
            self._assign_win = open_assign_window(self, proj)
        except Exception:
            messagebox.showerror("エラー", traceback.format_exc())

    # ------------------------------------------------------------------
    # ハンドラ
    # ------------------------------------------------------------------
    def _pick_input(self) -> None:
        path = filedialog.askopenfilename(
            title="音声ファイルを選択",
            filetypes=[
                ("音声ファイル", "*.m4a *.mp3 *.wav *.aac *.flac *.ogg *.mp4"),
                ("すべて", "*.*"),
            ],
        )
        if path:
            self.var_input.set(path)
            self._on_toggle_use_input_dir()

    def _pick_output(self) -> None:
        initial = self.var_output.get() or os.path.dirname(self.var_input.get() or "")
        path = filedialog.askdirectory(title="出力フォルダを選択", initialdir=initial or None)
        if path:
            self.var_output.set(path)
            self.var_use_input_dir.set(False)
            self._on_toggle_use_input_dir()

    def _on_toggle_use_input_dir(self) -> None:
        if self.var_use_input_dir.get():
            self.entry_output.configure(state="disabled")
            self.btn_pick_out.configure(state="disabled")
            in_path = self.var_input.get()
            if in_path:
                self.var_output.set(str(Path(in_path).parent))
            else:
                self.var_output.set("")
        else:
            self.entry_output.configure(state="normal")
            self.btn_pick_out.configure(state="normal")

    def _toggle_api_visibility(self) -> None:
        current = self.entry_api.cget("show")
        self.entry_api.configure(show="" if current else "●")

    def _save_api_key(self) -> None:
        api = self.var_api.get().strip()
        if not api:
            messagebox.showwarning("API キー", "API キーが空です。")
            return
        self.cfg["api_key"] = api
        save_config(self.cfg)
        messagebox.showinfo("API キー", "API キーを保存しました。")

    def _show_credits(self) -> None:
        """同梱している部品と、その利用条件を出す。

        **TitaNet が CC-BY-4.0 なので、表示は配布の条件である。**ファイルを
        同梱しただけでは表示にならないので、画面から辿れるようにしてある。
        """
        win = tk.Toplevel(self)
        win.title("使っている部品と表示")
        win.geometry("620x520")
        frm = ttk.Frame(win)
        frm.pack(fill="both", expand=True, padx=10, pady=10)
        txt = tk.Text(frm, wrap="word", font=("", 10))
        bar = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=bar.set)
        txt.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        txt.insert("1.0", credits_text())
        txt.configure(state="disabled")
        ttk.Button(win, text="閉じる", command=win.destroy).pack(pady=(0, 10))

    def _open_output_dir(self) -> None:
        path = self.var_output.get().strip()
        if not path or not Path(path).exists():
            messagebox.showinfo("出力フォルダ", "出力フォルダが存在しません。")
            return
        try:
            if sys.platform == "win32":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", path])
            else:
                subprocess.run(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("エラー", str(e))

    def _get_roster(self) -> str:
        """名簿を「名前(役職)」1 行 1 人の形にして返す(保存と転写経路に渡す形)"""
        return self.tbl_roster.to_text()

    def _split_roster(self) -> None:
        """名前の欄に肩書ごと入っている行を、名前と役職に分ける提案。"""
        done = self.tbl_roster.auto_split()
        self.var_roster_note.set(
            f"{done} 人を分けました。外れたものは直してください。" if done
            else "分けられる行がありませんでした。")

    def _paste_roster(self) -> None:
        """一覧をまとめて貼り込む小窓。1 行 1 人で読み、名前と役職に分ける。"""
        dlg = tk.Toplevel(self)
        dlg.title("一覧をまとめて貼り付け")
        dlg.transient(self)
        dlg.grab_set()
        dlg.columnconfigure(0, weight=1)
        dlg.rowconfigure(1, weight=1)
        ttk.Label(
            dlg,
            text=("1 行に 1 人ぶんを貼り付けてください。"
                  "名前と役職は端末内の規則で分けます(外部には送りません)。"
                  + chr(10) + "分け方が外れたものは、貼り付けたあとに直せます。"),
            foreground="#555", justify="left",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
        txt = tk.Text(dlg, width=60, height=12, wrap="none")
        txt.grid(row=1, column=0, sticky="nsew", padx=12)
        txt.insert("1.0", self.tbl_roster.to_text())
        txt.focus_set()
        btns = ttk.Frame(dlg)
        btns.grid(row=2, column=0, sticky="ew", padx=12, pady=12)

        def ok() -> None:
            self.tbl_roster.set_text(txt.get("1.0", "end"))
            self.var_roster_note.set(
                f"{len(self.tbl_roster.values())} 人を読み込みました。"
                "分け方が外れたものは直してください。")
            dlg.destroy()

        ttk.Button(btns, text="読み込む", command=ok).pack(side="right")
        ttk.Button(btns, text="キャンセル", command=dlg.destroy)            .pack(side="right", padx=6)
        dlg.bind("<Escape>", lambda e: dlg.destroy())

    # ------------------------------------------------------------------
    # 実行
    # ------------------------------------------------------------------
    def _start(self) -> None:
        in_path = self.var_input.get().strip()
        if not in_path or not Path(in_path).is_file():
            messagebox.showwarning("入力", "音声ファイルを選択してください。")
            return

        out_dir = self.var_output.get().strip() or str(Path(in_path).parent)
        engine_mode = self.var_engine.get()
        api = self.var_api.get().strip()
        # 鍵が要るのはクラウドのときだけ。ローカルは端末内で完結する。
        if engine_mode == ENGINE_CLOUD and not api:
            messagebox.showwarning("API キー", "Gemini の API キーを入力してください。")
            return

        mode = self.var_mode.get()
        with_ts = bool(self.var_timestamps.get())
        with_diar = bool(self.var_diarization.get())
        verbatim = bool(self.var_verbatim.get())
        force = bool(self.var_force.get())
        roster = self._get_roster()
        if with_diar:
            with_ts = True  # 強制

        # 話者分離に渡す上限。名簿があればその人数、無ければ人に聞く。
        # 0 のままなら pipeline 側が名簿から数える。
        num_speakers = 0
        if mode == MODE_MANUAL and not roster.strip():
            wants_diarize = (engine_mode == ENGINE_LOCAL
                             and bool(self.var_diarize_local.get()))
            if wants_diarize:
                # **ここで聞く。**転写が終わってからでは 67 分の会議で 25 分後に
                # なり、席を外していればそこで処理が止まる(話者分離設計書 §7)。
                num_speakers = _ask_speaker_count(self)
                if num_speakers is None:
                    return
            elif not messagebox.askyesno(
                "出席者リストが空です",
                "出席者を入力しておくと、割当画面ですぐ候補から選べます。\n"
                "(あとから割当画面で追加することもできます)\n\nこのまま進めますか?",
            ):
                return

        # 経路ごとのモデルを両方覚えておく(切り替えて戻したときのため)
        self._model_by_engine[engine_mode] = self.var_model.get()
        self.cfg.update({
            "api_key": api,
            "engine": engine_mode,
            "diarize": bool(self.var_diarize_local.get()),
            "model": self._model_by_engine[ENGINE_CLOUD],
            "local_model": self._model_by_engine[ENGINE_LOCAL],
            "chunk_minutes": int(self.var_chunk.get()),
            "mode": mode,
            "with_timestamps": with_ts,
            "with_diarization": with_diar,
            # ローカルの間はチェックを外してあるので、そのまま保存すると
            # クラウドで選んでいた設定を False で潰してしまう
            "verbatim": verbatim if engine_mode == ENGINE_CLOUD
                        else self._verbatim_for_cloud,
            "roster": roster,
            "last_input": in_path,
        })
        save_config(self.cfg)

        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.configure(state="disabled")

        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.var_status.set("実行中...")
        self.progress.configure(value=0, maximum=1)
        self._scroll_to_log()

        self._cancel_flag.clear()
        self._worker = threading.Thread(
            target=self._run_worker,
            args=(
                Path(in_path),
                Path(out_dir),
                EngineSpec(mode=engine_mode, model=self.var_model.get(),
                           api_key=api,
                           diarize=bool(self.var_diarize_local.get()),
                           num_speakers=num_speakers),
                int(self.var_chunk.get()),
                with_ts,
                with_diar,
                roster,
                verbatim,
                mode,
                force,
            ),
            daemon=True,
        )
        self._worker.start()

    def _cancel(self) -> None:
        if self._worker and self._worker.is_alive():
            self._cancel_flag.set()
            self.var_status.set("キャンセル要求中...")
            self._post("log", "キャンセル要求を送信しました。現在のチャンク完了後に停止します。")

    def _run_worker(
        self,
        in_path: Path,
        out_dir: Path,
        engine: EngineSpec,
        chunk_minutes: int,
        with_timestamps: bool,
        with_diarization: bool,
        roster: str,
        verbatim: bool,
        mode: str = MODE_AUTO,
        force_retranscribe: bool = False,
    ) -> None:
        try:
            if mode == MODE_MANUAL:
                result = run_segment_pipeline(
                    audio_path=in_path,
                    output_dir=out_dir,
                    engine=engine,
                    chunk_minutes=chunk_minutes,
                    on_log=lambda m: self._post("log", m),
                    on_progress=lambda c, t: self._post("progress", (c, t)),
                    is_cancelled=self._cancel_flag.is_set,
                    verbatim=verbatim,
                    roster=roster,
                    force_retranscribe=force_retranscribe,
                )
            else:
                # 従来方式(AI に実名まで任せる)はクラウド専用。
                # 画面側でもローカル選択時は選べないようにしてある。
                result = run_pipeline(
                    audio_path=in_path,
                    output_dir=out_dir,
                    api_key=engine.api_key,
                    model=engine.resolved_model(),
                    chunk_minutes=chunk_minutes,
                    on_log=lambda m: self._post("log", m),
                    on_progress=lambda c, t: self._post("progress", (c, t)),
                    is_cancelled=self._cancel_flag.is_set,
                    with_timestamps=with_timestamps,
                    with_diarization=with_diarization,
                    roster=roster,
                    verbatim=verbatim,
                    force_retranscribe=force_retranscribe,
                )
            self._post("done", result)
        except FatalTranscriptionError as e:
            # 残高切れ・APIキー不正。traceback ではなく対処方法を見せる。
            self._post("fatal", str(e))
        except Exception:
            self._post("error", traceback.format_exc())

    # ------------------------------------------------------------------
    # ワーカ→UI のメッセージ受信
    # ------------------------------------------------------------------
    def _post(self, kind: str, data: object) -> None:
        self.msg_queue.put((kind, data))

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, data = self.msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(str(data))
                elif kind == "progress":
                    cur, total = data  # type: ignore[misc]
                    self.progress.configure(maximum=max(1, total), value=cur)
                    self.var_status.set(f"{cur}/{total}")
                elif kind == "done":
                    self._on_done(data)  # type: ignore[arg-type]
                elif kind == "fatal":
                    self._on_fatal(str(data))
                elif kind == "error":
                    self._on_error(str(data))
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _append_log(self, msg: str) -> None:
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", msg + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _on_done(self, result: object) -> None:
        self.btn_start.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        if result is None:
            self.var_status.set("キャンセル")
            return
        self.var_status.set("完了")

        # 手動割当モード: そのまま割当画面を開く
        if isinstance(result, Project):
            n = result.total_count
            # 擬似クラスタ(? / *)は「声のまとまり」ではない。数に入れると、
            # ローカル転写のように全部が未判別のときでも
            # 「まとまり 10 種類」と出て、使える手がかりがあるように見える。
            real = {s.cluster for s in result.segments if not s.is_pseudo_cluster}
            if real:
                self._append_log(
                    f"{n} 区間 / 声のまとまり {len(real)} 種類。割当画面を開きます。")
            else:
                self._append_log(
                    f"{n} 区間(声のまとまりなし・全区間が未判別)。割当画面を開きます。")
            self._open_assign(result)
            return

        result = Path(str(result))
        if messagebox.askyesno("完了", f"文字起こしが完了しました。\n\n{result}\n\nファイルを開きますか?"):
            try:
                if sys.platform == "win32":
                    os.startfile(str(result))  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(result)])
                else:
                    subprocess.run(["xdg-open", str(result)])
            except Exception as e:
                messagebox.showerror("エラー", str(e))

    def _on_fatal(self, message: str) -> None:
        """再試行しても直らないエラー。原因と対処だけを見せる。"""
        self.btn_start.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        self.var_status.set("中断")
        self._append_log("=== 中断 ===\n" + message)
        messagebox.showerror("処理を中断しました", message)

    def _on_error(self, tb: str) -> None:
        self.btn_start.configure(state="normal")
        self.btn_cancel.configure(state="disabled")
        self.var_status.set("エラー")
        self._append_log("=== エラー ===\n" + tb)
        messagebox.showerror("エラー", tb.splitlines()[-1] if tb.strip() else "不明なエラー")

    def _on_close(self) -> None:
        if self._worker and self._worker.is_alive():
            if not messagebox.askyesno("確認", "処理中です。終了しますか?"):
                return
            self._cancel_flag.set()
        # メインウィンドウを destroy すると子ウィンドウの終了処理が走らないので、
        # 割当画面に「保存してから閉じる」を先に実行させる(編集中の本文が消える)
        win = self._assign_win
        if win is not None:
            try:
                if win.winfo_exists():
                    win._on_close()
            except Exception:
                pass
        self.destroy()
