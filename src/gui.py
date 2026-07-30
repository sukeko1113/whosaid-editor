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

from .assign_gui import open_assign_window
from .audio import audio_fingerprint
from .config import load_config, save_config
from .pipeline import run_pipeline, run_segment_pipeline
from .segments import Project


APP_TITLE = "Gemini 文字起こし"
MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

MODE_MANUAL = "manual"
MODE_AUTO = "auto"

MODE_MANUAL_DESC = (
    "AI は声質で発言者を A/B/C… に分けるだけ。実名は、区間ごとに音声を聴いて"
    "候補リストから選んで確定します(確定するたびに候補の並び順を学習)。"
)
MODE_AUTO_DESC = (
    "AI が名簿を見て実名まで推定します(v1 の方式)。速いが誤りが多く、"
    "後から直すのが大変です。"
)
ROSTER_HINT = (
    "1行に1人、「名前(役職)」の形式で入力(例: 佐藤(理事長))。"
    "よく発言する人を上に置くと、初期の候補順が良くなります。"
)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("780x900")
        self.minsize(700, 720)

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
        self.columnconfigure(0, weight=1)

        # === 入力ファイル ===
        frm_in = ttk.LabelFrame(self, text="音声ファイル")
        frm_in.grid(row=0, column=0, sticky="ew", **pad)
        frm_in.columnconfigure(0, weight=1)
        self.var_input = tk.StringVar()
        ttk.Entry(frm_in, textvariable=self.var_input).grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        ttk.Button(frm_in, text="参照...", command=self._pick_input).grid(row=0, column=1, padx=(0, 6), pady=6)

        # === 出力フォルダ ===
        frm_out = ttk.LabelFrame(self, text="出力フォルダ")
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

        # === 話者の決め方(v2.0.0) ===
        frm_mode = ttk.LabelFrame(self, text="話者の決め方")
        frm_mode.grid(row=2, column=0, sticky="ew", **pad)
        frm_mode.columnconfigure(0, weight=1)
        self.var_mode = tk.StringVar(value=MODE_MANUAL)
        ttk.Radiobutton(
            frm_mode, text="区間ごとに聴いて割り当てる(推奨)",
            value=MODE_MANUAL, variable=self.var_mode, command=self._update_mode_state,
        ).grid(row=0, column=0, sticky="w", padx=6, pady=(6, 0))
        ttk.Label(frm_mode, text=MODE_MANUAL_DESC, foreground="#666", wraplength=700)\
            .grid(row=1, column=0, sticky="w", padx=28, pady=(0, 4))
        ttk.Radiobutton(
            frm_mode, text="AI にすべて任せる(従来方式)",
            value=MODE_AUTO, variable=self.var_mode, command=self._update_mode_state,
        ).grid(row=2, column=0, sticky="w", padx=6)
        ttk.Label(frm_mode, text=MODE_AUTO_DESC, foreground="#666", wraplength=700)\
            .grid(row=3, column=0, sticky="w", padx=28, pady=(0, 6))
        ttk.Button(
            frm_mode, text="保存済みの割当作業を開く...", command=self._open_saved_project,
        ).grid(row=4, column=0, sticky="w", padx=6, pady=(0, 8))

        # === 詳細設定 ===
        frm_adv = ttk.LabelFrame(self, text="詳細設定")
        frm_adv.grid(row=3, column=0, sticky="ew", **pad)
        frm_adv.columnconfigure(1, weight=1)

        ttk.Label(frm_adv, text="API キー:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.var_api = tk.StringVar()
        self.entry_api = ttk.Entry(frm_adv, textvariable=self.var_api, show="●")
        self.entry_api.grid(row=0, column=1, sticky="ew", padx=6, pady=4)
        ttk.Button(frm_adv, text="表示", command=self._toggle_api_visibility).grid(row=0, column=2, padx=(0, 4), pady=4)
        ttk.Button(frm_adv, text="保存", command=self._save_api_key).grid(row=0, column=3, padx=(0, 6), pady=4)

        ttk.Label(frm_adv, text="モデル:").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.var_model = tk.StringVar(value=MODELS[0])
        ttk.Combobox(frm_adv, values=MODELS, textvariable=self.var_model, state="readonly")\
            .grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=4)

        ttk.Label(frm_adv, text="チャンク長(分):").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        # 長いほど声のまとまりが減り、割当の手数も減る。既定を 15 分にしている。
        self.var_chunk = tk.IntVar(value=15)
        ttk.Spinbox(frm_adv, from_=1, to=30, textvariable=self.var_chunk, width=6)\
            .grid(row=2, column=1, sticky="w", padx=6, pady=4)
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

        # やり直しチェックボックス(v2.0.1)
        # 通常は音声の指紋で自動判定するが、手動で強制できる逃げ道も用意する。
        self.var_force = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm_adv,
            text="キャッシュを使わず最初からやり直す(結果がおかしいときに)",
            variable=self.var_force,
        ).grid(row=7, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 6))

        # === 出席者(候補者リスト) ===
        self.frm_roster = ttk.LabelFrame(self, text="出席者(候補者リスト)")
        self.frm_roster.grid(row=4, column=0, sticky="ew", **pad)
        self.frm_roster.columnconfigure(0, weight=1)
        ttk.Label(self.frm_roster, text=ROSTER_HINT, foreground="#666", wraplength=700)\
            .grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 0))
        self.txt_roster = tk.Text(self.frm_roster, height=5, wrap="word")
        self.txt_roster.grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        sb_roster = ttk.Scrollbar(self.frm_roster, orient="vertical", command=self.txt_roster.yview)
        sb_roster.grid(row=1, column=1, sticky="ns", pady=6)
        self.txt_roster.configure(yscrollcommand=sb_roster.set)

        # === 操作ボタン ===
        frm_btn = ttk.Frame(self)
        frm_btn.grid(row=5, column=0, sticky="ew", **pad)
        self.btn_start = ttk.Button(frm_btn, text="文字起こし開始", command=self._start)
        self.btn_start.pack(side="left")
        self.btn_cancel = ttk.Button(frm_btn, text="キャンセル", command=self._cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=8)
        self.btn_open_out = ttk.Button(frm_btn, text="出力フォルダを開く", command=self._open_output_dir)
        self.btn_open_out.pack(side="right")

        # === 進捗 ===
        frm_prog = ttk.Frame(self)
        frm_prog.grid(row=6, column=0, sticky="ew", **pad)
        frm_prog.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(frm_prog, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew")
        self.var_status = tk.StringVar(value="待機中")
        ttk.Label(frm_prog, textvariable=self.var_status, width=14, anchor="e")\
            .grid(row=0, column=1, padx=(8, 0))

        # === ログ ===
        frm_log = ttk.LabelFrame(self, text="ログ")
        frm_log.grid(row=7, column=0, sticky="nsew", **pad)
        frm_log.columnconfigure(0, weight=1)
        frm_log.rowconfigure(0, weight=1)
        self.rowconfigure(7, weight=1)
        self.txt_log = tk.Text(frm_log, height=10, wrap="word", state="disabled")
        self.txt_log.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        sb = ttk.Scrollbar(frm_log, orient="vertical", command=self.txt_log.yview)
        sb.grid(row=0, column=1, sticky="ns", pady=6)
        self.txt_log.configure(yscrollcommand=sb.set)

    def _populate_from_config(self) -> None:
        if api := self.cfg.get("api_key"):
            self.var_api.set(api)
        if model := self.cfg.get("model"):
            if model in MODELS:
                self.var_model.set(model)
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
            self.txt_roster.delete("1.0", "end")
            self.txt_roster.insert("1.0", str(roster))
        if last_in := self.cfg.get("last_input"):
            if Path(last_in).exists():
                self.var_input.set(last_in)
        self._on_toggle_use_input_dir()

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
            self.lbl_diar_note.configure(
                text="※ この方式では名簿を AI に渡しません。声質だけで区切った区間を、"
                     "あとの割当画面で 1 区間ずつ確定します。"
            )
            self.txt_roster.configure(state="normal", background="white")
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
            self.txt_roster.configure(state="normal", background="white")
        else:
            self.chk_timestamps.configure(state="normal")
            self.txt_roster.configure(state="disabled", background="#f0f0f0")

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
        """名簿欄のテキストを取得(disabled 状態でも読めるように一時的に戻す)"""
        state = self.txt_roster.cget("state")
        if state == "disabled":
            self.txt_roster.configure(state="normal")
            text = self.txt_roster.get("1.0", "end").strip()
            self.txt_roster.configure(state="disabled")
        else:
            text = self.txt_roster.get("1.0", "end").strip()
        return text

    # ------------------------------------------------------------------
    # 実行
    # ------------------------------------------------------------------
    def _start(self) -> None:
        in_path = self.var_input.get().strip()
        if not in_path or not Path(in_path).is_file():
            messagebox.showwarning("入力", "音声ファイルを選択してください。")
            return

        out_dir = self.var_output.get().strip() or str(Path(in_path).parent)
        api = self.var_api.get().strip()
        if not api:
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

        if mode == MODE_MANUAL and not roster.strip():
            if not messagebox.askyesno(
                "出席者リストが空です",
                "出席者を入力しておくと、割当画面ですぐ候補から選べます。\n"
                "(あとから割当画面で追加することもできます)\n\nこのまま進めますか?",
            ):
                return

        self.cfg.update({
            "api_key": api,
            "model": self.var_model.get(),
            "chunk_minutes": int(self.var_chunk.get()),
            "mode": mode,
            "with_timestamps": with_ts,
            "with_diarization": with_diar,
            "verbatim": verbatim,
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

        self._cancel_flag.clear()
        self._worker = threading.Thread(
            target=self._run_worker,
            args=(
                Path(in_path),
                Path(out_dir),
                api,
                self.var_model.get(),
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
        api_key: str,
        model: str,
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
                    api_key=api_key,
                    model=model,
                    chunk_minutes=chunk_minutes,
                    on_log=lambda m: self._post("log", m),
                    on_progress=lambda c, t: self._post("progress", (c, t)),
                    is_cancelled=self._cancel_flag.is_set,
                    verbatim=verbatim,
                    roster=roster,
                    force_retranscribe=force_retranscribe,
                )
            else:
                result = run_pipeline(
                    audio_path=in_path,
                    output_dir=out_dir,
                    api_key=api_key,
                    model=model,
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
            clusters = len(result.clusters())
            self._append_log(f"{n} 区間 / 声のまとまり {clusters} 種類。割当画面を開きます。")
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
