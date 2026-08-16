"""逐語の正解を作る画面(段階 2)。

    python tools\\verbatim_truth.py <作業ファイル.speakers.json> <帯の名前>

設計書 `claude/claude_正解ラベルの作り方_設計書.md` の段階 2。
製品には含めない(検証のための道具)。

**落ちている発話を足せることが、この道具の要点である。**本文を直すだけの
作りにすると、ASR が拾わなかった発話は永遠に見つからない——画面に無いものは
直しようがないからである。逐語保持率は「正解に含まれる相づち・フィラーのうち
何割が出力に残ったか」なので、正解の側に落ちた発話が入っていなければ測れない。

段階 1 で `M` を押した区間には印を出す。そこには聞こえたのに本文に無い声が
あると、既に人が判断している。

出力は実名を含むのでリポジトリの外(`truth/` 配下)に置く。
"""
from __future__ import annotations

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.player import SegmentPlayer  # noqa: E402
from src.segments import Project, fmt_hms  # noqa: E402

# 帯の定義(秒)。脱落の分布から選んだ——前半は説明が続いて脱落ゼロ、
# 後半は質疑の応酬で 17〜23%。両方を含めないと偏る。
BANDS: dict[str, tuple[float, float]] = {
    # 説明が続く帯。脱落が無いはずの基準線として置く
    "a-setsumei": (60.0, 180.0),
    # 脱落 19%・短い区間は少ない
    "b-chuban": (1440.0, 1560.0),
    # 応酬。同時発話が多い
    "c-ouchou": (2400.0, 2520.0),
    # 最も密で脱落が最多(23%)
    "d-missitsu": (3420.0, 3540.0),
}


class VerbatimWindow(tk.Tk):
    def __init__(self, proj: Project, band: str, out: Path,
                 noted: set[int]) -> None:
        super().__init__()
        lo, hi = BANDS[band]
        self.title(f"逐語の正解 — {band} ({fmt_hms(lo)}〜{fmt_hms(hi)})")
        self.geometry("980x700")
        self.minsize(820, 560)

        self.proj = proj
        self.audio = Path(proj.audio_path)
        self.out = out
        self.band = band
        self.lo, self.hi = lo, hi
        self.noted = noted

        ordered = sorted(proj.segments, key=lambda s: (s.start, s.index))
        self.all_segments = ordered
        self.pos_of = {s.index: i for i, s in enumerate(ordered)}
        self.targets = [s for s in ordered if lo <= s.start < hi]

        self.fixed: dict[int, str] = {}          # 区間 index → 直した本文
        self.missing: list[dict] = []            # 拾われていない発話
        self._load_existing()
        self.cur = self._first_untouched()

        self.player = SegmentPlayer()
        self._build_ui()
        self._bind_keys()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._show()

    # ------------------------------------------------------------------
    def _load_existing(self) -> None:
        if not self.out.exists():
            return
        try:
            data = json.loads(self.out.read_text(encoding="utf-8"))
        except Exception:
            return
        for row in data.get("segments", []):
            self.fixed[int(row["index"])] = row.get("truth", "")
        self.missing = list(data.get("missing", []))

    def _first_untouched(self) -> int:
        for i, s in enumerate(self.targets):
            if s.index not in self.fixed:
                return i
        return max(0, len(self.targets) - 1)

    def _save(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "project": Path(self.proj.json_path or "").name,
            "audio_sha256": self.proj.source_sha256,
            "made_by": "human-verbatim",
            "band": {"name": self.band, "start": self.lo, "end": self.hi},
            "segments": [
                {"index": s.index, "start": round(s.start, 2),
                 "end": round(s.end, 2), "asr": s.text,
                 "truth": self.fixed[s.index]}
                for s in self.targets if s.index in self.fixed
            ],
            "missing": self.missing,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.out)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 5}
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        head = ttk.Frame(self)
        head.grid(row=0, column=0, sticky="ew", **pad)
        head.columnconfigure(1, weight=1)
        self.var_progress = tk.StringVar()
        ttk.Label(head, textvariable=self.var_progress,
                  font=("", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.bar = ttk.Progressbar(head, mode="determinate",
                                   maximum=max(1, len(self.targets)))
        self.bar.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        guide = ttk.LabelFrame(self, text="やること")
        guide.grid(row=1, column=0, sticky="ew", **pad)
        ttk.Label(
            guide, wraplength=930, justify="left",
            text=(
                "① 音を聴く（区間を開くと自動で鳴ります。もう一度なら F1）\n"
                "② 下の「正解の本文」を、聞こえたとおりに直す\n"
                "     ・「あの」「えー」などの言い淀みも、言ったとおりに残す\n"
                "     ・**この人が言ったのに本文から落ちている言葉は、"
                "ここに直接書き足す**\n"
                "     ・**直すところが無ければ、何もしないで次へ。**"
                "それも「この区間は正しかった」という記録になります\n"
                "③ **別の人の声**が聞こえるのに本文に無いときは［＋ 聞こえた発話を足す］\n"
                "     （同じ人の言い足りない分は②で書き足す。"
                "③は「誰かの発話ごと落ちている」場合）\n"
                "④［確定して次へ］を押す（Ctrl+Enter でも同じ）"
            ),
        ).grid(row=0, column=0, sticky="w", padx=10, pady=8)

        info = ttk.Frame(self)
        info.grid(row=2, column=0, sticky="ew", **pad)
        self.var_when = tk.StringVar()
        ttk.Label(info, textvariable=self.var_when).pack(side="left")
        self.var_mark = tk.StringVar()
        ttk.Label(info, textvariable=self.var_mark,
                  foreground="#B26500").pack(side="left", padx=16)

        asr = ttk.LabelFrame(self, text="いまの本文（音声認識が出したもの・参考）")
        asr.grid(row=3, column=0, sticky="ew", **pad)
        asr.columnconfigure(0, weight=1)
        self.var_asr = tk.StringVar()
        ttk.Label(asr, textvariable=self.var_asr, foreground="#666",
                  wraplength=920).grid(row=0, column=0, sticky="w",
                                       padx=8, pady=6)

        fix = ttk.LabelFrame(
            self, text="正解の本文（ここを直す。直すところが無ければそのまま次へ）")
        fix.grid(row=4, column=0, sticky="nsew", **pad)
        fix.columnconfigure(0, weight=1)
        fix.rowconfigure(0, weight=1)
        self.txt = tk.Text(fix, height=5, wrap="word", font=("", 12))
        self.txt.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)

        miss = ttk.LabelFrame(
            self,
            text="この区間で足した「別の人の発話」（正解の一部として数えます）")
        miss.grid(row=5, column=0, sticky="ew", **pad)
        miss.columnconfigure(0, weight=1)
        self.var_missing = tk.StringVar()
        ttk.Label(miss, textvariable=self.var_missing, wraplength=920,
                  foreground="#B26500").grid(row=0, column=0, sticky="w",
                                             padx=8, pady=(6, 0))
        ttk.Label(
            miss, foreground="#888", wraplength=920, justify="left",
            text="※ 本文とは別に持ちます。集計では時刻の順に本文へ差し込んで"
                 "「正解の全文」にすると同時に、"
                 "**この分だけを取り出して「各エンジンが拾えているか」を測ります**。"
                 "本文に溶かし込むと、その測定ができなくなります。",
        ).grid(row=1, column=0, sticky="w", padx=8, pady=(2, 6))

        # ボタンを出す。キーだけにすると「何をすればよいか」が画面から
        # 読み取れない(実際に分からないと報告を受けた)。
        btns = ttk.Frame(self)
        btns.grid(row=6, column=0, sticky="ew", **pad)
        ttk.Button(btns, text="▶ もう一度聴く (F1)",
                   command=self._play_current).pack(side="left")
        ttk.Button(btns, text="◀ 前を聴く (F2)",
                   command=lambda: self._play_neighbour(-1)).pack(side="left", padx=6)
        ttk.Button(btns, text="次を聴く ▶ (F3)",
                   command=lambda: self._play_neighbour(+1)).pack(side="left")
        ttk.Button(btns, text="＋ 聞こえた発話を足す (F4)",
                   command=self._add_missing).pack(side="left", padx=16)
        ttk.Button(btns, text="1 つ戻る (F5)",
                   command=self._back).pack(side="left")
        self.btn_next = ttk.Button(btns, text="確定して次へ → (Ctrl+Enter)",
                                   command=self._commit)
        self.btn_next.pack(side="right")

        self.var_action = tk.StringVar()
        ttk.Label(self, textvariable=self.var_action, foreground="#1B5E20")\
            .grid(row=7, column=0, sticky="w", **pad)

    def _bind_keys(self) -> None:
        # 本文欄に文字を打つので、単独の文字キーは使えない。
        # **Ctrl+Space は使わない** —— 日本語 IME がオン/オフに使っており、
        # 押しても画面まで届かない(実機で確認)。F キーなら奪われない。
        self.bind("<F1>", lambda e: self._play_current())
        self.bind("<F2>", lambda e: self._play_neighbour(-1))
        self.bind("<F3>", lambda e: self._play_neighbour(+1))
        self.bind("<F4>", lambda e: self._add_missing())
        self.bind("<F5>", lambda e: self._back())
        self.bind("<Control-Return>", lambda e: self._commit())
        self.bind("<Escape>", lambda e: self._on_close())

    # ------------------------------------------------------------------
    def _target(self):
        if not self.targets:
            return None
        self.cur = max(0, min(self.cur, len(self.targets) - 1))
        return self.targets[self.cur]

    def _show(self) -> None:
        seg = self._target()
        self.var_progress.set(
            f"{len(self.fixed)} / {len(self.targets)} 件 "
            f"（いま {self.cur + 1} 件目）／ 足した発話 {len(self.missing)} 件")
        self.bar.configure(value=len(self.fixed))
        if seg is None:
            return
        self.var_when.set(f"#{seg.index}  {fmt_hms(seg.start)} 〜 "
                          f"{fmt_hms(seg.end)}（{seg.duration:.1f} 秒）")
        self.var_mark.set(
            "★ 段階 1 で「本文に無い声が聞こえた」と記録した区間"
            if seg.index in self.noted else "")
        self.var_asr.set(seg.text or "（本文なし）")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", self.fixed.get(seg.index, seg.text))
        self.txt.focus_set()
        self._refresh_missing()
        self._play_current()

    def _refresh_missing(self) -> None:
        seg = self._target()
        if seg is None:
            return
        here = [m for m in self.missing
                if seg.start <= m["at"] <= seg.end]
        self.var_missing.set(
            "／".join(f"{fmt_hms(m['at'])} 「{m['text']}」" for m in here)
            or "（なし。別の人の声が落ちていなければ、これで正しいです）")

    def _play_current(self) -> None:
        seg = self._target()
        if seg is not None:
            self.player.play(self.audio, seg.start, seg.end)

    def _play_neighbour(self, direction: int) -> None:
        seg = self._target()
        if seg is None:
            return
        pos = self.pos_of[seg.index] + direction
        if 0 <= pos < len(self.all_segments):
            other = self.all_segments[pos]
            self.player.play(self.audio, other.start, other.end)

    def _add_missing(self) -> None:
        """この区間で聞こえたのに本文に無い発話を足す。"""
        seg = self._target()
        if seg is None:
            return
        dlg = tk.Toplevel(self)
        dlg.title("拾われていない発話を足す")
        dlg.transient(self)
        ttk.Label(dlg, wraplength=440, justify="left",
                  text=f"{fmt_hms(seg.start)}〜{fmt_hms(seg.end)} の範囲で、"
                       "本文に出ていない**別の人の発話**が聞こえましたか。\n"
                       "聞こえたとおりに書いてください（「うん」「あー」等）。\n\n"
                       "※ 同じ人が言ったのに落ちている言葉なら、ここではなく"
                       "「正解の本文」に直接書き足してください。")\
            .grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(14, 6))
        var = tk.StringVar()
        ent = ttk.Entry(dlg, textvariable=var, width=44)
        ent.grid(row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 10))

        def ok() -> None:
            text = var.get().strip()
            if text:
                self.missing.append({
                    "after_index": seg.index,
                    "at": round((seg.start + seg.end) / 2, 2),
                    "text": text,
                })
                self._save()
                self.var_action.set(f"「{text}」を足しました")
            dlg.destroy()
            self._refresh_missing()

        btns = ttk.Frame(dlg)
        btns.grid(row=2, column=0, columnspan=2, sticky="e", padx=14, pady=(0, 14))
        ttk.Button(btns, text="足す", command=ok).pack(side="left")
        ttk.Button(btns, text="やめる", command=dlg.destroy).pack(side="left", padx=8)
        dlg.bind("<Return>", lambda e: ok())
        dlg.bind("<Escape>", lambda e: dlg.destroy())
        ent.focus_set()
        dlg.grab_set()
        self.wait_window(dlg)
        self.txt.focus_set()

    def _commit(self) -> str:
        seg = self._target()
        if seg is None:
            return "break"
        self.fixed[seg.index] = self.txt.get("1.0", "end").strip()
        self._save()
        if self.cur + 1 >= len(self.targets):
            self.player.stop()
            messagebox.showinfo(
                "この帯は終わりです",
                f"{len(self.fixed)} 区間を直し、{len(self.missing)} 件の"
                "発話を足しました。\n\n保存先:\n{}".format(self.out),
                parent=self)
            return "break"
        self.cur += 1
        self._show()
        return "break"

    def _back(self) -> str:
        if self.cur > 0:
            self.cur -= 1
            self._show()
        return "break"

    def _on_close(self) -> None:
        seg = self._target()
        if seg is not None:                   # 編集中の内容を捨てない
            self.fixed[seg.index] = self.txt.get("1.0", "end").strip()
        self._save()
        self.player.close()
        self.destroy()


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        print("帯の一覧:")
        for name, (lo, hi) in BANDS.items():
            print(f"  {name:<12} {fmt_hms(lo)}〜{fmt_hms(hi)}"
                  f"（{(hi-lo)/60:.0f} 分）")
        return 2
    proj_path, band = Path(sys.argv[1]), sys.argv[2]
    if band not in BANDS:
        print(f"知らない帯です: {band}（{', '.join(BANDS)}）")
        return 1

    proj = Project.load(proj_path)
    truth_dir = proj_path.parent / "truth"

    # 段階 1 で「本文に無い声」と印を付けた区間を読む(あれば)
    noted: set[int] = set()
    for path in sorted(truth_dir.glob("labels_blind.r*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in data.get("untranscribed_voice", []):
            noted.add(int(row["index"]))
    if noted:
        print(f"段階 1 の印を {len(noted)} 件読みました。")

    out = truth_dir / f"verbatim.{band}.json"
    win = VerbatimWindow(proj, band, out, noted)
    win.mainloop()
    print(f"保存先: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
