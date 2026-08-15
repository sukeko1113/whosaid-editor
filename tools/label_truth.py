"""白紙で話者ラベルを付ける画面(正解データ作り)。

    python tools\\label_truth.py <作業ファイル.speakers.json> [巡]

設計書 `claude/claude_正解ラベルの作り方_設計書.md` §3 の実装。
製品には含めない(検証のための道具)。

**話者分離の結果と、候補の並び順は見せない。** 見せると人の判断が引きずられ、
その正解で話者分離を測ることが循環になる。名簿は常に名簿順で固定し、
数字キーの位置は最後まで変わらない。

付けた結果は 1 件ごとに保存する。いつ中断しても続きから再開できる。
出力は実名を含むのでリポジトリの外に置く。
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
from src.segments import (  # noqa: E402
    SPECIAL_MULTI,
    SPECIAL_UNKNOWN,
    Project,
    fmt_hms,
)

CONTEXT_SECONDS = 2.0       # 前後の文脈をどれだけ鳴らすか


class TruthWindow(tk.Tk):
    def __init__(self, proj: Project, sample: dict, out: Path, round_no: int):
        super().__init__()
        self.title(f"正解ラベル（白紙） — {round_no} 巡目")
        self.geometry("900x620")
        self.minsize(760, 540)

        self.proj = proj
        self.audio = Path(proj.audio_path)
        self.out = out
        self.round_no = round_no
        self.by_index = {s.index: s for s in proj.segments}
        self.ordered = sorted(proj.segments, key=lambda s: (s.start, s.index))
        self.pos_of = {s.index: i for i, s in enumerate(self.ordered)}

        # 標本のうち、この巡のぶんだけを時間順に
        picked = [x for x in sample["samples"] if x.get("round", 1) == round_no]
        picked.sort(key=lambda x: (x["start"], x["index"]))
        self.targets = [x for x in picked if x["index"] in self.by_index]

        self.labels: dict[int, dict] = {}
        self._load_existing()
        self.cur = self._first_unlabeled()

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
        for row in data.get("labels", []):
            self.labels[int(row["index"])] = row

    def _first_unlabeled(self) -> int:
        for i, t in enumerate(self.targets):
            if t["index"] not in self.labels:
                return i
        return len(self.targets) - 1 if self.targets else 0

    def _save(self) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "project": Path(self.proj.json_path or "").name,
            "audio_sha256": self.proj.source_sha256,
            "made_by": "blind",
            "round": self.round_no,
            "labels": [self.labels[k] for k in sorted(self.labels)],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.out)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        pad = {"padx": 12, "pady": 6}
        self.columnconfigure(0, weight=1)

        head = ttk.Frame(self)
        head.grid(row=0, column=0, sticky="ew", **pad)
        head.columnconfigure(1, weight=1)
        self.var_progress = tk.StringVar()
        ttk.Label(head, textvariable=self.var_progress,
                  font=("", 11, "bold")).grid(row=0, column=0, sticky="w")
        self.bar = ttk.Progressbar(head, mode="determinate",
                                   maximum=max(1, len(self.targets)))
        self.bar.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        ttk.Label(
            self,
            text="この区間を話しているのは誰ですか。音を聴いて選んでください。"
                 "（話者分離の結果と候補の並びは、判断が引きずられないよう出していません）",
            foreground="#555", wraplength=860,
        ).grid(row=1, column=0, sticky="w", **pad)

        box = ttk.LabelFrame(self, text="この区間")
        box.grid(row=2, column=0, sticky="ew", **pad)
        box.columnconfigure(0, weight=1)
        self.var_when = tk.StringVar()
        ttk.Label(box, textvariable=self.var_when, foreground="#333")\
            .grid(row=0, column=0, sticky="w", padx=8, pady=(6, 0))
        self.var_text = tk.StringVar()
        ttk.Label(box, textvariable=self.var_text, wraplength=840,
                  font=("", 12)).grid(row=1, column=0, sticky="w",
                                      padx=8, pady=(2, 8))

        ctx = ttk.LabelFrame(self, text="前後（文脈の確認用）")
        ctx.grid(row=3, column=0, sticky="ew", **pad)
        ctx.columnconfigure(0, weight=1)
        self.var_prev = tk.StringVar()
        self.var_next = tk.StringVar()
        ttk.Label(ctx, textvariable=self.var_prev, foreground="#666",
                  wraplength=840).grid(row=0, column=0, sticky="w", padx=8, pady=(6, 0))
        ttk.Label(ctx, textvariable=self.var_next, foreground="#666",
                  wraplength=840).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

        cand = ttk.LabelFrame(self, text="出席者（名簿順・キーの位置は変わりません）")
        cand.grid(row=4, column=0, sticky="ew", **pad)
        cand.columnconfigure(0, weight=1)
        for i, sp in enumerate(self.proj.speakers[:9]):
            ttk.Label(cand, text=f"  {i + 1}  {sp.name}"
                                 f"{f'（{sp.note}）' if sp.note else ''}")\
                .grid(row=i, column=0, sticky="w", padx=8,
                      pady=(4 if i == 0 else 0, 0))
        n = min(9, len(self.proj.speakers))
        ttk.Label(cand, text="  0  分からない        *  複数人が同時に話している",
                  foreground="#555").grid(row=n, column=0, sticky="w",
                                          padx=8, pady=(6, 6))

        keys = ttk.Label(
            self,
            text="Space 再生 ／ , 前を再生 ／ . 次を再生 ／ "
                 "1〜9・0・* で確定して次へ ／ Backspace 1 つ戻る ／ Esc 終了",
            foreground="#555",
        )
        keys.grid(row=5, column=0, sticky="w", **pad)
        self.var_action = tk.StringVar()
        ttk.Label(self, textvariable=self.var_action, foreground="#1B5E20")\
            .grid(row=6, column=0, sticky="w", **pad)

    def _bind_keys(self) -> None:
        self.bind("<space>", lambda e: self._play_current())
        self.bind("<comma>", lambda e: self._play_neighbour(-1))
        self.bind("<period>", lambda e: self._play_neighbour(+1))
        self.bind("<BackSpace>", lambda e: self._back())
        self.bind("<Escape>", lambda e: self._on_close())
        for d in "0123456789":
            self.bind(d, self._on_digit)
        self.bind("*", lambda e: self._commit(SPECIAL_MULTI))
        self.bind("<asterisk>", lambda e: self._commit(SPECIAL_MULTI))

    # ------------------------------------------------------------------
    def _target(self):
        if not self.targets:
            return None
        self.cur = max(0, min(self.cur, len(self.targets) - 1))
        return self.by_index[self.targets[self.cur]["index"]]

    def _show(self) -> None:
        seg = self._target()
        done = len(self.labels)
        self.var_progress.set(f"{done} / {len(self.targets)} 件 "
                              f"（いま {self.cur + 1} 件目）")
        self.bar.configure(value=done)
        if seg is None:
            self.var_when.set("対象がありません")
            return
        self.var_when.set(f"#{seg.index}  {fmt_hms(seg.start)} 〜 "
                          f"{fmt_hms(seg.end)}  （{seg.duration:.1f} 秒）")
        self.var_text.set(seg.text or "（本文なし）")

        pos = self.pos_of[seg.index]
        prev_s = self.ordered[pos - 1] if pos > 0 else None
        next_s = self.ordered[pos + 1] if pos + 1 < len(self.ordered) else None
        self.var_prev.set("前: " + (prev_s.preview(70) if prev_s else "（なし）"))
        self.var_next.set("次: " + (next_s.preview(70) if next_s else "（なし）"))

        already = self.labels.get(seg.index)
        if already:
            name = self.proj.speaker_name(already["speaker_id"]) or already["speaker_id"]
            self.var_action.set(f"この区間は既に「{name}」で付けてあります"
                                "（上書きするならもう一度押してください）")
        else:
            self.var_action.set("")
        self._play_current()

    def _play_current(self) -> None:
        seg = self._target()
        if seg is None:
            return
        self.player.play(self.audio, seg.start, seg.end)

    def _play_neighbour(self, direction: int) -> None:
        seg = self._target()
        if seg is None:
            return
        pos = self.pos_of[seg.index] + direction
        if not (0 <= pos < len(self.ordered)):
            return
        other = self.ordered[pos]
        self.player.play(self.audio, other.start, other.end)
        self.var_action.set("前の区間を再生しています"
                            if direction < 0 else "次の区間を再生しています")

    def _on_digit(self, event) -> None:
        d = event.char
        if d == "0":
            self._commit(SPECIAL_UNKNOWN)
            return
        i = int(d) - 1
        if 0 <= i < len(self.proj.speakers):
            self._commit(self.proj.speakers[i].id)

    def _commit(self, speaker_id: str) -> None:
        seg = self._target()
        if seg is None:
            return
        self.labels[seg.index] = {
            "index": seg.index,
            "start": round(seg.start, 2),
            "duration": round(seg.duration, 2),
            "speaker_id": speaker_id,
        }
        self._save()
        name = self.proj.speaker_name(speaker_id) or speaker_id
        if self.cur + 1 >= len(self.targets):
            self.player.stop()
            self.var_action.set(f"「{name}」で確定しました。これで最後です。")
            self._show_done()
            return
        self.cur += 1
        self._show()
        self.var_action.set(f"前の区間を「{name}」で確定しました")

    def _back(self) -> None:
        if self.cur > 0:
            self.cur -= 1
            self._show()

    def _show_done(self) -> None:
        messagebox.showinfo(
            "この巡は終わりです",
            f"{len(self.labels)} 件に正解ラベルを付けました。\n\n"
            f"保存先: {self.out}\n\n"
            "集計は tools\\report_truth.py で行います。",
            parent=self,
        )

    def _on_close(self) -> None:
        self._save()
        self.player.close()
        self.destroy()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    proj_path = Path(sys.argv[1])
    round_no = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    proj = Project.load(proj_path)
    truth_dir = proj_path.parent / "truth"
    sample_path = truth_dir / "sample.json"
    if not sample_path.exists():
        print(f"標本がありません: {sample_path}")
        print("先に tools\\make_sample.py を実行してください"
              "（測定前に標本を固定するため）。")
        return 1
    sample = json.loads(sample_path.read_text(encoding="utf-8"))

    if sample.get("audio_sha256") and proj.source_sha256 \
            and sample["audio_sha256"] != proj.source_sha256:
        print("標本と作業ファイルの音声が一致しません。取り違えの恐れがあります。")
        return 1
    if not proj.speakers:
        print("名簿が空です。先に出席者を登録してください。")
        return 1

    out = truth_dir / f"labels_blind.r{round_no}.json"
    win = TruthWindow(proj, sample, out, round_no)
    win.mainloop()
    print(f"保存先: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
