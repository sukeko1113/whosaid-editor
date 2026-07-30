"""セグメント(発言区間)のデータモデルと永続化・Word 出力。

v2.0.0 で追加。従来の「参加者名簿を渡して AI に話者を推定させる」方式に代わり、
    1) AI は声質だけで話者クラスタ(発言者A/B/C…)に分けた区間リストを作る
    2) ユーザーがタイムラインを区間単位でたどりながら、音声を聴いて話者を確定する
という流れに変更したため、その中間成果物を保持する構造が必要になった。

中間成果物は `<出力フォルダ>/<音声名>.speakers.json` に保存する。
アプリを閉じても、あとから同じファイルを開いて割当作業を再開できる。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA_VERSION = 2

UNKNOWN_LABEL = "発言者不明"
MULTI_LABEL = "発言者複数・重複"

# 特別扱いの話者 ID(名簿の人物ではない)
SPECIAL_UNKNOWN = "__unknown__"
SPECIAL_MULTI = "__multi__"
SPECIAL_NOISE = "__noise__"

SPECIAL_SPEAKERS: dict[str, str] = {
    SPECIAL_UNKNOWN: UNKNOWN_LABEL,
    SPECIAL_MULTI: MULTI_LABEL,
    SPECIAL_NOISE: "発言なし・雑音",
}


def fmt_hms(seconds: float) -> str:
    """秒 → [HH:MM:SS] 用の 'HH:MM:SS' 文字列"""
    total = int(round(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def fmt_ms(seconds: float) -> str:
    """秒 → 'MM:SS'(1時間未満のときの簡易表示用)"""
    total = int(round(seconds))
    if total >= 3600:
        return fmt_hms(total)
    return f"{total // 60:02d}:{total % 60:02d}"


# ----------------------------------------------------------------------
# 話者
# ----------------------------------------------------------------------

@dataclass
class Speaker:
    """出席者(候補者リストに並ぶ 1 人)"""

    id: str
    name: str
    note: str = ""          # 役職・特徴などの補足
    order: int = 0          # 名簿上の並び順(初期の候補順に使う)

    @property
    def display(self) -> str:
        return f"{self.name}({self.note})" if self.note else self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Speaker":
        return cls(
            id=str(d["id"]),
            name=str(d.get("name", "")),
            note=str(d.get("note", "")),
            order=int(d.get("order", 0)),
        )


_ROSTER_LINE = re.compile(
    r"^\s*(?:[-*・]\s*)?"          # 行頭の箇条書き記号は無視
    r"(?P<name>[^(（:：]+?)"        # 名前
    r"(?:[(（](?P<note1>[^)）]*)[)）])?"   # (役職)
    r"\s*(?:[:：]\s*(?P<note2>.*))?$"      # : 補足
)


def parse_roster(text: str) -> list[Speaker]:
    """名簿テキスト(1行1人)を Speaker のリストにする。

    受け付ける形:
        佐藤
        佐藤(理事)
        佐藤(理事): 議長役。名乗ることが多い
        - 佐藤理事：会計担当
    """
    speakers: list[Speaker] = []
    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        m = _ROSTER_LINE.match(line)
        if not m:
            name, note = line, ""
        else:
            name = (m.group("name") or "").strip()
            parts = [p.strip() for p in (m.group("note1"), m.group("note2")) if p and p.strip()]
            note = " / ".join(parts)
        if not name:
            continue
        speakers.append(Speaker(id=f"sp{i + 1:02d}", name=name, note=note, order=len(speakers)))
    return speakers


def roster_to_text(speakers: Iterable[Speaker]) -> str:
    lines = []
    for sp in speakers:
        lines.append(f"{sp.name}({sp.note})" if sp.note else sp.name)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# セグメント
# ----------------------------------------------------------------------

@dataclass
class Segment:
    """1 つの発言区間"""

    index: int
    start: float                    # 元音声内の絶対秒
    end: float
    text: str
    cluster: str                    # 例 "0:A"(チャンク0の発言者A)。声質ベースの仮ラベル
    chunk: int = 0
    speaker_id: Optional[str] = None    # 確定した話者(None = 未確定)
    reviewed: bool = False              # ユーザーが目を通した(音声を聴いた)か
    note: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def cluster_label(self) -> str:
        """UI 表示用の短いクラスタ名 → 'C1-A'"""
        chunk, _, tail = self.cluster.partition(":")
        try:
            return f"C{int(chunk) + 1}-{tail}"
        except ValueError:
            return self.cluster

    def preview(self, width: int = 60) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= width else t[: width - 1] + "…"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Segment":
        return cls(
            index=int(d["index"]),
            start=float(d.get("start", 0.0)),
            end=float(d.get("end", 0.0)),
            text=str(d.get("text", "")),
            cluster=str(d.get("cluster", "")),
            chunk=int(d.get("chunk", 0)),
            speaker_id=d.get("speaker_id") or None,
            reviewed=bool(d.get("reviewed", False)),
            note=str(d.get("note", "")),
        )


# ----------------------------------------------------------------------
# プロジェクト(1 音声ファイル分の作業状態)
# ----------------------------------------------------------------------

@dataclass
class Project:
    audio_path: str
    duration: float = 0.0
    chunk_seconds: int = 600
    model: str = ""
    verbatim: bool = False
    speakers: list[Speaker] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    json_path: Optional[str] = None      # 保存先(load 時に設定)

    # -------------------------------------------------- 話者アクセス
    def speaker(self, speaker_id: Optional[str]) -> Optional[Speaker]:
        if not speaker_id:
            return None
        if speaker_id in SPECIAL_SPEAKERS:
            return Speaker(id=speaker_id, name=SPECIAL_SPEAKERS[speaker_id], order=999)
        for sp in self.speakers:
            if sp.id == speaker_id:
                return sp
        return None

    def speaker_name(self, speaker_id: Optional[str]) -> str:
        sp = self.speaker(speaker_id)
        return sp.name if sp else ""

    def add_speaker(self, name: str, note: str = "") -> Speaker:
        existing = {sp.id for sp in self.speakers}
        i = len(self.speakers) + 1
        while f"sp{i:02d}" in existing:
            i += 1
        sp = Speaker(id=f"sp{i:02d}", name=name, note=note, order=len(self.speakers))
        self.speakers.append(sp)
        return sp

    def remove_speaker(self, speaker_id: str) -> None:
        self.speakers = [sp for sp in self.speakers if sp.id != speaker_id]
        for seg in self.segments:
            if seg.speaker_id == speaker_id:
                seg.speaker_id = None
                seg.reviewed = False
        for i, sp in enumerate(self.speakers):
            sp.order = i

    # -------------------------------------------------- 統計
    @property
    def assigned_count(self) -> int:
        return sum(1 for s in self.segments if s.speaker_id)

    @property
    def total_count(self) -> int:
        return len(self.segments)

    def clusters(self) -> list[str]:
        seen: list[str] = []
        for s in self.segments:
            if s.cluster not in seen:
                seen.append(s.cluster)
        return seen

    def cluster_segments(self, cluster: str) -> list[Segment]:
        return [s for s in self.segments if s.cluster == cluster]

    # -------------------------------------------------- 永続化
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "audio_path": self.audio_path,
            "duration": self.duration,
            "chunk_seconds": self.chunk_seconds,
            "model": self.model,
            "verbatim": self.verbatim,
            "speakers": [sp.to_dict() for sp in self.speakers],
            "segments": [sg.to_dict() for sg in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Project":
        return cls(
            audio_path=str(d.get("audio_path", "")),
            duration=float(d.get("duration", 0.0)),
            chunk_seconds=int(d.get("chunk_seconds", 600)),
            model=str(d.get("model", "")),
            verbatim=bool(d.get("verbatim", False)),
            speakers=[Speaker.from_dict(x) for x in d.get("speakers", [])],
            segments=[Segment.from_dict(x) for x in d.get("segments", [])],
        )

    def save(self, path: Optional[Path | str] = None) -> Path:
        target = Path(path or self.json_path or "")
        if not str(target):
            raise ValueError("保存先が指定されていません。")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        tmp.replace(target)
        self.json_path = str(target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "Project":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        proj = cls.from_dict(data)
        proj.json_path = str(p)
        return proj

    @staticmethod
    def default_json_path(output_dir: Path | str, audio_path: Path | str) -> Path:
        return Path(output_dir) / f"{Path(audio_path).stem}.speakers.json"


# ----------------------------------------------------------------------
# Word 出力
# ----------------------------------------------------------------------

REVIEW_NOTE_TEMPLATE = (
    "※ 話者ラベルはユーザーが音声を聴いて割り当てたものです"
    "(確定 {done}/{total} 区間{unassigned_note})。"
)


def _merge_runs(
    proj: Project,
    merge_consecutive: bool = True,
) -> list[tuple[float, Optional[str], str]]:
    """(開始秒, 話者ID, 本文) の並びを作る。
    merge_consecutive=True なら、同一話者の連続区間を 1 段落にまとめる。
    """
    runs: list[tuple[float, Optional[str], list[str]]] = []
    for seg in proj.segments:
        text = seg.text.strip()
        if not text:
            continue
        if (
            merge_consecutive
            and runs
            and runs[-1][1] == seg.speaker_id
            and seg.speaker_id is not None
        ):
            runs[-1][2].append(text)
        else:
            runs.append((seg.start, seg.speaker_id, [text]))
    return [(start, sid, " ".join(parts)) for start, sid, parts in runs]


def write_docx(
    proj: Project,
    output_path: Path | str,
    title: Optional[str] = None,
    with_timestamps: bool = True,
    merge_consecutive: bool = True,
    include_note: bool = True,
) -> Path:
    """割当結果を Word ファイルに書き出す。"""
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    output_path = Path(output_path)
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "游明朝"
    style.font.size = Pt(11)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), "游明朝")

    doc.add_heading(title or Path(proj.audio_path).stem, level=1)

    if include_note:
        unassigned = proj.total_count - proj.assigned_count
        note = REVIEW_NOTE_TEMPLATE.format(
            done=proj.assigned_count,
            total=proj.total_count,
            unassigned_note=f"・未確定 {unassigned} 区間" if unassigned else "",
        )
        p = doc.add_paragraph()
        run = p.add_run(note)
        run.italic = True
        run.font.color.rgb = RGBColor(0x70, 0x70, 0x70)
        doc.add_paragraph()

    for start, sid, text in _merge_runs(proj, merge_consecutive):
        p = doc.add_paragraph()
        if with_timestamps:
            ts = p.add_run(f"[{fmt_hms(start)}] ")
            ts.bold = True
        sp = proj.speaker(sid)
        label = sp.name if sp else UNKNOWN_LABEL
        name_run = p.add_run(f"【{label}】 ")
        name_run.bold = True
        if sp is None:
            name_run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        p.add_run(text)

    doc.save(str(output_path))
    return output_path


def write_text(proj: Project, output_path: Path | str, merge_consecutive: bool = True) -> Path:
    """プレーンテキスト出力(確認・差分取り用)"""
    output_path = Path(output_path)
    lines = []
    for start, sid, text in _merge_runs(proj, merge_consecutive):
        sp = proj.speaker(sid)
        lines.append(f"[{fmt_hms(start)}] 【{sp.name if sp else UNKNOWN_LABEL}】 {text}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
