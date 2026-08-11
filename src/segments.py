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


SCHEMA_VERSION = 3

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

# 擬似クラスタの記号(transcribe.normalize_cluster_label の出力と対応)
PSEUDO_UNKNOWN = "?"
PSEUDO_MULTI = "*"

# 人が時刻を直したり区間を分けたりするときに許す最短の長さ。
# 0 にすると start == end の区間ができて、再生も出力も意味を失う。
MIN_SEGMENT_SECONDS = 0.1


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


def fmt_hms_frac(seconds: float) -> str:
    """秒 → 'HH:MM:SS.s'(0.1 秒精度)。時刻を人が直接編集するときの表示形式。

    一覧の表示に使う fmt_hms() は 1 秒に丸めるが、区間の境目を耳で合わせる
    作業では 0.1 秒が要る。負の秒は 0 として扱う(時刻に負は無い)。
    """
    tenths = max(0, int(round(seconds * 10)))
    total, frac = divmod(tenths, 10)
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}.{frac}"


# 全角のまま打たれても受ける(日本語入力の途中で切り替え忘れが起きやすい)
_ZEN_TO_HAN = str.maketrans("０１２３４５６７８９：．", "0123456789:.")
_INT_FIELD = re.compile(r"^\d+$", re.ASCII)
_SEC_FIELD = re.compile(r"^\d+(?:\.\d*)?$", re.ASCII)


def parse_hms(text: str) -> float:
    """'HH:MM:SS.s' / 'MM:SS.s' / 'SS.s' のいずれかを秒に直す(0.1 秒精度)。

    時刻入力欄から受ける。読めない文字列は ValueError にして、
    呼び出し側が「編集前の値に戻す」判断をできるようにする。
    最上位の桁だけは 60 以上を許す('90' = 1分30秒、'75:00' = 75分)。
    """
    s = str(text).strip().translate(_ZEN_TO_HAN).replace(" ", "").replace("　", "")
    if not s:
        raise ValueError("時刻が空です。")
    parts = s.split(":")
    if len(parts) > 3:
        raise ValueError(f"時刻の形式が読めません: {text!r}")
    if not all(_INT_FIELD.match(p) for p in parts[:-1]) or not _SEC_FIELD.match(parts[-1]):
        raise ValueError(f"時刻の形式が読めません: {text!r}")
    values = [float(p) for p in parts]
    for v in values[1:]:
        if v >= 60.0:
            raise ValueError(f"分・秒は 60 未満で指定してください: {text!r}")
    total = 0.0
    for v in values:
        total = total * 60.0 + v
    return round(total, 1)


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
    reviewed: bool = False              # この区間の音声を実際に聴いて確定したか
    note: str = ""
    text_edited: bool = False           # ユーザーが本文を手直ししたか(再実行時に保護)
    time_edited: bool = False           # ユーザーが時刻を直したか
    # パイプライン(AI)が出した元の時刻。start/end をユーザーが直しても動かさない。
    # 再実行したときに新旧の区間を突き合わせる鍵と、「元に戻す」の戻り先に使う。
    # None を渡すと start/end で埋める(新規生成時と、これを持たない旧ファイル)。
    orig_start: Optional[float] = None
    orig_end: Optional[float] = None

    def __post_init__(self) -> None:
        if self.orig_start is None:
            self.orig_start = self.start
        if self.orig_end is None:
            self.orig_end = self.end

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def cluster_tail(self) -> str:
        """クラスタ記号だけ('0:A' → 'A')"""
        return self.cluster.partition(":")[2] or self.cluster

    @property
    def is_pseudo_cluster(self) -> bool:
        """『誰か判別できない』『複数人が重なっている』の擬似クラスタか。

        これらは「同じ声のまとまり」ではなく雑多な寄せ集めなので、
        学習にも一括適用にも使ってはいけない。
        """
        return self.cluster_tail in (PSEUDO_UNKNOWN, PSEUDO_MULTI)

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
        # orig_start / orig_end は schema 3 から。旧ファイルには無いので None を
        # 渡し、__post_init__ に start / end を入れさせる(移行処理は不要)。
        orig_start = d.get("orig_start")
        orig_end = d.get("orig_end")
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
            text_edited=bool(d.get("text_edited", False)),
            time_edited=bool(d.get("time_edited", False)),
            orig_start=float(orig_start) if orig_start is not None else None,
            orig_end=float(orig_end) if orig_end is not None else None,
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
    # 音声の中身から作った指紋。ファイル名が同じでも中身が変われば別物として
    # 扱うために使う(空文字なら未記録 = 古い形式の作業ファイル)。
    audio_fingerprint: str = ""
    # 再生位置のずれ補正(秒)。Gemini の時刻推定が実音声より早い/遅いときに
    # 使う。録音ごとに傾向が違うので、設定ではなく作業ファイルに持たせる。
    time_offset: float = 0.0
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

    # -------------------------------------------------- 区間の分割・結合
    def renumber(self) -> None:
        """index を 0..n-1 に振り直す。区間を増減させたら必ず呼ぶ。"""
        for i, seg in enumerate(self.segments):
            seg.index = i

    def split_segment(self, index: int, boundary: float, cut: int) -> tuple[Segment, Segment]:
        """1 つの区間を、境界時刻と本文の位置で 2 つに分ける。

        1 区間に 2 人の発言が(重なりなしで)順に混ざっているときに使う。
        転写から丸ごと落ちた発言も、隣の区間を分割して空いた側に本文を
        書き足せば復元できる。

        boundary: 境界の秒(実音声の時刻)。区間の内側に収める。
        cut:      本文を切る位置(文字数。ダイアログのカーソル位置)。

        後半のクラスタを擬似不明にするのは、元の声質ラベルが区間全体に付いた
        もので、後半の声には保証がないため。擬似クラスタは候補学習からも一括
        適用からも自動的に外れる(Segment.is_pseudo_cluster)ので、分割で
        生まれた不確かな区間が学習を汚したり誤って伝播したりするのを防げる。
        """
        seg = self.segments[index]
        if seg.duration < MIN_SEGMENT_SECONDS * 2:
            raise ValueError("この区間は短すぎて分割できません。")
        lo = seg.start + MIN_SEGMENT_SECONDS
        hi = seg.end - MIN_SEGMENT_SECONDS
        boundary = min(max(round(float(boundary), 1), lo), hi)
        cut = max(0, min(int(cut), len(seg.text)))

        head = Segment(
            index=index,
            start=seg.start,
            end=boundary,
            text=seg.text[:cut],
            cluster=seg.cluster,            # 前半は元の声のまとまりのまま
            chunk=seg.chunk,
            speaker_id=seg.speaker_id,
            reviewed=False,                 # 範囲が変わったので聴き直し対象
            note=seg.note,
            text_edited=seg.text_edited,
            time_edited=True,
            # 再実行時に「元は 1 つだった」と分かるよう、親の値を両方が共有する
            orig_start=seg.orig_start,
            orig_end=seg.orig_end,
        )
        tail = Segment(
            index=index + 1,
            start=boundary,
            end=seg.end,
            text=seg.text[cut:],
            cluster=f"{seg.chunk}:{PSEUDO_UNKNOWN}",
            chunk=seg.chunk,
            speaker_id=None,                # 後半の声は別人かもしれない
            reviewed=False,
            note="",
            text_edited=seg.text_edited,
            time_edited=True,
            orig_start=seg.orig_start,
            orig_end=seg.orig_end,
        )
        self.segments[index:index + 1] = [head, tail]
        self.renumber()
        return head, tail

    def merge_segments(self, index: int) -> Segment:
        """index の区間と、その次の区間を 1 つにまとめる。

        分割のやり直し(逆操作)と、同じ発言が 2 行に割れているときの整理に使う。
        話者が食い違うときは、どちらが正しいか機械には決められないので未確定に
        落とす。呼び出し側はその旨をユーザーに確認してから呼ぶこと。
        """
        if index < 0 or index + 1 >= len(self.segments):
            raise ValueError("次の区間がないので結合できません。")
        a, b = self.segments[index], self.segments[index + 1]
        notes = [n for n in (a.note, b.note) if n]
        merged = Segment(
            index=index,
            start=min(a.start, b.start),
            end=max(a.end, b.end),
            text=a.text + b.text,           # 日本語なので空白を挟まない
            cluster=a.cluster,              # 前側の声のまとまりを採用する
            chunk=a.chunk,
            speaker_id=a.speaker_id if a.speaker_id == b.speaker_id else None,
            reviewed=False,                 # 範囲が変わったので聴き直し対象
            note=" / ".join(notes),
            text_edited=a.text_edited or b.text_edited,
            time_edited=True,
            orig_start=a.orig_start,        # 系譜の始まりは前側
            orig_end=b.orig_end,            # 終わりは後側(再実行時の吸収に要る)
        )
        self.segments[index:index + 2] = [merged]
        self.renumber()
        return merged

    # -------------------------------------------------- 統計
    @property
    def assigned_count(self) -> int:
        return sum(1 for s in self.segments if s.speaker_id)

    @property
    def reviewed_count(self) -> int:
        """実際に音声を聴いて確定した区間の数。

        一括適用で埋めた区間は「確定はしているが未確認」なので、ここには入らない。
        あとから未確認だけを拾い直せるようにするための区別。
        """
        return sum(1 for s in self.segments if s.speaker_id and s.reviewed)

    @property
    def unreviewed_count(self) -> int:
        return sum(1 for s in self.segments if s.speaker_id and not s.reviewed)

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
            "audio_fingerprint": self.audio_fingerprint,
            "time_offset": self.time_offset,
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
            audio_fingerprint=str(d.get("audio_fingerprint", "")),
            time_offset=float(d.get("time_offset", 0.0) or 0.0),
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

def _merge_runs(
    proj: Project,
    merge_consecutive: bool = True,
    drop_noise: bool = True,
) -> list[tuple[float, Optional[str], str]]:
    """(開始秒, 話者ID, 本文) の並びを作る。
    merge_consecutive=True なら、同一話者の連続区間を 1 段落にまとめる。
    drop_noise=True なら「発言なし・雑音」と印を付けた区間は出力しない。
    """
    runs: list[tuple[float, Optional[str], list[str]]] = []
    for seg in proj.segments:
        text = seg.text.strip()
        if not text:
            continue
        if drop_noise and seg.speaker_id == SPECIAL_NOISE:
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
    # 日本語なので連結時に空白を挟まない
    return [(start, sid, "".join(parts)) for start, sid, parts in runs]


def build_note(proj: Project) -> str:
    """docx 冒頭に入れる但し書き。何がどこまで人手で確認されたかを明示する。"""
    parts = [
        "※ 話者ラベルはユーザーが音声を聴いて割り当てたものです",
        f"(全 {proj.total_count} 区間中、聴いて確定 {proj.reviewed_count} 区間",
    ]
    if proj.unreviewed_count:
        parts.append(f"、まとめて適用 {proj.unreviewed_count} 区間")
    unassigned = proj.total_count - proj.assigned_count
    if unassigned:
        parts.append(f"、未確定 {unassigned} 区間")
    parts.append(")。")
    return "".join(parts)


def write_docx(
    proj: Project,
    output_path: Path | str,
    title: Optional[str] = None,
    with_timestamps: bool = True,
    merge_consecutive: bool = True,
    include_note: bool = True,
    include_attendees: bool = True,
    drop_noise: bool = True,
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

    if include_attendees and proj.speakers:
        p = doc.add_paragraph()
        head = p.add_run("出席者: ")
        head.bold = True
        p.add_run("、".join(sp.display for sp in proj.speakers))

    if include_note:
        p = doc.add_paragraph()
        run = p.add_run(build_note(proj))
        run.italic = True
        run.font.color.rgb = RGBColor(0x70, 0x70, 0x70)
        doc.add_paragraph()

    for start, sid, text in _merge_runs(proj, merge_consecutive, drop_noise):
        p = doc.add_paragraph()
        if with_timestamps:
            ts = p.add_run(f"[{fmt_hms(start)}] ")
            ts.bold = True
        sp = proj.speaker(sid)
        label = sp.name if sp else UNKNOWN_LABEL
        name_run = p.add_run(f"【{label}】 ")
        name_run.bold = True
        if sid is None:
            # 一度も触れられていない区間。要確認なので赤で目立たせる
            name_run.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        elif sid in SPECIAL_SPEAKERS:
            # ユーザーが意図的に「不明」等と判断した区間はグレー
            name_run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
        p.add_run(text)

    doc.save(str(output_path))
    return output_path


def write_text(
    proj: Project,
    output_path: Path | str,
    merge_consecutive: bool = True,
    drop_noise: bool = True,
) -> Path:
    """プレーンテキスト出力(自分のテンプレートに貼り込む場合や、差分取り用)"""
    output_path = Path(output_path)
    lines = []
    for start, sid, text in _merge_runs(proj, merge_consecutive, drop_noise):
        sp = proj.speaker(sid)
        lines.append(f"[{fmt_hms(start)}] 【{sp.name if sp else UNKNOWN_LABEL}】 {text}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
