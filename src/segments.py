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
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


SCHEMA_VERSION = 5

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

# 処理経路(Project.engine["mode"])。作業ファイルにそのまま入る値なので、
# 画面・Word の検証要約・パイプラインで同じ文字列を使う。
ENGINE_CLOUD = "cloud"
ENGINE_LOCAL = "local"
ENGINE_LABELS = {ENGINE_CLOUD: "クラウド", ENGINE_LOCAL: "ローカル"}

# 人が時刻を直したり区間を分けたりするときに許す最短の長さ。
# 0 にすると start == end の区間ができて、再生も出力も意味を失う。
MIN_SEGMENT_SECONDS = 0.1


def audio_span(seg: "Segment", time_offset: float) -> tuple[float, float]:
    """その区間が実音声のどこで鳴っているか(開始, 終了)。

    再生の規約: 実音声の位置 = 保存時刻 + ずれ補正。ただし一度直した区間の
    start/end は実音声の時刻そのものなので、補正を足さない。
    画面の再生・時刻編集の初期値・自動点検の照合窓が、すべてこの 1 つの
    規約を見るようにしてある(散らばると必ず食い違う)。
    """
    if seg.time_edited:
        return seg.start, seg.end
    return max(0.0, seg.start + time_offset), max(0.0, seg.end + time_offset)


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

@dataclass(frozen=True)
class Utterance:
    """転写エンジンが返す 1 発言。Segment を組み立てる前の中間の形。

    クラウド(Gemini)とローカル(faster-whisper)で、ここまでは同じ形に揃える。
    どちらの経路も「チャンク → Utterance の並び」を返し、オフセットの足し込みと
    通し番号の付与から先は共通の後段が引き受ける
    (claude/claude_ローカル転写_設計書.md §4.2)。

    時刻はチャンクの先頭からの相対秒。絶対秒にするのは後段の仕事。
    """

    rel_start: float
    rel_end: float
    text: str
    # 声のまとまりの記号。"A"/"B"… のほか、判別不能は "?"、複数人同時は "*"。
    # チャンク番号を頭に付けた "0:A" の形にするのは後段(Segment を作るとき)。
    cluster: str = PSEUDO_UNKNOWN


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
    # 時刻を「自分の耳で確かめた」か。機械が出した時刻を当てただけの区間と区別する。
    # 話者の reviewed と同じ考え方で、あとから未確認だけを拾い直せるようにする。
    time_reviewed: bool = False
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
        """UI 表示用の短いクラスタ名。

        チャンク内で閉じたクラスタ(Gemini)は 'C1-A'。
        全長で分けたクラスタ(話者分離)は '声A' —— チャンク番号を出しても
        意味が無く、むしろ「C1-A と C2-A は別」という誤解を招く。
        """
        chunk, _, tail = self.cluster.partition(":")
        if chunk == "g":
            return f"声{tail}"
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
        # schema 3 までは「時刻を直した = 自分の耳で合わせた」しかなかった。
        # 機械が出した時刻を当てただけの区間と区別する印は 4 で足したので、
        # 古いファイルの time_edited は確認済みとして読む。
        time_edited = bool(d.get("time_edited", False))
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
            time_edited=time_edited,
            time_reviewed=bool(d.get("time_reviewed", time_edited)),
            orig_start=float(orig_start) if orig_start is not None else None,
            orig_end=float(orig_end) if orig_end is not None else None,
        )


def utterances_to_segments(
    utterances: Iterable[Utterance],
    *,
    chunk_index: int = 0,
    offset_seconds: float = 0.0,
    start_index: int = 0,
) -> list[Segment]:
    """Utterance の並びを Segment に組み立てる(両経路の共通の後段)。

    クラウド(Gemini)もローカル(faster-whisper)も、チャンク単位で Utterance を
    作るところまでが経路ごとの仕事で、その先——チャンク先頭からの相対秒に
    オフセットを足して絶対秒にする / クラスタ記号にチャンク番号の名前空間を
    付ける / 通し番号を振る——は同じ。1 か所に集めておかないと、経路を足す
    たびに同じ処理が増える(設計書 §4.2)。

    ここで時刻をいじるのはオフセットの足し込みだけ。文字数按分
    (redistribute_times)は通さない。按分は Gemini のタイムスタンプが
    ドリフトする既知バグへの対策であって、実測時刻にかけるものではない。
    クラウド経路では parse_utterances の中で既に済ませてある。
    """
    out: list[Segment] = []
    for i, u in enumerate(utterances):
        start = offset_seconds + u.rel_start
        end = offset_seconds + u.rel_end
        # 長さ 0 の区間は聴き直せない(再生しても何も鳴らない)ので、
        # 最低限の長さを与えて操作できる状態にする。
        if end <= start:
            end = start + 1.0
        out.append(Segment(
            index=start_index + i,
            start=round(start, 2),
            end=round(end, 2),
            text=u.text,
            cluster=f"{chunk_index}:{u.cluster}",
            chunk=chunk_index,
        ))
    return out


# ----------------------------------------------------------------------
# 話者分離の結果を区間へ落とす
# (claude/claude_話者分離_設計書.md §4・§5・§8)
# ----------------------------------------------------------------------

# 区間の何割が話者区間と重なれば、その話者とみなすか。
# 自動クラスタリングが作る「1 区間だけの幽霊話者」は、この閾値でどの区間も
# 取れずに自然に消える(PoC で確認)。
MIN_SPEAKER_OVERLAP = 0.5

# 連結してよい間隔。これ以上空いていれば別の発言とみなす。
SPEAKER_MERGE_MAX_GAP = 0.3

# 全長の名前空間。チャンク内で閉じる "0:A" と区別する(設計書 §5)。
GLOBAL_NAMESPACE = "g"

_SENTENCE_ENDS = "。．！？!?」』…"


def _speaker_letters(turns: Iterable[Any]) -> dict[int, str]:
    """話者番号を A/B/C… に写す。**先に話した人から順**に振る。

    分離器が返す番号は連番とは限らない(実測で 0/1/3/4/7/22/33… のように飛ぶ)。
    番号をそのまま見せると意味の無い数字が並ぶので、出てきた順に文字を当てる。
    """
    order: list[int] = []
    for t in sorted(turns, key=lambda x: (x.start, x.end)):
        if t.speaker not in order:
            order.append(t.speaker)
    letters: dict[int, str] = {}
    for i, spk in enumerate(order):
        letters[spk] = chr(ord("A") + i) if i < 26 else f"S{i + 1}"
    return letters


def assign_speaker_clusters(
    segments: Sequence[Segment],
    turns: Sequence[Any],
    min_overlap: float = MIN_SPEAKER_OVERLAP,
) -> list[str]:
    """区間ごとの `cluster` 文字列を作る(設計書 §4・§5)。

    区間の時間範囲と**最も重なる話者区間**の話者を採る。重なりが区間の長さの
    `min_overlap` に満たなければ `?`(判別不能)にする。

    返すのは文字列の並びだけで、区間には書き込まない。書き込む場所を 1 つに
    保つため(呼び出し側で `seg.cluster = ...` する)。

    話者区間は重なりうるので、「最も重なる 1 つ」を選ぶ。同時に話している
    区間には、より長く重なっていたほうが入る。**それを `*` に直すのは人**
    ——重なりから機械的に判定しようとすると適合率 50% にしかならないことを
    実測した(設計書 §6)。
    """
    letters = _speaker_letters(turns)
    out: list[str] = []
    for seg in segments:
        best_spk, best_ov = None, 0.0
        for t in turns:
            ov = min(seg.end, t.end) - max(seg.start, t.start)
            if ov > best_ov:
                best_spk, best_ov = t.speaker, ov
        span = max(0.01, seg.end - seg.start)
        if best_spk is None or best_ov / span < min_overlap:
            out.append(f"{GLOBAL_NAMESPACE}:{PSEUDO_UNKNOWN}")
        else:
            out.append(f"{GLOBAL_NAMESPACE}:{letters[best_spk]}")
    return out


def merge_same_speaker(
    segments: Sequence[Segment],
    max_gap: float = SPEAKER_MERGE_MAX_GAP,
) -> list[Segment]:
    """同じ話者の、文の途中で切れた区間を連結する(設計書 §8)。

    ローカル転写は 74% の区間が文の途中で終わる(実測)。話者ラベルが付けば
    安全に連結できる。条件は 3 つとも満たすこと:

      1. 同じ話者(擬似クラスタ `?` `*` は連結しない)
      2. 前の区間が句点等で終わっていない(文が続いている)
      3. 間隔が max_gap 未満

    **人が手を付けた区間は連結しない。**割当・本文の手直し・時刻の修正が
    入っているものを勝手にまとめると、その作業が消える。

    `orig_start` / `orig_end` は前後の端を保つ。再実行時に新旧の区間を
    突き合わせる鍵なので、連結で失うと引き継ぎが壊れる。
    """
    out: list[Segment] = []
    for seg in segments:
        if out:
            prev = out[-1]
            touched = any((
                prev.speaker_id, seg.speaker_id,
                prev.text_edited, seg.text_edited,
                prev.time_edited, seg.time_edited,
            ))
            joinable = (
                not touched
                and prev.cluster == seg.cluster
                and not prev.is_pseudo_cluster
                and prev.text
                and prev.text[-1] not in _SENTENCE_ENDS
                and seg.start - prev.end < max_gap
            )
            if joinable:
                prev.text = prev.text + seg.text
                prev.end = seg.end
                prev.orig_end = seg.orig_end
                continue
        out.append(seg)
    for i, seg in enumerate(out):
        seg.index = i
    return out


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
    # ---- ここから v5(検証履歴)。追加は文書レベルのみで、区間の形は変えない ----
    # 元音声の SHA-256。「この書面はこの録音から作った」を第三者が
    # Get-FileHash / certutil / sha256sum で検算するための値。
    # 指紋(audio_fingerprint)とは役割が違う: あちらはキャッシュの同一性判定。
    source_sha256: str = ""
    # 処理経路の記録(自由形式)。慣例のキー: mode("cloud"/"local")・model・
    # app_version・at(UTC の ISO8601)。将来のモデル出所記録(Model BOM)にも
    # ここを拡張して使う。
    engine: dict[str, Any] = field(default_factory=dict)
    # Word を出力するたびに +1 し、書面に「版」として併記する。
    # ファイルを開いただけでは進めない(開いた事実は版ではない)。
    doc_revision: int = 0
    # 追記型の編集履歴。慣例のキー: at(UTC)・actor("user"/"inspect")・
    # kind(time/text/speaker/…)・target(orig_start)・before/after・batch_id。
    # v5 では器だけを定義し、記録の書き込みは編集履歴の実装(Day 45)で行う。
    edit_log: list[dict[str, Any]] = field(default_factory=list)
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
            # 境界は人が決めるが、外側の端は元の区間のまま。親が未確認なら
            # 子も未確認にする(分割したというだけで確認済みには昇格させない)。
            time_reviewed=seg.time_reviewed,
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
            time_reviewed=seg.time_reviewed,
            orig_start=seg.orig_start,
            orig_end=seg.orig_end,
        )
        self.segments[index:index + 1] = [head, tail]
        self.renumber()
        return head, tail

    # -------------------------------------------------- 相づちを足す
    def added_utterance_keys(self) -> set[float]:
        """人が足した区間の orig_start（丸め済み）。

        **再実行の引き継ぎがこれを見て、突き合わせから外す。**外さないと、
        足した区間の独自の時刻が近くの無関係な区間に誤って照合され、
        その区間を置き換えて消す（設計書 §4）。

        識別に区間のフラグを使わないのは、スキーマを増やさないため
        （v3 の一括移行まで待つ）。edit_log は既にあり、再実行でも
        引き継がれるので、ここに置くのが自然。
        """
        keys: set[float] = set()
        for rec in self.edit_log:
            op = rec.get("op")
            k = rec.get("orig_start")
            if k is None:
                continue
            if op == "add_utterance":
                keys.add(round(float(k), 3))
            elif op == "remove_added_utterance":
                keys.discard(round(float(k), 3))
        return keys

    def is_added_utterance(self, seg: Segment) -> bool:
        """この区間は人が足したものか（消してよいか）。"""
        key = float(seg.orig_start if seg.orig_start is not None else seg.start)
        return round(key, 3) in self.added_utterance_keys()

    def add_utterance(self, start: float, end: float, text: str,
                      cluster: str = "", cut: Optional[int] = None,
                      parent_orig: Optional[float] = None) -> Segment:
        """聞こえたのに本文に無い発話を、区間として足す（設計書 §2）。

        時刻と声のまとまりは機械（話者分離の turn）が用意し、本文は人が打つ。
        **重なりを禁止しない。**相づちは主発言と重なるのが本性なので、
        既存区間と時間的に重なってよい。

        話者は付けずに返す。付けるのは呼び出し側の通常の割当操作で、
        そのときの ✓/△ は既存の意味論に従う（機械が ✓ を立てる経路は無い）。
        """
        start = round(float(start), 3)
        end = round(max(float(end), start + MIN_SEGMENT_SECONDS), 3)
        text = (text or "").strip()
        if not text:
            raise ValueError("本文が空です。")
        # 前後の区間と同じチャンクに属させる（チャンク番号は再生や
        # クラスタ記号の表示に使われる）
        near = min(self.segments, key=lambda s: abs(s.start - start),
                   default=None)
        seg = Segment(
            index=0,                        # renumber で振り直す
            start=start,
            end=end,
            text=text,
            cluster=cluster or f"{near.chunk if near else 0}:{PSEUDO_UNKNOWN}",
            chunk=near.chunk if near else 0,
            speaker_id=None,
            reviewed=False,
            text_edited=True,               # 人が打った本文
            time_edited=False,              # 時刻は turn 由来の機械値
        )
        pos = len([s for s in self.segments if (s.start, s.index) < (start, 0)])
        self.segments.insert(pos, seg)
        self.renumber()
        self.edit_log.append({
            "op": "add_utterance",
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "actor": "user",
            "orig_start": float(seg.orig_start),
            "start": start,
            "end": end,
            "cluster": seg.cluster,
            # **どの区間の本文の、どこに割り込んだか。**
            # 区間は割らない(割ると直すときに元の本文を復元できない)。
            # 代わりにここを覚えておき、Word に出すときだけ差し込む。
            "cut": None if cut is None else int(cut),
            "parent_orig": None if parent_orig is None else float(parent_orig),
        })
        return seg

    def remove_added_utterance(self, index: int) -> None:
        """人が足した区間を消す。**それ以外は消せない。**

        短い相づちの自動削除をしない原則（CLAUDE.md）はそのまま。
        ここで消せるのは人がいま足したものだけで、音声認識が出した区間には
        削除の入口を作らない。
        """
        if not (0 <= index < len(self.segments)):
            raise ValueError("その区間はありません。")
        seg = self.segments[index]
        if not self.is_added_utterance(seg):
            raise ValueError("人が足した区間ではないので消せません。")
        del self.segments[index]
        self.renumber()
        self.edit_log.append({
            "op": "remove_added_utterance",
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "actor": "user",
            "orig_start": float(seg.orig_start),
        })

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
            # 両端とも耳で確かめてあったときだけ確認済みのまま(片方が未確認なら未確認)
            time_reviewed=a.time_reviewed and b.time_reviewed,
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
            "source_sha256": self.source_sha256,
            "engine": self.engine,
            "doc_revision": self.doc_revision,
            "edit_log": self.edit_log,
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
            # v4 以前のファイルには無い。既定値で読めば移行処理は不要
            source_sha256=str(d.get("source_sha256", "")),
            engine=dict(d.get("engine") or {}),
            doc_revision=int(d.get("doc_revision", 0) or 0),
            edit_log=list(d.get("edit_log") or []),
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
    # **人が足した発話は、割り込んだ位置で元の本文に差し込む。**
    # 区間そのものは割らない(割ると、直すときに元の本文を復元できない)。
    # 割り込み位置は add_utterance が edit_log に残している。
    # これをやらないと Word が「長い発言 → 相づち」の順になり、しかも
    # 同じ話者の相づちが 1 段落にまとまる(実機で判明・2026-08-18)。
    cuts: dict[float, list[tuple[int, float]]] = {}
    for rec in proj.edit_log:
        if rec.get("op") == "add_utterance" and rec.get("cut") is not None                 and rec.get("parent_orig") is not None:
            cuts.setdefault(round(float(rec["parent_orig"]), 3), []).append(
                (int(rec["cut"]), round(float(rec["orig_start"]), 3)))
        elif rec.get("op") == "remove_added_utterance":
            key = round(float(rec.get("orig_start", -1)), 3)
            for lst in cuts.values():
                lst[:] = [c for c in lst if c[1] != key]
    added_by_key = {round(float(s.orig_start), 3): s
                    for s in proj.segments if proj.is_added_utterance(s)}

    def pieces(seg: Segment) -> list[Segment]:
        """区間を、割り込みの位置で切った断片に分ける（出力のためだけ）。"""
        marks = sorted(
            (c for c in cuts.get(round(float(seg.orig_start), 3), [])
             if c[1] in added_by_key),
            key=lambda c: c[0])
        if not marks:
            return [seg]
        out: list[Segment] = []
        prev = 0
        for cut, key in marks:
            cut = max(0, min(len(seg.text), cut))
            head = seg.text[prev:cut]
            if head.strip():
                out.append(replace(seg, text=head))
            out.append(added_by_key[key])
            prev = cut
        tail = seg.text[prev:]
        if tail.strip():
            out.append(replace(seg, text=tail))
        return out

    ordered: list[Segment] = []
    for seg in proj.segments:
        if proj.is_added_utterance(seg):
            # 差し込み先で出すので、単独では出さない（先が無ければ出す）
            key = round(float(seg.orig_start), 3)
            if any(key in [k for _c, k in v] for v in cuts.values()):
                continue
        ordered.extend(pieces(seg))

    runs: list[tuple[float, Optional[str], list[str]]] = []
    for seg in ordered:
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


def build_verification(proj: Project, revision: int) -> list[tuple[str, str]]:
    """docx 末尾の「検証要約」の (項目, 値) を作る。

    書くのは確認の履歴であって、正しさの保証ではない。何をどの経路で処理し、
    人がどこまで確認したかを、第三者が検算できる形で残す(スキーマ v5)。
    """
    rows: list[tuple[str, str]] = [
        ("元音声", f"{Path(proj.audio_path).name} ({fmt_hms(proj.duration)})"),
    ]
    if proj.source_sha256:
        rows.append(("SHA-256", proj.source_sha256))
    if proj.engine:
        mode = ENGINE_LABELS.get(
            str(proj.engine.get("mode", "")), str(proj.engine.get("mode", "")))
        model = str(proj.engine.get("model", ""))
        at = str(proj.engine.get("at", ""))
        rows.append(("処理経路", " / ".join(x for x in (mode, model, at) if x)))
    rows.append(("版", f"revision {revision} (schema {SCHEMA_VERSION})"))

    # 数え方は「区間の数」で統一する。分母は必ず全区間数を書き、内訳は
    # 読点で区切る。区切りに「/」を使うと分数に見え、「聴いて確認 41 /
    # 適用のみ 40」が「41 分の 40」と読まれる(実出力で発生した)。
    # 検証要約は確認の履歴そのものなので、読み違えられる表示は信用を損なう。
    total = proj.total_count
    heard = proj.reviewed_count
    bulk = proj.unreviewed_count
    unassigned = total - proj.assigned_count
    rows.append((
        "話者の確認",
        f"全 {total} 区間 — 聴いて確定 {heard} 区間、"
        f"まとめて適用 {bulk} 区間、未確定 {unassigned} 区間",
    ))
    t_heard = sum(1 for s in proj.segments if s.time_edited and s.time_reviewed)
    t_bulk = sum(1 for s in proj.segments if s.time_edited and not s.time_reviewed)
    rows.append((
        "時刻の修正",
        f"全 {total} 区間中 {t_heard + t_bulk} 区間 — "
        f"聴いて確認 {t_heard} 区間、適用のみ {t_bulk} 区間",
    ))
    rows.append((
        "凡例",
        "「聴いて確定」「聴いて確認」＝その区間の音声を人が聴いて決めたもの。"
        "「まとめて適用」「適用のみ」＝機械の結果をまとめて当てただけで、"
        "その区間を個別には聴いていないもの。数はいずれも区間の数です。",
    ))
    rows.append(("注意", "本書の記載は確認の履歴であり、内容の正しさや"
                        "法的効力を保証するものではありません。"))
    return rows


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
    include_verification: bool = True,
    revision: Optional[int] = None,
) -> Path:
    """割当結果を Word ファイルに書き出す。

    include_verification: 末尾に検証要約(元音声・SHA-256・処理経路・版・
    確認状態)を付ける。revision はこの出力の版番号(省略時は記録済みの値)。
    """
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

    if include_verification:
        doc.add_paragraph()
        head = doc.add_paragraph()
        run = head.add_run("―― 検証要約 ――")
        run.bold = True
        for label, value in build_verification(
                proj, revision if revision is not None else proj.doc_revision):
            p = doc.add_paragraph()
            r = p.add_run(f"{label}: ")
            r.bold = True
            r.font.size = Pt(9)
            v = p.add_run(value)
            v.font.size = Pt(9)
            v.font.color.rgb = RGBColor(0x40, 0x40, 0x40)

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
