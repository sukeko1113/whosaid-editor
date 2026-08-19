"""「区間の中に、別の人の声がある箇所」を並べる（設計書 §10.3）。

**これは検出器ではない。**聴く場所を絞る道具であって、判定は出さない。
`listen_order.py` と同じ線で、印が無い＝脱落が無い、ではない。

## 定義を実測で入れ替えた経緯

もとの設計は「声はあるのに本文が無い箇所」＝ turn と区間の差集合だった。
**作る前に測ったら、それでは 1 件も拾えないことが分かった。**

    turn に覆われている              34/34   100%
    turn はあるが本文の区間が無い      0/34     0%   ← すきまはゼロ

**脱落は他人の発言の最中に埋もれている。**25:03 の区間は 14.5 秒あり、
相づち 5 件はその中にある。区間は存在するので「すきま」にならない。

そこで **「区間の中に、その区間の主たる話者と違う声がある箇所」** に変えた。

## 実測（2026-08-19・逐語正解 4 帯・脱落 34 件）

| | もとの設計 | 入れ替えた定義 |
| --- | --- | --- |
| 再現 | **0/34** | **31/34（91% ±10 点）** |
| 適合 | — | **35/51（69% ±13 点）** |
| 作業量 | — | 4 帯 8 分で候補 **51 個** |

比較: 既存の「聴きどころ」は高スコア 96 区間で **9/34** しか指せない。

**幅は大きい。**音声 1 本・34 件ぶんなので、91% / 69% を確定値として扱わない。
外れた 3 件のうち 1 件は重なりで**原理的に取れない**。空振り 16 個のうち
**12 個は理由を説明できていない**。

**閾値は測定に使った値をそのまま既定にしてある。**別の値を置くと、上の数字が
その設定に対する保証でなくなる。変えるときは測り直すこと。

数字を出し直す道具: `tools/report_candidates.py`
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .diarize import SpeakerTurn
from .segments import Segment, key_text, segment_key

# 実装のバージョン。上げると sidecar を作り直す。
CANDIDATES_VER = 2   # 鍵を (orig_start, start) に変えた

# 候補として数える最小の重なり（秒）。短すぎる被りは息継ぎや漏れ込みで出る。
# **この値で 31/34・35/51 を測った。**変えるなら測り直すこと。
MIN_OVERLAP_SECONDS = 0.2


@dataclass(frozen=True)
class VoiceCandidate:
    """1 件ぶんの「この区間の中に、別の人の声がある」。

    **区間を指す鍵は `parent_orig`。**`index` は分割・結合・再実行で振り直る
    ので鍵にしてはいけない（自動点検設計書 §6.1 と同じ理由）。
    """

    # **親区間を指す鍵は (orig_start, start) の組。**orig_start だけでは
    # 分割した 2 つを区別できない（segments.segment_key を見よ）。
    parent_orig: float
    parent_start: float
    at: float                   # その声が聞こえ始める時刻
    end: float
    speaker: int                # turn の話者番号（表示は「声B」等）
    overlap: float              # 区間と重なっている秒数
    index: int = -1             # 表示のためだけ。鍵には使わない

    @property
    def parent_key(self) -> tuple[float, float]:
        return (self.parent_orig, self.parent_start)

    @property
    def key(self) -> str:
        """却下の記録に使う識別子。時刻は 0.01 秒に丸める。"""
        return f"{key_text(self.parent_key)}@{self.at:.2f}"

    def to_dict(self) -> dict:
        return {"parent_orig": round(self.parent_orig, 3),
                "parent_start": round(self.parent_start, 3),
                "at": round(self.at, 3), "end": round(self.end, 3),
                "speaker": self.speaker, "overlap": round(self.overlap, 3),
                "index": self.index}

    @classmethod
    def from_dict(cls, d: dict) -> "VoiceCandidate":
        return cls(parent_orig=float(d["parent_orig"]),
                   parent_start=float(d.get("parent_start", d["parent_orig"])),
                   at=float(d["at"]),
                   end=float(d["end"]), speaker=int(d["speaker"]),
                   overlap=float(d.get("overlap", 0.0)),
                   index=int(d.get("index", -1)))


def main_speaker(seg: Segment, turns: Sequence[SpeakerTurn]) -> Optional[int]:
    """区間の「主たる話者」。最も長く重なる turn の話者。

    話者割当（`speaker_id`）ではなく turn を使う。割当がまだでも効かせたい
    のと、比べたいのは**同じ声かどうか**であって名前ではないため。
    """
    best, best_ov = None, 0.0
    for t in turns:
        ov = min(seg.end, t.end) - max(seg.start, t.start)
        if ov > best_ov:
            best, best_ov = t.speaker, ov
    return best


def find_candidates(
    segments: Sequence[Segment],
    turns: Sequence[SpeakerTurn],
    min_overlap: float = MIN_OVERLAP_SECONDS,
) -> list[VoiceCandidate]:
    """区間の中に、その区間の主たる話者と違う声がある箇所を並べる。

    turns が空なら空。**話者分離を通していない作業ファイルでは使えない**
    ことを、呼び出し側が利用者に伝えること（`listen_order` と同じ）。
    """
    if not segments or not turns:
        return []
    out: list[VoiceCandidate] = []
    for seg in segments:
        main = main_speaker(seg, turns)
        if main is None:
            continue
        for t in turns:
            ov = min(seg.end, t.end) - max(seg.start, t.start)
            if ov < min_overlap or t.speaker == main:
                continue
            k = segment_key(seg)
            out.append(VoiceCandidate(
                parent_orig=k[0], parent_start=k[1],
                at=max(seg.start, t.start), end=min(seg.end, t.end),
                speaker=t.speaker, overlap=ov, index=seg.index))
    out.sort(key=lambda c: c.at)
    return out


# 「もう足した」と見なす時間の余裕（秒）。
# 足した発話の時刻は小窓の文字数按分から決まるので、turn の開始とは
# ずれる。実データで **0.69〜1.07 秒** ずれていた（2026-08-19 実測）。
DONE_SLACK_SECONDS = 1.0


def done_keys(candidates: Sequence[VoiceCandidate],
              added: Sequence[Segment],
              slack: float = DONE_SLACK_SECONDS) -> set:
    """人がもう発話を足した位置の候補。

    **隠すのではなく印を付けるために使う。**隠すと「候補が無い＝やること
    が無い」と読まれる。この道具の性格（印が無い＝安全ではない）と合わない。
    """
    out = set()
    for c in candidates:
        for a in added:
            if a.start <= c.end + slack and a.end >= c.at - slack:
                out.add(c.key)
                break
    return out


def for_segment(candidates: Sequence[VoiceCandidate],
                seg: Segment) -> list[VoiceCandidate]:
    """その区間に属する候補。

    **鍵は (orig_start, start) の組。**orig_start だけで突き合わせると、
    分割した 2 つの区間が互いの候補まで出す（実データで発生・2026-08-19）。
    """
    key = segment_key(seg)
    return [c for c in candidates
            if abs(c.parent_orig - key[0]) < 0.005
            and abs(c.parent_start - key[1]) < 0.005]


def drop_dismissed(candidates: Sequence[VoiceCandidate],
                   dismissed: Sequence[str]) -> list[VoiceCandidate]:
    """人が × を付けたものを外す。**判断は残す**ので次も出てこない。"""
    gone = set(dismissed)
    return [c for c in candidates if c.key not in gone]


def coverage(candidates: Sequence[VoiceCandidate],
             segments: Sequence[Segment]) -> tuple[int, int]:
    """(候補のある区間の数, 全区間) — 何割を聴くことになるかの目安。"""
    keys = {c.parent_key for c in candidates}
    n = sum(1 for s in segments if segment_key(s) in keys)
    return n, len(segments)


# ----------------------------------------------------------------------
# sidecar（本体 JSON には入れない。派生データなので作り直せる）
# ----------------------------------------------------------------------
def dismissed_path(work_dir: Path | str, fingerprint: str) -> Optional[Path]:
    """却下（×）の置き場。指紋が無ければ保存しない。"""
    if not fingerprint:
        return None
    return Path(work_dir) / "inspect" / (
        f"candidates.{fingerprint}.v{CANDIDATES_VER}.json")


def load_dismissed(path: Optional[Path]) -> list[str]:
    """読む。無い・壊れている・別バージョンなら空（また出てくるだけ）。"""
    if path is None or not path.exists():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if int(d.get("version", 0)) != CANDIDATES_VER:
            return []
        return [str(x) for x in d.get("dismissed", [])]
    except Exception:
        return []


def save_dismissed(path: Optional[Path], keys: Sequence[str],
                   min_overlap: float = MIN_OVERLAP_SECONDS) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "version": CANDIDATES_VER,
        "min_overlap_seconds": min_overlap,
        # **この 2 行は消さないこと。**数字だけが残ると、あとで読んだ人が
        # 「候補が無い＝脱落が無い」と解釈する。
        "note": "聴く場所を絞るための一覧であり、脱落の有無を表さない",
        "measured": "再現 31/34(±10 点) / 適合 35/51(±13 点)"
                    "（2026-08-19・逐語正解 4 帯 34 件で評価）",
        "dismissed": list(dict.fromkeys(keys)),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
