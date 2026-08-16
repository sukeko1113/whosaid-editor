"""「どの区間から聴くか」の順番を出す（脱落の見つけやすさ順）。

**これは検出器ではない。**再現率は約 4 割で、**印が付かない区間にも脱落はある。**
「検出」と名乗ると利用者は「印が無い＝大丈夫」と読む。それは嘘になるので、
この道具が出すのは順番と根拠の数字だけで、判定は出さない。

## 何を手がかりにするか

話者分離（`diarize.py`）が出す `SpeakerTurn` の**切り替わりの密度**。
ASR とは独立に声を捉えているので、ASR が自分の取りこぼしを知らなくても効く。

## 実測（2026-08-16・段階 1 の印 300 区間中 40 件で評価）

| 手がかり | 適合率 | 再現率 | 提示する区間 |
| --- | --- | --- | --- |
| 何も見ずに全部 | 13% | 100% | 100% |
| 声はあるが区間が無い秒数 | 18% | 82% | 62% |
| **前後 3 秒の切り替わり数（採用）** | **36%** | **38%** | **14%** |

「声はあるが区間が無い秒数」は 13% → 18% でほとんど効かなかったため採らない。

**帯（2 分）単位では、切り替わりの密度が脱落件数と順位まで一致した**
（7.0/8.5/15.0/18.5 回毎分 に対して脱落 0/3/11/19 件）。ただし帯は 4 つしかない。

評価に使った印は、人が話者ラベルを付ける作業の"ついで"に気づいたものなので、
**見落としを含む。**上の再現率は下振れしている可能性がある。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .diarize import SpeakerTurn
from .segments import Segment

# 実装のバージョン。上げると sidecar を作り直す。
LISTEN_ORDER_VER = 1

# 区間の前後にどれだけ広げて数えるか（秒）。
# 3 / 5 / 10 秒で測り、3 秒が最も分離した（適合率 36% / 24% / 20%）。
WINDOW_SECONDS = 3.0

# この数以上なら「聴く価値が高い」。実測で適合率が最大になった値。
# **しきい値以下＝安全ではない。**再現率は 4 割なので、6 割はここに現れない。
HIGH_SCORE = 5


@dataclass(frozen=True)
class ListenHint:
    """1 区間ぶんの「聴く価値」。

    score は前後 WINDOW_SECONDS 秒に重なる話者 turn の数。
    **順番を決めるためだけの数字で、脱落の有無を表さない。**
    """

    index: int
    start: float
    score: int

    @property
    def is_high(self) -> bool:
        return self.score >= HIGH_SCORE

    def to_dict(self) -> dict:
        return {"index": self.index, "start": round(self.start, 3),
                "score": self.score}

    @classmethod
    def from_dict(cls, d: dict) -> "ListenHint":
        return cls(index=int(d["index"]), start=float(d["start"]),
                   score=int(d["score"]))


def score_segments(
    segments: Sequence[Segment],
    turns: Sequence[SpeakerTurn],
    window: float = WINDOW_SECONDS,
) -> list[ListenHint]:
    """区間ごとに「聴く価値」を付ける。並べ替えはしない（呼び出し側の仕事）。

    turns が空なら全区間 0。**話者分離を通していない作業ファイルでは
    この機能は使えない**ことを、呼び出し側が利用者に伝えること。
    """
    if not segments:
        return []
    # turn は重なりうるので、開始で並べて二分探索……までは要らない規模
    # (1 音声あたり数百件)。素直に数える。
    ordered = sorted(turns, key=lambda t: t.start)
    out: list[ListenHint] = []
    for seg in segments:
        lo, hi = seg.start - window, seg.end + window
        n = sum(1 for t in ordered if t.end > lo and t.start < hi)
        out.append(ListenHint(index=seg.index, start=seg.start, score=n))
    return out


def listen_first(hints: Sequence[ListenHint],
                 skip: Optional[Iterable[int]] = None) -> list[ListenHint]:
    """聴く順に並べる。同点なら時間順（前から聴くほうが自然なため）。

    skip: 既に人が聴いて確定した区間の index。もう一度勧めない。
    """
    done = set(skip or ())
    rest = [h for h in hints if h.index not in done]
    return sorted(rest, key=lambda h: (-h.score, h.start))


def coverage(hints: Sequence[ListenHint]) -> tuple[int, int]:
    """(しきい値以上の件数, 全体の件数)。画面に「上位 N 件」と出すため。"""
    return sum(1 for h in hints if h.is_high), len(hints)


# ----------------------------------------------------------------------
# sidecar（本体 JSON には入れない。派生データなので作り直せる）
# ----------------------------------------------------------------------
def hints_path(work_dir: Path | str, fingerprint: str) -> Optional[Path]:
    """置き場。指紋が無ければ保存しない（転写のキャッシュと同じ流儀）。"""
    if not fingerprint:
        return None
    return Path(work_dir) / "inspect" / (
        f"listen.{fingerprint}.v{LISTEN_ORDER_VER}.json")


def load_hints(path: Optional[Path]) -> Optional[list[ListenHint]]:
    """読む。無い・壊れている・別バージョンなら None（作り直す）。"""
    if path is None or not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if int(d.get("version", 0)) != LISTEN_ORDER_VER:
            return None
        return [ListenHint.from_dict(x) for x in d.get("hints", [])]
    except Exception:
        return None


def save_hints(path: Optional[Path], hints: Sequence[ListenHint],
               window: float = WINDOW_SECONDS) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({
        "version": LISTEN_ORDER_VER,
        "window_seconds": window,
        "high_score": HIGH_SCORE,
        # **この 2 行は消さないこと。**数字だけが残ると、あとで読んだ人が
        # 「印が無い＝脱落が無い」と解釈する。
        "note": "順番を決めるための数字であり、脱落の有無を表さない",
        "measured": "適合率 36% / 再現率 38%（2026-08-16・300 区間中 40 件で評価）",
        "hints": [h.to_dict() for h in hints],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
