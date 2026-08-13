"""inspection.py(提案づくりと sidecar)の検査。モデルも音声も要らない。

    .venv\\Scripts\\python.exe tests\\test_inspection.py

実測は偽の単語列を注入して作る。ここで守りたいのは速さや精度ではなく、
「機械が勝手に確定しない」という規約のほう。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.align import ALIGN_VER, Word                     # noqa: E402
from src.inspection import (                              # noqa: E402
    MIN_COVERAGE,
    PROPOSAL_TIME,
    Proposal,
    clip_to_neighbours,
    drop_conflicts,
    inspect_times,
    load_proposals,
    merge_history,
    proposals_path,
    save_proposals,
    target_segment,
)
from src.segments import MIN_SEGMENT_SECONDS, Project, Segment    # noqa: E402


SPOKEN = ["おはようございます", "きょうはさむいですね", "そうですねまったく"]


def evenly(text: str, start: float, per_char: float = 0.5) -> list[Word]:
    return [Word(text=ch, start=start + i * per_char,
                 end=start + (i + 1) * per_char)
            for i, ch in enumerate(text)]


def measured_words(shift: float = 0.0) -> list[Word]:
    """実測の単語列。3 つの発言が 0 / 6 / 12 秒から始まる。"""
    ws: list[Word] = []
    for text, at in zip(SPOKEN, (0.0, 6.0, 12.0)):
        ws += evenly(text, at + shift)
    return ws


def project(drift: float = 0.0, **kw) -> Project:
    """本文側。drift を足すと「時刻がずれている」状態になる。"""
    proj = Project(audio_path="a.m4a", duration=60.0, **kw)
    proj.segments = [
        Segment(index=i, start=at + drift, end=at + drift + 4.5,
                text=text, cluster="0:A")
        for i, (text, at) in enumerate(zip(SPOKEN, (0.0, 6.0, 12.0)))
    ]
    return proj


# ======================================================================
# 提案づくり
# ======================================================================

def test_no_proposal_when_times_are_right():
    """合っている時刻には口を出さない。"""
    got = inspect_times(project(), measured_words())
    assert got.proposals == []
    assert got.close_enough == 3


def test_proposes_when_drifted():
    got = inspect_times(project(drift=6.8), measured_words())
    assert len(got.proposals) == 3
    p = got.proposals[0]
    assert p.type == PROPOSAL_TIME
    assert abs(p.payload["start"] - 0.0) < 0.1      # 実測の位置に戻す
    assert p.confidence == 1.0
    assert "被覆 100%" in p.evidence and "-6.8 秒" in p.evidence
    assert p.status == "pending"


def test_target_is_orig_start_not_index():
    """index は分割で振り直る。指し先は動かない値にする。"""
    proj = project(drift=6.8)
    got = inspect_times(proj, measured_words())
    assert got.proposals[1].target_orig_start == proj.segments[1].orig_start
    # 前に区間が増えて index がずれても、同じ区間を指せる
    proj.segments.insert(0, Segment(index=0, start=0.0, end=0.5, text="あ",
                                    cluster="0:A", orig_start=-1.0, orig_end=-0.5))
    proj.renumber()
    found = target_segment(proj, got.proposals[1])
    assert found is not None and found.text == SPOKEN[1]


def test_reviewed_times_are_never_touched():
    """人が耳で確定した時刻は、機械が上書き提案しない(§3.4)。"""
    proj = project(drift=6.8)
    for seg in proj.segments:
        seg.time_edited = True
        seg.time_reviewed = True
    got = inspect_times(proj, measured_words())
    assert got.proposals == []
    assert got.reviewed == 3
    # 黙って捨てるのではなく、食い違いは記録に残す
    assert len(got.notes) == 3 and "確認済み" in got.notes[0]


def test_unreviewed_edits_are_still_proposed():
    """まとめて適用しただけの時刻(✎△)は、新しい実測で上書きしてよい。"""
    proj = project(drift=6.8)
    for seg in proj.segments:
        seg.time_edited = True
        seg.time_reviewed = False
    got = inspect_times(proj, measured_words())
    assert len(got.proposals) == 3 and got.reviewed == 0


def test_speaker_reviewed_is_not_a_reason_to_skip():
    """話者を確定した区間でも、時刻の提案は出す(時刻と話者は別物)。"""
    proj = project(drift=6.8)
    for seg in proj.segments:
        seg.speaker_id, seg.reviewed = "sp01", True
    assert len(inspect_times(proj, measured_words()).proposals) == 3


def test_low_coverage_is_not_proposed():
    """偶然の数文字一致から時刻を作らない。"""
    proj = project(drift=6.8)
    proj.segments[0].text = "まったくちがうながいはつげんがここにあります"
    got = inspect_times(proj, measured_words())
    assert len(got.proposals) == 2
    assert got.low_coverage + got.short_match + got.unmatched == 1


def test_full_match_on_short_text_is_not_trusted():
    """本文が短いと被覆率 100% でも根拠にならない。

    実会議の聴き取りで完全に外れた 2 件は、どちらも一致 3 文字だった。
    3 文字の本文が 3 文字一致すれば被覆率は 100% になる。被覆率では防げない。
    """
    proj = Project(audio_path="a.m4a", duration=60.0)
    proj.segments = [Segment(index=0, start=6.8, end=9.8,
                             text="はいそうです", cluster="0:A")]
    got = inspect_times(proj, evenly("はいそうです", 0.0))
    assert got.proposals == []
    assert got.short_match == 1


def test_end_only_drift_is_not_proposed():
    """終了だけのずれでは提案しない。

    記録に出るのも頭出しに効くのも開始時刻。終了は照合の中で最も精度が
    弱い部分なので、終了のずれだけで人を呼び出さない。
    """
    proj = project()
    for seg in proj.segments:
        seg.end += 3.0
    got = inspect_times(proj, measured_words())
    assert got.proposals == []
    assert got.close_enough == 3


def test_unheard_tail_keeps_current_duration():
    """末尾が聞き取れていない区間は、終了を動かさずいまの長さを保つ。

    終了は「最後に一致した文字」から取るので、末尾が欠けると手前に出て
    発言の途中で切れる(聴き取りで確認: 欠け 3 字以上は終了が外れる)。
    """
    heard = "あきのてんきはとてもすごしやすい"       # 16 字は実測にある
    proj = Project(audio_path="a.m4a", duration=60.0)
    proj.segments = [Segment(index=0, start=6.8, end=16.8,
                             text=heard + "ですよね", cluster="0:A")]
    got = inspect_times(proj, evenly(heard, 0.0))
    assert len(got.proposals) == 1
    p = got.proposals[0]
    assert abs(p.payload["start"] - 0.0) < 0.1          # 開始は実測に直す
    duration = p.payload["end"] - p.payload["start"]
    assert abs(duration - 10.0) < 0.1                   # 長さは変えない
    assert "終わりは測れず" in p.evidence


def _prop(pid: str, start: float, end: float) -> Proposal:
    return Proposal(id=pid, type=PROPOSAL_TIME, target_orig_start=0.0,
                    payload={"start": start, "end": end},
                    evidence="", confidence=1.0)


def test_conflicting_proposals_are_all_dropped():
    """同じ音声位置を取り合う提案は、どれも出さない。

    挨拶の応酬では複数の区間が同じ実測位置に当たる(実測: 3 区間が全く同じ
    時刻に当たった)。正しいのは高々 1 つで、どれかは機械には決められない。
    """
    same = [_prop("a", 10.0, 12.0), _prop("b", 10.0, 12.0), _prop("c", 10.1, 12.0)]
    kept, dropped = drop_conflicts(same)
    assert kept == [] and dropped == 3


def test_adjacent_proposals_are_not_conflicts():
    """接しているだけ(隣り合う発言)の提案は正当。引っ込めない。"""
    row = [_prop("a", 10.0, 12.0), _prop("b", 12.0, 15.0), _prop("c", 15.1, 16.0)]
    kept, dropped = drop_conflicts(row)
    assert len(kept) == 3 and dropped == 0
    # 余白ぶんの僅かな食い込み(半分未満)も正当なまま
    graze = [_prop("a", 10.0, 12.1), _prop("b", 12.0, 15.0)]
    kept, dropped = drop_conflicts(graze)
    assert len(kept) == 2 and dropped == 0


def test_conflict_drop_is_counted():
    """取り合いは実際に起き、引っ込めた数は内訳に出る。

    一致した範囲が単調掃引の戻り幅(1 秒)より短いと、次の区間の走査が
    同じ場所まで戻れてしまい、同じ音声を 2 区間が取り合う。早口の挨拶
    (1 秒未満)が該当する。実測でも 3 区間が全く同じ時刻に当たった。
    """
    heard = "よろしくおねがいいたします"
    proj = Project(audio_path="a.m4a", duration=60.0)
    proj.segments = [
        Segment(index=0, start=27.0, end=28.0, text=heard, cluster="0:A"),
        Segment(index=1, start=45.0, end=46.0, text=heard, cluster="0:B"),
    ]
    # 実測には 30.0 秒に 1 回だけ、0.8 秒で早口に言った挨拶がある
    got = inspect_times(proj, evenly(heard, 30.0, per_char=0.06))
    assert got.proposals == []
    assert got.conflicted == 2


def test_unheard_head_is_compensated():
    """頭の欠けは、その区間の実測発話速度で秒に換算して手前に戻す。

    開始は「最初に一致した文字」から取るので、頭が聞き取られていないと
    その分だけ遅く出る(聴き取り 2 回目: 換算 5.3 秒 ≒ 人の判定「約 5 秒」)。
    """
    heard = "きょうのてんきはとてもすごしやすいですね"     # 20 字が実測にある
    proj = Project(audio_path="a.m4a", duration=120.0)
    # 頭の 6 字(えーとあのー相当)は聞き取られていない
    proj.segments = [Segment(index=0, start=40.0, end=50.0,
                             text="えーとあのー" + heard, cluster="0:A")]
    got = inspect_times(proj, evenly(heard, 30.0, per_char=0.2))    # 5 字/秒
    assert len(got.proposals) == 1
    p = got.proposals[0]
    # 30.0 − 換算 1.2 秒 − 余白 0.3 秒 = 28.5
    assert abs(p.payload["start"] - 28.5) < 0.05
    assert "頭の欠け 6 字を 1.2 秒手前に補正" in p.evidence


def test_hopeless_head_is_not_proposed():
    """頭の欠けが大きすぎる区間は、開始を測れていないので提案しない。

    補正が実測より大きくなったら、それはもう推定であって実測ではない。
    """
    heard = "きょうのてんきはとてもすごしやすいですねこのごろは"
    proj = Project(audio_path="a.m4a", duration=120.0)
    # 頭の 13 字が聞き取られていない(換算 6.5 秒 > 上限 6 秒)
    proj.segments = [Segment(index=0, start=40.0, end=55.0,
                             text="あ" * 13 + heard, cluster="0:A")]
    got = inspect_times(proj, evenly(heard, 30.0, per_char=0.5))    # 2 字/秒
    assert got.proposals == []
    assert got.head_lost == 1


def test_scattered_match_is_not_proposed():
    """一致が広い範囲に散らばりすぎた区間は提案しない。

    本文が発話と食い違っていると、あちこちの数文字がまばらに当たって
    「一致」に見える(聴き取り 2 回目: 内容違いの区間は 0.8 字/秒だった)。
    """
    text = "はいそれではじかいのかいごう"
    proj = Project(audio_path="a.m4a", duration=120.0)
    proj.segments = [Segment(index=0, start=20.0, end=36.0,
                             text=text, cluster="0:A")]
    # 本文の文字が 2 秒おきにまばらに現れる(密度 0.5 字/秒)
    ws = [Word(text=ch, start=30.0 + i * 2.0, end=30.2 + i * 2.0)
          for i, ch in enumerate(text)]
    got = inspect_times(proj, ws)
    assert got.proposals == []
    assert got.scattered == 1


def test_heard_tail_uses_measured_end():
    """末尾まで乗っていれば、終了も実測で直す(従来どおり)。"""
    got = inspect_times(project(drift=6.8), measured_words())
    p = got.proposals[0]
    assert "終わりは測れず" not in p.evidence
    assert abs(p.payload["end"] - 4.55) < 0.2       # 実測の終了(4.5 + 余白)


def test_unmatched_segments_are_counted():
    proj = project(drift=6.8)
    proj.segments[0].text = "ぜんぜんべつのはなし"
    got = inspect_times(proj, measured_words())
    assert got.unmatched == 1 and len(got.proposals) == 2


def test_time_offset_is_taken_into_account():
    """ずれ補正が効いている区間は、補正後の位置を基準に見る(§3.3)。"""
    # 保存値は 6.8 秒ずれているが、補正 +6.8 秒で聴くと合っている
    proj = project(drift=-6.8, time_offset=6.8)
    got = inspect_times(proj, measured_words())
    assert got.proposals == [] and got.close_enough == 3


def test_threshold_can_be_tightened():
    """0.6 秒のずれは既定(0.75 秒)では誤差の範囲。厳しくすれば拾う。"""
    proj = project(drift=0.6)
    assert inspect_times(proj, measured_words()).proposals == []
    tight = inspect_times(proj, measured_words(), time_delta=0.5)
    assert len(tight.proposals) == 3


def test_result_counts_every_segment():
    got = inspect_times(project(drift=6.8), measured_words())
    assert got.checked == 3


def test_no_words_means_no_proposals():
    assert inspect_times(project(drift=6.8), []).proposals == []


# ======================================================================
# 適用のための計算
# ======================================================================

def test_clip_to_neighbours():
    assert clip_to_neighbours(10.0, 20.0, None, None) == (10.0, 20.0)
    assert clip_to_neighbours(10.0, 20.0, 12.0, None) == (12.0, 20.0)
    assert clip_to_neighbours(10.0, 20.0, None, 18.0) == (10.0, 18.0)
    assert clip_to_neighbours(10.0, 20.0, 12.0, 18.0) == (12.0, 18.0)


def test_clip_refuses_to_make_a_crushed_segment():
    """切り詰めて潰れるくらいなら、当てない。"""
    assert clip_to_neighbours(10.0, 20.0, 19.99, None) is None
    ok = clip_to_neighbours(10.0, 20.0, 20.0 - MIN_SEGMENT_SECONDS, None)
    assert ok is not None


def test_target_segment_prefers_the_front_of_a_split_family():
    proj = project()
    proj.split_segment(1, proj.segments[1].start + 2.0, 3)
    p = Proposal(id="x", type=PROPOSAL_TIME,
                 target_orig_start=float(proj.segments[1].orig_start),
                 payload={}, evidence="", confidence=1.0)
    found = target_segment(proj, p)
    assert found is not None and found.index == 1       # 分割した前側


def test_target_segment_missing_returns_none():
    p = Proposal(id="x", type=PROPOSAL_TIME, target_orig_start=999.0,
                 payload={}, evidence="", confidence=1.0)
    assert target_segment(project(), p) is None


# ======================================================================
# sidecar
# ======================================================================

def test_proposals_path_needs_a_fingerprint():
    assert proposals_path("w", "") is None
    p = proposals_path("w", "abc123")
    assert p is not None and p.name == f"proposals.abc123.a{ALIGN_VER}.json"
    assert p.parent.name == "inspect"


def test_save_and_load_round_trip():
    got = inspect_times(project(drift=6.8), measured_words())
    with tempfile.TemporaryDirectory() as d:
        path = proposals_path(d, "ff00")
        save_proposals(path, got.proposals)
        again = load_proposals(path)
    assert [p.id for p in again] == [p.id for p in got.proposals]
    assert again[0].payload == got.proposals[0].payload
    assert again[0].evidence == got.proposals[0].evidence


def test_load_tolerates_a_broken_file():
    with tempfile.TemporaryDirectory() as d:
        path = proposals_path(d, "ff00")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("これは JSON ではない", encoding="utf-8")
        assert load_proposals(path) == []
    assert load_proposals(None) == []


def test_same_evidence_gives_the_same_id():
    """再点検しても、同じ根拠なら同じ id。却下を覚えておける。"""
    a = inspect_times(project(drift=6.8), measured_words())
    b = inspect_times(project(drift=6.8), measured_words())
    assert [p.id for p in a.proposals] == [p.id for p in b.proposals]


def test_rejected_proposals_are_not_shown_again():
    fresh = inspect_times(project(drift=6.8), measured_words()).proposals
    history = [Proposal(**{**p.to_dict(), "status": "rejected"})
               for p in fresh[:1]]
    left = merge_history(fresh, history)
    assert [p.id for p in left] == [p.id for p in fresh[1:]]


def test_accepted_proposals_are_not_shown_again():
    fresh = inspect_times(project(drift=6.8), measured_words()).proposals
    history = [Proposal(**{**p.to_dict(), "status": "accepted"}) for p in fresh]
    assert merge_history(fresh, history) == []


def test_undecided_history_does_not_hide_anything():
    fresh = inspect_times(project(drift=6.8), measured_words()).proposals
    assert len(merge_history(fresh, list(fresh))) == len(fresh)      # pending


def test_new_evidence_comes_back_even_after_rejection():
    """実測が変われば別の提案。改めて判断してもらう。"""
    fresh = inspect_times(project(drift=6.8), measured_words()).proposals
    history = [Proposal(**{**p.to_dict(), "status": "rejected"}) for p in fresh]
    moved = inspect_times(project(drift=6.8), measured_words(shift=2.0)).proposals
    assert merge_history(moved, history) == moved


# ======================================================================
# pytest が無い環境向けの簡易ランナー
# ======================================================================

if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except Exception as e:
                failures += 1
                import traceback
                print(f"  FAIL {name}: {e}")
                traceback.print_exc()
    print(f"\n{'FAILED' if failures else 'ALL PASSED'} ({failures} failures)")
    sys.exit(1 if failures else 0)
