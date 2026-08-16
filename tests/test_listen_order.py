"""聴く順番（listen_order.py）の検査。

**この機能は検出器ではない。**再現率は 4 割で、印が無い区間にも脱落はある。
その性格が壊れていないこと（判定を出さない・注意書きが消えない）も見る。

実データでの再現（段階 1 の印 300 区間）は tests/test_listen_order_real.py
ではなく、このファイルの最後で、データがある場合のみ動かす。
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.diarize import SpeakerTurn  # noqa: E402
from src.listen_order import (  # noqa: E402
    HIGH_SCORE,
    LISTEN_ORDER_VER,
    WINDOW_SECONDS,
    ListenHint,
    coverage,
    hints_path,
    match,
    listen_first,
    load_hints,
    save_hints,
    score_segments,
)
from src.segments import Segment  # noqa: E402


def _seg(index: int, start: float, end: float) -> Segment:
    return Segment(index=index, start=start, end=end, text="本文",
                   cluster="g:A", chunk=0)


def _turns(*spans) -> list[SpeakerTurn]:
    return [SpeakerTurn(start=a, end=b, speaker=s) for a, b, s in spans]


# ======================================================================
# 点数の付け方
# ======================================================================

def test_score_counts_turns_around_the_segment():
    """前後 WINDOW_SECONDS 秒に重なる turn を数える。"""
    segs = [_seg(0, 10.0, 12.0)]
    t = _turns((10.5, 11.0, 0),          # 中
               (8.0, 9.0, 1),            # 前 3 秒以内
               (14.0, 14.5, 2),          # 後 3 秒以内
               (30.0, 31.0, 3))          # 遠い
    (h,) = score_segments(segs, t)
    assert h.score == 3
    assert h.orig_start == 10.0


def test_key_is_orig_start_not_index():
    """区間を指す鍵は orig_start。**index は分割・再実行で振り直る**ので
    鍵にしない(自動点検設計書 §6.1 と同じ理由)。"""
    seg = _seg(7, 50.0, 52.0)
    seg.orig_start = 30.0                # 分割・時刻編集で本来の位置が別にある
    (h,) = score_segments([seg], _turns((50.0, 51.0, 0)))
    assert h.orig_start == 30.0 and h.start == 50.0


def test_match_uses_orig_start():
    """番号が振り直されても、同じ区間に当たる。"""
    a = _seg(0, 10.0, 11.0)
    hints = score_segments([a], _turns((10.0, 11.0, 0)))
    moved = _seg(5, 10.0, 11.0)          # 番号だけ変わった
    got = match(hints, [moved])
    assert set(got) == {5} and got[5].score == 1


def test_match_skips_segments_without_a_hint():
    """当たらない区間は入らない。**呼び出し側はそれを安全とみなさないこと。**"""
    hints = score_segments([_seg(0, 10.0, 11.0)], _turns((10.0, 11.0, 0)))
    other = _seg(1, 900.0, 901.0)
    assert match(hints, [other]) == {}


def test_score_is_zero_without_diarization():
    """turn が無ければ全区間 0。

    話者分離を通していない作業ファイルでは使えない。**0 を「安全」と
    読ませないため、呼び出し側が使えない旨を出すこと。**
    """
    hints = score_segments([_seg(0, 1.0, 2.0), _seg(1, 5.0, 6.0)], [])
    assert [h.score for h in hints] == [0, 0]


def test_window_is_the_measured_one():
    """窓は実測で選んだ 3 秒。3/5/10 秒で測り、3 秒が最も分離した。"""
    assert WINDOW_SECONDS == 3.0


def test_overlapping_turns_are_counted_separately():
    """同時発話は別々に数える。pyannote は重なる turn を出す。"""
    segs = [_seg(0, 10.0, 11.0)]
    (h,) = score_segments(segs, _turns((10.0, 11.0, 0), (10.2, 10.8, 1)))
    assert h.score == 2


def test_empty_input_is_empty_output():
    assert score_segments([], _turns((1.0, 2.0, 0))) == []


# ======================================================================
# 並べ方
# ======================================================================

def test_listen_first_sorts_by_score_then_time():
    hints = [ListenHint(100.0, 100.0, 2), ListenHint(10.0, 10.0, 7),
             ListenHint(50.0, 50.0, 7), ListenHint(20.0, 20.0, 5)]
    got = [h.start for h in listen_first(hints)]
    assert got == [10.0, 50.0, 20.0, 100.0]   # 7点(時間順) → 5点 → 2点


def test_listen_first_skips_confirmed_segments():
    """人が聴いて確定した区間は、もう勧めない。skip も orig_start で指す。"""
    hints = [ListenHint(10.0, 10.0, 9), ListenHint(20.0, 20.0, 8)]
    assert [h.start for h in listen_first(hints, skip={10.0})] == [20.0]


def test_coverage_reports_how_many_are_high():
    hints = [ListenHint(1.0, 1.0, HIGH_SCORE),
             ListenHint(2.0, 2.0, HIGH_SCORE - 1),
             ListenHint(3.0, 3.0, HIGH_SCORE + 3)]
    assert coverage(hints) == (2, 3)


def test_high_score_is_the_measured_threshold():
    """しきい値は実測で適合率が最大になった値。勝手に動かさない。"""
    assert HIGH_SCORE == 5


# ======================================================================
# sidecar（本体 JSON を汚さない）
# ======================================================================

def test_hints_path_needs_a_fingerprint():
    """指紋が無ければ保存しない。別の音声の結果を使い回さないため。"""
    assert hints_path("work", "") is None
    p = hints_path("work", "abc123")
    assert p is not None and f"v{LISTEN_ORDER_VER}" in p.name
    assert "abc123" in p.name


def test_save_and_load_round_trip():
    with tempfile.TemporaryDirectory() as d:
        p = hints_path(d, "fp")
        save_hints(p, [ListenHint(12.5, 12.5, 7, 3)])
        got = load_hints(p)
        assert got == [ListenHint(12.5, 12.5, 7, 3)]


def test_saved_file_keeps_the_warning():
    """**注意書きが消えないこと。**数字だけが残ると、あとで読んだ人が
    「印が無い＝脱落が無い」と解釈する。"""
    with tempfile.TemporaryDirectory() as d:
        p = hints_path(d, "fp")
        save_hints(p, [ListenHint(1.0, 1.0, 9)])
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "脱落の有無を表さない" in data["note"]
        assert "再現率" in data["measured"]


def test_old_version_is_discarded():
    """版が違えば読まない。作り直せる派生データなので捨ててよい。"""
    with tempfile.TemporaryDirectory() as d:
        p = hints_path(d, "fp")
        save_hints(p, [ListenHint(1.0, 1.0, 9)])
        data = json.loads(p.read_text(encoding="utf-8"))
        data["version"] = LISTEN_ORDER_VER + 1
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        assert load_hints(p) is None


def test_missing_or_broken_file_is_none():
    with tempfile.TemporaryDirectory() as d:
        assert load_hints(Path(d) / "無い.json") is None
        bad = Path(d) / "壊れ.json"
        bad.write_text("{ではない", encoding="utf-8")
        assert load_hints(bad) is None
    assert load_hints(None) is None


def test_save_with_no_path_does_nothing():
    save_hints(None, [ListenHint(1.0, 1.0, 5)])      # 例外を出さない


# ======================================================================
# 実データでの再現（データがある場合のみ）
# ======================================================================

_TA = Path(r"C:\dev\01\test-audio")


def test_measured_numbers_still_hold():
    """段階 1 の印 300 区間に対して、実測した数字が再現するか。

    **手がかりを変えたときに、ここが落ちれば気づける。**
    データが無い環境では飛ばす（CI にはこの音声を置かない）。
    """
    proj_p = _TA / "01+02edited.speakers.json"
    turn_p = _TA / "01+02edited.diarize.auto.4036s.json"
    lab_p = _TA / "truth" / "labels_blind.r1.json"
    if not (proj_p.exists() and turn_p.exists() and lab_p.exists()):
        print("       (実データが無いので飛ばします)")
        return

    from src.segments import Project
    proj = Project.load(proj_p)
    turns = [SpeakerTurn(**t) for t in
             json.loads(turn_p.read_text(encoding="utf-8"))["turns"]]
    lab = json.loads(lab_p.read_text(encoding="utf-8"))
    sampled = {int(r["index"]) for r in lab["labels"]}
    marked = {int(r["index"]) for r in lab["untranscribed_voice"]}

    by_index = match(score_segments(proj.segments, turns), proj.segments)
    hints = [h for i, h in by_index.items() if i in sampled]
    high = [h for h in hints if h.is_high]
    hit = sum(1 for i, h in by_index.items()
              if i in sampled and h.is_high and i in marked)
    prec = hit / len(high)
    rec = hit / len(marked & sampled)
    base = len(marked & sampled) / len(hints)
    print(f"       適合率 {prec*100:.0f}% / 再現率 {rec*100:.0f}% / "
          f"下限 {base*100:.0f}% / 提示 {len(high)/len(hints)*100:.0f}%")
    # 実測は適合率 36% / 再現率 38%。丸めと turn の版で少し動くので幅を持たせる
    assert prec > base * 2, "下限の 2 倍を割った。手がかりが効いていない"
    assert 0.30 <= rec <= 0.50, f"再現率が想定から外れた: {rec:.2f}"


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
