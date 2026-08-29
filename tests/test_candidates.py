"""候補の一覧（candidates.py）の検査。

**この機能は検出器ではない。**適合 35/51・再現 31/34 で、候補が無い区間にも
脱落はある。その性格が壊れていないこと（判定を出さない・注意書きが消えない・
測定に使った閾値が黙って変わらない）も見る。

実データでの再現は、逐語正解がある場合のみ最後で動かす。
数字を出し直す道具は tools/report_candidates.py。
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

from src.candidates import (  # noqa: E402
    CANDIDATES_VER,
    MIN_OVERLAP_SECONDS,
    VoiceCandidate,
    coverage,
    dismissed_path,
    done_keys,
    DONE_SLACK_SECONDS,
    drop_dismissed,
    find_candidates,
    for_segment,
    load_dismissed,
    main_speaker,
    save_dismissed,
)
from src.diarize import SpeakerTurn  # noqa: E402
from src.segments import Project, Segment, parse_roster, segment_key  # noqa: E402


def _seg(index: int, start: float, end: float, text: str = "本文") -> Segment:
    return Segment(index=index, start=start, end=end, text=text,
                   cluster="g:A", chunk=0)


def _turn(start: float, end: float, speaker: int) -> SpeakerTurn:
    return SpeakerTurn(start=start, end=end, speaker=speaker)


# --- 主たる話者 -------------------------------------------------------
def test_main_speaker_is_the_longest_overlap():
    """区間の主たる話者は、最も長く重なる turn の話者。"""
    seg = _seg(0, 100.0, 110.0)
    turns = [_turn(100.0, 109.0, 3), _turn(104.0, 104.8, 7)]
    assert main_speaker(seg, turns) == 3


def test_main_speaker_is_none_without_turns():
    """話者分離を通していなければ決められない（この機能は使えない）。"""
    assert main_speaker(_seg(0, 0.0, 1.0), []) is None


# --- 候補を見つける ---------------------------------------------------
def test_finds_another_voice_inside_a_segment():
    """**これが入れ替えた定義そのもの。**区間の中の、別の声。

    もとの設計(turn と区間の差集合)は逐語正解 34 件に対して 0/34 だった。
    脱落は他人の発言の最中に埋もれており、区間は存在するので
    「すきま」にならない(設計書 §10.3)。
    """
    seg = _seg(0, 100.0, 110.0)
    turns = [_turn(100.0, 110.0, 3), _turn(104.0, 104.8, 7)]
    got = find_candidates([seg], turns)
    assert len(got) == 1, got
    assert got[0].speaker == 7
    assert abs(got[0].at - 104.0) < 1e-6
    assert got[0].parent_orig == seg.orig_start


def test_does_not_propose_the_main_speaker():
    """主たる話者そのものは候補にしない（本人が喋っているだけ）。"""
    seg = _seg(0, 100.0, 110.0)
    turns = [_turn(100.0, 105.0, 3), _turn(105.0, 110.0, 3)]
    assert find_candidates([seg], turns) == []


def test_ignores_a_too_short_overlap():
    """**息継ぎや漏れ込みで出る短い被りは数えない。**"""
    seg = _seg(0, 100.0, 110.0)
    turns = [_turn(100.0, 110.0, 3), _turn(104.0, 104.0 + 0.1, 7)]
    assert find_candidates([seg], turns) == []
    turns2 = [_turn(100.0, 110.0, 3), _turn(104.0, 104.0 + 0.3, 7)]
    assert len(find_candidates([seg], turns2)) == 1


def test_min_overlap_is_the_value_we_measured_with():
    """**閾値を黙って変えない。**変えると 31/34・35/51 が保証でなくなる。"""
    assert MIN_OVERLAP_SECONDS == 0.2


def test_returns_nothing_without_diarization():
    """turn が無ければ空。呼び出し側が「使えない」と伝えること。"""
    assert find_candidates([_seg(0, 0.0, 10.0)], []) == []
    assert find_candidates([], [_turn(0.0, 1.0, 1)]) == []


def test_candidates_come_out_in_time_order():
    """聴く順に流せるよう、時刻順で返す。"""
    segs = [_seg(0, 100.0, 110.0), _seg(1, 110.0, 120.0)]
    turns = [_turn(100.0, 120.0, 3), _turn(115.0, 116.0, 7),
             _turn(104.0, 105.0, 8)]
    got = find_candidates(segs, turns)
    assert [round(c.at) for c in got] == [104, 115], got


def test_key_is_orig_start_not_index():
    """**index は分割・結合・再実行で振り直る。**鍵にしてはいけない。"""
    seg = _seg(5, 100.0, 110.0)
    turns = [_turn(100.0, 110.0, 3), _turn(104.0, 105.0, 7)]
    got = find_candidates([seg], turns)[0]
    seg.index = 99                      # 振り直しを模す
    assert for_segment([got], seg) == [got], "index に依存している"


def test_for_segment_only_returns_its_own():
    segs = [_seg(0, 100.0, 110.0), _seg(1, 200.0, 210.0)]
    turns = [_turn(100.0, 110.0, 3), _turn(104.0, 105.0, 7),
             _turn(200.0, 210.0, 3), _turn(204.0, 205.0, 8)]
    got = find_candidates(segs, turns)
    assert len(for_segment(got, segs[0])) == 1
    assert for_segment(got, segs[0])[0].speaker == 7


def test_split_halves_do_not_share_candidates():
    """**分割した 2 つが互いの候補まで出さない（設計書 §10.3.4）。**

    `split_segment` は「元は 1 つだった」と分かるよう、分割した両方に親の
    `orig_start` を与える（再実行の引き継ぎのため。これ自体は正しい）。
    その結果 `orig_start` は一意でなくなり、実データで候補が二重に出て、
    片方で × を押すともう片方からも消えていた（2026-08-19 に実測）。
    鍵を `(orig_start, start)` の組にして分けた。
    """
    proj = Project(audio_path="a.m4a", duration=300.0)
    proj.speakers = parse_roster("佐藤")
    proj.segments = [_seg(0, 100.0, 120.0, "まえのはなしとあとのはなし")]
    head, tail = proj.split_segment(0, 110.0, 6)
    assert head.orig_start == tail.orig_start, "前提が変わった（分割の設計）"
    assert segment_key(head) != segment_key(tail), "**鍵が分けられていない**"

    turns = [_turn(100.0, 120.0, 3), _turn(104.0, 105.0, 7),
             _turn(115.0, 116.0, 8)]
    got = find_candidates(proj.segments, turns)
    a = for_segment(got, proj.segments[0])
    b = for_segment(got, proj.segments[1])
    assert len(a) == 1 and abs(a[0].at - 104.0) < 0.01, a
    assert len(b) == 1 and abs(b[0].at - 115.0) < 0.01, b
    assert a[0].key != b[0].key, "× が両方から消える"


def test_true_duplicates_still_share_a_key():
    """**本物の重複は鍵では分けられない。**

    `start` も `orig_start` も同じ区間が 2 つあると、この鍵でも区別できない。
    実データに 1 組あった（2026-08-19）。**これは鍵の問題ではなく、
    区間が重複して作られている問題**なので、鍵を凝っても解決しない。
    分かっていることを検査に残しておく。
    """
    a = _seg(0, 100.0, 110.0, "同じ本文")
    b = _seg(1, 100.0, 110.0, "同じ本文")
    assert segment_key(a) == segment_key(b)


# --- 却下（×）--------------------------------------------------------
def test_dismissed_ones_do_not_come_back():
    """**空振りは 3 割ある。**捨てた判断が残らないと毎回出てきて使えない。"""
    seg = _seg(0, 100.0, 110.0)
    turns = [_turn(100.0, 110.0, 3), _turn(104.0, 105.0, 7),
             _turn(107.0, 108.0, 8)]
    got = find_candidates([seg], turns)
    assert len(got) == 2
    kept = drop_dismissed(got, [got[0].key])
    assert len(kept) == 1 and kept[0].speaker == 8


def test_key_survives_a_rebuild():
    """作り直しても同じ鍵になる（でないと却下が効かない）。"""
    seg = _seg(0, 100.0, 110.0)
    turns = [_turn(100.0, 110.0, 3), _turn(104.0, 105.0, 7)]
    a = find_candidates([seg], turns)[0]
    b = find_candidates([_seg(9, 100.0, 110.0)], turns)[0]
    assert a.key == b.key


def test_dismissed_sidecar_round_trip():
    with tempfile.TemporaryDirectory() as d:
        p = dismissed_path(d, "ff00")
        assert p is not None
        assert load_dismissed(p) == []          # 無ければ空
        save_dismissed(p, ["100.000@104.00", "100.000@107.00"])
        assert load_dismissed(p) == ["100.000@104.00", "100.000@107.00"]


def test_dismissed_sidecar_needs_a_fingerprint():
    """指紋が無ければ保存しない（転写のキャッシュと同じ流儀）。"""
    with tempfile.TemporaryDirectory() as d:
        assert dismissed_path(d, "") is None
        save_dismissed(None, ["x"])             # 落ちない
        assert load_dismissed(None) == []


def test_dismissed_sidecar_is_rebuilt_on_a_version_change():
    with tempfile.TemporaryDirectory() as d:
        p = dismissed_path(d, "ff00")
        save_dismissed(p, ["a"])
        raw = json.loads(p.read_text(encoding="utf-8"))
        raw["version"] = CANDIDATES_VER + 1
        p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        assert load_dismissed(p) == []


def test_dismissed_sidecar_survives_a_broken_file():
    with tempfile.TemporaryDirectory() as d:
        p = dismissed_path(d, "ff00")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{壊れている", encoding="utf-8")
        assert load_dismissed(p) == []          # また出てくるだけ


def test_dismissed_keeps_no_duplicates():
    with tempfile.TemporaryDirectory() as d:
        p = dismissed_path(d, "ff00")
        save_dismissed(p, ["a", "a", "b"])
        assert load_dismissed(p) == ["a", "b"]


def test_sidecar_keeps_the_warning():
    """**数字だけが残ると「候補が無い＝脱落が無い」と読まれる。**

    listen_order と同じ理由で、注意書きを消させない。
    """
    with tempfile.TemporaryDirectory() as d:
        p = dismissed_path(d, "ff00")
        save_dismissed(p, [])
        raw = json.loads(p.read_text(encoding="utf-8"))
        assert "脱落の有無を表さない" in raw["note"]
        assert "31/34" in raw["measured"] and "51" in raw["measured"]
        assert raw["min_overlap_seconds"] == MIN_OVERLAP_SECONDS


# --- もう足した位置（設計書 §10.3.2）----------------------------------
def _added(start: float, end: float) -> Segment:
    return Segment(index=99, start=start, end=end, text="はい",
                   cluster="g:B", chunk=0)


def test_done_marks_where_something_was_already_added():
    """人がもう足した位置は「済」。**隠すのではなく印を付ける。**

    隠すと「候補が無い＝やることが無い」と読まれ、この道具の性格
    （印が無い＝安全ではない）と食い違う。
    """
    seg = _seg(0, 100.0, 110.0)
    turns = [_turn(100.0, 110.0, 3), _turn(104.0, 105.0, 7),
             _turn(107.0, 108.0, 8)]
    got = find_candidates([seg], turns)
    done = done_keys(got, [_added(104.2, 105.0)])
    assert len(done) == 1
    assert got[0].key in done and got[1].key not in done
    assert len(got) == 2, "**候補そのものは減らさない**"


def test_done_allows_for_the_time_gap():
    """**足した時刻は turn の開始とずれる。**実データで 0.69〜1.07 秒。

    小窓の時刻は本文の位置からの文字数按分で決まるので、声の始まりとは
    一致しない。余裕を持たせないと、足したのにまた出てくる。
    """
    seg = _seg(0, 100.0, 110.0)
    turns = [_turn(100.0, 110.0, 3), _turn(104.0, 104.5, 7)]
    got = find_candidates([seg], turns)
    assert done_keys(got, [_added(105.4, 106.2)]), "1 秒のずれを許していない"
    assert not done_keys(got, [_added(108.0, 108.8)]), "離れすぎたものまで済にした"


def test_done_slack_is_the_measured_value():
    """**余裕を黙って変えない。**広げると別の相づちまで済にしてしまう。"""
    assert DONE_SLACK_SECONDS == 1.0


def test_done_is_empty_without_added_utterances():
    seg = _seg(0, 100.0, 110.0)
    turns = [_turn(100.0, 110.0, 3), _turn(104.0, 105.0, 7)]
    assert done_keys(find_candidates([seg], turns), []) == set()


# --- 作業量の目安 -----------------------------------------------------
def test_coverage_counts_segments_not_candidates():
    """「何割の区間を聴くことになるか」。候補の数ではなく区間の数。"""
    segs = [_seg(0, 100.0, 110.0), _seg(1, 200.0, 210.0)]
    turns = [_turn(100.0, 110.0, 3), _turn(104.0, 105.0, 7),
             _turn(106.0, 107.0, 8), _turn(200.0, 210.0, 3)]
    got = find_candidates(segs, turns)
    assert len(got) == 2                        # 候補は 2 件だが
    assert coverage(got, segs) == (1, 2)        # 聴く区間は 1 つ


def test_module_does_not_claim_detection():
    """**「検出」と名乗らない。**印が無い＝大丈夫、と読まれるため。"""
    import src.candidates as mod
    doc = mod.__doc__ or ""
    assert "検出器ではない" in doc
    assert "31/34" in doc and "35/51" in doc
    assert "±" in doc, "幅を落として書いている"


# --- 実データ（あるときだけ）------------------------------------------
def test_real_data_reproduces_the_measured_numbers():
    """逐語正解があるときだけ動く。**設計書の数字が再現するか見張る。**"""
    truth_dir = Path(r"C:\dev\01\test-audio\truth")
    # **見る先は diarized/ の作業ファイル。**本体は人が毎日編集しており、
    # 区間を分けたり時刻を直したりするたびに候補が増えて数字が動く
    # (実測: 候補 49→73、適合 69%→55%)。**編集される file を検査の
    # 基準にすると、機能を壊していないのに落ちる。**
    proj_path = Path(
        r"C:\dev\01\test-audio\diarized\01+02edited.speakers.json")
    turns_path = Path(
        r"C:\dev\01\test-audio\.work_01+02edited\diarize"
        r"\turns.ca1fb4d464e99c16.pyannote3-titanet.n9.v1.json")
    if not (truth_dir.exists() and proj_path.exists() and turns_path.exists()):
        print("       (実データが無いので飛ばす)")
        return
    from src.segments import Project
    proj = Project.load(str(proj_path))
    turns = [SpeakerTurn.from_dict(t) for t in
             json.loads(turns_path.read_text(encoding="utf-8"))["turns"]]

    bands, missing = {}, []
    for name in ("a-setsumei", "b-chuban", "c-ouchou", "d-missitsu"):
        f = truth_dir / f"verbatim.{name}.json"
        if not f.exists():
            print("       (逐語正解がそろっていないので飛ばす)")
            return
        d = json.loads(f.read_text(encoding="utf-8"))
        bands[name] = d["band"]
        missing.extend(d.get("missing") or [])

    cands = find_candidates(proj.segments, turns)
    in_band = [c for c in cands
               if any(b["start"] <= c.at <= b["end"] for b in bands.values())]
    hit = sum(1 for c in in_band
              if any(abs(c.at - m["at"]) <= 3.0 for m in missing))
    found = sum(1 for m in missing
                if any(abs(c.at - m["at"]) <= 3.0 for c in in_band))
    print(f"       候補 {len(in_band)} 個 / 再現 {found}/{len(missing)} / "
          f"適合 {hit}/{len(in_band)}")
    # 安定したファイルでの実測は 候補 49 / 再現 31/34 / 適合 34/49(69%)。
    # 設計書の 51 / 31/34 / 35/51(69%) と**割合は同じ**。
    assert found / len(missing) >= 0.85, f"再現が落ちた: {found}/{len(missing)}"
    assert hit / len(in_band) >= 0.60, f"適合が落ちた: {hit}/{len(in_band)}"
    assert len(in_band) <= 60, f"候補が増えすぎた: {len(in_band)}"


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
