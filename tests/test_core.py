"""コア部分(GUI 以外)のユニットテスト。

    python -m pytest tests/ -q
    python tests/test_core.py     # pytest が無くても動く

GUI・Gemini API・ffmpeg には依存しない。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.segments import (  # noqa: E402
    Project,
    SPECIAL_UNKNOWN,
    Segment,
    fmt_hms,
    parse_roster,
    roster_to_text,
)
from src.suggest import SpeakerSuggester, next_unassigned  # noqa: E402
from src.transcribe import (  # noqa: E402
    build_prompt,
    normalize_cluster_label,
    parse_segments,
)


# ======================================================================
# 名簿パース
# ======================================================================

def test_parse_roster_formats():
    text = """
    佐藤(理事長)
    田中（事務局長）: 議事進行を担当
    - 鈴木
    高橋: 会計
    """
    sp = parse_roster(text)
    assert [s.name for s in sp] == ["佐藤", "田中", "鈴木", "高橋"]
    assert sp[0].note == "理事長"
    assert sp[1].note == "事務局長 / 議事進行を担当"
    assert sp[2].note == ""
    assert sp[3].note == "会計"
    assert [s.order for s in sp] == [0, 1, 2, 3]
    # ID は一意
    assert len({s.id for s in sp}) == 4


def test_roster_round_trip():
    sp = parse_roster("佐藤(理事長)\n鈴木")
    assert roster_to_text(sp) == "佐藤(理事長)\n鈴木"


# ======================================================================
# クラスタラベルの正規化
# ======================================================================

def test_normalize_cluster_label():
    assert normalize_cluster_label("A") == "A"
    assert normalize_cluster_label("発言者B") == "B"
    assert normalize_cluster_label("話者 c") == "C"
    assert normalize_cluster_label("Speaker D") == "D"
    assert normalize_cluster_label("A(男性)") == "A"
    assert normalize_cluster_label("発言者不明") == "?"
    assert normalize_cluster_label("発言者複数・重複") == "*"
    assert normalize_cluster_label("?") == "?"
    assert normalize_cluster_label("*") == "*"
    assert normalize_cluster_label(None) == "?"
    assert normalize_cluster_label("") == "?"


# ======================================================================
# セグメント分解
# ======================================================================

SAMPLE = """[00:00] 【A】 本日はお忙しい中ありがとうございます。
[00:25] 【B】 前回の議事録は配布済みですか?
続きの行です。
[00:32] 【A】 はい、配布済みです。
[01:10] 【?】 (聴取不能)
"""


def test_parse_segments_basic():
    segs = parse_segments(SAMPLE, chunk_index=0, offset_seconds=0, chunk_seconds=600)
    assert len(segs) == 4
    assert segs[0]["start"] == 0.0
    assert segs[0]["cluster"] == "0:A"
    assert segs[1]["start"] == 25.0
    # 時刻の無い行は直前に連結される
    assert segs[1]["text"].endswith("続きの行です。")
    # 終了時刻は次の開始
    assert segs[1]["end"] == 32.0
    # 最終区間はチャンク末尾まで
    assert segs[3]["end"] == 600.0
    assert segs[3]["cluster"] == "0:?"
    assert [s["index"] for s in segs] == [0, 1, 2, 3]


def test_parse_segments_offset_and_chunk_namespacing():
    segs = parse_segments(SAMPLE, chunk_index=2, offset_seconds=1200, chunk_seconds=600,
                          start_index=10)
    assert segs[0]["start"] == 1200.0
    assert segs[1]["start"] == 1225.0
    assert segs[0]["cluster"] == "2:A"        # チャンクごとに別クラスタ扱い
    assert segs[0]["index"] == 10
    assert segs[-1]["end"] == 1800.0


def test_parse_segments_handles_messy_output():
    messy = """
    ここに前置きが入ってしまった
    [0:05]【発言者A】ラベルが全角括弧でない場合
    ［01:00］ 【B】 全角の括弧
    [00:30] 【C】 時刻が巻き戻っているケース
    [02:00]
    [02:10] 【B】 空行のあとの発言
    """
    segs = parse_segments(messy, chunk_index=1, offset_seconds=600, chunk_seconds=600)
    starts = [s["start"] for s in segs]
    # 単調増加が保たれる(巻き戻りは直前に合わせる)
    assert starts == sorted(starts)
    clusters = [s["cluster"] for s in segs]
    assert "1:A" in clusters and "1:B" in clusters and "1:C" in clusters
    # 本文が空の行は落ちる
    assert all(s["text"] for s in segs)


def test_parse_segments_empty():
    assert parse_segments("", 0, 0, 600) == []


def test_build_prompt_cluster_only_has_no_names():
    p = build_prompt(True, True, roster="佐藤(理事長)", verbatim=False, cluster_only=True)
    assert "佐藤" not in p              # 名簿は渡さない
    assert "【A】" in p
    assert "名前や役職は絶対に書かない" in p


# ======================================================================
# 候補の学習・並べ替え
# ======================================================================

def _make_project() -> Project:
    proj = Project(audio_path="dummy.m4a", duration=600.0, chunk_seconds=600)
    proj.speakers = parse_roster("佐藤\n田中\n鈴木")
    texts = ["あ", "い", "う", "え", "お", "か", "が", "き", "く", "け"]
    clusters = ["0:A", "0:B", "0:A", "0:B", "0:A", "0:C", "0:A", "0:B", "0:A", "0:C"]
    proj.segments = [
        Segment(index=i, start=i * 10.0, end=i * 10.0 + 9, text=t, cluster=c, chunk=0)
        for i, (t, c) in enumerate(zip(texts, clusters))
    ]
    return proj


def test_cluster_vote_dominates():
    proj = _make_project()
    sato, tanaka, suzuki = proj.speakers
    # クラスタ 0:A を佐藤で 2 回確定
    proj.segments[0].speaker_id = sato.id
    proj.segments[2].speaker_id = sato.id
    s = SpeakerSuggester(proj)

    # 同じクラスタの未確定区間 → 佐藤が 1 位
    top = s.rank(4)[0]
    assert top.speaker.id == sato.id
    assert "同じ声のまとまり" in top.reason_text

    # 別クラスタでは佐藤が 1 位にはならない(頻度の加点だけ)
    ranked_b = s.rank(1)
    assert ranked_b[0].speaker.id != sato.id or ranked_b[0].score < top.score


def test_learning_improves_with_more_assignments():
    proj = _make_project()
    sato, tanaka, suzuki = proj.speakers
    s = SpeakerSuggester(proj)
    # 何も確定していない段階では名簿順
    assert [c.speaker.id for c in s.rank(1)] == [sato.id, tanaka.id, suzuki.id]

    # 0:B を田中で確定していくと、0:B の未確定は田中が 1 位になる
    proj.segments[1].speaker_id = tanaka.id
    proj.segments[3].speaker_id = tanaka.id
    s.refresh()
    assert s.rank(7)[0].speaker.id == tanaka.id
    assert s.dominant_speaker("0:B") == tanaka.id


def test_repeat_penalty_and_transition():
    proj = _make_project()
    sato, tanaka, suzuki = proj.speakers
    # 直前が佐藤。遷移統計が薄いので佐藤は減点されるはず
    proj.segments[0].speaker_id = sato.id
    s = SpeakerSuggester(proj)
    ranked = {c.speaker.id: c.score for c in s.rank(1)}
    assert ranked[sato.id] < ranked[tanaka.id]


def test_special_labels_excluded_from_stats():
    proj = _make_project()
    proj.segments[0].speaker_id = SPECIAL_UNKNOWN
    proj.segments[2].speaker_id = SPECIAL_UNKNOWN
    s = SpeakerSuggester(proj)
    # 不明だけのクラスタは、実在の話者を押し上げない
    ranked = s.rank(4)
    assert all(not c.reasons for c in ranked)
    assert s.dominant_speaker("0:A") is None


def test_rank_ignores_own_current_assignment():
    """付け替えたいときに、自分自身の割当が候補を固着させないこと"""
    proj = _make_project()
    sato, tanaka, _ = proj.speakers
    proj.segments[0].speaker_id = sato.id   # このクラスタで唯一の確定
    s = SpeakerSuggester(proj)
    ranked = s.rank(0)                      # 自分自身を評価
    assert not ranked[0].reasons            # クラスタ投票の根拠は消えている


def test_next_unassigned():
    proj = _make_project()
    proj.segments[0].speaker_id = "sp01"
    proj.segments[1].speaker_id = "sp01"
    assert next_unassigned(proj, 0) == 2
    assert next_unassigned(proj, 1) == 2
    assert next_unassigned(proj, 9) is None
    assert next_unassigned(proj, 3, forward=False) == 2


# ======================================================================
# 保存・読み込み・出力
# ======================================================================

def test_save_load_round_trip():
    proj = _make_project()
    proj.segments[0].speaker_id = proj.speakers[0].id
    proj.segments[0].reviewed = True
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.speakers.json"
        proj.save(p)
        loaded = Project.load(p)
    assert loaded.total_count == proj.total_count
    assert loaded.assigned_count == 1
    assert loaded.segments[0].reviewed is True
    assert [s.name for s in loaded.speakers] == ["佐藤", "田中", "鈴木"]
    assert loaded.clusters() == ["0:A", "0:B", "0:C"]


def test_remove_speaker_clears_assignments():
    proj = _make_project()
    sid = proj.speakers[0].id
    proj.segments[0].speaker_id = sid
    proj.remove_speaker(sid)
    assert proj.segments[0].speaker_id is None
    assert proj.assigned_count == 0
    assert [s.order for s in proj.speakers] == [0, 1]


def test_write_docx_merges_consecutive():
    from docx import Document
    from src.segments import write_docx

    proj = _make_project()
    sid = proj.speakers[0].id
    for i in (0, 1, 2):
        proj.segments[i].speaker_id = sid
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.docx"
        write_docx(proj, out)
        doc = Document(str(out))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
    body = [p for p in paras if p.startswith("[")]
    # 連続する 3 区間が 1 段落にまとまる
    assert body[0].startswith("[00:00:00] 【佐藤】")
    assert "あ い う" in body[0]
    # 未確定は【発言者不明】
    assert "【発言者不明】" in body[1]


def test_fmt_hms():
    assert fmt_hms(0) == "00:00:00"
    assert fmt_hms(59.6) == "00:01:00"
    assert fmt_hms(3725) == "01:02:05"


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
