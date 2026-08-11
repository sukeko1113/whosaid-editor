"""コア部分(GUI 以外)のユニットテスト。

    python -m pytest tests/ -q
    python tests/test_core.py     # pytest が無くても動く

GUI・Gemini API・ffmpeg には依存しない。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows の既定コンソールは cp932 / cp1252 なので、日本語のラベルを print した
# 時点で UnicodeEncodeError になる(テストの中身とは無関係に落ちる)。
# 出力先のエンコーディングをここで固定しておく。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


from src.segments import (  # noqa: E402
    SCHEMA_VERSION,
    Project,
    SPECIAL_UNKNOWN,
    Segment,
    fmt_hms,
    fmt_hms_frac,
    parse_hms,
    parse_roster,
    roster_to_text,
)
from src.suggest import (  # noqa: E402
    SpeakerSuggester,
    next_unassigned,
    next_unreviewed,
)
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


# ======================================================================
# 同じ話者の連続をまとめる(v2.0.3)
#
# Gemini は指示しても息継ぎごとに行を分けてくることがある。
# 実機では 52 分の音声が 615 区間に割れ、割当作業が現実的でなくなった。
# ======================================================================

FRAGMENTED = """[00:00] 【A】 吉沢と申しますけども
[00:01] 【A】 えー、まずその6つのですね
[00:24] 【A】 工程表
[00:26] 【A】 えー、財源計画
[00:29] 【A】 年度別資金繰り
[00:47] 【A】 あの、第1の、あの、お願いでございます。
[01:05] 【B】 はい。
[01:06] 【A】 はい、以上です。
[01:06] 【B】 で、は、あの、代わりまして西村と申します。
[01:11] 【B】 えー、校長をしておりました。はい。
"""


def test_merge_collapses_same_speaker_run():
    merged = parse_segments(FRAGMENTED, 0, 0, 600)
    clusters = [s["cluster"] for s in merged]
    assert clusters == ["0:A", "0:B", "0:A", "0:B"], clusters
    # 1件目は 6 行分がまとまり、開始と終了が run 全体を覆う
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 65.0
    assert "吉沢と申しますけども" in merged[0]["text"]
    assert "お願いでございます。" in merged[0]["text"]


def test_merge_can_be_disabled():
    raw = parse_segments(FRAGMENTED, 0, 0, 600, merge_same_speaker=False)
    assert len(raw) == 10


def test_merge_inserts_reading_comma():
    """断片をそのまま繋ぐと「けどもえー」になる。間には読点を補う。"""
    merged = parse_segments(FRAGMENTED, 0, 0, 600)
    text = merged[0]["text"]
    assert "けども、えー" in text
    assert "けどもえー" not in text
    # 既に句読点で終わっていれば二重にしない
    assert "。、" not in text


def test_merge_skips_pseudo_clusters():
    """【?】【*】は中身がばらばらなので連結しない"""
    text = """[00:00] 【?】 うん
[00:02] 【?】 ええ
[00:04] 【*】 (重なり)
[00:06] 【*】 (重なり)
"""
    merged = parse_segments(text, 0, 0, 600)
    assert len(merged) == 4


def test_merge_respects_gap_and_max_length():
    from src.transcribe import merge_consecutive

    raw = [
        {"rel": 0.0, "rel_end": 10.0, "cluster": "A", "text": "あ"},
        {"rel": 60.0, "rel_end": 70.0, "cluster": "A", "text": "い"},   # 50秒の間
    ]
    assert len(merge_consecutive(raw, max_gap=20.0)) == 2
    assert len(merge_consecutive(raw, max_gap=90.0)) == 1

    long_run = [
        {"rel": float(i * 30), "rel_end": float(i * 30 + 30), "cluster": "A", "text": f"t{i}"}
        for i in range(10)      # 合計 300 秒
    ]
    out = merge_consecutive(long_run, max_seconds=120.0)
    assert len(out) > 1
    assert all(s["rel_end"] - s["rel"] <= 130.0 for s in out)


def test_merge_keeps_absolute_times_with_offset():
    merged = parse_segments(FRAGMENTED, chunk_index=2, offset_seconds=1200,
                            chunk_seconds=600)
    assert merged[0]["start"] == 1200.0
    assert merged[0]["cluster"] == "2:A"
    starts = [s["start"] for s in merged]
    assert starts == sorted(starts)


# ======================================================================
# 時刻が詰まった箇所の割り振り直し(v2.0.4 / v2.0.6)
#
# Gemini は発言が立て込むと 1 秒刻みで時刻を振る。そのまま使うと
# 長い発言が 1 秒で切られ、単に延ばすと隣同士の再生窓が重なって
# 同じ音声が聞こえる。本文の長さで按分して、重ならないようにする。
# ======================================================================

def test_estimate_speech_seconds():
    from src.transcribe import MIN_SEGMENT_SECONDS, estimate_speech_seconds

    assert estimate_speech_seconds("") == MIN_SEGMENT_SECONDS
    assert estimate_speech_seconds("うん。") == MIN_SEGMENT_SECONDS   # 短い相づちは下限
    long_line = "彼が要するに院外理事で、あの、理事で、あとはみんな学園の教授の方とか。"
    assert 6.0 < estimate_speech_seconds(long_line) < 12.0
    # 長さに比例する
    assert estimate_speech_seconds("あ" * 100) > estimate_speech_seconds("あ" * 50)


def test_overlapping_backchannel_does_not_truncate():
    """相づちが発言に重なると、終了時刻が次の開始で潰されていた。

    実機で確認された事例: 37文字の発言の再生窓が 1 秒になり、
    再生すると「彼」だけ聞こえて切れた。
    """
    text = """[07:02] 【C】 卒業生。
[07:03] 【A】 彼が要するに院外理事で、あの、理事で、あとはみんな学園の教授の方とか。
[07:04] 【C】 うん。
[07:30] 【A】 次の発言。
"""
    segs = parse_segments(text, chunk_index=5, offset_seconds=2700, chunk_seconds=540)
    long_seg = next(s for s in segs if "院外理事" in s["text"])
    # 次の開始(07:04)で切られず、本文に見合う長さが確保されている
    assert long_seg["end"] - long_seg["start"] > 6.0, long_seg
    # 後ろに余裕があるので開始は動かない
    assert long_seg["start"] == 2700 + 7 * 60 + 3
    # 余裕があっても必要以上には伸ばさない
    assert long_seg["end"] - long_seg["start"] < 12.0


def test_no_overlapping_playback_windows():
    """隣り合う区間が重なると、再生したときに同じ音声が聞こえてしまう。

    実機報告: 51:47 と 51:48、51:50 と 51:51、52:03 と 52:04 で
    同じ音声が流れた。
    """
    text = """[06:47] 【B】 それができればいいんだけど。
[06:48] 【A】 理事がもう。
[06:49] 【B】 はい。
[06:50] 【A】 ま、梅田さんとあともう1人の県議以外は全部内部理事。
[06:51] 【C】 うん。
[06:56] 【B】 県議で。
[06:57] 【A】 はい。
"""
    segs = parse_segments(text, chunk_index=5, offset_seconds=2700, chunk_seconds=540)
    for a, b in zip(segs, segs[1:]):
        assert b["start"] >= a["end"] - 0.01, (a["text"], b["text"], a["end"], b["start"])
    # 詰まった範囲でも、長い発言には短い相づちより多くの時間が割り当てられる
    # (最後の区間はチャンク末尾まで伸びるので比較から外す)
    long_seg = next(s for s in segs if "梅田" in s["text"])
    short = [s for s in segs[:-1] if s["text"].strip() in ("はい。", "うん。")]
    assert short, "比較対象の相づちが無い"
    assert all(long_seg["end"] - long_seg["start"] > s["end"] - s["start"] for s in short)
    assert long_seg["end"] - long_seg["start"] > 3.0


def test_roomy_timestamps_are_left_alone():
    """余裕がある箇所は Gemini の時刻をそのまま使う"""
    text = """[00:00] 【A】 短い。
[00:30] 【B】 これも短い。
[01:00] 【A】 やはり短い。
"""
    segs = parse_segments(text, 0, 0, 600)
    assert [s["start"] for s in segs] == [0.0, 30.0, 60.0]
    assert segs[0]["end"] == 30.0 and segs[1]["end"] == 60.0


def test_extension_is_clamped_to_chunk_end():
    """チャンク(音声)の末尾を越えて伸ばさない"""
    text = "[08:58] 【A】 " + "あ" * 300 + "\n"
    segs = parse_segments(text, chunk_index=0, offset_seconds=0, chunk_seconds=540)
    assert segs[-1]["end"] == 540.0


def test_time_is_shared_in_proportion_to_text():
    """詰まった範囲では、時間を本文の長さの比で分け合う。

    実際より時間が足りない以上、全員に必要な長さを与えることはできない。
    せめて長い発言に多く、短い相づちに少なく配ることで、
    再生したときに隣と同じ音にならないようにする。
    """
    text = """[00:00] 【A】 これはかなり長い発言でして、相づちが重なっても最後まで再生できる必要があります。
[00:01] 【B】 うん。
[00:02] 【A】 続けてもう一つ、これも長めの発言をしておきます。切られては困ります。
[00:03] 【C】 はい。
[00:20] 【A】 次の発言です。
"""
    from src.transcribe import estimate_speech_seconds

    segs = parse_segments(text, 0, 0, 600)
    dur = {s["text"][:8]: s["end"] - s["start"] for s in segs}

    # 0〜2 秒に 2 発言。時間が足りないので比で分ける
    assert dur["これはかなり長い"] > dur["うん。"] * 3

    # 後ろに余裕がある発言は、必要な長さを丸ごと確保できる
    need2 = estimate_speech_seconds(
        "続けてもう一つ、これも長めの発言をしておきます。切られては困ります。")
    assert dur["続けてもう一つ、"] >= need2 - 0.05

    # 隙間なく敷き詰められ、重ならない
    for a, b in zip(segs, segs[1:]):
        assert abs(b["start"] - a["end"]) < 0.02


def test_cluster_prompt_forbids_fragmenting():
    p = build_prompt(True, True, cluster_only=True)
    assert "同じ人が話し続けている間は、絶対に行を分けない" in p
    assert "息継ぎ" in p


def test_classify_api_error():
    """再試行しても直らないエラーを見分けられること"""
    from src.transcribe import classify_api_error

    depleted = (
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': "
        "'Your prepayment credits are depleted. Please go to AI Studio at "
        "https://ai.studio/projects to manage your project and billing.', "
        "'status': 'RESOURCE_EXHAUSTED'}}"
    )
    msg = classify_api_error(Exception(depleted))
    assert msg is not None
    assert "残高が尽きています" in msg
    assert "aistudio.google.com" in msg

    quota = classify_api_error(Exception("You exceeded your current quota, please check your plan"))
    assert quota is not None and "利用上限" in quota

    for key_err in ("API key not valid. Please pass a valid API key.",
                    "403 PERMISSION_DENIED",
                    "UNAUTHENTICATED: request is missing credentials"):
        m = classify_api_error(Exception(key_err))
        assert m is not None, key_err
        assert "API キー" in m

    # 一時的なものは再試行させる(None)
    assert classify_api_error(Exception("Connection reset by peer")) is None
    assert classify_api_error(Exception("503 Service Unavailable")) is None
    assert classify_api_error(Exception("500 Internal error")) is None
    # 素の 429(分あたりのレート上限)は待てば通るので再試行対象
    assert classify_api_error(Exception("429 Too Many Requests")) is None


def test_rate_limit_detection():
    from src.transcribe import _is_rate_limited

    assert _is_rate_limited(Exception("429 Too Many Requests")) is True
    assert _is_rate_limited(Exception("RESOURCE_EXHAUSTED")) is True
    assert _is_rate_limited(Exception("Connection reset")) is False


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


def test_pseudo_clusters_are_not_learned():
    """【?】【*】は「同じ声のまとまり」ではないので、投票にも一括適用にも使わない"""
    proj = _make_project()
    sato = proj.speakers[0]
    for i, seg in enumerate(proj.segments):
        seg.cluster = "0:?"
    proj.segments[0].speaker_id = sato.id
    s = SpeakerSuggester(proj)

    top = s.rank(4)[0]
    assert not top.reasons, "判別不能の区間から他の区間を推測してはいけない"
    assert s.dominant_speaker("0:?") is None
    assert proj.segments[4].is_pseudo_cluster is True
    assert "一括適用は不可" in s.cluster_summary("0:?")


def test_boundary_continuity_bonus():
    """チャンク境界をまたぐ隣接区間は同一話者の可能性が高い"""
    proj = Project(audio_path="x.m4a", duration=1200.0, chunk_seconds=600)
    proj.speakers = parse_roster("佐藤\n田中\n鈴木")
    sato, tanaka, _ = proj.speakers
    proj.segments = [
        Segment(index=0, start=560.0, end=595.0, text="…そこで", cluster="0:B", chunk=0),
        Segment(index=1, start=600.0, end=630.0, text="申し上げた", cluster="1:A", chunk=1),
        Segment(index=2, start=630.0, end=660.0, text="別の発言", cluster="1:B", chunk=1),
    ]
    proj.segments[0].speaker_id = tanaka.id
    s = SpeakerSuggester(proj)

    ranked = s.rank(1)
    assert ranked[0].speaker.id == tanaka.id
    assert "チャンク境界" in ranked[0].reason_text
    # 境界でない区間には効かない
    assert "チャンク境界" not in " ".join(c.reason_text for c in s.rank(2))


def test_letter_prior_across_chunks():
    """前のチャンクの A が誰だったかは、次のチャンクの A の弱い手がかりになる"""
    proj = Project(audio_path="x.m4a", duration=1800.0, chunk_seconds=600)
    proj.speakers = parse_roster("佐藤\n田中\n鈴木")
    sato = proj.speakers[0]
    segs = []
    for i in range(6):
        chunk = i // 2
        segs.append(Segment(index=i, start=i * 100.0, end=i * 100.0 + 50,
                            text=f"t{i}", cluster=f"{chunk}:{'AB'[i % 2]}", chunk=chunk))
    proj.segments = segs
    # チャンク 0 と 1 の A を佐藤で確定
    proj.segments[0].speaker_id = sato.id
    proj.segments[2].speaker_id = sato.id
    s = SpeakerSuggester(proj)

    # チャンク 2 の A(未確定・クラスタ票なし)でも佐藤が 1 位
    ranked = s.rank(4)
    assert ranked[0].speaker.id == sato.id
    assert "別チャンクの 【A】" in ranked[0].reason_text


def test_next_unreviewed():
    proj = _make_project()
    sid = proj.speakers[0].id
    proj.segments[0].speaker_id = sid
    proj.segments[0].reviewed = True
    proj.segments[1].speaker_id = sid     # 一括適用ぶん(未確認)
    assert proj.reviewed_count == 1
    assert proj.unreviewed_count == 1
    assert next_unreviewed(proj, 0) == 1
    assert next_unassigned(proj, 0) == 2


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
    from src.segments import SPECIAL_NOISE, write_docx

    proj = _make_project()
    sid = proj.speakers[0].id
    for i in (0, 1, 2):
        proj.segments[i].speaker_id = sid
    proj.segments[3].speaker_id = SPECIAL_NOISE    # 出力から省かれるはず
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.docx"
        write_docx(proj, out, title="第1回理事会")
        doc = Document(str(out))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
    body = [p for p in paras if p.startswith("[")]
    # 連続する 3 区間が 1 段落にまとまる(日本語なので空白を挟まない)
    assert body[0].startswith("[00:00:00] 【佐藤】")
    assert "あいう" in body[0]
    # 表題・出席者一覧が入る
    assert paras[0] == "第1回理事会"
    assert any(p.startswith("出席者: 佐藤、田中、鈴木") for p in paras)
    # 雑音と印を付けた区間は出ない
    assert not any("え" == p[-1] for p in body if "【発言なし" in p)
    assert all("発言なし・雑音" not in p for p in paras)
    # 未確定は【発言者不明】
    assert "【発言者不明】" in body[1]


def test_write_docx_options():
    from docx import Document
    from src.segments import write_docx

    proj = _make_project()
    sid = proj.speakers[0].id
    for i in (0, 1):
        proj.segments[i].speaker_id = sid
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.docx"
        write_docx(proj, out, with_timestamps=False, merge_consecutive=False,
                   include_attendees=False, include_note=False)
        paras = [p.text for p in Document(str(out)).paragraphs if p.text.strip()]
    assert not any(p.startswith("[") for p in paras)
    assert not any(p.startswith("出席者:") for p in paras)
    assert paras[1] == "【佐藤】 あ"      # まとめない
    assert paras[2] == "【佐藤】 い"


def test_build_note_distinguishes_reviewed():
    from src.segments import build_note

    proj = _make_project()
    sid = proj.speakers[0].id
    proj.segments[0].speaker_id = sid
    proj.segments[0].reviewed = True
    proj.segments[1].speaker_id = sid          # 一括適用ぶん
    note = build_note(proj)
    assert "聴いて確定 1 区間" in note
    assert "まとめて適用 1 区間" in note
    assert "未確定 8 区間" in note


def test_fmt_hms():
    assert fmt_hms(0) == "00:00:00"
    assert fmt_hms(59.6) == "00:01:00"
    assert fmt_hms(3725) == "01:02:05"


# ======================================================================
# 時刻の表示・入力(0.1 秒精度)
# ======================================================================

def test_fmt_hms_frac():
    assert fmt_hms_frac(0) == "00:00:00.0"
    assert fmt_hms_frac(2631.5) == "00:43:51.5"
    assert fmt_hms_frac(3725.04) == "01:02:05.0"
    assert fmt_hms_frac(59.96) == "00:01:00.0"      # 秒への繰り上がり
    assert fmt_hms_frac(-3.0) == "00:00:00.0"       # 負は 0 に丸める


def test_parse_hms_formats():
    assert parse_hms("01:02:05.4") == 3725.4
    assert parse_hms("43:51.5") == 2631.5
    assert parse_hms("7.5") == 7.5
    assert parse_hms("7") == 7.0
    assert parse_hms("00:00:00.0") == 0.0
    # 最上位の桁だけは 60 以上を許す(90 秒、75 分)
    assert parse_hms("90") == 90.0
    assert parse_hms("75:00") == 4500.0
    # 前後の空白と、全角のまま打たれた数字・区切りも受ける
    assert parse_hms("  01:02:05.4  ") == 3725.4
    assert parse_hms("００:４３:５１．５") == 2631.5


def test_parse_hms_rejects_bad_values():
    bad = [
        "", "   ",
        "abc",
        "01:02:03:04",       # 桁が多い
        "01::05",            # 空フィールド
        ":30",
        "-5",                # 負
        "1.5:30",            # 小数は最下位だけ
        "01:75:00",          # 分が 60 以上
        "43:75.0",           # 秒が 60 以上
        "12:34:5x",
    ]
    for s in bad:
        try:
            parse_hms(s)
        except ValueError:
            continue
        raise AssertionError(f"拒否されるべき入力が通った: {s!r}")


def test_hms_round_trip():
    for sec in (0.0, 0.1, 7.5, 59.9, 2631.5, 3725.4, 36000.0):
        assert parse_hms(fmt_hms_frac(sec)) == sec


# ======================================================================
# 時刻編集のスキーマ(schema 3)
# ======================================================================

def test_orig_times_default_to_start_end():
    seg = Segment(index=0, start=10.0, end=20.0, text="あ", cluster="0:A")
    assert seg.orig_start == 10.0
    assert seg.orig_end == 20.0
    assert seg.time_edited is False
    # 明示指定したときはそちらが勝つ(分割で親の値を共有させるため)
    child = Segment(index=1, start=15.0, end=20.0, text="い", cluster="0:A",
                    orig_start=10.0, orig_end=20.0)
    assert (child.orig_start, child.orig_end) == (10.0, 20.0)


def test_load_schema2_file_fills_orig_times():
    """orig_* を持たない旧ファイルは既定値だけで読め、保存は v3 になる。"""
    old = {
        "schema": 2,
        "audio_path": "a.m4a",
        "duration": 100.0,
        "speakers": [{"id": "sp01", "name": "佐藤", "note": "", "order": 0}],
        "segments": [
            {"index": 0, "start": 10.0, "end": 20.0, "text": "あ", "cluster": "0:A",
             "chunk": 0, "speaker_id": "sp01", "reviewed": True, "note": "",
             "text_edited": False},
        ],
    }
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "old.speakers.json"
        p.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")

        proj = Project.load(p)
        seg = proj.segments[0]
        assert (seg.orig_start, seg.orig_end) == (10.0, 20.0)
        assert seg.time_edited is False

        # 時刻を直して保存 → schema 3 で書かれ、orig は元の時刻のまま残る
        seg.start, seg.end = 16.0, 26.0
        seg.time_edited = True
        proj.save(p)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["schema"] == SCHEMA_VERSION == 3

        again = Project.load(p)
        s2 = again.segments[0]
        assert (s2.start, s2.end) == (16.0, 26.0)
        assert (s2.orig_start, s2.orig_end) == (10.0, 20.0)
        assert s2.time_edited is True
        assert s2.reviewed is True and s2.speaker_id == "sp01"


# ======================================================================
# 区間の分割・結合
# ======================================================================

def _splittable() -> Project:
    proj = Project(audio_path="a.m4a", duration=300.0)
    proj.speakers = parse_roster("佐藤\n田中")
    proj.segments = [
        Segment(index=0, start=90.0, end=100.0, text="前の発言", cluster="0:A"),
        Segment(index=1, start=100.0, end=110.0, text="そうですねいいと思います",
                cluster="0:B", chunk=0, speaker_id="sp01", reviewed=True, note="めも"),
        Segment(index=2, start=110.0, end=120.0, text="次の発言", cluster="0:C"),
    ]
    return proj


def test_split_segment_basics():
    proj = _splittable()
    head, tail = proj.split_segment(1, 105.0, len("そうですね"))
    assert len(proj.segments) == 4
    assert [s.index for s in proj.segments] == [0, 1, 2, 3]     # 振り直される
    assert (head.start, head.end) == (100.0, 105.0)
    assert (tail.start, tail.end) == (105.0, 110.0)
    assert (head.text, tail.text) == ("そうですね", "いいと思います")
    # 前半は元の声のまとまりと話者を維持、後半は擬似不明で未確定
    assert head.cluster == "0:B" and head.speaker_id == "sp01"
    assert tail.cluster == "0:?" and tail.speaker_id is None
    assert tail.is_pseudo_cluster is True
    # 範囲が変わったので、どちらも聴き直し対象
    assert head.reviewed is False and tail.reviewed is False
    assert head.time_edited is True and tail.time_edited is True
    # 再実行時の系譜キーは親の値を両方が共有する
    assert (head.orig_start, head.orig_end) == (100.0, 110.0)
    assert (tail.orig_start, tail.orig_end) == (100.0, 110.0)
    # 隣の区間には触らない
    assert (proj.segments[0].start, proj.segments[3].start) == (90.0, 110.0)


def test_split_segment_clamps_boundary_and_cut():
    proj = _splittable()
    # 区間の外を指されても内側に収める(前後に最短の長さを残す)
    head, tail = proj.split_segment(1, 999.0, 999)
    assert head.end == 109.9 and tail.start == 109.9
    assert tail.text == ""                     # cut は本文の長さで頭打ち

    proj = _splittable()
    head, tail = proj.split_segment(1, 0.0, -5)
    assert head.end == 100.1 and head.text == ""


def test_split_segment_rejects_too_short():
    proj = _splittable()
    proj.segments[1].end = proj.segments[1].start + 0.15    # 2 つに割れない
    try:
        proj.split_segment(1, 100.05, 3)
    except ValueError:
        return
    raise AssertionError("短すぎる区間が分割できてしまった")


def test_merge_segments_basics():
    proj = _splittable()
    proj.segments[1].speaker_id = "sp01"
    proj.segments[2].speaker_id = "sp01"
    proj.segments[2].note = "あとのメモ"
    merged = proj.merge_segments(1)
    assert len(proj.segments) == 2
    assert [s.index for s in proj.segments] == [0, 1]
    assert (merged.start, merged.end) == (100.0, 120.0)
    assert merged.text == "そうですねいいと思います次の発言"   # 空白を挟まない
    assert merged.cluster == "0:B"                            # 前側を採用
    assert merged.speaker_id == "sp01"                        # 同じ話者なら維持
    assert merged.reviewed is False
    assert merged.time_edited is True
    assert merged.note == "めも / あとのメモ"
    # 系譜は前側の始まりと後側の終わり(再実行時の吸収に要る)
    assert (merged.orig_start, merged.orig_end) == (100.0, 120.0)


def test_merge_segments_drops_conflicting_speaker():
    proj = _splittable()
    proj.segments[1].speaker_id = "sp01"
    proj.segments[2].speaker_id = "sp02"
    merged = proj.merge_segments(1)
    assert merged.speaker_id is None      # どちらが正しいかは機械には決められない


def test_merge_segments_rejects_last():
    proj = _splittable()
    try:
        proj.merge_segments(len(proj.segments) - 1)
    except ValueError:
        return
    raise AssertionError("次の区間が無いのに結合できてしまった")


def test_split_then_merge_restores_shape():
    """分割の取り消しは結合で行う(Ctrl+Z の対象外)。"""
    proj = _splittable()
    before = proj.segments[1]
    span, text = (before.start, before.end), before.text
    proj.split_segment(1, 105.0, 5)
    proj.merge_segments(1)
    after = proj.segments[1]
    assert len(proj.segments) == 3
    assert (after.start, after.end) == span
    assert after.text == text
    assert (after.orig_start, after.orig_end) == span


# ======================================================================
# 再実行時の引き継ぎ(carry-over)
# ======================================================================

def _carry(old_segments, new_segments):
    """_carry_over_assignments を単体で回す。戻り値は (新しい区間リスト, ログ)"""
    from src.pipeline import _carry_over_assignments      # google-genai を引くので局所 import

    old = Project(audio_path="a.m4a")
    old.segments = old_segments
    logs: list[str] = []
    return _carry_over_assignments(old, new_segments, logs.append), logs


def test_carry_over_keeps_edited_times():
    """時刻を直した区間は orig_start で照合され、直した時刻のまま残る。"""
    old = [
        Segment(index=0, start=106.0, end=116.0, text="あ", cluster="0:A",
                speaker_id="sp01", reviewed=True, time_edited=True,
                orig_start=100.0, orig_end=110.0),
    ]
    new = [
        Segment(index=0, start=100.0, end=110.0, text="あ", cluster="0:A"),
        Segment(index=1, start=110.0, end=120.0, text="い", cluster="0:B"),
    ]
    result, _ = _carry(old, new)
    assert len(result) == 2
    assert (result[0].start, result[0].end) == (106.0, 116.0)
    assert result[0].time_edited is True
    assert result[0].speaker_id == "sp01" and result[0].reviewed is True
    assert result[1].start == 110.0        # 隣は再生成側のまま
    assert [s.index for s in result] == [0, 1]


def test_carry_over_replaces_with_split_family():
    """分割で 2 つになった区間が、再生成の 1 区間を丸ごと置き換える。"""
    old = [
        Segment(index=0, start=100.0, end=105.0, text="そうですね", cluster="0:A",
                speaker_id="sp01", reviewed=True, time_edited=True,
                orig_start=100.0, orig_end=110.0),
        Segment(index=1, start=105.0, end=110.0, text="いいと思います", cluster="0:?",
                time_edited=True, orig_start=100.0, orig_end=110.0),
    ]
    new = [
        Segment(index=0, start=100.0, end=110.0, text="そうですねいいと思います", cluster="0:A"),
        Segment(index=1, start=110.0, end=120.0, text="次の議題です", cluster="0:B"),
    ]
    result, _ = _carry(old, new)
    assert [s.text for s in result] == ["そうですね", "いいと思います", "次の議題です"]
    assert result[1].cluster == "0:?"      # 擬似クラスタが保たれる
    assert [s.index for s in result] == [0, 1, 2]


def test_carry_over_absorbs_merged_range():
    """結合した区間は、再生成で分かれて出てきたぶんを取り込む(区間が戻らない)。"""
    old = [
        Segment(index=0, start=100.0, end=110.0, text="そうですねいいと思います",
                cluster="0:A", speaker_id="sp01", time_edited=True,
                orig_start=100.0, orig_end=110.0),
    ]
    new = [
        Segment(index=0, start=100.0, end=105.0, text="そうですね", cluster="0:A"),
        Segment(index=1, start=105.0, end=110.0, text="いいと思います", cluster="0:A"),
        Segment(index=2, start=110.0, end=120.0, text="次の議題です", cluster="0:B"),
    ]
    result, logs = _carry(old, new)
    assert [s.text for s in result] == ["そうですねいいと思います", "次の議題です"]
    assert [s.index for s in result] == [0, 1]
    # 消したことをユーザーが気づけるようログに出す
    assert any("取り込みました" in m for m in logs)


def test_carry_over_legacy_behaviour_unchanged():
    """orig_* を持たない旧ファイル(= orig が start と同値)は従来どおり動く。"""
    old = [
        Segment(index=0, start=100.0, end=110.0, text="あ(手直し)", cluster="0:A",
                speaker_id="sp01", reviewed=True, note="めも", text_edited=True),
        Segment(index=1, start=110.0, end=120.0, text="い", cluster="0:B",
                speaker_id="sp02"),
    ]
    # 再実行で時刻がわずかに動いた(照合の許容 ±2 秒の内側)
    new = [
        Segment(index=0, start=101.0, end=111.0, text="あ", cluster="0:A"),
        Segment(index=1, start=111.0, end=121.0, text="い", cluster="0:B"),
    ]
    result, _ = _carry(old, new)
    assert len(result) == 2
    # 時刻には手が入っていないので、再生成側の時刻をそのまま使う
    assert (result[0].start, result[0].end) == (101.0, 111.0)
    # 人が入れた情報だけが移る
    assert result[0].speaker_id == "sp01" and result[0].reviewed is True
    assert result[0].text == "あ(手直し)" and result[0].text_edited is True
    assert result[0].note == "めも"
    assert result[1].speaker_id == "sp02"


def test_carry_over_absorbs_only_matched_families():
    """照合できなかった旧区間の範囲では吸収しない(穴が開くのを防ぐ)。"""
    old = [
        # 再生成側に対応する区間が無い(±2 秒に何も無い)結合済み区間
        Segment(index=0, start=300.0, end=330.0, text="消えた区間", cluster="0:A",
                speaker_id="sp01", time_edited=True, orig_start=300.0, orig_end=330.0),
    ]
    new = [
        Segment(index=0, start=310.0, end=320.0, text="残すべき区間", cluster="0:B"),
    ]
    result, _ = _carry(old, new)
    assert [s.text for s in result] == ["残すべき区間"]


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
