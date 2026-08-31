"""割当エディタのヘッドレス動作確認。

X サーバが必要なので、CI/Linux では xvfb-run 経由で実行する:

    xvfb-run -a python3.12 tests/test_gui_smoke.py

実際に AssignWindow を組み立て、キー操作相当の処理(割当・一括適用・
取り消し・絞り込み・保存・Word 出力)を呼んで例外が出ないことを確認する。
音声再生は行わない(ffplay も音声ファイルも不要)。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ======================================================================
# **利用者の設定に触れないようにする（src を読み込む前に）。**
#
# 差し替えの戻し忘れが 1 箇所あっただけで、実際に利用者の config.json が
# 上書きされ、API キーと名簿が消えた（2026-08-22）。個々の差し替えに頼らず、
# 設定の置き場そのものをテスト用に振り替える。`config_dir()` は毎回
# 環境変数を読むので、ここで変えれば全部の経路に効く。
# ======================================================================
import os                                                   # noqa: E402

REAL_APPDATA = os.environ.get("APPDATA", "")
_SANDBOX = tempfile.mkdtemp(prefix="whosaid-test-appdata-")
os.environ["APPDATA"] = _SANDBOX
os.environ["XDG_CONFIG_HOME"] = _SANDBOX        # Linux/CI 側も同じく


def real_config_path() -> Path | None:
    """本物の config.json（触っていないことを確かめるために使う）。"""
    if not REAL_APPDATA:
        return None
    return Path(REAL_APPDATA) / "WhosaidEditor" / "config.json"


# ======================================================================
# **押し出されて消えた部品を、幅の検査から見逃さない。**
#
# 幅の検査は winfo_ismapped() の部品だけを数えていた。説明文と候補の
# 選択肢は排他で同時には出ないので（設計書 §10.3）、隠れているものを
# 数えないこと自体は正しい。
#
# **だが pack は、入りきらない部品も unmap する。**そうなると要求幅が
# その部品のぶんだけ小さくなり、**押し出された結果として検査が通る**。
# 実測（1024x768・125% 表示）で［取り消し (Ctrl+Z)］が消え、6 個中 5 個で
# 「収まる」と出ていた。一番ひどい状態で OK が出ていたことになる。
#
# 見分けはつく。意図して隠した部品（pack_forget）は pack の管理下から
# 外れるが、入りきらずに消えた部品は管理下に残ったまま unmap される。
# 実測で確認した（pack_slaves に残り、ismapped は False）。
#
# ----------------------------------------------------------------------
# **画素数は機械ごとに違う。CI の数字を手元の数字と比べないこと。**
#
# 同じ 1024x768・同じ拡大率でも、右ペインの幅が CI では 535px、開発機では
# 620px だった（2026-08-29 に実測）。字形とテーマがランナーと手元で違う
# ため。**「手元で 30px 余っているから CI も通る」とは言えない。**
#
# だから検査は絶対値を書かず、その場で測った 要 と 幅 を比べる形にして
# ある。落ちたときにどちらがどれだけ足りなかったかがログに出るのは、
# 手元で再現できない差をログだけで追うため。
# ----------------------------------------------------------------------
# ======================================================================

def squeezed_out(parent) -> list:
    """入りきらずに消えた部品。意図して隠したものは含まない。"""
    try:
        managed = list(parent.pack_slaves())
    except Exception:                       # pack 以外で並べている行
        managed = []
    return [c for c in managed if not c.winfo_ismapped()]


def visible_need(parent, pad: int = 0) -> tuple[int, list]:
    """その行に要る幅と、押し出されて消えた部品。

    幅は**見えている部品**から出す（排他の部品を二重に数えないため）。
    消えた部品は呼び出し側が別に落とす——ここで幅に足すと「閾値の緩和の
    裏返し」になり、何が起きたのかが読めなくなる。
    """
    lost = squeezed_out(parent)
    need = sum(c.winfo_reqwidth() for c in parent.winfo_children()
               if c.winfo_ismapped()) + pad
    return need, lost


def lost_names(lost: list) -> str:
    """消えた部品を人が読める形に。"""
    out = []
    for c in lost:
        try:
            out.append(str(c.cget("text")) or c.winfo_class())
        except Exception:
            # 文字列を持たない部品や、すでに壊した窓。**名前が出せなくても
            # 落とさない**——ここで例外を投げると、検査の失敗そのものが
            # 読めない traceback に化ける。
            out.append(str(c))
    return " / ".join(out)


_REAL_CONFIG_STAMP = (real_config_path().stat().st_mtime
                      if real_config_path() and real_config_path().exists()
                      else None)


def run_user_config_untouched() -> int:
    """**本物の設定を書き換えていないこと。**

    実害が出た欠陥なので、毎回確かめる。ここが落ちたら、どこかの
    差し替えが漏れている。
    """
    print("\n[利用者の設定]")
    p = real_config_path()
    if p is None or _REAL_CONFIG_STAMP is None:
        print("  ok   本物の設定が無いので確かめようがない（素通り）")
        return 0
    now = p.stat().st_mtime if p.exists() else None
    ok = now == _REAL_CONFIG_STAMP
    print(f"  {'ok  ' if ok else 'FAIL'} 本物の config.json を書き換えていない")
    return 0 if ok else 1


# Windows の既定コンソールは cp932 / cp1252 なので、日本語のラベルを print した
# 時点で UnicodeEncodeError になる(テストの中身とは無関係に落ちる)。
# 出力先のエンコーディングをここで固定しておく。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


import tkinter as tk  # noqa: E402

import src.assign_gui as assign_gui  # noqa: E402
from src.assign_gui import (  # noqa: E402
    FILTER_ALL,
    FILTER_UNASSIGNED,
    FILTER_UNREVIEWED,
    PREVIEW_FLOOR_SECONDS,
    PREVIEW_MAX_SECONDS,
    PREVIEW_MIN_SECONDS,
    AssignWindow,
    ProposalDialog,
    ReplaceWordsDialog,
    ReplaceSpeakerDialog,
    fmt_short_time,
    RosterDialog,
    ProposalRow,
    SplitDialog,
    clamp_times,
    move_edge,
    plan_roster_rows,
    plan_roster_text,
    shift_span,
    playback_window,
    preview_length,
    tail_window,
    time_edit_base,
)
from src.align import Word as AlignWord  # noqa: E402
from src.inspection import (  # noqa: E402
    Proposal,
    inspect_times,
    load_proposals,
    merge_history,
    proposals_path,
    save_proposals,
)
from src.segments import (  # noqa: E402
    Project,
    SPECIAL_UNKNOWN,
    Segment,
    fmt_hms_frac,
    parse_roster,
)


def make_project(tmp: Path) -> Project:
    proj = Project(audio_path=str(tmp / "meeting.m4a"), duration=1200.0, chunk_seconds=600)
    proj.speakers = parse_roster("佐藤(理事長)\n田中(事務局長)\n鈴木")
    segs = []
    for i in range(40):
        chunk = i // 20
        cluster = f"{chunk}:{'ABC'[i % 3]}"
        segs.append(Segment(
            index=i, start=i * 30.0, end=i * 30.0 + 28,
            text=f"これは {i} 番目の発言です。", cluster=cluster, chunk=chunk,
        ))
    proj.segments = segs
    proj.json_path = str(tmp / "meeting.speakers.json")
    proj.save()
    return proj


def install_dialog_guard() -> list:
    """**差し替え忘れたモーダルで止まらないようにする。**

    tkinter の messagebox / filedialog は人が押すまで返らない。検査の中で
    出ると、そこで止まって**以降が一度も走らない**。しかも出力はバッファ
    されるので、止まった位置すら分からない。

    実際に 2 回起きた(2026-08-28):
      - 一括適用で全区間が埋まり showinfo(完了) が出た
      - 出力の最後に askyesno(出力完了 / ファイルを開きますか?) が出た

    そこで既定を「記録して、いちばん害のない値を返す」に置き換える。
    - 問い合わせ(askyesno 等) → False。**「いいえ」が最も害がない**
      (True を返すと、出力の後に Word が本当に起動する)
    - 選択(asksaveasfilename 等) → 空文字。取り消しと同じ
    - 通知(showinfo 等) → None

    個別に差し替えている箇所はこれまでどおりそちらが勝つ(あとから代入し、
    finally で戻すときも同じ関数に戻る)。差し替え忘れは止まらずに進み、
    末尾に一覧が出るので、そこで気づける。
    """
    leaked: list = []

    def record(name, ret):
        def f(*a, **k):
            leaked.append((name, a[0] if a else ""))
            return ret
        return f

    for name, ret in (("showinfo", None), ("showwarning", None),
                      ("showerror", None), ("askyesno", False),
                      ("askokcancel", False), ("askretrycancel", False)):
        setattr(assign_gui.messagebox, name, record(name, ret))
    for name in ("asksaveasfilename", "askopenfilename", "askdirectory"):
        setattr(assign_gui.filedialog, name, record(name, ""))
    return leaked


def run() -> int:
    failures = []
    leaked = install_dialog_guard()

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        if not cond:
            failures.append(label)

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        proj = make_project(tmp)

        root = tk.Tk()
        root.withdraw()
        win = AssignWindow(root, proj)
        win.var_autoplay.set(False)      # 実音声を用意しないので自動再生は切る
        win.var_apply_cluster.set(False)  # 個別確定から検証する
        win.update()

        # --- 再生する長さ -------------------------------------------------
        check("短い相づちは区間が長くても再生を切り上げる",
              preview_length("うん。", 26.0) == PREVIEW_MIN_SECONDS)
        check("極端に短い区間でも聞き取れる長さは鳴らす",
              preview_length("うん。", 0.4) == PREVIEW_FLOOR_SECONDS)
        check("区間が下限より長ければその長さで鳴らす",
              preview_length("うん。", 3.0) == 3.0)
        check("長い発言は本文に見合う長さを再生する",
              10.0 < preview_length("あ" * 60, 60.0) < 30.0)
        check("極端に長い区間でも上限で止まる",
              preview_length("あ" * 3000, 900.0) == PREVIEW_MAX_SECONDS)

        # --- ずれ補正 -------------------------------------------------------
        check("ずれ補正の初期値は 0", proj.time_offset == 0.0)
        win._set_offset(1.4)
        win.update()
        check("ずれ補正を設定できる", proj.time_offset == 1.4)
        check("ずれ補正が保存対象になる", win._dirty is True)
        win._nudge_offset(-0.2)
        check("キーで微調整できる", abs(proj.time_offset - 1.2) < 1e-6)
        win._set_offset(99.0)
        check("上限で頭打ちになる", proj.time_offset == 10.0)
        win._set_offset(0.0)
        check("0 に戻せる", proj.time_offset == 0.0)

        # --- 区間の時刻を直す(範囲の計算) --------------------------------
        plain = Segment(index=0, start=100.0, end=110.0, text="あ" * 50, cluster="0:A")
        edited = Segment(index=0, start=100.0, end=110.0, text="あ" * 50, cluster="0:A",
                         time_edited=True)
        check("未編集の区間は入力欄にずれ補正込みの値を出す",
              time_edit_base(plain, 4.0) == (104.0, 114.0))
        check("直した区間はそのままの値を出す",
              time_edit_base(edited, 4.0) == (100.0, 110.0))
        check("直した区間は再生にずれ補正を足さない",
              playback_window(edited, 4.0) == (100.0, 110.0))
        check("直した区間は本文の長さで切らず終了時刻まで鳴らす",
              playback_window(edited, 4.0)[1] == edited.end)
        check("直していない区間はずれ補正が効く",
              playback_window(plain, 4.0)[0] == 104.0)
        check("直していない区間は本文に見合う長さで切る",
              playback_window(plain, 4.0)[1]
              == 100.0 + preview_length(plain.text, plain.duration) + 4.0)
        check("この先30秒は区間の終わりから延ばす",
              playback_window(plain, 4.0, extend=30.0)[1] == 110.0 + 4.0 + 30.0)
        check("5秒前からはさかのぼる", playback_window(plain, 0.0, back=5.0)[0] == 95.0)

        check("開始が終了を追い越さない",
              clamp_times(120.0, 110.0, 1200.0, "start") == (109.9, 110.0))
        check("終了が開始を下回らない",
              clamp_times(100.0, 90.0, 1200.0, "end") == (100.0, 100.1))
        check("負の開始は 0 で止まる", clamp_times(-5.0, 10.0, 1200.0, "start") == (0.0, 10.0))
        check("音声の長さを超えない",
              clamp_times(100.0, 9999.0, 1200.0, "end") == (100.0, 1200.0))

        # 開始・終了は片側だけ動かす(長さが変わる = 境界の微調整)
        check("開始を後ろへ詰めると区間が縮む",
              move_edge(100.0, 110.0, "start", 106.8, 1200.0) == (106.8, 110.0))
        check("開始を前へ出すと区間が伸びる",
              move_edge(100.0, 110.0, "start", 97.0, 1200.0) == (97.0, 110.0))
        check("終了を前へ詰めると区間が縮む",
              move_edge(100.0, 110.0, "end", 104.0, 1200.0) == (100.0, 104.0))
        check("終了を後ろへ伸ばすと区間が伸びる",
              move_edge(100.0, 110.0, "end", 115.0, 1200.0) == (100.0, 115.0))
        # ナッジは追い越さずに突き当たりで止まる(短い区間でも幅を調整できる)
        check("短い区間でもナッジは片側だけ動く",
              move_edge(100.0, 100.7, "start", 101.0, 1200.0) == (100.6, 100.7))
        check("短い区間でも終了のナッジは片側だけ動く",
              move_edge(100.0, 100.7, "end", 99.7, 1200.0) == (100.0, 100.1))
        # 入力欄に直接打たれた時刻はそのまま置く(区間ごとそこへ移す)
        check("追い越す時刻を打てば区間ごとそこへ移る",
              move_edge(100.0, 101.1, "start", 106.8, 1200.0, shift_if_past=True)
              == (106.8, 107.9))
        check("下回る時刻を打っても区間ごと移る",
              move_edge(100.0, 101.1, "end", 90.0, 1200.0, shift_if_past=True)
              == (88.9, 90.0))

        # 「区間ごと」ボタンは長さを保ったまま前後にずらす。
        # 相づちのような短い区間ほど、長さより大きく動かす必要がある
        check("区間ごと後ろへずらす", shift_span(100.0, 101.1, +6.8, 1200.0) == (106.8, 107.9))
        check("区間ごと前へずらす", shift_span(100.0, 110.0, -1.0, 1200.0) == (99.0, 109.0))
        check("長い区間でも長さは変わらない",
              shift_span(100.0, 130.0, +6.0, 1200.0) == (106.0, 136.0))
        check("音声の先頭より前には出ない", shift_span(0.5, 1.6, -9.0, 1200.0) == (0.0, 1.1))
        check("音声の末尾を超えない", shift_span(1195.0, 1196.1, +9.0, 1200.0) == (1198.9, 1200.0))

        # --- 区間の時刻を直す(画面操作) ----------------------------------
        keep_current = win.current
        win.goto(3)
        seg3 = proj.segments[3]
        orig3 = (seg3.start, seg3.end)                  # (90.0, 118.0)
        check("入力欄に元の時刻が出る", win.var_start.get() == "00:01:30.0")

        win.var_start.set("00:01:36.5")
        win._commit_time("start")
        check("入力した開始時刻が入る", seg3.start == 96.5)
        check("時刻を直した印が付く", seg3.time_edited is True)
        check("終了は動かない(区間が縮む)", seg3.end == orig3[1])
        check("保存対象になる", win._dirty is True)

        win._nudge_time("end", +1.0)
        check("終了をナッジできる", abs(seg3.end - (orig3[1] + 1.0)) < 1e-6)
        check("そのとき開始は動かない", abs(seg3.start - 96.5) < 1e-6)
        check("一覧に ✎ が出る", win.tree.item("s3", "values")[0].startswith("✎"))
        check("画面で直した時刻は確認済みになる", seg3.time_reviewed is True)
        check("確認済みは ✎ で、✎△ にはしない",
              win.tree.item("s3", "values")[0].startswith("✎ "))
        check("区間ヘッダにも確認済みと出る", "✎時刻を修正済み" in win.var_seginfo.get())

        # 終了を直したら、頭からではなく終わりだけを鳴らす(長い区間で待たない)
        lo, hi = tail_window(seg3, proj.time_offset)
        check("終わりの再生は終了時刻で終わる", abs(hi - seg3.end) < 1e-6)
        check("終わりの再生は 3 秒", abs((hi - lo) - 3.0) < 1e-6)
        short = Segment(index=0, start=10.0, end=11.2, text="はい。", cluster="0:A")
        s_lo, s_hi = tail_window(short, 0.0)
        check("短い区間は区間の長さまで", abs((s_hi - s_lo) - 1.2) < 1e-6)
        off = Segment(index=0, start=10.0, end=30.0, text="あ", cluster="0:A")
        o_lo, o_hi = tail_window(off, 4.0)
        check("ずれ補正込みの終了から数える", abs(o_hi - 34.0) < 1e-6)

        win.var_start.set("ここは時刻を書く欄")
        win._commit_time("start")
        check("読めない入力は入力欄を元に戻す", win.var_start.get() == "00:01:36.5")
        check("読めない入力では区間を変えない", seg3.start == 96.5)

        win.revert_time()
        check("元の時刻に戻せる", (seg3.start, seg3.end) == orig3)
        check("印が外れる", seg3.time_edited is False)
        check("元の時刻は保持されている", (seg3.orig_start, seg3.orig_end) == orig3)
        check("一覧の ✎ も消える", not win.tree.item("s3", "values")[0].startswith("✎"))
        check("確認済みの印も外れる", seg3.time_reviewed is False)

        # --- 時刻は入れたが自分の耳では未確認(✎△) --------------------------
        # 機械が出した時刻を当てただけの状態。いまこれを作る操作は無いので直に
        # 組み立てる(点検の提案をまとめて適用したときにこの状態になる)。
        seg3.start, seg3.end = orig3[0] + 6.0, orig3[1] + 6.0
        seg3.time_edited, seg3.time_reviewed = True, False
        win._update_row(3)
        win.goto(3)
        check("未確認の時刻は一覧で ✎△",
              win.tree.item("s3", "values")[0].startswith("✎△"))
        check("区間ヘッダにも未確認と出る",
              "✎△時刻は推定(未確認)" in win.var_seginfo.get())
        win._shift_time(+0.1)
        check("画面で直せば確認済みに変わる", seg3.time_reviewed is True)
        check("一覧の印も ✎ に変わる",
              win.tree.item("s3", "values")[0].startswith("✎ "))
        win.revert_time()                                   # 後片付け
        check("元に戻すと時刻も印も消える",
              (seg3.start, seg3.end) == orig3 and seg3.time_reviewed is False)

        # --- 聴いて確かめた印を上げる([この時刻で確認]) --------------------
        seg3.start, seg3.end = orig3[0] + 6.0, orig3[1] + 6.0
        seg3.time_edited, seg3.time_reviewed = True, False
        win.goto(3)
        check("未確認なら押せる", "disabled" not in win.btn_confirm_time.state())
        before = (seg3.start, seg3.end)
        win.confirm_time()
        check("時刻の値は変わらない", (seg3.start, seg3.end) == before)
        check("確認済みになる", seg3.time_reviewed is True)
        check("一覧の印が ✎ に変わる",
              win.tree.item("s3", "values")[0].startswith("✎ "))
        check("確認済みなら押せない", "disabled" in win.btn_confirm_time.state())
        win.revert_time()

        # まだ直していない区間でも「聴いた、この時刻でいい」と言える
        win._set_offset(4.0)
        win.goto(7)
        seg7 = proj.segments[7]
        win.confirm_time()
        check("補正込みの時刻がそのまま確定する",
              abs(seg7.start - (seg7.orig_start + 4.0)) < 1e-6)
        check("確認済みとして入る", seg7.time_reviewed is True)
        win.revert_time()
        win._set_offset(0.0)

        # --- 点検の提案を当てる --------------------------------------------
        win.goto(9)
        seg9 = proj.segments[9]
        orig9 = (seg9.start, seg9.end)
        prop = Proposal(id="p1", type="time",
                        target_orig_start=float(seg9.orig_start),
                        payload={"start": orig9[0] + 6.8, "end": orig9[1] + 6.8},
                        evidence="一致 22/26 文字", confidence=0.85)
        check("まとめて適用は未確認のまま入る",
              win.apply_proposal(prop, reviewed=False) is True
              and seg9.time_edited is True and seg9.time_reviewed is False)
        check("提案の時刻が入る", abs(seg9.start - (orig9[0] + 6.8)) < 0.15)
        check("一覧では ✎△", win.tree.item("s9", "values")[0].startswith("✎△"))
        # 聴いて承認したぶんは ✎ になる
        check("聴いて承認は確認済みで入る",
              win.apply_proposal(prop, reviewed=True) is True
              and seg9.time_reviewed is True)
        check("一覧の印も ✎", win.tree.item("s9", "values")[0].startswith("✎ "))

        # 隣と重なる提案は接点で切り詰める(同時に当てても重ならない)
        seg10_start = proj.segments[10].start
        over = Proposal(id="p2", type="time",
                        target_orig_start=float(seg9.orig_start),
                        payload={"start": orig9[0], "end": seg10_start + 30.0},
                        evidence="", confidence=0.9)
        win.apply_proposal(over, reviewed=False)
        check("隣を侵さない", seg9.end <= seg10_start + 1e-6)

        # 指し先が見つからない提案は当てない
        gone = Proposal(id="p3", type="time", target_orig_start=99999.0,
                        payload={"start": 1.0, "end": 2.0}, evidence="",
                        confidence=1.0)
        check("対象が無ければ当てない", win.apply_proposal(gone, reviewed=True) is False)

        # --- まとめて適用は順序に依存しない ---------------------------------
        # ドリフト帯: 隣り合う 4 区間が全部 +7 秒ずれている。1 件ずつ前から
        # 当てると、まだ動いていない隣の古い位置に切り詰められて潰れる。
        # 行き先を先に全部解いてから書くので、何件でも同時に動かせる。
        band = []
        for i in (16, 17, 18, 19):
            s = proj.segments[i]
            band.append(Proposal(
                id=f"b{i}", type="time", target_orig_start=float(s.orig_start),
                payload={"start": float(s.orig_start) + 7.0,
                         "end": float(s.orig_end) + 7.0},
                evidence="", confidence=0.9))
        ok, failed = win.apply_proposals_bulk(band)
        check("ドリフト帯を全件まとめて当てられる",
              len(ok) == 4 and not failed)
        check("全件が提案どおりの位置に入る",
              all(abs(proj.segments[i].start - (proj.segments[i].orig_start + 7.0))
                  < 1e-6 for i in (16, 17, 18, 19)))
        check("まとめて適用は全件 ✎△",
              all(proj.segments[i].time_edited and not proj.segments[i].time_reviewed
                  for i in (16, 17, 18, 19)))
        check("承認済みとして記録される",
              all(p.status == "accepted" for p in band))
        # 隣どうしは重ならない(接点で切り詰め済み)
        check("適用後も隣と重ならない",
              all(proj.segments[i].end <= proj.segments[i + 1].start + 1e-6
                  for i in (16, 17, 18)))

        # --- 一括適用をまとめて元に戻す -------------------------------------
        check("一括適用の直後は取り消せる",
              "disabled" not in win.btn_undo_bulk.state())
        win.undo_bulk_times()
        check("全件が適用前の時刻に戻る",
              all(abs(proj.segments[i].start - proj.segments[i].orig_start) < 1e-6
                  for i in (16, 17, 18, 19)))
        check("印も適用前に戻る",
              all(not proj.segments[i].time_edited
                  and not proj.segments[i].time_reviewed
                  for i in (16, 17, 18, 19)))
        check("二度は取り消せない",
              "disabled" in win.btn_undo_bulk.state())
        # 分割・結合で並びが変わったら、index で戻すのは危ないので拒む
        win.apply_proposals_bulk(band)
        total_before = len(proj.segments)
        proj.split_segment(30, proj.segments[30].start + 2.0, 3)
        real_ask2 = assign_gui.messagebox.showinfo
        told: list = []
        assign_gui.messagebox.showinfo = lambda *a, **k: told.append(a)
        try:
            win.undo_bulk_times()
        finally:
            assign_gui.messagebox.showinfo = real_ask2
        check("並びが変わったら取り消さずに知らせる",
              bool(told) and abs(proj.segments[16].start
                                 - (proj.segments[16].orig_start + 7.0)) < 1e-6)
        proj.merge_segments(30)                             # 後片付け
        check("区間の数が戻る", len(proj.segments) == total_before)
        s30 = proj.segments[30]                 # 結合は time_edited を立てる
        s30.start, s30.end = float(s30.orig_start), float(s30.orig_end)
        s30.time_edited = s30.time_reviewed = False
        win.reload_tree()
        # band(区間 16〜19)は +7 秒のまま次の検査へ渡す

        # 人が耳で確定した区間には、まとめて適用でも触れない
        seg16 = proj.segments[16]
        seg16.time_reviewed = True
        again = Proposal(id="b16x", type="time",
                         target_orig_start=float(seg16.orig_start),
                         payload={"start": float(seg16.orig_start) + 9.0,
                                  "end": float(seg16.orig_end) + 9.0},
                         evidence="", confidence=0.9)
        ok, failed = win.apply_proposals_bulk([again])
        check("確定済み(✎)はまとめて適用でも動かない",
              not ok and len(failed) == 1
              and abs(seg16.start - (seg16.orig_start + 7.0)) < 1e-6)
        for i in (16, 17, 18, 19):                          # 後片付け
            s = proj.segments[i]
            s.start, s.end = float(s.orig_start), float(s.orig_end)
            s.time_edited = s.time_reviewed = False
        win.reload_tree()
        for i in (9,):                                      # 後片付け
            s = proj.segments[i]
            s.start, s.end = s.orig_start, s.orig_end
            s.time_edited = s.time_reviewed = False
        win.reload_tree()

        # 実データで詰まった 1.1 秒の相づちを 6.8 秒ずらす、と同じ形。
        # 「区間ごと」ボタンなら、長さより大きく動かしても潰れない
        win.goto(8)
        short = proj.segments[8]
        short.start, short.end = 240.0, 241.1
        short.orig_start, short.orig_end = 240.0, 241.1
        win.show_current()
        for _ in range(6):
            win._shift_time(+1.0)
        win._shift_time(+0.8)
        check("短い区間を長さ以上にずらせる", abs(short.start - 246.8) < 1e-6)
        check("ずらしても長さは変わらない", abs(short.duration - 1.1) < 1e-6)
        check("入力欄にも反映される", win.var_start.get() == "00:04:06.8")
        # ナッジは片側だけ動かす。短い区間で大きく押しても、区間ごとスライド
        # せずに突き当たりで止まる(ここを間違えると幅を調整できなくなる)
        win._nudge_time("start", +0.1)
        check("開始のナッジは区間を縮める", abs(short.duration - 1.0) < 1e-6)
        win._nudge_time("start", +1.0)
        check("大きく押しても区間ごとスライドしない", abs(short.end - 247.9) < 1e-6)
        check("突き当たりで止まる", abs(short.start - 247.8) < 1e-6)
        win._nudge_time("end", +1.0)
        check("終了を伸ばして幅を戻せる", abs(short.duration - 1.1) < 1e-6)
        short.start, short.end = 240.0, 268.0          # 後片付け
        short.orig_start, short.orig_end = 240.0, 268.0
        short.time_edited = False

        # 通り過ぎただけ(フォーカスが外れただけ)では修正済みにしない
        win.goto(6)
        seg6 = proj.segments[6]
        win._commit_time("start")
        check("通り過ぎただけでは修正済みにしない", seg6.time_edited is False)

        # ずれ補正込みの初期値を Enter でそのまま確定できる
        win._set_offset(4.0)
        win.goto(5)
        seg5 = proj.segments[5]                          # start=150.0
        check("未編集の区間はずれ補正込みで表示する", win.var_start.get() == "00:02:34.0")
        win._commit_time("start", explicit=True)
        check("そのまま確定すると補正込みの時刻が固定される", seg5.start == 154.0)
        check("固定した区間は以後ずれ補正の影響を受けない",
              playback_window(seg5, proj.time_offset)[0] == 154.0)
        win.revert_time()
        win._set_offset(0.0)
        check("後片付け: 時刻の修正が残っていない",
              not any(s.time_edited for s in proj.segments))
        win.goto(keep_current)          # 以降のテストは選択位置を引き継ぐ

        check("初期表示: 区間 40 件", len(win.tree.get_children()) == 40)
        check("初期表示: 候補は名簿順", [c.speaker.name for c in win._candidates][:3]
              == ["佐藤", "田中", "鈴木"])

        # --- 1 件確定 ---------------------------------------------------
        sato = proj.speakers[0].id
        win.assign(sato)
        win.update()
        check("確定後に 1 件カウント", proj.assigned_count == 1)
        check("自分で聴いた区間は確認済み", proj.segments[0].reviewed is True)
        check("確定したら次の未確定へ進む", win.current == 1)

        # --- 学習: 同じクラスタで佐藤が 1 位に -------------------------
        win.goto(3)          # index 3 は index 0 と同じクラスタ (0:A)
        win.update()
        check("同一クラスタで学習が効く", win._candidates[0].speaker.id == sato)
        check("根拠が表示される", "同じ声のまとまり" in win._candidates[0].reason_text)

        # --- クラスタ一括適用 -------------------------------------------
        win.var_apply_cluster.set(True)
        win.goto(3)
        win.assign(sato)
        win.update()
        cluster_size = len(proj.cluster_segments("0:A"))
        check("クラスタ一括適用", proj.assigned_count == cluster_size)
        check("一括適用ぶんは未確認のまま",
              proj.unreviewed_count == cluster_size - 2)   # index 0 と 3 は聴いて確定
        check("一括適用の件数が画面に残る", "まとめて適用" in win.var_action.get())
        win.var_apply_cluster.set(False)

        # --- 取り消し ----------------------------------------------------
        before = proj.assigned_count
        win.undo()
        win.update()
        check("取り消しで元に戻る", proj.assigned_count == before - (cluster_size - 1))

        # --- 特別ラベル --------------------------------------------------
        win.goto(1)
        win.assign(SPECIAL_UNKNOWN)
        win.update()
        check("不明ラベルを割り当てられる",
              proj.segments[1].speaker_id == SPECIAL_UNKNOWN)

        # --- 未確定に戻す ------------------------------------------------
        win.goto(1)
        win.unassign()
        win.update()
        check("未確定に戻せる", proj.segments[1].speaker_id is None)

        # --- 絞り込み ----------------------------------------------------
        win.var_filter.set(FILTER_UNASSIGNED)
        win._on_filter_change()
        win.update()
        check("未確定のみ表示",
              len(win.tree.get_children()) == proj.total_count - proj.assigned_count)

        win.var_apply_cluster.set(True)
        win.var_filter.set(FILTER_ALL)
        win._on_filter_change()
        win.goto(1)
        win.assign(proj.speakers[1].id)     # 0:B を一括で埋める
        win.update()
        win.var_filter.set(FILTER_UNREVIEWED)
        win._on_filter_change()
        win.update()
        check("未確認のみ表示に一括適用ぶんが出る",
              len(win.tree.get_children()) == proj.total_count - proj.reviewed_count)
        win.var_apply_cluster.set(False)
        win.var_filter.set(FILTER_ALL)
        win._on_filter_change()
        win.update()

        # --- 絞り込み中の 1 件移動が飛ばさない ---------------------------
        win.var_filter.set(FILTER_UNASSIGNED)
        win._on_filter_change()
        vis = win._visible_indexes()
        win.current = -1                    # わざと表示外に置く
        win.move(1)
        check("絞り込み中でも 1 件ずつ進む", win.current == vis[0])
        win.var_filter.set(FILTER_ALL)
        win._on_filter_change()
        win.update()

        # --- 本文の編集 --------------------------------------------------
        win.goto(5)
        win.txt_body.delete("1.0", "end")
        win.txt_body.insert("1.0", "編集しました")
        win._commit_text()
        check("本文編集が反映される", proj.segments[5].text == "編集しました")
        check("編集した印が付く", proj.segments[5].text_edited is True)

        # --- 出席者の編集(下見してから適用) -----------------------------
        plan = plan_roster_text(proj, "佐藤(理事長)\n田中(事務局長)\n鈴木\n高橋")
        check("出席者追加の下見", plan.added == ["高橋"] and not plan.removed)
        plan.apply(proj)
        check("既存 ID が保持される", proj.speakers[0].id == sato)
        check("確定済みの割当が壊れない", proj.segments[0].speaker_id == sato)

        # 下見だけでは何も変わらないこと(『いいえ』を選んだ場合に相当)
        snapshot = [(s.speaker_id, s.reviewed) for s in proj.segments]
        plan = plan_roster_text(proj, "田中(事務局長)\n鈴木")
        check("下見の時点では割当が消えない",
              [(s.speaker_id, s.reviewed) for s in proj.segments] == snapshot)
        check("削除の影響件数を数えられる", plan.affected_segments > 0)
        plan.apply(proj)
        check("適用すると割当が外れる", proj.segments[0].speaker_id is None)

        # --- 行が ID を持ち回る(設計書 §11.8)-----------------------------
        # **名前の一致で引き継ぐと、名前を直しただけで割当が外れる。**
        # 「山本学　文科省…室長」を「山本学」に縮めたときに起きる。
        rp = Project(audio_path="r.m4a")
        rp.speakers = parse_roster(chr(10).join(
            ["山本学　文科省　高等教育局私学部参事官付　企画官", "西村香介"]))
        yamamoto, nishimura = rp.speakers[0].id, rp.speakers[1].id
        rp.segments = [Segment(index=i, start=i, end=i + 1, text="a",
                               cluster="0:A") for i in range(3)]
        rp.segments[0].speaker_id = yamamoto
        rp.segments[0].reviewed = True
        rp.segments[1].speaker_id = nishimura

        # 名前を縮めて、役職を右の欄へ移す
        pl = plan_roster_rows(rp, [
            (yamamoto, "山本学", "文科省 高等教育局私学部参事官付 企画官"),
            (nishimura, "西村香介", ""),
        ])
        check("名前を直しても誰も消えない", pl.removed == [] and pl.added == [])
        check("名前を直しても割当は無傷", pl.affected_segments == 0)
        pl.apply(rp)
        check("ID が保たれる", rp.speakers[0].id == yamamoto)
        check("確定済みの割当が残る",
              rp.segments[0].speaker_id == yamamoto
              and rp.segments[0].reviewed is True)
        check("名前が短くなっている", rp.speakers[0].name == "山本学")
        check("役職は別に持つ", rp.speakers[0].note.startswith("文科省"))

        # **名前の一致だと壊れる**ことを、縮める前の器で示しておく。
        # この道を選び直したら、この検査が落ちて気付ける。
        was = Project(audio_path="w.m4a")
        was.speakers = parse_roster(
            "山本学　文科省　高等教育局私学部参事官付　企画官")
        was.segments = [Segment(index=0, start=0, end=1, text="a",
                                cluster="0:A")]
        was.segments[0].speaker_id = was.speakers[0].id
        was.segments[0].reviewed = True
        broken = plan_roster_text(was, "山本学(文科省 高等教育局)")
        check("名前一致だと、名前を直した人が消える扱いになる",
              [sp.id for sp in broken.removed] == [was.speakers[0].id])
        check("名前一致だと、確定済みの割当が巻き添えになる",
              broken.affected_segments == 1)
        # 同じ入力でも、行が ID を持てば無傷
        safe = plan_roster_rows(
            was, [(was.speakers[0].id, "山本学", "文科省 高等教育局")])
        check("行が ID を持てば、同じ直しでも無傷",
              safe.removed == [] and safe.affected_segments == 0)

        # 並べ替えても ID は行についてくる
        pl2 = plan_roster_rows(rp, [
            (nishimura, "西村香介", ""),
            (yamamoto, "山本学", "文科省"),
        ])
        pl2.apply(rp)
        check("並べ替えても割当は無傷",
              rp.segments[0].speaker_id == yamamoto
              and rp.segments[1].speaker_id == nishimura)
        check("並び順は入れ替わる", rp.speakers[0].id == nishimura)

        # 新しい行は ID を空にする / 渡さなかった人は削除
        pl3 = plan_roster_rows(rp, [(nishimura, "西村香介", ""), ("", "新人", "")])
        check("新しい行に ID が振られる", pl3.added == ["新人"])
        check("渡さなかった人は削除になる",
              [sp.id for sp in pl3.removed] == [yamamoto])
        check("削除の影響件数を数える", pl3.affected_segments == 1)
        pl3.apply(rp)
        check("削除された人の割当は外れる", rp.segments[0].speaker_id is None)
        check("関係ない人の割当は残る", rp.segments[1].speaker_id == nishimura)
        check("新しい ID は既存とぶつからない",
              len({sp.id for sp in rp.speakers}) == len(rp.speakers))

        # 空の名前の行は無視する(表に空行が残っていても落ちない)
        pl4 = plan_roster_rows(rp, [(nishimura, "西村香介", ""), ("", "  ", "")])
        check("空の行は人として数えない", len(pl4.speakers) == 1)

        # --- 時刻欄の事故を防ぐ（2026-08-19 の実機の指摘）-----------------
        # ［時刻へ飛ぶ］のつもりで 00:35:09 を［開始］に打ち、Enter を
        # 押していないのに記録へ入って区間が 2125 秒に膨らんだ。
        tp = Project(audio_path=str(tmp / "meeting.m4a"),
                     duration=3600.0, chunk_seconds=600)
        tp.speakers = parse_roster("佐藤")
        tp.segments = [
            Segment(index=i, start=i * 20.0, end=i * 20.0 + 15.8,
                    text=f"発言 {i}。", cluster="g:A", chunk=0)
            for i in range(6)]
        tp.json_path = str(tmp / "time.speakers.json")
        tp.save()
        twin = AssignWindow(root, tp)
        twin.var_autoplay.set(False)
        twin.update()

        # ウ: 欄から離れただけでは書かない
        twin.goto(0)
        twin.var_start.set("00:35:09")
        twin._discard_time_edit("start")
        check("**欄から離れただけでは書かない**", tp.segments[0].start == 0.0)
        check("捨てたことを黙らない", "Enter" in twin.var_action.get())
        check("表示は元に戻る", twin.var_start.get().startswith("00:00:00"))

        # イ: どの区間の欄かを見張る（本文欄と同じ守り）
        twin.goto(0)
        twin.var_end.set("00:00:18.0")
        twin._time_index = 3                 # 別の区間の欄だったことにする
        twin._commit_time("end", explicit=True)
        check("**別の区間の欄なら書かない**", tp.segments[0].end == 15.8)
        check("いまの区間も汚さない", tp.segments[3].end == 3 * 20.0 + 15.8)

        # ア: ほかの区間をまたぐ長さは確認する
        twin.goto(0)
        twin._time_index = 0
        asked_wide: list = []
        real_yn = assign_gui.messagebox.askyesno
        assign_gui.messagebox.askyesno = lambda *a, **k: (
            asked_wide.append(a), False)[1]
        try:
            twin.var_end.set("00:01:40.0")   # 100 秒 = 区間 1〜4 をまたぐ
            twin._commit_time("end", explicit=True)
        finally:
            assign_gui.messagebox.askyesno = real_yn
        check("**またぐ長さは確認する**", bool(asked_wide))
        check("何区間またぐか伝える",
              asked_wide and "区間" in str(asked_wide[0]))
        check("いいえなら書かない", tp.segments[0].end == 15.8)

        # はいなら書く
        assign_gui.messagebox.askyesno = lambda *a, **k: True
        try:
            twin.var_end.set("00:01:40.0")
            twin._commit_time("end", explicit=True)
        finally:
            assign_gui.messagebox.askyesno = real_yn
        check("はいなら書く", abs(tp.segments[0].end - 100.0) < 0.05)

        # 短い直しでは聞かない（毎回聞かれると使えない）
        twin.goto(2)
        twin._time_index = 2
        asked2: list = []
        assign_gui.messagebox.askyesno = lambda *a, **k: (asked2.append(1), True)[1]
        try:
            twin.var_end.set("00:00:56.0")   # 元 55.8 → 56.0（0.2 秒）
            twin._commit_time("end", explicit=True)
        finally:
            assign_gui.messagebox.askyesno = real_yn
        check("短い直しでは聞かない", not asked2)
        twin.player.close()
        twin.destroy()

        # --- 名簿を 2 列の表で編集する(設計書 §11.8)-----------------------
        rd = RosterDialog(root, parse_roster(chr(10).join([
            "三ツ林衆議院議員", "山本学　文科省　高等教育局私学部参事官付 企画官",
            "梅田茂(加茂暁星学園理事)"])))
        rd.withdraw()
        rd.update()
        check("既存の出席者が行に入る", len(rd.rows) == 3)
        check("名前と役職が別の欄に入る",
              rd.rows[2]["name"].get() == "梅田茂"
              and rd.rows[2]["note"].get() == "加茂暁星学園理事")
        check("行は話者 ID を覚えている",
              all(r["sid"] for r in rd.rows))
        scr_w = rd.winfo_screenwidth()
        scr_h = rd.winfo_screenheight()
        geo = rd.geometry().split("+")[0].split("x")
        check("窓が画面に収まる",
              int(geo[0]) <= scr_w - 40 and int(geo[1]) <= scr_h - 90)

        # 自動で分ける
        n = rd.auto_split()
        check("肩書ごと入っている行を分ける", n == 2)
        check("役職語で切る", rd.rows[0]["name"].get() == "三ツ林"
              and rd.rows[0]["note"].get() == "衆議院議員")
        check("空白で切る", rd.rows[1]["name"].get() == "山本学"
              and rd.rows[1]["note"].get().startswith("文科省"))
        check("**すでに役職が入っている行は触らない**",
              rd.rows[2]["name"].get() == "梅田茂"
              and rd.rows[2]["note"].get() == "加茂暁星学園理事")
        check("何人分けたか知らせる", "2 人" in rd.var_note.get())

        # 行を足す・消す
        added_row = rd.add_row()
        check("行を足せる", len(rd.rows) == 4)
        check("足した行は新しい人(ID が空)", added_row["sid"] == "")
        check("名前が空の行は人として数えない", len(rd.values()) == 3)
        added_row["name"].set("新人")
        check("名前を入れれば数える", len(rd.values()) == 4)
        rd.del_row(added_row)
        check("行を消せる", len(rd.rows) == 3)

        vals = rd.values()
        check("(ID, 名前, 役職) の形で返す",
              len(vals[0]) == 3 and vals[0][1] == "三ツ林")

        # 確定・取り消し
        rd._ok()
        check("OK で中身を返す", rd.result is not None and len(rd.result) == 3)

        rd2 = RosterDialog(root, parse_roster("佐藤"))
        rd2.withdraw()
        rd2.update()
        rd2._cancel()
        check("キャンセルなら何も返さない", rd2.result is None)

        rd3 = RosterDialog(root, [])
        rd3.withdraw()
        rd3.update()
        check("出席者がいなくても空の行が 1 つ出る", len(rd3.rows) == 1)
        warned3: list = []
        real_w3 = assign_gui.messagebox.showwarning
        assign_gui.messagebox.showwarning = lambda *a, **k: warned3.append(1)
        try:
            rd3._ok()
        finally:
            assign_gui.messagebox.showwarning = real_w3
        check("全員空なら確定させない", bool(warned3) and rd3.result is None)
        rd3.rows[0]["name"].set("佐藤")
        rd3.del_row(rd3.rows[0])
        check("最後の 1 行を消しても空の行が残る", len(rd3.rows) == 1)
        rd3._cancel()

        # --- 同姓の人がいても取りこぼさない ------------------------------
        dup = Project(audio_path="x.m4a")
        dup.speakers = parse_roster("佐藤(理事)\n佐藤(監事)")
        dup.segments = [Segment(index=0, start=0, end=1, text="a", cluster="0:A"),
                        Segment(index=1, start=1, end=2, text="b", cluster="0:B")]
        dup.segments[0].speaker_id = dup.speakers[0].id
        dup.segments[1].speaker_id = dup.speakers[1].id
        p2 = plan_roster_text(dup, "佐藤(理事)\n佐藤(監事)")
        p2.apply(dup)
        check("同姓 2 人がそのまま維持される",
              [s.id for s in dup.speakers] == ["sp01", "sp02"]
              and dup.segments[0].speaker_id == "sp01"
              and dup.segments[1].speaker_id == "sp02")
        p3 = plan_roster_text(dup, "佐藤(理事)")
        check("同姓 1 人を消すと 1 区間だけ外れる", p3.affected_segments == 1)
        p3.apply(dup)
        check("残った側の割当は保たれる",
              dup.segments[0].speaker_id == "sp01" and dup.segments[1].speaker_id is None)

        # --- 次の対象へジャンプ ------------------------------------------
        win.goto(0)
        win._goto_next_target()
        win.update()
        check("次の未確定へ移動", not proj.segments[win.current].speaker_id)

        # --- 時刻の点検(画面から) ------------------------------------------
        # 実測は偽の単語列を注入する(whisper もモデルも要らない)。
        check("作業ディレクトリの置き方",
              win._work_dir().name == ".work_meeting")

        def fake_words(indexes, shift=0.0):
            """その区間を実際に喋った、という単語列を作る。

            発話速度は実測に寄せて約 6.7 字/秒にする。区間の長さで按分すると
            0.9 字/秒という不自然な遅さになり、密度フィルタに正しく弾かれて
            しまう(それはそれで正しい挙動)。
            """
            out = []
            for i in indexes:
                seg = proj.segments[i]
                text = seg.text.replace("。", "")
                for n, ch in enumerate(text):
                    at = seg.start + shift + n * 0.15
                    out.append(AlignWord(text=ch, start=at, end=at + 0.15))
            return out

        # この見本は全区間が「これは N 番目の発言です。」でほぼ同じ本文なので、
        # どの区間の単語列にも当たってしまう(実際の会議録では起きない形)。
        # 点検を試す区間だけ、それらしく別々の本文にする。
        spoken = [
            "本日はお忙しい中お集まりいただきありがとうございます",
            "それでは第一号議案について事務局から説明をお願いします",
            "お手元の資料の三ページをご覧ください",
            "前回の会議で出された意見を踏まえて修正しております",
            "この点について何かご質問はございますでしょうか",
            "特にないようですので次の議題に進みます",
        ]
        kept_text = [proj.segments[i].text for i in range(10, 16)]
        for i, text in zip(range(10, 16), spoken):
            proj.segments[i].text = text

        # 実測はすべて正しい位置にある。ずれているのは本文側の時刻のほう
        # (区間 12 の時刻だけが 5 秒早い)。実測をずらすと隣の発言と音声が
        # 重なってしまい、実際には起こらない形になる。
        words = fake_words(range(10, 16))
        seg12 = proj.segments[12]
        seg12.start, seg12.end = seg12.start - 5.0, seg12.end - 5.0
        result = inspect_times(proj, words)
        moved = [p for p in result.proposals
                 if abs(p.target_orig_start - proj.segments[12].orig_start) < 1e-6]
        check("ずれている区間だけ提案が出る", len(moved) == 1)
        check("合っている区間には出ない", len(result.proposals) == 1)

        rows = win._proposal_rows(result.proposals)
        check("一覧の行ができる", len(rows) == 1)
        check("対象は区間 13(1 始まり)", rows[0].target == "区間 13")
        check("ずれの向きが出る", rows[0].delta.startswith("+"))
        check("根拠が入る", "被覆" in rows[0].evidence)
        check("信頼度が語で出る", rows[0].confidence in ("高", "中", "低"))

        # まとめて適用 → ✎△、聴いて承認 → ✎
        win.decide_proposal(moved[0], "bulk")
        # 提案には頭出しの余白 0.3 秒が付くので、その分だけ手前に入る
        check("まとめて適用で実測の時刻が入る",
              abs(seg12.start - seg12.orig_start) < 0.5)
        check("まとめて適用は未確認", seg12.time_reviewed is False)
        check("承認済みとして記録される", moved[0].status == "accepted")
        win.decide_proposal(moved[0], "accept")
        check("聴いて承認は確認済み", seg12.time_reviewed is True)

        reject_me = Proposal(id="r1", type="time",
                             target_orig_start=float(proj.segments[14].orig_start),
                             payload={"start": 1.0, "end": 2.0},
                             evidence="", confidence=0.5)
        win.decide_proposal(reject_me, "reject")
        check("却下は区間を変えない",
              proj.segments[14].time_edited is False
              and reject_me.status == "rejected")

        lost = Proposal(id="r2", type="time", target_orig_start=88888.0,
                        payload={"start": 1.0, "end": 2.0}, evidence="",
                        confidence=0.5)
        win.decide_proposal(lost, "accept")
        # 却下ではなく pending のまま残す。隣が動いた後なら当てられたかも
        # しれず、却下として記録すると正当な提案が二度と出なくなる
        check("当てられない提案は pending のまま", lost.status == "pending")

        # 却下した提案は次の点検で出し直さない(sidecar に判断が残る)
        path = proposals_path(win._work_dir(), "ff00")
        save_proposals(path, [reject_me])
        again = inspect_times(proj, words)
        check("判断済みは再提示しない",
              merge_history(again.proposals, load_proposals(path)) == again.proposals
              or reject_me.id not in [p.id for p in
                                      merge_history(again.proposals,
                                                    load_proposals(path))])
        seg12.start, seg12.end = seg12.orig_start, seg12.orig_end   # 後片付け
        seg12.time_edited = seg12.time_reviewed = False
        for i, text in zip(range(10, 16), kept_text):
            proj.segments[i].text = text
        win.reload_tree()

        # --- 保存・再読込 ------------------------------------------------
        win.save()
        reloaded = Project.load(proj.json_path)
        check("保存と再読込", reloaded.total_count == 40
              and reloaded.segments[5].text == "編集しました")
        win._set_offset(-0.6)
        win.save()
        check("ずれ補正が JSON に残る",
              abs(Project.load(proj.json_path).time_offset + 0.6) < 1e-6)

        # --- 分割ダイアログ ----------------------------------------------
        win.goto(10)
        target = proj.segments[10]
        dlg = SplitDialog(win, target)
        dlg.update()
        check("境界の初期値は本文の真ん中あたり",
              abs(dlg.boundary - (target.start + target.duration / 2)) < 1.0)
        check("音声が無くても波形の枠は描ける", len(dlg.canvas.find_all()) > 0)
        check("カーソル位置が本文の分割点", dlg._cursor_pos() == len(target.text) // 2)
        dlg._set_boundary(target.start + 3.0)
        check("境界を動かせる", abs(dlg.boundary - (target.start + 3.0)) < 1e-6)
        dlg._set_boundary(0.0)
        check("境界は区間の内側に収まる", dlg.boundary >= target.start + 0.1)
        dlg._set_boundary(9999.0)
        check("境界は区間の終わりを越えない", dlg.boundary <= target.end - 0.1)
        dlg.var_bound.set("ここは時刻を書く欄")
        dlg._commit_bound()
        check("読めない境界は元に戻す", dlg.var_bound.get() == fmt_hms_frac(dlg.boundary))
        dlg._ok()
        win.update()
        check("OK で (境界, 本文の位置) を返す",
              dlg.result is not None and len(dlg.result) == 2)

        # --- 点検の提案一覧(骨格) ----------------------------------------
        # 提案を作る側(inspect.py)はまだ無いので、表示用の行を直に組み立てる
        rows = [
            ProposalRow(key=f"p{i}", kind="時刻", target=f"区間 {i}",
                        now="00:43:51.0", measured="00:43:57.8", delta="+6.8秒",
                        evidence="一致 42/48 文字", confidence="高",
                        text=f"提案 {i} の発言")
            for i in range(3)
        ]
        played: list[str] = []
        accepted: list[str] = []
        bulked: list[tuple[str, ...]] = []
        rejected: list[str] = []
        played_now: list[str] = []
        pdlg = ProposalDialog(win, rows, on_play=played.append,
                              on_play_now=played_now.append,
                              on_accept=accepted.append, on_bulk=bulked.append,
                              on_reject=rejected.append)
        pdlg.update()
        check("提案がすべて並ぶ", len(pdlg.tree.get_children()) == 3)
        check("根拠の列がある", pdlg.tree.heading("evidence")["text"] == "根拠")
        check("残り件数を出す", pdlg.var_status.get() == "残り 3 件")

        pdlg.tree.selection_set("p0")
        pdlg.update()
        check("行を選ぶと提案の時刻で再生を頼む", played == ["p0"])
        pdlg._play_now()
        check("いまの時刻でも聴き比べられる", played_now == ["p0"])
        pdlg._accept()
        pdlg.update()
        check("聴いて承認したことを伝える", accepted == ["p0"])
        check("決めた行は一覧から消える", not pdlg.tree.exists("p0"))
        check("残りが減る", pdlg.var_status.get() == "残り 2 件")

        pdlg.tree.selection_set("p1")
        pdlg._reject()
        check("却下したことを伝える", rejected == ["p1"])
        check("却下した行も一覧から消える", not pdlg.tree.exists("p1"))

        pdlg.tree.selection_remove(*pdlg.tree.get_children())
        pdlg._accept()
        check("選ばずに承認はできない", accepted == ["p0"])

        real_ask = assign_gui.messagebox.askyesno
        assign_gui.messagebox.askyesno = lambda *a, **k: True    # 確認は「はい」
        try:
            pdlg._bulk()
        finally:
            assign_gui.messagebox.askyesno = real_ask
        check("残りをまとめて適用できる", bulked == [("p2",)])
        check("まとめて適用すると提案が残らない", pdlg.var_status.get() == "残り 0 件")
        check("何をどう決めたか記録する",
              [kind for kind, _ in pdlg.decisions] == ["accept", "reject", "bulk"])
        pdlg._close()
        win.update()

        # --- 区間の分割・結合 --------------------------------------------
        total = len(proj.segments)
        win.goto(10)
        seg10 = proj.segments[10]
        span, text10 = (seg10.start, seg10.end), seg10.text
        win.apply_split(seg10.start + 12.0, 4)
        win.update()
        check("分割で区間が 1 つ増える", len(proj.segments) == total + 1)
        check("一覧も作り直される", len(win.tree.get_children()) == total + 1)
        check("分割後は後半を選ぶ", win.current == 11)
        check("後半は擬似クラスタで未確定",
              proj.segments[11].is_pseudo_cluster
              and proj.segments[11].speaker_id is None)
        check("本文がカーソル位置で切れる",
              proj.segments[10].text == text10[:4]
              and proj.segments[11].text == text10[4:])

        win.apply_merge(10)
        win.update()
        check("結合で元の数に戻る", len(proj.segments) == total)
        check("結合後はその区間を選ぶ", win.current == 10)
        check("分割前の範囲に戻る",
              (proj.segments[10].start, proj.segments[10].end) == span)
        check("本文もつながる", proj.segments[10].text == text10)

        # --- 取り消しスタックの付け替え ------------------------------------
        win.goto(20)
        win.assign(sato)
        check("取り消しが積まれる", win._undo[-1][0][0] == 20)
        win.goto(10)
        win.apply_split(proj.segments[10].start + 12.0, 4)
        check("分割より後ろの取り消しは 1 つずれる", win._undo[-1][0][0] == 21)
        win.undo()
        win.update()
        check("取り消しが正しい区間に効く", proj.segments[21].speaker_id is None)
        win.apply_merge(10)             # 分割を元に戻す

        depth = len(win._undo)
        win.goto(15)
        win.assign(sato)
        check("取り消しが 1 世代増える", len(win._undo) == depth + 1)
        win.apply_merge(14)             # 15 が 14 に吸収されて消える
        check("消えた区間ぶんの取り消しは残さない", len(win._undo) == depth)
        check("空の世代が残っていない", all(win._undo))
        win.apply_split(proj.segments[14].start + 12.0, 4)   # 数を元に戻す

        # --- 分割・結合・時刻修正を保存して読み直す ------------------------
        win.goto(20)
        win._apply_time("start", proj.segments[20].start + 3.0, explicit=True)
        win.save()
        again = Project.load(proj.json_path)
        check("区間の数がそのまま保存される", again.total_count == len(proj.segments))
        check("時刻の修正が保存される", any(s.time_edited for s in again.segments))
        check("分割で生まれた擬似クラスタが保存される",
              any(s.is_pseudo_cluster for s in again.segments))
        check("元の時刻はすべての区間に残る",
              all(s.orig_start is not None and s.orig_end is not None
                  for s in again.segments))

        # --- タイムライン描画 --------------------------------------------
        win._draw_timeline()
        win.update()
        check("タイムライン描画", len(win.canvas.find_all()) > 0)

        # --- Word 出力 ---------------------------------------------------
        from src.segments import write_docx
        out = tmp / "out.docx"
        write_docx(reloaded, out)
        check("Word 出力", out.exists() and out.stat().st_size > 0)

        # --- 元音声の SHA-256(記録と照合) --------------------------------
        # 見本の音声ファイルは無いので、実体を作って記録→照合→改変を試す
        audio_file = Path(proj.audio_path)
        audio_file.write_bytes(b"RIFF-dummy-audio-content")
        check("未記録なら計算して記録する",
              win.ensure_source_sha() is True
              and len(proj.source_sha256) == 64)
        first_sha = proj.source_sha256
        check("記録済みなら再計算しない",
              win.ensure_source_sha() is True
              and proj.source_sha256 == first_sha)

        infos: list = []
        real_info = assign_gui.messagebox.showinfo
        real_warn = assign_gui.messagebox.showwarning
        assign_gui.messagebox.showinfo = lambda t, *a, **k: infos.append(("i", t))
        assign_gui.messagebox.showwarning = lambda t, *a, **k: infos.append(("w", t))
        try:
            win.verify_source_audio()
            check("一致すれば一致と伝える", infos[-1] == ("i", "一致しました"))
            audio_file.write_bytes(b"RIFF-dummy-audio-CHANGED")
            win.verify_source_audio()
            check("中身が変われば不一致と警告する",
                  infos[-1] == ("w", "一致しません"))
            check("警告しても記録は書き換えない", proj.source_sha256 == first_sha)
        finally:
            assign_gui.messagebox.showinfo = real_info
            assign_gui.messagebox.showwarning = real_warn
            audio_file.unlink(missing_ok=True)

        win.player.close()
        win.destroy()

        # --- 声のまとまりが無い作業ファイル(ローカル転写) -----------------
        # 全区間が擬似クラスタなので、一括適用は成り立たない。ON のまま
        # 残すと確定のたびに警告が出る(1219 区間なら 1219 回)。
        local_proj = Project(audio_path=str(tmp / "meeting.m4a"),
                             duration=600.0, chunk_seconds=420)
        local_proj.speakers = parse_roster("佐藤\n田中")
        local_proj.segments = [
            Segment(index=i, start=i * 10.0, end=i * 10.0 + 8,
                    text=f"発言 {i}。", cluster=f"{i // 5}:?", chunk=i // 5)
            for i in range(10)
        ]
        local_proj.json_path = str(tmp / "local.speakers.json")
        local_proj.save()

        lwin = AssignWindow(root, local_proj)
        lwin.var_autoplay.set(False)
        lwin.update()
        check("まとまりが無いことを見分ける", lwin.has_real_clusters() is False)
        check("一括適用は最初から外れている",
              lwin.var_apply_cluster.get() is False)
        check("一括適用のチェックは押せない",
              str(lwin.chk_apply_cluster.cget("state")) == "disabled")

        # A キーを押しても ON にならない(ON になると警告が出る状態に戻る)
        lwin._toggle_cluster_mode()
        check("A キーでも一括適用は入らない",
              lwin.var_apply_cluster.get() is False)

        # 確定しても警告ダイアログが出ないこと
        warned: list[str] = []
        real_warn = assign_gui.messagebox.showwarning
        assign_gui.messagebox.showwarning = lambda t, *a, **k: warned.append(t)
        try:
            lwin.current = 0
            lwin.assign(local_proj.speakers[0].id)
            check("確定しても警告が出ない", warned == [])
            check("その区間だけが確定する",
                  local_proj.segments[0].speaker_id == local_proj.speakers[0].id
                  and all(s.speaker_id is None for s in local_proj.segments[1:]))
            check("自分で聴いた 1 件は ✓ になる",
                  local_proj.segments[0].reviewed is True)
        finally:
            assign_gui.messagebox.showwarning = real_warn
        # 聴く順: 順番の sidecar が無い作業ファイルでは選べない。
        # **0 点や「無し」を安全と読ませないため、灰色にして出さない。**
        check("順番が無ければ聴く順は選べない",
              str(lwin.chk_listen.cget("state")) == "disabled")
        lwin.player.close()
        lwin.destroy()

        # --- 聴く順(順番の sidecar がある場合) ---------------------------
        from src import listen_order as lo
        lp = Project(audio_path=str(tmp / "meeting.m4a"),
                     duration=600.0, chunk_seconds=420)
        lp.speakers = parse_roster("佐藤\n田中")
        lp.segments = [
            Segment(index=i, start=i * 10.0, end=i * 10.0 + 8,
                    text=f"発言 {i}。", cluster="g:A", chunk=0)
            for i in range(4)
        ]
        lp.audio_fingerprint = "fp-listen"
        lp.json_path = str(tmp / "listen.speakers.json")
        lp.save()
        work = tmp / ".work_meeting"
        # 2 番の区間だけ点数が高い順番を置く
        lo.save_hints(lo.hints_path(work, "fp-listen"), [
            lo.ListenHint(orig_start=float(s.start), start=float(s.start),
                          score=(9 if s.index == 2 else 1), index=s.index)
            for s in lp.segments
        ])
        wwin = AssignWindow(root, lp)
        wwin.var_autoplay.set(False)
        wwin.update()
        check("順番があれば聴く順を選べる",
              str(wwin.chk_listen.cget("state")) == "normal")
        check("既定は時間順のまま",
              wwin.var_listen_order.get() is False
              and wwin._visible_indexes() == [0, 1, 2, 3])
        wwin.var_listen_order.set(True)
        wwin._on_listen_order_change()
        check("聴く順にすると点数の高い区間が先頭に来る",
              wwin._visible_indexes()[0] == 2)
        check("言い切らない(取りこぼしが残ると添える)",
              "取りこぼしはあります" in wwin.var_action.get())
        wwin.var_listen_order.set(False)
        wwin._on_listen_order_change()
        check("時間順に戻せる", wwin._visible_indexes() == [0, 1, 2, 3])
        wwin.player.close()
        wwin.destroy()

        # --- 相づちを足す ------------------------------------------------
        ap = Project(audio_path=str(tmp / "meeting.m4a"),
                     duration=600.0, chunk_seconds=420)
        ap.speakers = parse_roster("佐藤\n田中")
        ap.segments = [
            Segment(index=i, start=i * 10.0, end=i * 10.0 + 8,
                    text=f"発言 {i}。", cluster="g:A", chunk=0)
            for i in range(4)
        ]
        ap.json_path = str(tmp / "add.speakers.json")
        ap.save()
        awin = AssignWindow(root, ap)
        awin.var_autoplay.set(False)
        awin.update()

        # **転写が出した区間でも押せる。**禁じているのは自動削除であって、
        # 聴いて機械の重複だと判断した人が消すのは別のこと（実機の指摘・
        # 2026-08-22。結合を 2 回してから分割する遠回りを強いられた）。
        check("転写の区間でも消すボタンが押せる",
              str(awin.btn_del_added.cget("state")) == "normal")

        # --- 足した発話は一覧で目立つか ------------------------------
        # 「＋」だけでは埋もれて分からないという指摘（2026-08-22）。
        # **背景・文字色・括弧の 3 つ**で示す。色だけでは印刷や色覚の
        # 条件で消える。
        _asr_vals, _asr_tags = awin._row_values(ap.segments[0])
        check("転写の区間は括弧で囲まない", "＋（" not in _asr_vals[3])
        check("同じ種類の色を 2 つ付けない",
              sum(1 for t in _asr_tags if t.startswith("bg_")) <= 1
              and sum(1 for t in _asr_tags if t.startswith("fg_")) <= 1)

        # 小窓は差し替える(開くと応答待ちで止まる)
        real_ask_utt = awin._ask_utterance
        awin._ask_utterance = lambda seg, turns, existing=None, **_k: (
            [{"cut": 2, "at": 14.0, "end": 14.6, "text": "はいはい",
              "cluster": "g:B", "sid": ap.speakers[0].id}], True)
        try:
            awin.goto(1)
            awin.add_utterance()
        finally:
            awin._ask_utterance = real_ask_utt

        added = [s for s in ap.segments if s.text == "はいはい"]
        check("足した発話が入る", len(added) == 1)
        if added:
            a = added[0]
            check("時間順の位置に入る",
                  [s.start for s in ap.segments]
                  == sorted(s.start for s in ap.segments))
            check("話者を選んだので ✓ になる",
                  a.speaker_id == ap.speakers[0].id and a.reviewed is True)
            check("人が足した区間だと分かる", ap.is_added_utterance(a) is True)
            check("足した区間へ移動している", awin.current == a.index)
            check("消すボタンが押せるようになる",
                  str(awin.btn_del_added.cget("state")) == "normal")
            _vals, _tags = awin._row_values(a)
            check("足した発話は括弧で囲んで示す", _vals[3].startswith("＋（"))
            check("足した発話は背景と文字色でも示す",
                  "bg_added" in _tags and "fg_added" in _tags)
            check("足した行も同じ種類の色は 1 つずつ",
                  sum(1 for t in _tags if t.startswith("bg_")) == 1
                  and sum(1 for t in _tags if t.startswith("fg_")) == 1)
            # 消す(確認ダイアログは「はい」に差し替える)
            real_ask = assign_gui.messagebox.askyesno
            assign_gui.messagebox.askyesno = lambda *a2, **k2: True
            try:
                awin.remove_added()
            finally:
                assign_gui.messagebox.askyesno = real_ask
            check("足した発話を消せる",
                  not any(s.text == "はいはい" for s in ap.segments))
            check("元の 4 区間は残る", len(ap.segments) == 4)

        # --- 候補の一覧（設計書 §10.3）------------------------------------
        # **検出器ではない。**適合 35/51 で 3 割は空振り、再現 31/34 なので
        # 候補の無い区間にも取りこぼしはある。言い切っていないことも見る。
        from src.diarize import SpeakerTurn

        check("話者分離が無ければ候補は空", awin._voice_candidates == [])
        awin.var_filter.set(assign_gui.FILTER_CANDIDATES)
        awin._on_filter_change()
        check("使えないと知らせる（黙って 0 件にしない）",
              "話者分離" in awin.var_action.get())

        # turn を差し替えて候補を作る（区間 1 の中に別の声）
        real_turns = awin._load_turns
        awin._load_turns = lambda: [
            SpeakerTurn(start=10.0, end=18.0, speaker=1),
            SpeakerTurn(start=13.0, end=13.9, speaker=2),
            SpeakerTurn(start=15.5, end=16.2, speaker=3),
            SpeakerTurn(start=20.0, end=28.0, speaker=1),
        ]
        try:
            awin._voice_candidates = awin._load_voice_candidates()
            check("区間の中の別の声を拾う", len(awin._voice_candidates) == 2)
            awin.goto(1)
            awin.update()
            check("選択肢に時刻と声が出る",
                  len(awin.cmb_cand.cget("values")) == 2
                  and "声" in awin.cmb_cand.cget("values")[0])
            check("候補があればボタンが押せる",
                  str(awin.btn_cand_add.cget("state")) == "normal")
            check("**言い切らない**（空振りがあると書く）",
                  "空振り" in str(awin.lbl_cand_note.cget("text")))
            check("選択肢は時刻と声で読める",
                  ":" in awin.cmb_cand.cget("values")[0])
            # **選択肢が切れないこと。**HH:MM:SS だと幅に収まらず
            # 「00:00:29 声!」と切れた(実機で確認・2026-08-19)。
            check("選択肢が選択欄の幅に収まる",
                  all(len(v) <= int(awin.cmb_cand.cget("width"))
                      for v in awin.cmb_cand.cget("values")))
            check("先頭の 00: を出さない",
                  not awin.cmb_cand.cget("values")[0].startswith("00:"))

            # **もう足した位置は「済」。隠さずに印を付けて後ろに回す。**
            ap.add_utterance(13.2, 13.9, "はい", cluster="g:B")
            awin._show_voice_candidates()
            awin.update()
            _vals = list(awin.cmb_cand.cget("values"))
            check("足した位置に「済」が付く",
                  any(v.startswith("済") for v in _vals))
            check("候補そのものは消さない", len(_vals) == 2)
            check("済は後ろに回す", not _vals[0].startswith("済"))
            check("残りの数を出す", "残" in str(awin.lbl_cand_note.cget("text")))
            # 待ち行列（候補のみ）からは、済んだものだけの区間は外れる
            ap.add_utterance(15.2, 15.9, "ええ", cluster="g:C")
            awin.var_filter.set(assign_gui.FILTER_CANDIDATES)
            awin._on_filter_change()
            check("全部済んだ区間は待ち行列から外れる",
                  1 not in awin._visible_indexes())
            for _s in [s for s in ap.segments
                       if ap.is_added_utterance(s)]:
                ap.remove_added_utterance(_s.index)
            awin.var_filter.set(assign_gui.FILTER_ALL)
            awin._on_filter_change()
            awin.goto(1)
            awin.update()

            # --- ［ここから］が位置を渡す（設計書 §10.3.3）------
            _seen: dict = {}

            def _fake_ask(seg, turns, existing=None,
                          initial_cut=None, initial_note=""):
                _seen["cut"] = initial_cut
                _seen["note"] = initial_note
                _seen["seg"] = seg
                return ([], False)

            _real_ask3 = awin._ask_utterance
            awin._ask_utterance = _fake_ask
            try:
                awin.cmb_cand.current(0)
                awin.add_from_voice_candidate()
            finally:
                awin._ask_utterance = _real_ask3
            check("候補から開くと本文の位置を渡す",
                  isinstance(_seen.get("cut"), int))
            check("位置は本文の範囲に収まる",
                  0 <= _seen["cut"] <= len(_seen["seg"].text))
            check("どの声かを案内する",
                  "声" in _seen.get("note", ""))
            check("**位置が目安だと書く**（実測ではない）",
                  "目安" in _seen.get("note", ""))
            # **話者は機械が選ばない。**選ぶと ✓ が機械の判断で立つ
            _dlg3 = assign_gui.AddUtteranceDialog(
                awin, ap.segments[1], awin._load_turns(), None,
                initial_cut=2, initial_note="候補: 0:13 に声B")
            _dlg3.withdraw()
            _dlg3.update()
            check("渡した位置にカーソルが立つ", _dlg3._cut == 2)
            check("案内が小窓に出る",
                  "声B" in _dlg3.var_from_cand.get())
            check("**話者は機械が選ばない**（✓ の意味を守る）",
                  _dlg3.cmb_sp.current() == 0)
            _dlg3.destroy()
            awin.var_filter.set(assign_gui.FILTER_ALL)
            awin._on_filter_change()

            # 絞り込み
            awin.var_filter.set(assign_gui.FILTER_CANDIDATES)
            awin._on_filter_change()
            check("候補のある区間だけ出す", awin._visible_indexes() == [1])
            check("取りこぼしの言い方を弱めない",
                  "空振り" in awin.var_action.get()
                  and "取りこぼし" in awin.var_action.get())

            # × で捨てる
            awin.goto(1)
            first = awin._current_voice_candidates[0]
            awin.cmb_cand.current(0)
            awin.dismiss_voice_candidate()
            check("× で候補が減る", len(awin._voice_candidates) == 1)
            check("消したと知らせる", "外しました" in awin.var_action.get())
            check("**判断が残る**（作り直しても出てこない）",
                  first.key not in
                  [c.key for c in awin._load_voice_candidates()])

            # 候補が無くなれば、説明に戻る
            awin.cmb_cand.current(0)
            awin.dismiss_voice_candidate()
            awin._show_voice_candidates()
            awin.update()
            check("候補が尽きたらボタンを止める",
                  str(awin.btn_cand_add.cget("state")) == "disabled")
            check("元の説明に戻る", awin.lbl_cand.winfo_ismapped()
                  and not awin.frm_cand.winfo_ismapped())
        finally:
            awin._load_turns = real_turns
            awin.var_filter.set(assign_gui.FILTER_ALL)
            awin._on_filter_change()

        # 話者の候補（suggest.py）と混ざっていないこと
        check("話者の候補とは別物のまま",
              awin._candidates is not awin._voice_candidates)

        # --- 特別な選択肢が名簿に押し出されないこと -----------------------
        # 「発言なし・雑音 はないです」(実機の指摘・2026-08-19)。出席者が
        # 9 人だと名簿のボタンに押し出され、下 2 つが画面から切れていた。
        from src.segments import SPECIAL_MULTI as _SM, SPECIAL_NOISE as _SN
        from src.segments import SPECIAL_UNKNOWN as _SU
        many = Project(audio_path=str(tmp / "meeting.m4a"),
                       duration=600.0, chunk_seconds=420)
        many.speakers = parse_roster(chr(10).join(
            [f"出席者{i}(とても長い所属と役職の名前がここに入ります)"
             for i in range(9)]))
        many.segments = [
            Segment(index=i, start=i * 10.0, end=i * 10.0 + 8,
                    text=f"発言 {i}。", cluster="g:A", chunk=0)
            for i in range(6)]
        many.json_path = str(tmp / "many.speakers.json")
        many.save()
        mwin = AssignWindow(root, many)
        mwin.var_autoplay.set(False)
        mwin.update()
        check("名簿が 9 人でも候補が出る", len(mwin._candidates) >= 1)
        for _sid, _name in ((_SU, "発言者不明"), (_SM, "複数人が同時"),
                            (_SN, "発言なし・雑音")):
            _b = mwin.special_buttons[_sid]
            check(f"「{_name}」が見えている",
                  _b.winfo_ismapped() and _b.winfo_height() > 1)
            check(f"「{_name}」が窓の中にある",
                  0 <= _b.winfo_rooty() - mwin.winfo_rooty()
                  < mwin.winfo_height())
        check("特別な選択肢は名簿と別枠にある",
              mwin.special_buttons[_SN].master is not mwin.cand_holder)
        check("押せば雑音として確定できる",
              callable(mwin.special_buttons[_SN].cget("command"))
              or bool(mwin.special_buttons[_SN].cget("command")))
        # **完了の知らせはモーダル。**この見本は 6 区間とも同じまとまり(g:A)で、
        # 一括適用が既定(var_apply_cluster=True)なので、1 回押すと全区間が
        # 埋まる。すると _after_change() が messagebox.showinfo("完了", ...) を
        # 出し、誰も押さないので**ここで検査が止まって返ってこない**
        # (2026-08-28 に実際に止まり、以降の検査が一度も走っていなかった)。
        # アプリの動作は正しい。止まるのは検査の側の落ち度なので、この
        # ファイルの他の箇所と同じやり方で差し替える。
        done_msgs: list = []
        real_done = assign_gui.messagebox.showinfo
        assign_gui.messagebox.showinfo = lambda *a, **k: done_msgs.append(a)
        try:
            mwin.special_buttons[_SN].invoke()
        finally:
            assign_gui.messagebox.showinfo = real_done
        check("雑音に確定できる", many.segments[0].speaker_id == _SN)
        # 止まっていた間、ここから先は一度も検査されていなかった。
        # 通りがかりに 2 つ足しておく。
        check("一括適用で同じまとまりが全部埋まる",
              all(s.speaker_id == _SN for s in many.segments))
        check("全部確定したら完了を知らせる",
              any("完了" in str(a) for a in done_msgs))
        mwin.player.close()
        mwin.destroy()

        # --- 既定の窓幅で、右ペインのボタンが全部見えること ---------------
        # 「右側が切れて表示されます。［＋この声を足す...］ボタンを表示させる
        # には、ウィンドウを広げる必要がある」(実機の指摘・2026-08-18)。
        # **画面より大きくしない。**アプリ本体と同じ規則で決める。
        # 1320x800 と決め打ちすると、狭い画面(実際に 1280x720 で動かした)
        # では窓が画面をはみ出し、右ペインが縮んで検査が落ちる。
        # 検査したいのは「既定の大きさで収まるか」であって、特定の画素数ではない。
        _sw, _sh = awin.winfo_screenwidth(), awin.winfo_screenheight()
        awin.geometry(f"{min(1320, _sw - 40)}x{min(800, _sh - 90)}")
        awin.update()
        _frm = awin.btn_del_added.master.master
        _rows = [r for r in _frm.winfo_children() if r.winfo_children()]
        # **見えている部品だけ数える。**説明文と候補の選択肢は排他で、
        # 同時には出ない(設計書 §10.3)。両方を足すと実際より広く見積もる。
        # **押し出されて消えたぶんは別に落とす**(visible_need の注記)。
        _needs = [visible_need(r, 20) for r in _rows]
        _need = max(n for n, _ in _needs)
        _lost = [c for _, lost in _needs for c in lost]
        _have = _frm.master.winfo_width()
        check(f"既定の幅で右ペインのボタン列が収まる（要 {_need} / 幅 {_have}）",
              _have >= _need)
        check(f"右ペインで押し出された部品が無い（{lost_names(_lost) or 'なし'}）",
              not _lost)
        check("＋この声を足す が右端をはみ出さない",
              awin.btn_del_added.winfo_x() + awin.btn_del_added.winfo_width()
              <= _have)
        check("行を分けてある（1 行に詰めない）", len(_rows) >= 4)

        # **注記をチェックボックスの文字列に埋めない。**埋めたら 353px に
        # なり、右端の[取り消し]を押し出した(要 734px / 幅 713px。実機の
        # 指摘・2026-08-21)。§10.3.6・GPU の注記と同じ種類の失敗で 3 度目。
        # **確定まわりは 2 行ある**(切り替え / 操作)。片方だけ見ていると、
        # もう片方が溢れても気づけない。両方測る。
        _opts = awin.chk_apply_cluster.master
        _opts_btn = awin.btn_undo.master
        check("確定まわりを 2 行に分けてある（1 行に詰めない）",
              _opts is not _opts_btn)
        _kids = []
        for _label, _row in (("切り替え", _opts), ("操作", _opts_btn)):
            _need, _lost = visible_need(_row, 20)
            _have = _row.master.winfo_width()
            _kids += [c for c in _row.winfo_children() if c.winfo_ismapped()]
            check(f"確定の行（{_label}）が収まる（要 {_need} / 幅 {_have}）",
                  _have >= _need)
            # **この 2 行に排他の部品は無い**(assign_gui.py で 6 つとも
            # 素直に pack している)。消えていたら押し出されたということ。
            check(f"確定の行（{_label}）で押し出された部品が無い"
                  f"（{lost_names(_lost) or 'なし'}）", not _lost)
        check("チェックの文字列に注記を埋めていない",
              len(str(awin.chk_apply_cluster.cget("text"))) <= 32)
        check("[取り消し]が窓の右端をはみ出さない",
              all(c.winfo_rootx() + c.winfo_reqwidth()
                  <= awin.winfo_rootx() + awin.winfo_width() for c in _kids))

        # 候補を出した状態でも収まること（説明文と排他にしてある）
        from src.diarize import SpeakerTurn as _ST
        _real_turns2 = awin._load_turns
        awin._load_turns = lambda: [
            _ST(start=10.0, end=18.0, speaker=1),
            _ST(start=13.0, end=13.9, speaker=2)]
        _real_dismissed = awin._dismissed
        awin._dismissed = []      # 上の検査で × を付けたぶんを外す
        try:
            awin._voice_candidates = awin._load_voice_candidates()
            awin.goto(1)
            awin.update()
            check("候補が出ている（この検査の前提）",
                  len(awin._current_voice_candidates) >= 1)
            _row = awin.btn_cand_add.master.master
            _need_c, _lost_c = visible_need(_row, 20)
            check("候補を出しても収まる（要 "
                  + str(_need_c) + " / 幅 " + str(_have) + "）",
                  _have >= _need_c)
            check("候補の行で押し出された部品が無い"
                  f"（{lost_names(_lost_c) or 'なし'}）", not _lost_c)
            check("候補が出ている間は説明文を出さない（幅の取り合い）",
                  not awin.lbl_cand.winfo_ismapped())
        finally:
            awin._load_turns = _real_turns2
            awin._dismissed = _real_dismissed
            awin._voice_candidates = []
            awin._show_voice_candidates()
            awin.update()

        # 高さ: **画面より大きくしない**(はみ出すと下の［保存］が画面外に出る)
        check("窓が画面からはみ出さない",
              awin.winfo_width() <= awin.winfo_screenwidth()
              and awin.winfo_height() <= awin.winfo_screenheight())
        # **末尾の子で取らない。**帯を 2 行にしたとき、これは新しく足した
        # 行を指してしまい、編集系と出力系の両方を出力系で測っていた。
        _bottom_bar = awin.bottom_edit
        # **これは素通りしていた。**下の帯は grid で最下段に固定されるので、
        # 「下端 <= 窓の高さ」はどんな条件でも真になる(実測: どの画面・
        # どの拡大率でも余り 0)。高さが足りているかを測っていなかった。
        #
        # 測るべきは**縮まない行が最後まで見えているか**。時刻・確定などは
        # 高さが固定で、入らなければ下から切れる(本文と候補は縮む側なので
        # 対象にしない)。行を足す変更を重ねているので、ここが効いていないと
        # 「幅は直したが下が見えない」に気づけない。
        # **下の帯 2 行もここで見る。**「帯が窓の中にある」という検査を
        # 別に置いていたが、帯は grid の最下段に固定されるので常に真だった。
        # 素通りを残さないため消し、この検査に含める。
        _fixed_rows = (list(awin._time_rows)
                       + [awin.btn_confirm_time.master,
                          awin.btn_del_added.master,
                          awin.chk_apply_cluster.master,
                          awin.btn_undo.master,
                          _bottom_bar, awin.bottom_out])
        _clipped = []
        for _r in _fixed_rows:
            if not _r.winfo_ismapped():
                _clipped.append(_r)
                continue
            # 窓の中に、下端まで入っていること
            _bot = _r.winfo_rooty() + _r.winfo_height()
            if _bot > awin.winfo_rooty() + awin.winfo_height():
                _clipped.append(_r)
        check(f"縮まない行が下まで見えている（切れ {len(_clipped)} 行）",
              not _clipped)
        # 下の帯は既に混んでいる。[語句をまとめて直す...]を足したので、
        # **横に押し出されていないか**を数える(§10.3.6 で特別な選択肢が
        # 窓の外に出ていた前例がある)。
        # **帯は 2 行ある**(編集系 / 出力系)。両方測る。
        for _name, _bar in (("編集系", _bottom_bar), ("出力系", awin.bottom_out)):
            _need_bar, _lost_bar = visible_need(_bar, 24)
            check(f"下の帯（{_name}）が既定の幅に収まる（要 {_need_bar} / 幅 "
                  f"{_bar.winfo_width()}）",
                  _bar.winfo_width() >= _need_bar)
            check(f"下の帯（{_name}）で押し出された部品が無い"
                  f"（{lost_names(_lost_bar) or 'なし'}）", not _lost_bar)
        # **［保存］は空いている行に置く。**混んでいる編集系の行に戻すと、
        # 狭い画面で押し出される(1024x768@125% で実際に消えていた)。
        #
        # 「見えているか」だけでは検査にならない。窓を 100px まで詰めても
        # ［保存］は消えなかった——右寄せの並び順で［元音声と照合］が先に
        # 落ちるため、押し出しのガードが必ず先に発火する(実測)。
        # **置き場所を留めるほうが、実際に落ちうる検査になる。**
        check("［保存］が空いているほうの行にある",
              awin.btn_save.master is awin.bottom_out)
        check("［保存］が見えている",
              awin.btn_save.winfo_ismapped()
              and awin.btn_save.winfo_rootx() + awin.btn_save.winfo_width()
              <= awin.winfo_rootx() + awin.winfo_width())
        _left = [c for c in _bottom_bar.winfo_children()
                 if c.winfo_ismapped() and c.pack_info().get("side") == "left"]
        _rep = next((c for c in _left
                     if "語句" in str(c.cget("text"))), None)
        check("［語句をまとめて直す...］がある", _rep is not None)
        check("［語句をまとめて直す...］が窓の右端をはみ出さない",
              _rep is not None
              and _rep.winfo_rootx() + _rep.winfo_reqwidth()
              <= awin.winfo_rootx() + awin.winfo_width())

        # --- 語句をまとめて直す（設計書 §16.3）---------------------------
        # **同じ語でも直してよい箇所とそうでない箇所が混ざる。**
        # 実データで「資格」は 10 回のうち 1 回が本物の資格だった。
        _keep = [s.text for s in ap.segments]
        ap.segments[0].text = "防災士の資格も取ってらっしゃって"
        ap.segments[1].text = "県は資格のことですからと言うだけで"
        ap.segments[0].reviewed = ap.segments[1].reviewed = False
        _dlg = ReplaceWordsDialog(awin)
        try:
            _dlg.update()
            _dlg.var_before.set("資格")
            _dlg.var_after.set("私学")
            _dlg.search()
            _dlg.update()
            check("見つかった箇所が並ぶ", len(_dlg.hits) == 2)
            _lines = [_dlg.tree.set(i, "text") for i in _dlg.tree.get_children()]
            check("前後の文脈が出る（時刻と語だけでは判断できない）",
                  any("防災士" in x for x in _lines)
                  and all("【資格】" in x for x in _lines))
            check("はじめは全部○", all(_dlg.marks))
            check("ボタンに件数が出る", "2 箇所" in str(_dlg.btn_ok.cget("text")))
            _dlg._toggle(0)                     # 本物の資格を外す
            _dlg.update()
            check("×にできる", _dlg.marks == [False, True])
            check("件数が減る", "1 箇所" in str(_dlg.btn_ok.cget("text")))
            check("×の行に×印が出る",
                  _dlg.tree.set("0", "mark") == _dlg.MARK_OFF)
            _dlg._mark_all(True)
            check("全部に○を押せる", all(_dlg.marks))
            _dlg._mark_all(False)
            check("全部に×にすると押せなくなる",
                  str(_dlg.btn_ok.cget("state")) == "disabled")
            _dlg._toggle(1)
            _dlg._ok()
            check("選んだぶんだけ返す",
                  _dlg.result is not None and len(_dlg.result[2]) == 1)
        finally:
            try:
                _dlg.destroy()
            except tk.TclError:
                pass
        _n = ap.replace_text(*_dlg.result) if _dlg.result else 0
        check("選ばなかった箇所は変わらない",
              ap.segments[0].text == "防災士の資格も取ってらっしゃって")
        check("選んだ箇所だけ直る",
              _n == 1 and ap.segments[1].text == "県は私学のことですからと言うだけで")
        check("聴いていないので確定の印は付かない",
              ap.segments[1].reviewed is False)
        check("人が直した本文なので再実行で戻されない",
              ap.segments[1].text_edited is True)
        for _i, _t in enumerate(_keep):
            ap.segments[_i].text = _t
        awin.reload_tree()

        # --- 話者をまとめて置き換える（途中退席）------------------------
        # **退席の直前は本人の発言。**「中座させてもらいます」まで
        # 付け替えると壊れる。
        _snap = [(x.speaker_id, x.reviewed) for x in ap.segments]
        _yo, _ni = ap.speakers[0].id, ap.speakers[1].id
        for _x in ap.segments:
            _x.speaker_id, _x.reviewed = _yo, False
        ap.segments[2].reviewed = True          # ✓ が 1 つある状態
        _sd = ReplaceSpeakerDialog(awin)
        try:
            _sd.update()
            _sd.cmb_before.current(0)
            _sd.cmb_after.current(1)
            _sd.var_after_time.set(fmt_short_time(ap.segments[1].start))
            _sd.search()
            _sd.update()
            check("時刻より後だけを拾う",
                  bool(_sd.rows)
                  and all(x.start > ap.segments[1].start for x in _sd.rows))
            check("その時刻以前は入らない（退席の発言は本人）",
                  ap.segments[1] not in _sd.rows)
            check("✓ の区間には印が出る",
                  any(_sd.tree.set(i, "seen") == "✓"
                      for i in _sd.tree.get_children()))
            check("△ に戻ることを画面で伝える",
                  "△" in _sd.var_status.get())
            _sd._toggle(0)
            _sd.update()
            check("×にできる", _sd.marks[0] is False)
            _sd._mark_all(True)
            _sd._ok()
            check("選んだぶんを返す",
                  _sd.result is not None
                  and len(_sd.result[2]) == len(_sd.rows))
        finally:
            try:
                _sd.destroy()
            except tk.TclError:
                pass
        _moved = ap.replace_speaker(*_sd.result) if _sd.result else 0
        check("退席より後だけ付け替わる",
              _moved == len(_sd.rows)
              and ap.segments[1].speaker_id == _yo
              and all(x.speaker_id == _ni for x in _sd.rows))
        check("付け替えた区間は ✓ を持たない（まとめて適用なので △）",
              all(x.reviewed is False for x in _sd.rows))
        for _i, (_sp, _rv) in enumerate(_snap):
            ap.segments[_i].speaker_id, ap.segments[_i].reviewed = _sp, _rv
        awin.reload_tree()

        # --- 本文欄の書き戻しは、読み込んだ区間にだけ効く ----------------
        # current を直接動かした直後に _commit_text が呼ばれると、**別の区間へ
        # 前の本文を上書きしていた**(この検査を書いていて見つかった)。
        awin.goto(0)
        before = [s.text for s in ap.segments]
        keep_body = awin.txt_body.get("1.0", "end").strip()
        awin.current = 1                      # 本文欄を更新せずに動かす
        awin._commit_text()
        check("本文欄と食い違う区間には書き戻さない",
              [s.text for s in ap.segments] == before)
        awin.goto(1)                          # 正しい経路なら書き戻す
        awin.txt_body.delete("1.0", "end")
        awin.txt_body.insert("1.0", "直した本文")
        awin._commit_text()
        check("読み込んだ区間には書き戻す",
              ap.segments[1].text == "直した本文")
        # 後の検査のために戻す。**本文欄も一緒に戻す**——欄に「直した本文」が
        # 残っていると、次の _commit_text がまた書き戻してしまう。
        awin.txt_body.delete("1.0", "end")
        awin.txt_body.insert("1.0", before[1])
        awin._commit_text()
        check("戻せている", ap.segments[1].text == before[1])
        _ = keep_body

        # --- 足したものを開き直して直せる（2026-08-18 の指摘）------------
        # 「修正しようとすると、最初の画面が出てきても、挿入されたものは
        # なくなっています」。位置は小窓でしか決められないのに、開くと
        # 空から始まるので直しようがなかった。
        base = next(s for s in ap.segments if s.text == "発言 1。")
        # まず 1 件足す
        awin._ask_utterance = lambda seg, turns, existing=None, **_k: (
            [{"cut": 2, "at": 14.0, "end": 14.6, "text": "もとの",
              "cluster": "g:B", "sid": None}], True)
        awin.goto(base.index)
        awin.add_utterance()
        check("下ごしらえ: 1 件入った",
              any(s.text == "もとの" for s in ap.segments))

        # 開き直すと、その 1 件が既存として渡る
        seen: list = []
        awin._ask_utterance = lambda seg, turns, existing=None, **_k: (
            seen.append(list(existing or [])),
            ([{"cut": 2, "at": 14.0, "end": 14.6, "text": "なおした",
               "cluster": "g:B", "sid": None}], True))[1]
        base = next(s for s in ap.segments if s.text == "発言 1。")
        awin.goto(base.index)
        awin.add_utterance()
        check("開き直すと既に足したものが渡る",
              seen and len(seen[-1]) == 1
              and seen[-1][0]["text"] == "もとの")
        check("位置(cut)も渡す", "cut" in seen[-1][0])
        check("入れ替えで古いほうは消える",
              not any(s.text == "もとの" for s in ap.segments))
        check("新しいほうが入る",
              any(s.text == "なおした" for s in ap.segments))
        check("増えていない（置き換えになっている）",
              sum(1 for s in ap.segments if ap.is_added_utterance(s)) == 1)

        # 空にして確定すると全部消える
        awin._ask_utterance = lambda seg, turns, existing=None, **_k: ([], True)
        awin.goto(base.index)
        awin.add_utterance()
        check("空にして確定すると全部消える",
              not any(ap.is_added_utterance(s) for s in ap.segments))

        # やめた場合は何も変えない
        awin._ask_utterance = lambda seg, turns, existing=None, **_k: (
            [{"cut": 2, "at": 14.0, "end": 14.6, "text": "入れない",
              "cluster": "g:B", "sid": None}], False)
        n_before = len(ap.segments)
        awin.goto(base.index)
        awin.add_utterance()
        check("やめたら何も変わらない",
              len(ap.segments) == n_before
              and not any(s.text == "入れない" for s in ap.segments))
        awin._ask_utterance = real_ask_utt

        # 話者を選ばなかった場合は ✓ を立てない
        awin._ask_utterance = lambda seg, turns, existing=None, **_k: (
            [{"cut": 2, "at": 24.0, "end": 24.6, "text": "うん",
              "cluster": "g:C", "sid": None}], True)
        try:
            awin.goto(2)
            awin.add_utterance()
        finally:
            awin._ask_utterance = real_ask_utt
        got = [s for s in ap.segments if s.text == "うん"]
        check("話者を選ばなければ ✓ を立てない",
              len(got) == 1 and got[0].speaker_id is None
              and got[0].reviewed is False)


        # --- 足す小窓: 位置をクリックして①②③と積む（2026-08-18）--------
        # 実機の指摘 3 件への対応。壊れたら落とす。
        #   (1) どこで入れるか分からない → 本文を出してクリックで決める
        #   (2) 全部聴けない → ［全部聴く］
        #   (3) 挿入位置が分からない・複数入れられない → ①②③を本文に出して積む
        dseg = Segment(index=0, start=100.0, end=110.0,
                       text="あいうえおかきくけこ", cluster="g:A", chunk=0)

        class _T:
            def __init__(self, s, e, spk):
                self.start, self.end, self.speaker = s, e, spk

        dlg = assign_gui.AddUtteranceDialog(awin, dseg, [_T(104.0, 104.8, 1)])
        dlg.withdraw()
        dlg.update()
        check("小窓に本文が出る",
              dlg.txt.get("1.0", "end").strip() == "あいうえおかきくけこ")
        check("本文は書き換えられない",
              dlg._on_key(type("E", (), {"keysym": "a"})()) == "break")
        check("移動キーは通す",
              dlg._on_key(type("E", (), {"keysym": "Right"})()) is None)

        # 位置 → 時刻（文字数按分）
        dlg.txt.mark_set("insert", "1.0")
        dlg._on_move()
        check("先頭なら区間の開始", abs(dlg._estimate() - 100.0) < 0.05)
        dlg.txt.mark_set("insert", "1.0 + 5 chars")
        dlg._on_move()
        check("真ん中なら区間の中央（文字数按分）",
              abs(dlg._estimate() - 105.0) < 0.05)
        check("入る場所を文字で示す", "【ここ】" in dlg.var_where.get())
        check("時刻は目安だと明記する", "目安" in dlg.var_at.get())
        check("入る場所に色が付く", bool(dlg.txt.tag_ranges("here")))

        # 1 件目を積む → 本文に ① が出る
        dlg.var_text.set("えー")
        dlg._add_item()
        check("積むと本文に①が出る", "①" in dlg.txt.get("1.0", "end"))
        check("①は本文の位置に入る",
              dlg.txt.get("1.0", "end").strip() == "あいうえお①かきくけこ")
        check("積んだら言葉の欄は空になる", dlg.var_text.get() == "")
        check("一覧に出る", "えー" in dlg.lb.get(0))

        # 2 件目をもっと前の位置に積む → 本文の並び順で①②が振り直る
        dlg.txt.mark_set("insert", "1.0 + 2 chars")
        dlg._on_move()
        dlg.var_text.set("はい")
        dlg.cmb_sp.current(1)
        dlg._add_item()
        disp = dlg.txt.get("1.0", "end").strip()
        check("2 件目も積める", disp.count("①") + disp.count("②") == 2)
        check("印は本文の並び順に振る", disp == "あい①うえお②かきくけこ")
        check("一覧も本文の順に並ぶ",
              "はい" in dlg.lb.get(0) and "えー" in dlg.lb.get(1))
        check("何件積んだかボタンに出る", "2 件" in dlg.btn_ok.cget("text"))

        # 印があってもクリック位置は本文の位置に戻せる
        dlg.txt.mark_set("insert", "1.0 + 3 chars")   # ① の直後（表示上）
        dlg._on_move()
        check("印の分を差し引いて位置を取る", dlg._cut == 2)

        # 消す・直す
        dlg.lb.selection_clear(0, "end")
        dlg.lb.selection_set(1)
        dlg._del_item()
        check("積んだものを消せる", len(dlg._items) == 1)
        check("消すと本文の印も減る", "②" not in dlg.txt.get("1.0", "end"))
        dlg.lb.selection_set(0)
        dlg._edit_item()
        check("直すと組み立て欄に戻る",
              dlg.var_text.get() == "はい" and len(dlg._items) == 0)

        # 全部聴く / この辺だけ
        played: list = []
        real_play = awin.player.play
        awin.player.play = lambda *a4, **k4: played.append((a4, k4))
        try:
            dlg._play_all()
            check("全部聴くは区間の頭から終わりまで",
                  played and played[-1][0][1] == 100.0
                  and played[-1][0][2] == 110.0)
            played.clear()
            dlg._play()
            check("この辺だけは挿入位置の周りだけ",
                  played and played[-1][0][2] - played[-1][0][1] < 6.0)
            check("速さが再生に渡る",
                  played[-1][1].get("speed") == dlg._speed())
        finally:
            awin.player.play = real_play
            dlg._stop_follow()
        check("小窓で 0.5 倍が選べる", "0.5x" in assign_gui.DIALOG_SPEEDS)
        dlg._toggle_pause()      # 鳴っていなくても落ちない

        # まとめて入れる
        dlg.var_text.set("")
        dlg._items = []
        warned2: list = []
        real_warn2 = assign_gui.messagebox.showwarning
        assign_gui.messagebox.showwarning = lambda *a5, **k5: warned2.append(1)
        try:
            dlg._ok()
        finally:
            assign_gui.messagebox.showwarning = real_warn2
        check("何も積んでいなければ入れさせない",
              warned2 and dlg.result == [])
        dlg.txt.mark_set("insert", "1.0 + 1 chars")
        dlg._on_move()
        dlg.var_text.set("うん")
        dlg._add_item()
        dlg.txt.mark_set("insert", "1.0 + 6 chars")
        dlg._on_move()
        dlg.var_text.set("ええ")     # 打ちかけのまま押しても拾う
        dlg._ok()
        check("まとめて複数件返す", len(dlg.result) == 2)
        check("打ちかけの 1 件も拾う",
              any(r["text"] == "ええ" for r in dlg.result))
        check("本文のどこかを返す（区間を割る位置になる）",
              all("cut" in r for r in dlg.result))


        # --- 出力の書き方を選ぶ（設計書 §11・2026-08-18）------------------
        # 標準の表記法を調べたら BTSJ に規定があった。データは 1 つのまま、
        # 出し方だけを 2 通りから選ぶ。小窓を実際に開いて Word まで出す。
        Path(ap.audio_path).write_bytes(b"RIFF-dummy")
        base_seg = ap.segments[1]
        ap.add_utterance(14.0, 14.6, "はいはい", cluster="g:B", cut=2,
                         parent_orig=base_seg.orig_start)
        added_one = [s for s in ap.segments if s.text == "はいはい"][0]
        added_one.speaker_id = ap.speakers[0].id

        def _all_widgets(w):
            out = [w]
            for c in w.winfo_children():
                out.extend(_all_widgets(c))
            return out

        def _export(style, out_path):
            """出力設定の小窓を開き、書き方を選んで［保存先を選ぶ］を押す。"""
            seen = {}

            def fake_wait(dlg):
                dlg.update()
                ws = _all_widgets(dlg)
                seen["radios"] = [w for w in ws
                                  if w.winfo_class() == "TRadiobutton"]
                if style is not None:
                    awin.var_insert_style.set(style)
                for w in ws:
                    if (w.winfo_class() == "TButton"
                            and "保存先" in str(w.cget("text"))):
                        w.invoke()
                        return
                dlg.destroy()

            real_wait = awin.wait_window
            real_save = assign_gui.filedialog.asksaveasfilename
            real_err = assign_gui.messagebox.showerror
            real_ask = assign_gui.messagebox.askyesno
            # 出力で落ちるとエラーの小窓で止まる。記録して先へ進める
            assign_gui.messagebox.showerror = (
                lambda *a, **k: seen.setdefault('error', a))
            # **出力の最後に askyesno(出力完了) が出る。**差し替えないと
            # ここで止まる(2026-08-28 に実際に止まった)。
            # **必ず False を返す。**True にすると Word が本当に起動する。
            assign_gui.messagebox.askyesno = (
                lambda *a, **k: (seen.setdefault('done', a), False)[1])
            awin.wait_window = fake_wait
            assign_gui.filedialog.asksaveasfilename = (
                lambda *a, **k: str(out_path))
            try:
                awin.export_docx()
            finally:
                awin.wait_window = real_wait
                assign_gui.filedialog.asksaveasfilename = real_save
                assign_gui.messagebox.showerror = real_err
                assign_gui.messagebox.askyesno = real_ask
            return seen

        def _docx_text(path):
            from docx import Document
            return chr(10).join(
                par.text for par in Document(str(path)).paragraphs)

        seen = _export(assign_gui.INSERT_STYLE_LINE, tmp / "style_line.docx")
        check("出力で落ちない", "error" not in seen)
        check("差し込みがあれば書き方を選べる", len(seen.get("radios", [])) == 2)
        body = _docx_text(tmp / "style_line.docx")
        check("行を分ける形で出る（,, でつなぐ）", "発言,," in body)
        check("読み方の説明が付く", "BTSJ" in body and "表記について" in body)
        check("相づちが別の行に出る",
              any(ln.strip().endswith("はいはい") for ln in body.splitlines()))
        check("割られた後半に時刻を書かない",
              any("1。" in ln and "[" not in ln for ln in body.splitlines()))

        seen2 = _export(assign_gui.INSERT_STYLE_INLINE, tmp / "style_inline.docx")
        body2 = _docx_text(tmp / "style_inline.docx")
        check("行に埋め込む形も出せる", "(佐藤：はいはい)" in body2)
        check("埋め込みでは ,, を使わない", ",," not in body2)
        check("埋め込みにも読み方の説明が付く", "表記について" in body2)
        check("同じ作業ファイルから両方出せる",
              (tmp / "style_line.docx").exists()
              and (tmp / "style_inline.docx").exists())
        check("言葉そのものは変わらない",
              ("はいはい" in body) and ("はいはい" in body2))

        # 差し込みが無ければ、選びようが無いので出さない（凡例も出さない）
        for s in [s for s in ap.segments if ap.is_added_utterance(s)]:
            ap.remove_added_utterance(s.index)
        seen3 = _export(None, tmp / "style_none.docx")
        check("差し込みが無ければ選択肢を出さない", seen3.get("radios") == [])
        # 「凡例」は検証要約にもある見出しなので、それでは判定できない
        check("差し込みが無ければ読み方の説明も出さない",
              "表記について" not in _docx_text(tmp / "style_none.docx"))

        # --- 時刻へ飛ぶ（2026-08-18 の指摘への対応）----------------------
        # 「聴く順にすると時間順でないので探せない」。700 区間をスクロールで
        # 探すのは現実的でない。逐語正解の道具にはあった機能で、本体に
        # 無いのは抜けだった。
        jp = Project(audio_path=str(tmp / "meeting.m4a"),
                     duration=3600.0, chunk_seconds=600)
        jp.speakers = parse_roster("佐藤\n田中")
        jp.segments = [
            Segment(index=i, start=1490.0 + i * 6.0, end=1490.0 + i * 6.0 + 5.0,
                    text=f"発言 {i}。", cluster="g:A", chunk=0)
            for i in range(6)
        ]
        jp.json_path = str(tmp / "jump.speakers.json")
        jp.save()
        jwin = AssignWindow(root, jp)
        jwin.var_autoplay.set(False)
        jwin.update()

        # **［時刻へ飛ぶ］が画面から押し出されていないこと。**
        # 「候補のみ」を足したぶんで絞り込みの行が 840px になり、左ペイン
        # (775px)から入力欄が消えて押せなくなった(実機の指摘・2026-08-19)。
        # **本体と同じ式にする**(assign_gui.py:653)。800 と書いていたので
        # 本体より低い窓で測っていた。厳しい側なので害は無かったが、
        # 別の式が 2 つあると、片方だけ直したときに食い違う。
        jwin.geometry(f"{min(1320, jwin.winfo_screenwidth() - 40)}"
                      f"x{min(900, jwin.winfo_screenheight() - 90)}")
        jwin.update()
        _je = None
        _stack = [jwin]
        while _stack:
            _w = _stack.pop()
            _stack.extend(_w.winfo_children())
            if (_w.winfo_class() == "TEntry"
                    and str(_w.cget("textvariable")) == str(jwin.var_jump)):
                _je = _w
        check("［時刻へ飛ぶ］の入力欄が見えている",
              _je is not None and _je.winfo_ismapped() and _je.winfo_width() > 10)
        _frows = [r for r in jwin.tree.master.winfo_children()
                  if r is not jwin.tree and r.winfo_children()
                  and r.winfo_class() == "TFrame"]
        check("絞り込みの行が左ペインに収まる",
              all(visible_need(r, 10)[0] <= jwin.tree.master.winfo_width()
                  for r in _frows))
        _lost_f = [c for r in _frows for c in squeezed_out(r)]
        check(f"絞り込みの行で押し出された部品が無い（{lost_names(_lost_f) or 'なし'}）",
              not _lost_f)
        check("一覧が見えている", jwin.tree.winfo_ismapped())

        jwin.var_jump.set("25:02")          # 1502 秒 = 3 番目の区間の先頭
        jwin.jump_to_time()
        check("分:秒 で飛べる", jp.segments[jwin.current].start == 1502.0)

        jwin.var_jump.set("00:25:20")       # 1520 秒 = 6 番目
        jwin.jump_to_time()
        check("時:分:秒 でも飛べる", jp.segments[jwin.current].start == 1520.0)

        jwin.var_jump.set("1508")           # 秒だけでも
        jwin.jump_to_time()
        check("秒だけでも飛べる", jp.segments[jwin.current].start == 1508.0)

        jwin.var_jump.set("9999")           # どの区間にも入らない → 一番近い
        jwin.jump_to_time()
        check("どの区間にも入らなければ一番近くへ",
              jp.segments[jwin.current].start == 1520.0)

        jwin.var_jump.set("あいうえお")
        jwin.jump_to_time()
        check("読めない時刻は知らせるだけ（落ちない）",
              "読めません" in jwin.var_action.get())

        # 絞り込みで隠れている区間にも飛ぶ（探しに来たのに出ないのは困る）
        jp.segments[0].speaker_id = jp.speakers[0].id
        jp.segments[0].reviewed = True
        jwin.var_filter.set(FILTER_UNASSIGNED)
        jwin.reload_tree()
        jwin.var_jump.set("24:50")          # 1490 秒 = 確定済みの 0 番目
        jwin.jump_to_time()
        check("絞り込みで隠れていても飛ぶ", jwin.current == 0)
        check("絞り込みを戻したと知らせる",
              jwin.var_filter.get() == FILTER_ALL)
        jwin.player.close()
        jwin.destroy()

        awin.player.close()
        awin.destroy()
        root.destroy()

    if leaked:
        print(f"\n※ 差し替えていないモーダルが {len(leaked)} 件出ました"
              f"(止血して先へ進めています。検査の側で差し替えてください):")
        for name, title in leaked:
            print(f"    {name}({title!r})")
    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


# ======================================================================
# メイン画面(処理経路の選択)
#
# **差し替えは本物と同じ形にする。**save_config は「包めずに落とした項目の
# 並び」を返す。None を返す差し替えのままだと、画面側でそれを見る処理が
# 落ちる（2026-08-29 に実際に踏んだ）。
# 設定ファイルには触らない。load_config / save_config を差し替えて、
# 利用者の config.json を書き換えないようにする。
# ======================================================================

def run_inline_inserts() -> int:
    """**足した発言は、一覧に行を増やさず親の本文へ埋め込む。**

    Word の出力は前から「割り込んだ位置に差し込む」形だったのに、一覧だけ
    独立した行として並べていた。同じものが 2 か所に見えて分かりにくい、と
    指摘を受けた（実機・2026-08-24）。

    **画面と出力で同じ計算を使うこと**が要点。別々に実装すると「画面では
    ここ、出力では別のところ」という食い違いが起きる。
    """
    failures: list[str] = []

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        if not cond:
            failures.append(label)

    print("\n[足した発言を本文に埋め込む]")
    root = tk.Tk()
    root.withdraw()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proj = make_project(tmp)
            proj.save()
            win = AssignWindow(root, proj)
            win.var_autoplay.set(False)
            win.update()
            rows_before = len(win.tree.get_children())

            parent = proj.segments[1]
            add = proj.add_utterance(
                parent.start + 0.1, parent.start + 0.4, "はい",
                cluster=parent.cluster, cut=3,
                parent_orig=parent.orig_start, parent_start=parent.start)
            add.speaker_id = proj.speakers[0].id
            win.reload_tree()
            win.update()

            check("一覧の行が増えない",
                  len(win.tree.get_children()) == rows_before)
            vals, tags = win._row_values(parent)
            check("親の本文の、割り込んだ位置に入る",
                  vals[3].startswith("これは（") and "：はい）" in vals[3])
            check("親の行にも印が付く", "bg_added" in tags)

            # **画面と出力が同じ位置を指すこと**
            from src.segments import inserted_marks, marks_for_segment
            marks, _ = inserted_marks(proj)
            out_cut = marks_for_segment(marks, parent)[0][0]
            gui_cut = win.inserted_in(parent)[0][0]
            check("画面と出力が同じ位置を使う", out_cut == gui_cut == 3)

            # 右ペインが唯一の入口になる
            win.goto(parent.index)
            win.update()
            check("右ペインに「重なって入っている発言」が出る",
                  win.row_inserted.winfo_ismapped())

            # 行で出す設定に戻せる
            win.var_added_rows.set(True)
            win._on_added_rows_toggled()
            win.update()
            check("行で出す設定にすると行が増える",
                  len(win.tree.get_children()) == rows_before + 1)
            check("行で出すときは埋め込まない",
                  "：はい）" not in win._row_values(parent)[0][3])
            win.var_added_rows.set(False)
            win._on_added_rows_toggled()
            win.update()

            # 消す入口が効く
            real_yes = assign_gui.messagebox.askyesno
            assign_gui.messagebox.askyesno = lambda *a2, **k2: True
            try:
                win._remove_inserted(add.index)
            finally:
                assign_gui.messagebox.askyesno = real_yes
            check("右ペインから消せる",
                  not any(s.text == "はい" for s in proj.segments))
            check("消したら埋め込みも消える",
                  "：はい）" not in win._row_values(proj.segments[1])[0][3])

            win.destroy()
    finally:
        root.destroy()

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


def run_candidate_scroll() -> int:
    """**出席者が多くても、全員を選べるか。**

    候補のボタンは人数ぶん作られていたのに、入れ物の高さが足りず下が
    切れていた。9 人のうち 6 人しか出ず、残りを選べなかった（実機の指摘・
    2026-08-24）。**画面からは「そもそも居ない」ようにしか見えない**種類の
    欠陥なので、数だけでなく「送れば見える」ことまで確かめる。
    """
    failures: list[str] = []

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        if not cond:
            failures.append(label)

    print("\n[候補が多いときのスクロール]")
    root = tk.Tk()
    root.withdraw()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proj = make_project(tmp)
            names = ["三ツ林", "志村", "山口", "山本", "高原", "山下",
                     "梅田", "吉沢", "西村", "佐藤", "田中", "鈴木"]
            proj.speakers = parse_roster("\n".join(names))
            proj.save()
            win = AssignWindow(root, proj)
            win.var_autoplay.set(False)
            win.update()

            check(f"候補が人数ぶん作られる（{len(names)} 人）",
                  len(win._candidates) == len(names))

            win.update_idletasks()
            inner = win.cand_holder.winfo_height()
            view = win.cand_canvas.winfo_height()
            check("入れ物より中身が高い（切れる状況を再現できている）",
                  inner > view)
            check("スクロールできる",
                  win.cand_canvas.yview()[1] < 1.0)
            check("要るときはスクロールバーが出る",
                  win.cand_scroll.winfo_ismapped())

            # **いちばん下の候補まで届く**
            win.cand_canvas.yview_moveto(1.0)
            win.update_idletasks()
            check("下端まで送れる", win.cand_canvas.yview()[1] >= 0.999)

            # 選ばれている候補が隠れていたら、見える位置へ送る
            last = win._candidates[-1]
            here = win.current          # 割当後は自動で次へ進む
            win.assign(last.speaker.id)
            win.update()
            check("最後の人も選べる",
                  proj.segments[here].speaker_id == last.speaker.id)

            # **隠れている候補は、見える位置へ送る。**割当後は並び順が
            # 学習で変わる（選んだ人が上に来る）ので、送る処理そのものを
            # 直接確かめる。
            win.cand_canvas.yview_moveto(0.0)
            win.update_idletasks()
            before = win.cand_canvas.yview()[0]
            bottom_btn = win._cand_widgets[len(win._candidates) - 1]
            win._scroll_candidate_into_view(bottom_btn)
            win.update_idletasks()
            check("隠れている候補は見える位置へ送られる",
                  win.cand_canvas.yview()[0] > before)

            win.destroy()
    finally:
        root.destroy()

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


def run_multi_select() -> int:
    """**複数選択で話者をまとめて指定できるか。**

    Shift で並び、Ctrl で飛び飛びに選び、一度に変える（実機の要望・
    2026-08-23）。**選んだ全部が △ になること**が要点——Shift+クリックで
    20 区間を一度に選べる操作で ✓ を立てると、「✓＝人が耳で聴いて確定」
    が崩れる（CLAUDE.md）。
    """
    failures: list[str] = []

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        if not cond:
            failures.append(label)

    print("\n[複数選択で話者を一括指定]")
    root = tk.Tk()
    root.withdraw()
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            proj = make_project(tmp)
            proj.save()
            win = AssignWindow(root, proj)
            win.var_autoplay.set(False)
            win.update()

            check("一覧が複数選択できる",
                  str(win.tree.cget("selectmode")) == "extended")

            sid = proj.speakers[0].id
            iids = [f"s{i}" for i in (0, 2, 3)]
            win.tree.selection_set(iids)
            win.tree.focus(iids[0])
            win.update()
            check("選んだ数を取り出せる",
                  win.selected_indices() == [0, 2, 3])

            win.assign(sid)
            win.update()
            got = [proj.segments[i] for i in (0, 2, 3)]
            check("選んだ全部に話者が入る",
                  all(s.speaker_id == sid for s in got))
            check("選ばなかった区間は変わらない",
                  not proj.segments[1].speaker_id)
            # **ここが要点**
            check("選んだ全部が △（未確認）になる",
                  all(s.reviewed is False for s in got))

            check("編集履歴は 1 件の判断として残る",
                  proj.edit_log[-1]["op"] == "assign_bulk"
                  and proj.edit_log[-1]["count"] == 3)

            check("適用しても選択が外れない",
                  set(win.tree.selection()) == set(iids))

            win.undo()
            win.update()
            check("Ctrl+Z で 3 件まとめて戻る",
                  all(not proj.segments[i].speaker_id for i in (0, 2, 3)))

            # 1 つだけ選んだときは、これまでどおり ✓
            win.tree.selection_set("s1")
            win.tree.focus("s1")
            win.update()
            win.assign(sid)
            check("1 つだけ選べば ✓ のまま",
                  proj.segments[1].reviewed is True)

            win.destroy()
    finally:
        root.destroy()

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


def grid_overlaps(root) -> list[str]:
    """grid で同じマスに 2 つ以上置かれている箇所を全部挙げる。

    **同じ場所で 2 度目だったので、機械に見張らせる。**

    2026-08-20 に「モデル欄を下の行(チャンク長)に直接置いたら重なった」を
    直したが、**枠に入れる対処で、行の衝突そのものは残っていた。**
    2026-08-31 に実機で再発（モデル欄が「タイムスタンプを付ける」に半分
    かぶる／チャンク長の行が「話者を識別する」と重なる）。

    ----------------------------------------------------------------------
    ## この検査が守れる範囲・守れない範囲

    **守れる**: 同じ親の中で、(行, 列) の矩形が重なっていること。
    `columnspan` / `rowspan` も見る。

    **守れない**: **重なっていない崩れ。**幅が足りずに押し出される、
    文字がはみ出す、余白が詰まりすぎる——どれも grid の座標は正しいので
    ここでは捕まらない（幅の検査は別にある）。

    **守れない(2)**: `grid_remove()` で外したものは対象外。**外し忘れて
    見えている**場合は捕まるが、**外すべきなのに外していない**という設計上の
    判断は機械には分からない。
    """
    problems: list[str] = []

    def name_of(w) -> str:
        for attr in ("cget",):
            try:
                t = w.cget("text")
                if t:
                    return f"{w.winfo_class()}({str(t)[:26]})"
            except Exception:
                pass
        return f"{w.winfo_class()}@{w.winfo_name()}"

    def walk(parent) -> None:
        try:
            slaves = parent.grid_slaves()
        except Exception:
            slaves = []
        boxes = []
        for w in slaves:
            try:
                info = w.grid_info()
            except Exception:
                continue
            if not info:
                continue            # grid_remove 済み
            r = int(info.get("row", 0))
            c = int(info.get("column", 0))
            rs = int(info.get("rowspan", 1) or 1)
            cs = int(info.get("columnspan", 1) or 1)
            boxes.append((w, r, r + rs - 1, c, c + cs - 1))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                _wa, r0a, r1a, c0a, c1a = boxes[i]
                _wb, r0b, r1b, c0b, c1b = boxes[j]
                if max(r0a, r0b) <= min(r1a, r1b) and \
                   max(c0a, c0b) <= min(c1a, c1b):
                    problems.append(
                        f"{parent.winfo_class()}@{parent.winfo_name()}: "
                        f"行{max(r0a, r0b)}〜{min(r1a, r1b)}／"
                        f"列{max(c0a, c0b)}〜{min(c1a, c1b)} で "
                        f"{name_of(boxes[i][0])} と {name_of(boxes[j][0])} が重なる")
        for w in parent.winfo_children():
            walk(w)

    walk(root)
    return problems

def run_main_window() -> int:
    failures: list[str] = []

    def check(label: str, cond: bool) -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {label}")
        if not cond:
            failures.append(label)

    import src.gui as main_gui
    from src.segments import ENGINE_CLOUD, ENGINE_LOCAL

    saved: dict = {}
    real_load, real_save = main_gui.load_config, main_gui.save_config
    main_gui.load_config = lambda: {}
    main_gui.save_config = lambda d: (saved.update(d), [])[1]

    print("\n[メイン画面 / 処理経路]")
    try:
        app = main_gui.App()
        try:
            # --- 画面に収まるか ---------------------------------------
            # 中身は 1000px を超える。窓を画面より高くすると、下端の
            # 進捗とログがタスクバーの裏へ回って処理の様子が見えなくなる。
            app.update()
            # **grid のマスが重なっていないか。**同じ場所で 2 度目の再発を
            # 機械に見張らせる(2026-08-20 / 2026-08-31)。
            _ov = grid_overlaps(app)
            for _p in _ov:
                print(f"       {_p}")
            check("grid のマスが重なっていない", not _ov)
            check("窓が画面に収まる",
                  app.winfo_height() <= app.winfo_screenheight() - 40)
            bbox = app._canvas.bbox("all")
            content_h = (bbox[3] - bbox[1]) if bbox else 0
            check("中身が窓より高いときはスクロールできる",
                  content_h > app._canvas.winfo_height()
                  and app._canvas.yview()[1] < 1.0)
            app._scroll_to_log()
            app.update()
            log_y = app.txt_log.winfo_rooty() - app._canvas.winfo_rooty()
            check("送ればログ欄まで届く",
                  0 <= log_y < app._canvas.winfo_height())

            # --- 名簿も 2 列の表(設計書 §11.8)---------------------------
            # 文字起こし画面と割当画面で同じ部品を使う。**どちらも外部には
            # 送らない**(分割は端末内の規則)。
            app.tbl_roster.set_text(chr(10).join([
                "三ツ林衆議院議員",
                "山本学　文科省　高等教育局私学部参事官付 企画官",
                "梅田茂(加茂暁星学園理事)"]))
            app.update()
            check("貼り込んだ一覧が行に分かれる", len(app.tbl_roster.rows) == 3)
            check("貼り込んだ時点で名前と役職に分かれる",
                  app.tbl_roster.rows[0]["name"].get() == "三ツ林"
                  and app.tbl_roster.rows[0]["note"].get() == "衆議院議員")
            check("すでに分かれている行はそのまま",
                  app.tbl_roster.rows[2]["name"].get() == "梅田茂")
            check("転写経路には「名前(役職)」の形で渡す",
                  "三ツ林(衆議院議員)" in app._get_roster())
            check("入れ子の括弧があっても往復する",
                  "山本学" in app._get_roster())

            # 往復しても壊れない(保存 → 読み直しに相当)
            once = app._get_roster()
            app.tbl_roster.set_text(once)
            check("保存して読み直しても同じ", app._get_roster() == once)

            app.tbl_roster.add_row(focus=False)
            check("行を足せる", len(app.tbl_roster.rows) == 4)
            check("名前が空の行は人として数えない",
                  len(app.tbl_roster.values()) == 3)
            app.tbl_roster.del_row(app.tbl_roster.rows[-1])
            check("行を消せる", len(app.tbl_roster.rows) == 3)

            app._split_roster()
            check("分けるものが無ければそう言う",
                  "ありません" in app.var_roster_note.get())
            app.tbl_roster.set_enabled(False)
            app.tbl_roster.set_enabled(True)
            check("有効・無効を切り替えても落ちない",
                  len(app.tbl_roster.values()) == 3)
            # 後の検査は名簿が空である前提(空なら話者分離の人数を聞く)。
            # ここで戻しておかないと、そちらが落ちる。
            app.tbl_roster.set_text("")
            check("空にできる", app._get_roster().strip() == "")

            # --- ホイールの行き先 ---------------------------------------
            # 窓がスクロールするようになると、設定欄の上でホイールを回す
            # 機会が増える。中身と窓が同時に動いたり、値が黙って変わったり
            # しないことを見る(チャンク長はキャッシュキーに入るので、
            # 変わると転写がまるごとやり直しになる)。
            app.txt_log.configure(state="normal")
            app.txt_log.insert("end", "".join(f"行 {i}\n" for i in range(200)))
            app.txt_log.configure(state="disabled")
            app._canvas.bind_all("<MouseWheel>", app._on_wheel)   # 乗った状態を作る
            app.txt_log.yview_moveto(0.5)
            app.update()

            before_c, before_t = app._canvas.yview()[0], app.txt_log.yview()[0]
            app.txt_log.event_generate("<MouseWheel>", delta=120, x=10, y=10)
            app.update()
            check("ログ欄の上では中身だけ動く",
                  app.txt_log.yview()[0] != before_t
                  and app._canvas.yview()[0] == before_c)

            before_chunk = app.var_chunk.get()
            before_c = app._canvas.yview()[0]
            app.spin_chunk.event_generate("<MouseWheel>", delta=120, x=5, y=5)
            app.update()
            check("チャンク長はホイールで変わらない",
                  app.var_chunk.get() == before_chunk)
            check("チャンク長の上でも画面は送れる",
                  app._canvas.yview()[0] != before_c)

            before_model = app.var_model.get()
            app.cmb_model.event_generate("<MouseWheel>", delta=120, x=5, y=5)
            app.update()
            check("モデルはホイールで変わらない",
                  app.var_model.get() == before_model)

            # --- GPU があるのに小さいモデルを選んでいたら知らせる ---------
            # **勝手に切り替えない。**設定に前回の値が残るので、GPU 機でも
            # small のまま走り出すことがある(実機で発生・2026-08-20)。
            from src import align as _al
            _real_default = _al.default_model
            try:
                _al.default_model = lambda device=None: "large-v3"
                app.var_engine.set(main_gui.ENGINE_LOCAL)
                app.var_model.set("small")
                app._update_gpu_hint()
                app.update()
                _txt = str(app.lbl_gpu_hint.cget("text"))
                check("GPU があるのに small なら注記が出る", "large-v3" in _txt)
                # 2026-08-31 まで「誤字」と「速く」で見ていたが、**どちらも
                # 撤回した主張**である(誤字は定義が復元できず、測り直すと
                # 向きが逆。速さは機械で変わるので比を書かない。
                # 設計書 §9.5.1 の注記)。**残った根拠は固有名詞だけ。**
                check("理由も書いてある（判断できるように）",
                      "固有名詞" in _txt)
                check("勝手に切り替えない", app.var_model.get() == "small")

                # **断ったときにも案内を出す。**3.6GB なので断るほうが多い
                # 経路だが、これまで断った側には何も出ていなかった(取得に
                # 失敗したときだけ「small のまま使えます」が出た)。
                # 「取得しないと使えないのか」と読まれると、この製品で
                # いちばん誤解されたくない点(ローカル完結)が崩れる。
                # **断った直後は large-v3 が手元に無い。**上の差し替え
                # (default_model が large-v3 を返す)のままだと GPU の案内が
                # 優先され、ありえない状態を検査することになる。実態に戻す。
                _al.default_model = lambda device=None: _al.DEFAULT_MODEL
                app._gpu_declined = True
                app._update_gpu_hint()
                app.update()
                _dec = str(app.lbl_gpu_hint.cget("text"))
                check("断ったら「そのまま使える」と出る",
                      "このまま使えます" in _dec and _al.DEFAULT_MODEL in _dec)
                # **ログには出さない。**ログ欄は起動直後は窓の外(実測)。
                check("その案内が窓の中に見えている",
                      app.lbl_gpu_hint.winfo_ismapped()
                      and app.lbl_gpu_hint.winfo_rooty()
                      < app.winfo_rooty() + app.winfo_height())
                app._gpu_declined = False

                # **選んだら、その場で覚える。**覚える処理が「経路を
                # 切り替えたとき」と「転写を開始したとき」にしか無く、
                # 選んで放置すると元に戻っていた(実機の指摘・2026-08-20)。
                _saved = {}
                _real_save = main_gui.save_config
                # **検査環境では large-v3 が「未取得」に見える**(APPDATA を
                # 差し替えてあるため)。そのままモデル欄で選ぶと**取得ダイアログが
                # 実際に開き、検査が止まる**——aac6d05 で CI が 30 分で
                # 打ち切られたのと同じ形。取得済みに見せてから確かめる。
                _real_find = _al.find_bundled_model
                _real_ask = main_gui.messagebox.askyesno
                try:
                    main_gui.save_config = lambda d: (_saved.update(d), [])[1]
                    _al.find_bundled_model = lambda m: (
                        Path("x") if m == "large-v3" else _real_find(m))
                    # 万一ここを通っても止まらないようにする(二重の備え)
                    main_gui.messagebox.askyesno = lambda *a, **k: False
                    app.var_model.set("large-v3")
                    app._on_model_chosen()
                    check("選んだモデルを覚える",
                          app._model_by_engine[main_gui.ENGINE_LOCAL]
                          == "large-v3")
                    check("設定にも書く", _saved.get("local_model") == "large-v3")
                    # 経路を往復しても戻らない
                    app.var_engine.set(main_gui.ENGINE_CLOUD)
                    app._update_engine_state()
                    app.var_engine.set(main_gui.ENGINE_LOCAL)
                    app._update_engine_state()
                    check("経路を往復しても戻らない",
                          app.var_model.get() == "large-v3")
                    # **他方の経路の欄に漏らさない。**確かめずに書いて
                    # いたため、ローカルで選んだ large-v3 がクラウドの欄に
                    # 入り、設定ファイルが壊れていた(実機で発生・
                    # 2026-08-20。model が 'large-v3' になっていた)。
                    check("クラウドの欄に漏れていない",
                          app._model_by_engine[main_gui.ENGINE_CLOUD]
                          in main_gui.MODELS)
                    check("設定のクラウド側も汚れていない",
                          _saved.get("model", main_gui.MODELS[0])
                          in main_gui.MODELS)
                    # **未取得のものを選んだら、その場で取得を尋ねる。**
                    # 黙って選ばせると、転写を始めてから 3GB を落としに行く。
                    # 断ったら標準へ戻す(選んだままにすると、次の転写で
                    # やはり落としに行く)。
                    _asked = []
                    _al.find_bundled_model = lambda m: (
                        None if m == "large-v3" else _real_find(m))
                    main_gui.messagebox.askyesno = lambda *a, **k: (
                        _asked.append(a[0] if a else ""), False)[1]
                    app.var_model_label.set(main_gui.MODEL_LABEL_NEEDS_FETCH)
                    app._on_model_chosen()
                    check("未取得の高精度を選んだら取得を尋ねる", bool(_asked))
                    check("断ったら標準に戻る",
                          app.var_model.get() == _al.DEFAULT_MODEL)
                finally:
                    main_gui.save_config = _real_save
                    _al.find_bundled_model = _real_find
                    main_gui.messagebox.askyesno = _real_ask
                    app.var_model.set("small")
                    app._model_by_engine[main_gui.ENGINE_LOCAL] = "small"
                    app._update_gpu_hint()

                # **重なっていないかを見る。**文字の中身だけを検査していた
                # ため、チャンク長の行に重ねたまま通っていた(実機の指摘・
                # 2026-08-20)。§10.3.6 と同じ種類の見落とし。
                #
                # **窓が見えているうちにやること。**withdraw のあとだと
                # 部品の大きさが取れず、重なり判定が素通りする(これも
                # 一度やった)。「見えている」の検査を必ず先に置く。
                def _box(w):
                    return (w.winfo_rootx(), w.winfo_rooty(),
                            w.winfo_rootx() + w.winfo_width(),
                            w.winfo_rooty() + w.winfo_height())

                def _hits(x, y):
                    ax1, ay1, ax2, ay2 = _box(x)
                    bx1, by1, bx2, by2 = _box(y)
                    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2

                check("注記が見えている（これが偽なら下の検査は無意味）",
                      app.lbl_gpu_hint.winfo_ismapped()
                      and app.lbl_gpu_hint.winfo_width() > 10)
                for _name, _w in (("モデル欄", app.cmb_model),
                                  ("チャンク長", app.spin_chunk)):
                    check(f"注記が{_name}に重ならない",
                          not _hits(app.lbl_gpu_hint, _w))
                check("注記はモデル欄の下にある",
                      app.lbl_gpu_hint.winfo_rooty()
                      >= app.cmb_model.winfo_rooty()
                      + app.cmb_model.winfo_height())

                app.var_model.set("large-v3")
                app._update_gpu_hint()
                check("選び直したら注記は消える",
                      str(app.lbl_gpu_hint.cget("text")) == "")
                _al.default_model = lambda device=None: _al.DEFAULT_MODEL
                app.var_model.set("small")
                app._update_gpu_hint()
                check("GPU が無ければ出さない",
                      str(app.lbl_gpu_hint.cget("text")) == "")
            finally:
                _al.default_model = _real_default
                app.var_model.set(main_gui.DEFAULT_LOCAL_MODEL)
                app._update_gpu_hint()

            app._canvas.unbind_all("<MouseWheel>")
            app.withdraw()
            # 既定はローカル。録音を外へ出す判断を既定に紛れ込ませない
            check("既定はローカル", app.var_engine.get() == ENGINE_LOCAL)
            check("ローカルでは API キー欄を触れない",
                  str(app.entry_api.cget("state")) == "disabled")
            check("ローカルでは逐語モードを触れない",
                  str(app.chk_verbatim.cget("state")) == "disabled")
            # 入ったまま灰色にすると「有効なのに触れない」と読める。
            # 実際には効かないので、チェックも外しておく。
            check("ローカルでは逐語モードのチェックも外れる",
                  app.var_verbatim.get() is False)
            check("ローカルでは従来方式を選べない",
                  str(app.rdo_mode_auto.cget("state")) == "disabled")
            check("ローカルのモデル候補が出る",
                  app.var_model.get() in main_gui.LOCAL_MODELS)

            app.var_engine.set(ENGINE_CLOUD)
            app._update_engine_state()
            check("クラウドにすると API キー欄が使える",
                  str(app.entry_api.cget("state")) == "normal")
            check("クラウドにすると逐語モードが使える",
                  str(app.chk_verbatim.cget("state")) == "normal")
            check("クラウドにすると従来方式が選べる",
                  str(app.rdo_mode_auto.cget("state")) == "normal")
            check("クラウドのモデル候補に入れ替わる",
                  app.var_model.get() in main_gui.MODELS)

            # クラウドで入れた逐語モードは、ローカルへ行って戻ると復活する
            app.var_verbatim.set(True)
            app.var_engine.set(ENGINE_LOCAL)
            app._update_engine_state()
            check("ローカルに移ると逐語モードが外れる",
                  app.var_verbatim.get() is False)
            app.var_engine.set(ENGINE_CLOUD)
            app._update_engine_state()
            check("クラウドへ戻すと逐語モードも戻る",
                  app.var_verbatim.get() is True)
            app.var_verbatim.set(False)

            # 選び直させないために、経路ごとのモデルを覚えている
            app.var_model.set("gemini-2.5-pro")
            app.var_engine.set(ENGINE_LOCAL)
            app._update_engine_state()
            app.var_engine.set(ENGINE_CLOUD)
            app._update_engine_state()
            check("経路を戻すとモデルの選択も戻る",
                  app.var_model.get() == "gemini-2.5-pro")

            # 名簿が空のときの人数の問い合わせ。**画面を出す処理なので必ず
            # 差し替える。**差し替え忘れると、テストが応答を待って止まる
            # (実際に止めた)。
            asked = {"n": 0}

            def fake_ask(parent):
                asked["n"] += 1
                return 4

            real_ask_count = main_gui._ask_speaker_count
            main_gui._ask_speaker_count = fake_ask

            # ローカルなら鍵が無くても開始できる(入力が無いことだけ言われる)
            warned: list[str] = []
            real_warn = main_gui.messagebox.showwarning
            main_gui.messagebox.showwarning = lambda t, *a, **k: warned.append(t)
            try:
                app.var_engine.set(ENGINE_LOCAL)
                app._update_engine_state()
                app.var_input.set("")
                app._start()
                check("ローカルでは API キーを求めない",
                      warned and warned[-1] == "入力")
            finally:
                main_gui.messagebox.showwarning = real_warn

            # ローカルで走らせても、クラウド用の逐語モード設定を潰さない
            app.var_engine.set(ENGINE_CLOUD)
            app._update_engine_state()
            app.var_verbatim.set(True)
            app.var_engine.set(ENGINE_LOCAL)
            app._update_engine_state()
            saved.clear()
            app.var_input.set(__file__)          # 実在するファイルなら先へ進む
            app.var_use_input_dir.set(True)
            real_ask = main_gui.messagebox.askyesno
            main_gui.messagebox.askyesno = lambda *a, **k: True
            # 転写そのものは走らせない。走らせるとテストの隣に
            # .work_<名前> フォルダを作ってしまう
            app._run_worker = lambda *a, **k: None       # type: ignore[method-assign]
            try:
                app._start()
            finally:
                main_gui.messagebox.askyesno = real_ask
            check("ローカルで走らせても逐語モードの設定を潰さない",
                  saved.get("verbatim") is True)
            check("経路も保存される", saved.get("engine") == ENGINE_LOCAL)
            check("名簿が空なら人数を聞く", asked["n"] == 1)
            check("話者分離の設定も保存される", saved.get("diarize") is True)

            # 話者分離を切れば人数は聞かない(使わない値を尋ねない)
            app.var_diarize_local.set(False)
            asked["n"] = 0
            main_gui.messagebox.askyesno = lambda *a, **k: True
            app._start()
            check("話者分離を切れば人数を聞かない", asked["n"] == 0)

            # --- 前回どこで作業したかを覚える -----------------------------
            # **覚えていないと毎回ちがう場所が開く。**実機で、G ドライブで
            # 作業したのに C ドライブが開いた(2026-08-22)。「出力フォルダ」は
            # いま選んでいる音声の出力先で、前回の作業場所とは関係がない。
            import tempfile as _tf
            with _tf.TemporaryDirectory() as _d:
                _proj = Path(_d) / "どこか" / "x.speakers.json"
                _proj.parent.mkdir()
                _proj.write_text("{}", encoding="utf-8")
                _saved3 = {}
                main_gui.save_config = lambda x: (_saved3.update(x), [])[1]
                app._remember_project(str(_proj))
                check("開いた作業ファイルを覚える",
                      app.cfg.get(app.LAST_PROJECT_KEY) == str(_proj))
                check("設定にも書く", app.LAST_PROJECT_KEY in _saved3)
                check("次に開くときはその場所",
                      app._last_project_dir() == str(_proj.parent))
                # 覚えた場所が消えていたら、出力フォルダに戻る
                app.cfg[app.LAST_PROJECT_KEY] = str(Path(_d) / "無い" / "y.json")
                app.var_output.set(_d)
                check("消えていたら出力フォルダに戻る",
                      app._last_project_dir() == _d)
            # **本物には戻さない。**ここで real_save に戻したせいで、
            # 続きの操作が利用者の config.json を書いた（2026-08-22）。
            # 戻すのは、この関数の最後の一箇所だけ。
            main_gui.save_config = lambda d: (saved.update(d), [])[1]

            # --- 出力先は覚える / 押す前に見える -----------------------
            # **起動のたびに空へ戻っていた。**毎回選び直しになり、
            # ［参照...］が「前回エクスプローラで開いた場所」を出すため、
            # 関係のないプロジェクトのフォルダへ書き出しかけた
            # （実機で発生・2026-08-22）。
            import tempfile as _tf4
            with _tf4.TemporaryDirectory() as _d4:
                app.var_use_input_dir.set(False)
                app._on_toggle_use_input_dir()
                app.var_output.set(_d4)
                app.update()
                check("開始ボタンのそばに出力先が出る",
                      _d4 in app.lbl_dest.cget("text"))
                check("出力先の表示は開始ボタンと同じ入れ物にある",
                      app.lbl_dest.master is app.btn_start.master)

                _saved4 = {}
                main_gui.save_config = lambda x: (_saved4.update(x), [])[1]
                app.cfg[app.OUTPUT_KEY] = _d4
                app.cfg[app.USE_INPUT_DIR_KEY] = False
                _app4 = main_gui.App.__new__(main_gui.App)
                main_gui.load_config = lambda: dict(app.cfg)
                try:
                    _app4.__init__()
                    _app4.update()
                    check("次に起動したとき出力先が戻る",
                          _app4.var_output.get() == _d4)
                    check("戻したときも表示に出ている",
                          _d4 in _app4.lbl_dest.cget("text"))
                finally:
                    _app4.destroy()
                    main_gui.load_config = lambda: {}

                # **消えていたら戻さない。**外したドライブや別 PC の設定を
                # そのまま使うと、書けない場所へ書きにいく。
                app.cfg[app.OUTPUT_KEY] = str(Path(_d4) / "無い場所")
                _app5 = main_gui.App.__new__(main_gui.App)
                main_gui.load_config = lambda: dict(app.cfg)
                try:
                    _app5.__init__()
                    check("消えていた出力先は戻さない",
                          "無い場所" not in _app5.var_output.get())
                finally:
                    _app5.destroy()
                    main_gui.load_config = lambda: {}
                main_gui.save_config = lambda d: (saved.update(d), [])[1]
                app.var_use_input_dir.set(True)
                app._on_toggle_use_input_dir()

            # --- API キー: 消せる / 保存しないを選べる -------------------
            # 保存できるのに消せないのは片手落ちで、共有のパソコンで
            # 使ったあとに困る（公開前チェック・2026-08-23）。
            check("鍵を消すボタンがある", hasattr(app, "btn_api_del"))
            check("保存しないを選べる", hasattr(app, "var_keep_api"))

            _saved_api = {}
            main_gui.save_config = lambda x: (_saved_api.update(x), [])[1]
            _real_yes = assign_gui.messagebox.askyesno
            _real_info = main_gui.messagebox.showinfo
            main_gui.messagebox.askyesno = lambda *a2, **k2: True
            main_gui.messagebox.showinfo = lambda *a2, **k2: None
            try:
                app.var_keep_api.set(True)
                app.var_api.set("AIzaSyTEST-0123456789")
                app._save_api_key()
                check("保存すると設定に載る", _saved_api.get("api_key"))

                app._delete_api_key()
                check("消すと設定から落ちる", not app.cfg.get("api_key"))
                check("消すと入力欄も空になる", app.var_api.get() == "")

                # **保存しない側へ倒したら、その場で消す**
                app.cfg["api_key"] = "AIzaSyTEST-0123456789"
                app.var_keep_api.set(False)
                app._on_keep_api_toggled()
                check("保存しないに倒すとその場で消える",
                      not app.cfg.get("api_key"))
                app.var_keep_api.set(True)
            finally:
                main_gui.messagebox.askyesno = _real_yes
                main_gui.messagebox.showinfo = _real_info
                main_gui.save_config = lambda d: (saved.update(d), [])[1]

            # **開始の経路からも漏らさない。**設定を扱う場所が複数あるので、
            # 画面のチェックだけ直しても、こちらから書けば元の木阿弥になる。
            import inspect as _insp
            _start = _insp.getsource(main_gui.App._start)
            check("開始時も「保存しない」を見ている",
                  "var_keep_api" in _start)
            check("保存しないときは鍵を書かない",
                  'self.cfg.pop("api_key", None)' in _start)

            # 名前が実態と合っているか（作業ファイルもここに置かれる）
            check("フォルダの名前が「出力」だけになっていない",
                  "作業" in str(app.entry_output.master.cget("text")))

            # --- 話者分離を切ったまま始めようとしたら止める ---------------
            # **注意文 1 行では気づけない。**実機で 30 分待ったあとに
            # 「1000 区間を 1 件ずつ」と分かった(2026-08-21)。
            _warned = {"n": 0, "text": ""}

            def _warn(title, msg, **kw):
                _warned["n"] += 1
                _warned["text"] = msg
                return False            # 「いいえ」= 始めない

            main_gui.messagebox.askyesno = _warn
            app.var_engine.set(main_gui.ENGINE_LOCAL)
            app.var_mode.set(main_gui.MODE_MANUAL)
            app.var_diarize_local.set(False)
            app._start()
            check("話者分離が切れていたら開始前に止める", _warned["n"] >= 1)
            check("何が起きるかを伝える",
                  "1 件ずつ" in _warned["text"])
            check("直し方も伝える", "話者分離" in _warned["text"])
            app.var_diarize_local.set(True)
            main_gui.messagebox.askyesno = lambda *a, **k: True

            # --- GPU 向けのモデルを一度だけ勧める -------------------------
            # **勝手に 3GB を落とさない。**断られたら二度と聞かない。
            from src import align as _al2, asr_fetch as _af
            _real_sug, _real_fetch = _al2.suggest_gpu_model, _af.fetch
            try:
                _asked = {"n": 0, "text": ""}

                def _ask(title, msg, **kw):
                    _asked["n"] += 1
                    _asked["text"] = msg
                    return False            # まず「いいえ」

                main_gui.messagebox.askyesno = _ask
                main_gui.align.suggest_gpu_model = lambda: "large-v3"
                app.cfg.pop(app.ASKED_KEY, None)
                _saved2 = {}
                main_gui.save_config = lambda d: (_saved2.update(d), [])[1]

                app._maybe_offer_gpu_model()
                check("GPU があって未取得なら一度は聞く", _asked["n"] == 1)
                check("大きさを先に伝える", "MB" in _asked["text"])
                check("録音を送らないと明記する",
                      "録音" in _asked["text"] and "送りません" in _asked["text"])
                check("断っても記録に残す", _saved2.get(app.ASKED_KEY) is True)

                # 二度目は聞かない
                app._maybe_offer_gpu_model()
                check("断ったら二度と聞かない", _asked["n"] == 1)

                # 取得できなくても止まらない
                app._on_fetch_failed("通信できません")
                check("失敗しても small のまま使えると伝える",
                      "small" in app.txt_log.get("1.0", "end"))

                # 取得できたら既定に上がる
                app.cfg.pop(app.ASKED_KEY, None)
                app._on_fetch_done("large-v3")
                check("取得したら選択に反映される",
                      app._model_by_engine[main_gui.ENGINE_LOCAL] == "large-v3")
            finally:
                main_gui.align.suggest_gpu_model = _real_sug
                _af.fetch = _real_fetch
                main_gui.save_config = lambda d: (saved.update(d), [])[1]
                app._model_by_engine[main_gui.ENGINE_LOCAL] = "small"
                app.var_model.set("small")

            # --- 設定の往復（保存 → 読み直し）-----------------------------
            # **今日の 4 件はここに固まっていた。**設定を扱う場所が 4 つ
            # あり(読み込み・経路の切替・モデルを選んだとき・開始時)、
            # そのうち 1 つだけ直しても別の経路から壊れた。
            # **1 か所ずつ見るのではなく、往復するかを見る。**
            _want = {
                "engine": main_gui.ENGINE_LOCAL,
                "local_model": "large-v3",
                "model": main_gui.MODELS[1],
                "chunk_minutes": 12,
                "diarize": False,
                "mode": main_gui.MODE_MANUAL,
            }
            # **検査環境では large-v3 が未取得に見える**(APPDATA を差し替えて
            # あるため)。取得済みに見せないと、モデル欄で選んだ時点で
            # **取得ダイアログが開いて検査が止まる**(aac6d05 と同じ形)。
            _rt_find = _al2.find_bundled_model
            _al2.find_bundled_model = lambda m: (
                Path("x") if m == "large-v3" else _rt_find(m))
            main_gui.messagebox.askyesno = lambda *a, **k: False
            app.var_engine.set(_want["engine"])
            app._update_engine_state()
            app.var_model.set(_want["local_model"])
            app._on_model_chosen()
            app._model_by_engine[main_gui.ENGINE_CLOUD] = _want["model"]
            app.var_chunk.set(_want["chunk_minutes"])
            app.var_diarize_local.set(_want["diarize"])
            app.var_mode.set(_want["mode"])

            _out = {}
            main_gui.save_config = lambda d: (_out.update(d), [])[1]
            main_gui.messagebox.askyesno = lambda *a, **k: True
            app._start()
            for _k, _v in _want.items():
                check(f"設定が保存される: {_k}", _out.get(_k) == _v)

            # 読み直したときに同じ状態になるか（別の App を作って確かめる）
            main_gui.load_config = lambda: dict(_out)
            _app2 = main_gui.App()
            try:
                _app2.update()
                check("読み直して経路が戻る",
                      _app2.var_engine.get() == _want["engine"])
                check("読み直してモデルが戻る",
                      _app2.var_model.get() == _want["local_model"])
                check("読み直してクラウドのモデルも保たれる",
                      _app2._model_by_engine[main_gui.ENGINE_CLOUD]
                      == _want["model"])
                check("読み直してチャンク長が戻る",
                      _app2.var_chunk.get() == _want["chunk_minutes"])
                check("読み直して話者分離の設定が戻る",
                      _app2.var_diarize_local.get() == _want["diarize"])
            finally:
                _app2.destroy()
                main_gui.load_config = real_load
                # **差し替えを戻す。**戻さないと後続の検査に漏れ、
                # 「large-v3 が取得済み」の世界のまま進む。
                _al2.find_bundled_model = _rt_find
            app.var_diarize_local.set(True)
            main_gui._ask_speaker_count = real_ask_count
        finally:
            app.destroy()
    finally:
        main_gui.load_config, main_gui.save_config = real_load, real_save

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    # 片方が落ちてももう片方を必ず走らせる(短絡すると検査が静かに減る)
    rc_assign = run()
    rc_inline = run_inline_inserts()
    rc_cand = run_candidate_scroll()
    rc_multi = run_multi_select()
    rc_main = run_main_window()
    rc_cfg = run_user_config_untouched()     # **最後に必ず確かめる**
    sys.exit(rc_assign or rc_inline or rc_cand or rc_multi or rc_main or rc_cfg)
