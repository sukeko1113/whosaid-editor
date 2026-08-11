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

# Windows の既定コンソールは cp932 / cp1252 なので、日本語のラベルを print した
# 時点で UnicodeEncodeError になる(テストの中身とは無関係に落ちる)。
# 出力先のエンコーディングをここで固定しておく。
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


import tkinter as tk  # noqa: E402

from src.assign_gui import (  # noqa: E402
    FILTER_ALL,
    FILTER_UNASSIGNED,
    FILTER_UNREVIEWED,
    PREVIEW_FLOOR_SECONDS,
    PREVIEW_MAX_SECONDS,
    PREVIEW_MIN_SECONDS,
    AssignWindow,
    SplitDialog,
    clamp_times,
    plan_roster_text,
    playback_window,
    preview_length,
    time_edit_base,
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


def run() -> int:
    failures = []

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
        check("終了は動かない", seg3.end == orig3[1])
        check("保存対象になる", win._dirty is True)

        win._nudge_time("end", +1.0)
        check("終了をナッジできる", abs(seg3.end - (orig3[1] + 1.0)) < 1e-6)
        check("一覧に ✎ が出る", win.tree.item("s3", "values")[0].startswith("✎"))

        win.var_start.set("ここは時刻を書く欄")
        win._commit_time("start")
        check("読めない入力は入力欄を元に戻す", win.var_start.get() == "00:01:36.5")
        check("読めない入力では区間を変えない", seg3.start == 96.5)

        win.revert_time()
        check("元の時刻に戻せる", (seg3.start, seg3.end) == orig3)
        check("印が外れる", seg3.time_edited is False)
        check("元の時刻は保持されている", (seg3.orig_start, seg3.orig_end) == orig3)
        check("一覧の ✎ も消える", not win.tree.item("s3", "values")[0].startswith("✎"))

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

        win.player.close()
        win.destroy()
        root.destroy()

    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'ALL PASSED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
