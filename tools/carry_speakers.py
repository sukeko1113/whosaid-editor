"""転写し直したときに、前の作業ファイルから話者の割当を写す道具。

モデルを変えると**区間の切れ目が変わる**ので、パイプラインの引き継ぎ
（`_carry_over_assignments`。`orig_start` が ±2 秒で一致するかを見る）では
何も引き継げない。実データで small → large-v3 に変えたところ、区間は
725 → 772 になり、時刻の対応が総入れ替わりになった（2026-08-20）。

そこで**時間の重なり**で対応づける。新しい区間の 6 割以上が 1 人の古い区間と
重なっていれば、その人を写す。届かないものは**未確定のまま残す**——迷った
ぶんまで埋めると、人が見直す手がかりが消える。

    python tools/carry_speakers.py <新しいファイル> --from <前のファイル>
    python tools/carry_speakers.py <新しいファイル> --from <前> --apply

**写したものはすべて △（まとめて適用）になる。**元が ✓ でも △ に落ちる。
区間の切れ目が変わっている以上、「その区間を聴いて確定した」とは言えない
（CLAUDE.md の ✓/△ の意味論）。**✓ は失われる**ので、必要なら聴き直すこと。

**足した発話は写さない。**「聞こえたのに本文に無い」ものなので、新しい転写
では既に本文になっているかもしれず、機械には判断できない。二重になる。

**音声が同じであることを確かめてから動く**（SHA-256 か指紋が一致すること）。
既定は下見で、--apply で書き換え、元のファイルは .bak に残す。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace                            # noqa: E402
from src.segments import Project, fmt_hms, segment_key      # noqa: E402

# 新しい区間の何割が 1 人の古い区間と重なっていれば写すか。
# **実データで 6 割にすると 772 区間中 516 区間（67%）が写せた**
# （2026-08-20）。上げると取りこぼし、下げると境目で他人を写す。
MIN_OVERLAP_RATIO = 0.6


def best_match(seg, olds) -> tuple[str | None, float]:
    """その区間といちばん長く重なる、話者の付いた古い区間。"""
    dur = max(1e-6, seg.end - seg.start)
    best, best_ov = None, 0.0
    for o in olds:
        if not o.speaker_id:
            continue
        ov = min(seg.end, o.end) - max(seg.start, o.start)
        if ov > best_ov:
            best, best_ov = o.speaker_id, ov
    return best, best_ov / dur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="転写し直した新しい作業ファイル")
    ap.add_argument("--from", dest="src", required=True, help="前の作業ファイル")
    ap.add_argument("--ratio", type=float, default=MIN_OVERLAP_RATIO)
    ap.add_argument("--take-roster", action="store_true",
                    help="名簿ごと前のファイルのものにする"
                         "（役職の書き方まで揃う。割当がまだ無いときだけ）")
    ap.add_argument("--apply", action="store_true", help="実際に書き換える")
    args = ap.parse_args()

    path, src_path = Path(args.path), Path(args.src)
    proj, src = Project.load(str(path)), Project.load(str(src_path))

    # **同じ音声か確かめる。**違う音声から写したら記録が壊れる。
    a, b = proj.source_sha256, src.source_sha256
    fa, fb = proj.audio_fingerprint, src.audio_fingerprint
    if (a and b and a != b) or (fa and fb and fa != fb):
        print("元の音声が違います。写せません。")
        return 1
    if not ((a and b) or (fa and fb)):
        print("※ 音声が同じかを確かめられません（SHA-256 も指紋も無い）。")

    # **名簿ごと引き継ぐ指定。**役職の書き方まで揃うので、名前で当てる
    # 必要がなくなる。割当が既にあると ID が指す先が変わって全員入れ替わる
    # ので、そのときは断る（CLAUDE.md「振り直すと全員入れ替わる」）。
    if args.take_roster:
        assigned = sum(1 for s in proj.segments if s.speaker_id)
        if assigned:
            print(f"すでに {assigned} 区間に割当があります。"
                  "名簿を入れ替えると指す先が変わるので、断ります。")
            return 1
        proj.speakers = [replace(sp) for sp in src.speakers]
        print(f"名簿を前のファイルのものにしました（{len(proj.speakers)} 人）。")

    # **ID は転写し直すと振り直る。**名簿を作り直すので当然だが、そのまま
    # 写すと全員が別人になる（CLAUDE.md「振り直すと全員入れ替わる」）。
    # 名前（＋所属）で対応づける。同じ表記が 2 人いたら、その人だけ写さない。
    here: dict[tuple[str, str], list[str]] = {}
    by_name: dict[str, list[str]] = {}
    for sp in proj.speakers:
        here.setdefault((sp.name, sp.note), []).append(sp.id)
        by_name.setdefault(sp.name, []).append(sp.id)
    mine = {sp.id for sp in proj.speakers}

    remap: dict[str, str] = {}
    unmapped: list[str] = []
    for sp in src.speakers:
        if sp.id in mine:
            remap[sp.id] = sp.id                    # ID がそのまま通じる
            continue
        # 役職の書き方が違うことがある（「文科省」と正式名称など）。
        # 組で当たらなければ**名前だけ**で当てる。同姓が 2 人いたら当てない。
        cand = here.get((sp.name, sp.note)) or by_name.get(sp.name)
        if cand and len(cand) == 1:
            remap[sp.id] = cand[0]
        else:
            unmapped.append(sp.display if hasattr(sp, "display") else sp.name)
    if unmapped:
        print("※ 新しいファイルに見当たらない出席者がいます: "
              + "、".join(unmapped))
        print("  その人の割当は写しません。先に名簿を揃えてください。")
    renamed = sum(1 for k, v in remap.items() if k != v)
    if renamed:
        print(f"※ 名簿の ID が振り直されていたので、名前で対応づけました"
              f"（{renamed} 人）。")

    # **足した発話は写さない**（二重になる。冒頭の注記）
    olds = [s for s in src.segments
            if s.speaker_id in remap and not src.is_added_utterance(s)]

    pairs, weak, nohit = [], [], []
    for seg in proj.segments:
        sid, ratio = best_match(seg, olds)
        if sid and ratio >= args.ratio:
            pairs.append((segment_key(seg), remap[sid]))
        elif sid:
            weak.append((seg, ratio))
        else:
            nohit.append(seg)

    print(f"新しいファイル {len(proj.segments)} 区間 / "
          f"前のファイル {len(src.segments)} 区間")
    print()
    print(f"  写せる（{args.ratio:.0%} 以上が 1 人と重なる） {len(pairs):>4} 区間")
    print(f"  重なるが届かない（未確定のまま）            {len(weak):>4} 区間")
    print(f"  重なる相手がいない（未確定のまま）          {len(nohit):>4} 区間")
    print()
    print("**写したものはすべて △（まとめて適用）になります。**")
    print(f"前のファイルの ✓ は "
          f"{sum(1 for s in src.segments if s.reviewed)} 区間ありますが、"
          "区間の切れ目が変わっているので引き継げません。")
    added = sum(1 for s in src.segments if src.is_added_utterance(s))
    if added:
        print(f"足した発話 {added} 件も写しません"
              "（新しい転写では本文になっている可能性があるため）。")

    if weak[:5]:
        print()
        print("届かなかった例（人が決めてください）:")
        for seg, r in weak[:5]:
            print(f"  {fmt_hms(seg.start)} 重なり {r:.0%}  「{seg.text[:40]}」")

    if not args.apply:
        print()
        print("下見だけです。実行するには --apply を付けてください。")
        return 0
    if not pairs:
        print("写すものがありません。")
        return 0

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    n = proj.carry_speakers(pairs, source=src_path.name)
    proj.save()
    print()
    print(f"{n} 区間に写しました（すべて △）。"
          f"元のファイルは {backup.name} に残してあります。")
    print("編集の履歴には 1 件（前の転写から写した割当）として残ります。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
