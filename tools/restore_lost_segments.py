"""再実行で消えた区間を、別の実行結果から戻す道具（設計書 §10.3.5）。

`_carry_over_assignments` の突き合わせが手前から順に貪欲だったため、
**本当の相手より手前の区間に先に当たり、その区間が黙って消えていた。**
原因は直したが、**すでに消えたものは再実行しない限り戻らない。**これは
その埋め合わせ。

同じ音声の別の作業ファイル（話者分離だけ通したもの等）を「元」として、
**元にあって今のファイルに無い区間だけ**を時間順の位置へ入れる。

    python tools/restore_lost_segments.py <今のファイル> --from <元のファイル>
    python tools/restore_lost_segments.py <今のファイル> --from <元> --apply

**話者は付けない。**元のファイルの割当は別の作業のものなので、持ってくると
「人が聴いて決めた」の意味が壊れる（CLAUDE.md）。時刻と本文だけを戻し、
割当は人がやり直す。

**音声が同じであることを確かめてから動く**（SHA-256 か指紋が一致すること）。
既定は下見で、--apply で書き換え、元のファイルは .bak に残す。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segments import Project, fmt_hms      # noqa: E402

# 「同じ区間」と見なす orig_start の差（秒）。転写が同じなら完全一致する
SAME_SECONDS = 0.02


def key_of(seg) -> float:
    return round(float(seg.orig_start if seg.orig_start is not None
                       else seg.start), 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--from", dest="src", required=True,
                    help="同じ音声の別の作業ファイル")
    ap.add_argument("--apply", action="store_true", help="実際に書き換える")
    args = ap.parse_args()

    path, src_path = Path(args.path), Path(args.src)
    proj = Project.load(str(path))
    src = Project.load(str(src_path))

    # **同じ音声か確かめる。**違う音声から区間を持ってきたら記録が壊れる。
    a, b = proj.source_sha256, src.source_sha256
    fa, fb = proj.audio_fingerprint, src.audio_fingerprint
    if (a and b and a != b) or (fa and fb and fa != fb):
        print("元の音声が違います。SHA-256 も指紋も一致しません。")
        return 1
    if not ((a and b) or (fa and fb)):
        print("※ 音声が同じかを確かめられません（SHA-256 も指紋も無い）。")

    mine = {key_of(s) for s in proj.segments}
    lost = [s for s in src.segments
            if not any(abs(key_of(s) - k) <= SAME_SECONDS for k in mine)]

    print(f"今のファイル {len(proj.segments)} 区間 / "
          f"元 {len(src.segments)} 区間")
    if not lost:
        print("戻すものはありません。")
        return 0

    print(f"元にあって今のファイルに無い区間: {len(lost)} 件")
    print()
    for s in lost:
        print(f"  {fmt_hms(s.start)}  {s.start:.2f}-{s.end:.2f}  「{s.text}」")
    print()
    print("**話者は付けません。**元の割当は別の作業のものなので、"
          "持ってくると「人が聴いて決めた」の意味が壊れます。")

    if not args.apply:
        print()
        print("下見だけです。実行するには --apply を付けてください。")
        return 0

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    for s in lost:
        # 割当・確認の印は持ってこない（時刻と本文だけ戻す）
        proj.segments.append(replace(
            s, index=0, speaker_id=None, reviewed=False, note="",
            time_reviewed=False))
    proj.segments.sort(key=lambda x: (x.start, x.end))
    proj.renumber()
    proj._log("restore_lost_segments",
              targets=[[key_of(s), round(float(s.start), 3)] for s in lost],
              count=len(lost), source=src_path.name)
    proj.save()
    print()
    print(f"{len(lost)} 件を戻しました。元のファイルは {backup.name} に"
          "残してあります。話者は未確定なので、聴いて割り当ててください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
