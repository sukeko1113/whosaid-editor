"""名簿が二重になった作業ファイルを直す(設計書 §11.8)。

「名前」と「企業・役職」に分けたあと文字起こし経路を通すと、名前だけの
照合では別人と判断され、さらに「消えた人も保持する」規則で古い方も残り、
**同じ人が 2 つの ID になる**。実データで 8 組できた(2026-08-18)。

原因側は `_merge_speakers` を直した(名前＋役職をつないだ形でも照合する)。
これは、すでにそうなってしまったファイルを直すための道具。

    python tools/merge_duplicate_speakers.py <file.speakers.json>          # 下見
    python tools/merge_duplicate_speakers.py <file.speakers.json> --apply  # 実行

同一人物の判定は「名前＋役職から空白を取り除いた形が一致する」ことだけ。
推測はしない。判定できない組があれば、そのまま残して報告する。
"""

from __future__ import annotations

import argparse
import collections
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.segments import Project, Speaker      # noqa: E402


def flat(sp: Speaker) -> str:
    return re.sub(r"\s", "", (sp.name or "") + (sp.note or ""))


def find_pairs(proj: Project) -> list[tuple[Speaker, list[Speaker]]]:
    """(残す人, 消す人たち) の組。残すのは**役職が分かれている方**。"""
    groups: dict[str, list[Speaker]] = collections.defaultdict(list)
    for sp in proj.speakers:
        groups[flat(sp)].append(sp)
    out: list[tuple[Speaker, list[Speaker]]] = []
    for _key, sps in groups.items():
        if len(sps) < 2:
            continue
        split = [sp for sp in sps if sp.note]
        if len(split) != 1:
            # 分かれている方が 1 人に決まらないなら触らない(推測しない)
            continue
        keep = split[0]
        out.append((keep, [sp for sp in sps if sp is not keep]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--apply", action="store_true", help="実際に書き換える")
    args = ap.parse_args()

    path = Path(args.path)
    proj = Project.load(str(path))
    counts = collections.Counter(
        s.speaker_id for s in proj.segments if s.speaker_id)
    reviewed = collections.Counter(
        s.speaker_id for s in proj.segments if s.speaker_id and s.reviewed)

    pairs = find_pairs(proj)
    if not pairs:
        print("二重になっている人はいません。")
        return 0

    print(f"話者 {len(proj.speakers)} 人 / 二重 {len(pairs)} 組")
    print()
    moved = 0
    for keep, drops in pairs:
        print(f"■ 残す  {keep.id}  {keep.name} ／ {keep.note}")
        for sp in drops:
            n, r = counts.get(sp.id, 0), reviewed.get(sp.id, 0)
            moved += n
            print(f"  消す  {sp.id}  {sp.name}")
            print(f"        割当 {n} 区間(うち聴いて確定 {r} 区間)を "
                  f"{keep.id} へ付け替え")
    print()
    print(f"付け替える区間: 合計 {moved}")
    print(f"話者: {len(proj.speakers)} 人 → "
          f"{len(proj.speakers) - sum(len(d) for _k, d in pairs)} 人")

    # 判定できなかった重複があれば黙って通さない
    groups: dict[str, list[Speaker]] = collections.defaultdict(list)
    for sp in proj.speakers:
        groups[flat(sp)].append(sp)
    unresolved = [sps for k, sps in groups.items()
                  if len(sps) > 1 and not any(
                      keep in sps for keep, _d in pairs)]
    for sps in unresolved:
        print("※ 判定できない組があります(触りません): "
              + "、".join(f"{sp.id} {sp.name}" for sp in sps))

    if not args.apply:
        print()
        print("下見だけです。実行するには --apply を付けてください。")
        return 0

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    remap = {sp.id: keep.id for keep, drops in pairs for sp in drops}
    for seg in proj.segments:
        if seg.speaker_id in remap:
            seg.speaker_id = remap[seg.speaker_id]      # reviewed は触らない
    drop_ids = set(remap)
    proj.speakers = [sp for sp in proj.speakers if sp.id not in drop_ids]
    for i, sp in enumerate(proj.speakers):
        sp.order = i
    proj.save()
    print()
    print(f"直しました。元のファイルは {backup.name} に残してあります。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
