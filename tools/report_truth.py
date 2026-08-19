"""正解ラベルに対して話者分離の質を集計する。

    python tools\\report_truth.py <作業ファイル.speakers.json> <diarize.json>

設計書 `claude/claude_正解ラベルの作り方_設計書.md` §1 の 2 指標を出す。

  1. クラスタ純度        … 話者分離そのものの質。候補学習に依存しない
  2. 第 1 候補的中率(H7-B) … 製品としての体験。実際の作業を再現して測る

**人が作業を始める前にこれを作っておく。**測る手順を後から決めると、
出た数字に合わせて手順のほうを選べてしまう。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src import evaluate  # noqa: E402
from src.segments import SPECIAL_SPEAKERS, Project  # noqa: E402
from src.suggest import SpeakerSuggester  # noqa: E402

LABELS = {evaluate.STRATUM_VERY_SHORT: "1 秒未満",
          evaluate.STRATUM_SHORT: "1〜2 秒",
          evaluate.STRATUM_LONG: "2 秒以上"}

MIN_OVERLAP = 0.5       # 設計書 §6.2(ローカル転写設計書)と同じ


def assign_turns(segments, turns, min_ratio=MIN_OVERLAP):
    """区間ごとに、最も重なる話者区間の話者を割り当てる。"""
    out = {}
    for s in segments:
        best, best_ov = None, 0.0
        for t in turns:
            ov = min(s.end, t["end"]) - max(s.start, t["start"])
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        span = max(0.01, s.end - s.start)
        out[s.index] = best if best_ov / span >= min_ratio else None
    return out


def load_truth(truth_dir: Path) -> dict[int, str]:
    """白紙で付けたラベルだけを読む(手直しで作ったものは混ぜない)。"""
    truth: dict[int, str] = {}
    for path in sorted(truth_dir.glob("labels_blind.r*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("made_by") != "blind":
            print(f"  飛ばします(白紙で作られていない): {path.name}")
            continue
        for row in data.get("labels", []):
            truth[int(row["index"])] = row["speaker_id"]
        print(f"  読み込み: {path.name}（{len(data.get('labels', []))} 件）")
    return truth


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    proj_path, diar_path = Path(sys.argv[1]), Path(sys.argv[2])
    proj = Project.load(proj_path)
    turns = json.loads(diar_path.read_text(encoding="utf-8"))["turns"]
    truth_dir = proj_path.parent / "truth"

    print("正解ラベル")
    truth = load_truth(truth_dir)
    if not truth:
        print("  ありません。先に tools\\label_truth.py で付けてください。")
        return 1
    # 「分からない」「複数人」は分母から外す(設計書 §1.1)
    usable = {k: v for k, v in truth.items() if v not in SPECIAL_SPEAKERS}
    print(f"  使える: {len(usable)} 件 "
          f"(分からない・複数人 {len(truth) - len(usable)} 件を除く)")

    by_index = {s.index: s for s in proj.segments}
    sizes = evaluate.stratum_sizes(proj.segments)
    cluster_of = assign_turns(proj.segments, turns)

    # ---------- 1. クラスタ純度 ----------
    print("\n【1】クラスタ純度（話者分離そのもの）")
    rates, counts, all_pairs = {}, {}, []
    for stratum in evaluate.STRATA:
        pairs = [(cluster_of.get(i), sid) for i, sid in usable.items()
                 if i in by_index
                 and evaluate.stratum_of(by_index[i].duration) == stratum]
        pairs = [(c, s) for c, s in pairs if c is not None]
        if not pairs:
            continue
        purity, _ = evaluate.cluster_purity(pairs)
        chance = evaluate.purity_chance_level(pairs)
        rates[stratum], counts[stratum] = purity, len(pairs)
        all_pairs.extend(pairs)
        print(f"  {LABELS[stratum]:<8}: {purity*100:>5.1f}%  "
              f"（{len(pairs)} 件／でたらめなら {chance*100:.1f}%）")
    if rates:
        overall = evaluate.weighted_rate(rates, sizes)
        half = evaluate.stratified_halfwidth(rates, sizes, counts)
        chance_all = evaluate.purity_chance_level(all_pairs)
        print(f"  {'全体':<8}: {overall*100:>5.1f}% ± {half*100:.1f} ポイント")
        print(f"  でたらめな正解でも {chance_all*100:.1f}% 出る"
              "（まとまりごとに多数派を採るため下駄がある）。"
              f"差は {(overall-chance_all)*100:+.1f} ポイント")

    # ---------- 2. 第 1 候補的中率 ----------
    # 実際の作業を再現する。時間順に 1 件ずつ確定させ、そのつど候補を作り直す。
    print("\n【2】第 1 候補的中率（H7-B・実際の作業の再現）")
    work = Project.load(proj_path)
    for s in work.segments:                       # まっさらから始める
        s.speaker_id, s.reviewed = None, False
    for s in work.segments:                       # 話者分離の結果を入れる
        c = cluster_of.get(s.index)
        s.cluster = f"g:{c}" if c is not None else f"{s.chunk}:?"
    sug = SpeakerSuggester(work)
    idx_of = {s.index: i for i, s in enumerate(work.segments)}

    hits = {k: 0 for k in evaluate.STRATA}
    tries = {k: 0 for k in evaluate.STRATA}
    for seg in sorted(work.segments, key=lambda s: (s.start, s.index)):
        want = truth.get(seg.index)
        if want is None:
            continue
        if seg.index in usable:
            sug.refresh()
            ranked = sug.rank(idx_of[seg.index])
            stratum = evaluate.stratum_of(seg.duration)
            tries[stratum] += 1
            if ranked and ranked[0].speaker.id == want:
                hits[stratum] += 1
        seg.speaker_id = want                     # 正解を入れて次へ進む

    rates2 = {k: hits[k] / tries[k] for k in evaluate.STRATA if tries[k]}
    counts2 = {k: tries[k] for k in evaluate.STRATA if tries[k]}
    for k in evaluate.STRATA:
        if tries[k]:
            print(f"  {LABELS[k]:<8}: {rates2[k]*100:>5.1f}%  "
                  f"（{hits[k]}/{tries[k]}）")
    if rates2:
        overall2 = evaluate.weighted_rate(rates2, sizes)
        half2 = evaluate.stratified_halfwidth(rates2, sizes, counts2)
        v = evaluate.verdict(overall2, half2, 0.70)
        print(f"  {'全体':<8}: {overall2*100:>5.1f}% ± {half2*100:.1f} ポイント")
        print(f"\n  H7-B（70% 以上）の判定: **{v}**")
        if v == "判定不能":
            print("  → 標本を 1 巡ぶん足して測り直す（基準は動かさない）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
