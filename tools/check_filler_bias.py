# -*- coding: utf-8 -*-
"""短い形を語のリストに足すと、フィラー保持率はどちらへ動くか。

記事の下書きは「実際の保持率はこれより高い可能性がある（＝いまの数字は
低すぎる）」としている。**保持率は 候補÷正解 なので分母も動く。**
向きは自明でないので、逐語正解で実測する。
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              write_through=True)
sys.path.insert(0, r"C:\dev\01\whosaid-editor")

from src import evaluate as ev            # noqa: E402

TRUTH = Path(r"C:\dev\01\test-audio\truth")
truth, hyp = [], []
for f in sorted(TRUTH.glob("verbatim.*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    for s in d["segments"]:
        truth.append(s.get("truth") or "")
        hyp.append(s.get("asr") or "")
TT, HH = "".join(truth), "".join(hyp)
print(f"  正解 {len(TT):,} 字 / 候補(既定モデル) {len(HH):,} 字 / "
      f"{len(truth)} 区間")
print()


def rate(terms):
    t = ev.count_terms(TT, terms)
    h = ev.count_terms(HH, terms)
    return h, t, (h / t * 100 if t else 0.0)


print("  --- いまの語のリスト ---")
h0, t0, r0 = rate(ev.FILLER_TERMS)
print(f"    正解 {t0} 個 / 候補 {h0} 個 → 保持率 {r0:.1f}%")
print("    内訳（正解 / 候補）:")
for w in ev.FILLER_TERMS:
    a, b = TT.count(w), HH.count(w)
    if a or b:
        print(f"      {w:6s} {a:4d} / {b:4d}")

SHORT = ("え、", "ま、", "あ、", "えと", "そのー")
print()
print("  --- 短い形（いまの一覧に無い形）---")
for w in SHORT:
    a, b = TT.count(w), HH.count(w)
    print(f"      {w:6s} {a:4d} / {b:4d}")

add = tuple(w for w in SHORT if w not in ev.FILLER_TERMS)
h1, t1, r1 = rate(ev.FILLER_TERMS + add)
print()
print("  --- 足したあと ---")
print(f"    正解 {t1} 個 / 候補 {h1} 個 → 保持率 {r1:.1f}%")

print()
print("=" * 64)
print(f"  保持率 {r0:.1f}% → {r1:.1f}%  "
      f"（{'下がる' if r1 < r0 else '上がる' if r1 > r0 else '変わらない'}）")
print("=" * 64)

print()
print("  --- 二重計上（「あのー」が「あの」にも数えられる）---")
d_t, d_h = TT.count("あのー"), HH.count("あのー")
print(f'    「あのー」 正解 {d_t} / 候補 {d_h} … それぞれ 2 回ずつ数えている')
no_dup = tuple(w for w in ev.FILLER_TERMS if w != "あの")
h2, t2, r2 = rate(no_dup)
print(f'    「あの」を外すと 正解 {t2} / 候補 {h2} → {r2:.1f}%')

print()
print("  --- 「あの」の指示語・「ちょっと」の副詞（実例）---")
import re
for w in ("あの", "ちょっと"):
    ex = [m.group(0) for m in re.finditer(rf"..{w}..", TT)][:4]
    print(f"    {w}: " + " / ".join(ex))
