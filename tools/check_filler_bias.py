# -*- coding: utf-8 -*-
r"""フィラー保持率の数え方が、どちらへ偏っているかを測る。

    .venv\Scripts\python.exe tools\check_filler_bias.py [逐語正解のフォルダ]

`evaluate.count_terms()` は単純な部分一致なので、**数え落とし**（`え、` `ま、`
のような短い形が一覧に無い）と**数えすぎ**（`あのー` が `あの` にも一致する
二重計上、`あの` の指示語、`ちょっと` の副詞）が両方ある。

**保持率は 候補÷正解 なので、分母も動く。向きは自明でない。**
「数え落としているから実際の保持率はもっと高いはず」は成り立たなかった
（実測で 9.2% → 8.5% と下がった。HANDOFF 参照）。

**`evaluate.FILLER_TERMS` を触る前に、これで向きを見ること。**
公表済みの 6 エンジンの数字はすべてこの物差しで出ているので、
一覧を変えるなら **6 系統を全部測り直す**必要がある。
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              write_through=True)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import evaluate as ev            # noqa: E402

# 逐語正解の置き場。引数で渡せる（既定は開発機の場所）
TRUTH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"C:\dev\01\test-audio\truth")
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
