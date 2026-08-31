# -*- coding: utf-8 -*-
"""手順書とチェックリストの照合。片方だけ直すと食い違うため。"""
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

root = Path(r"C:\dev\01\whosaid-editor\claude")
tej = (root / "処理時間の測定_手順書.md").read_text(encoding="utf-8")
chk = (root / "測定_実行チェックリスト.md").read_text(encoding="utf-8")
src = Path(r"C:\dev\01\whosaid-editor\src\pipeline.py").read_text(encoding="utf-8")

bad = []


def live(doc):
    '''「採用しなかった」の記録は指示ではないので、検査から外す。

    ここを外さないと、**やめた決定を書き残せなくなる**（消せば検査は通るが、
    なぜ変えたかが失われる）。
    '''
    s = doc.find("#### 当初の決定（採用しなかった）")
    if s < 0:
        return doc
    e = doc.find("####", s + 4)
    return doc[:s] + doc[e:]


def ck(label, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}{('   ' + detail) if detail and not cond else ''}")
    if not cond:
        bad.append(label)


print("[手作業のログ保存が、両方から消えているか]")
for name, whole in (("手順書", tej), ("チェックリスト", chk)):
    doc = live(whole)
    ck(f"{name}: 「ログ欄を全選択」が無い", "全選択" not in doc)
    ck(f"{name}: 測定n-n.txt の指定が無い",
       not re.search(r"測定\d-\d\.txt", doc),
       str(re.findall(r"測定\d-\d\.txt", doc)[:3]))
    ck(f"{name}: 「ログ26-08-31」への保存指示が無い", "ログ26-08-31" not in doc)
    # やめた決定が**残っている**ことも見る。消して通すのを防ぐ
    if name == "手順書":
        ck("手順書: やめた決定が記録に残っている", "採用しなかった" in whole)

print("\n[自動保存の説明が、両方に同じ形であるか]")
for name, doc in (("手順書", tej), ("チェックリスト", chk)):
    ck(f"{name}: run-*.log の形が書いてある", "run-" in doc and ".log" in doc)
    ck(f"{name}: .work_ の下だと書いてある", ".work_" in doc)
    ck(f"{name}: started_at と揃うと書いてある", "started_at" in doc)
    ck(f"{name}: timings の log の項に触れている",
       "`log` の項" in doc)
    ck(f"{name}: G ドライブへ同期される件に触れている",
       "同期" in doc and ("本文" in doc))
    ck(f"{name}: 途中で切れる件に触れている", "切れる" in doc)

print("\n[実装と食い違っていないか]")
ck("実装のひな型は run-%Y%m%d-%H%M%S.log",
   'f"run-{self._wall0:%Y%m%d-%H%M%S}.log"' in src)
ck("文書の例も同じ桁数",
   bool(re.search(r"run-\d{8}-\d{6}\.log", tej + chk)),
   "例が見つからない")
ck("timings に log を書いている（実装）", '"log": self.log_name' in src)
ck("包む位置が StageTimer の直後（実装）",
   "timer = StageTimer(on_log)\n" in src and "on_log = timer.tee(on_log, work_dir)" in src)

print("\n[段取りの数が合っているか]")
m = re.search(r"この(\d)ステップ", chk)
steps = sorted(set(re.findall(r"### ステップ(\d) —", chk)))
ck("チェックリストの宣言とステップ数が一致",
   m is not None and m.group(1) == str(len(steps)),
   f"宣言 {m.group(1) if m else '?'} / 実際 {steps}")
tsteps = sorted(set(re.findall(r"\*\*手順(\d) —", tej)))
ck("手順書のステップ番号が連番", tsteps == [str(i) for i in range(1, len(tsteps) + 1)],
   str(tsteps))
ck("両者のステップ数が同じ", len(steps) == len(tsteps), f"{steps} / {tsteps}")

print("\n[両方に同じ順序が書いてあるか]")
order = "G1 → C2 → G1 → C2 → G1 → C2 → C3"
ck("手順書に順序がある", order in tej)
ck("チェックリストに順序がある", order in chk)
for label in ("G1", "C2", "C3"):
    ck(f"{label} のパスが両方で同一",
       (re.search(rf"\*\*{label}\*\* \| `([^`]+)`", tej) or [None]) and
       (re.search(rf"\*\*{label}\*\* \| `([^`]+)`", tej).group(1)
        == re.search(rf"\*\*{label}\*\* \| `([^`]+)`", chk).group(1)))

print("\n[渡し方が両方で揃っているか]")
ck("手順書: ログは渡されないと書いてある", "ログは渡されない" in tej)
ck("チェックリスト: 渡すものはないと書いてある", "渡すものはありません" in chk)
ck("手順書: 実行順を並べる指示がある", "実行順" in tej)
ck("チェックリスト: 実行順を並べる指示がある", "実行順" in chk)

print()
if bad:
    print(f"食い違い {len(bad)} 件:")
    for b in bad:
        print("   -", b)
    sys.exit(1)
print("食い違いなし")
