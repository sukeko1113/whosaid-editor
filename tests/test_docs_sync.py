"""文書とコードの突き合わせ。

    .venv\\Scripts\\python.exe tests\\test_docs_sync.py

**このリポジトリで 4 回出ている型を機械に見張らせる。**「何かを足したときに、
それを列挙している場所を直し忘れる」形（HANDOFF.md「検査で縛れる型と、
縛れない型」）:

    GUI の「誤字」の文言    5 箇所と思って 10 箇所
    改名の残り              1 箇所と思って 20 箇所
    .gitignore の成果物     旧名のまま。新名 3 種が追跡対象だった（約 1.1GB）
    README のファイル構成    7 モジュール欠落（diarize.py を含む）

`CACHE_KEY_ENGINE_FIELDS` と `tests/test_cache_key.py` が鍵の要素で同じ縛りを
しているのと同じ形。**足したら分類するまで落ちる。**

---

## この検査が守れる範囲・守れない範囲

**守れる**: README のファイル構成のツリーに、`src/*.py` が全部載っていること。
逆に、実在しないモジュールが載っていないこと。

**守れない**: **ツリーに付いている一行説明が実態と合っているか。**
`diarize.py` が載っていても、説明が「話者分離」でなく別のことを書いていれば
害は同じである。機械には判定できない。

**「検査があるから安心」で説明文が腐るのが、この型の次の失敗の仕方。**
モジュールを足してこの検査に叱られたときは、名前を並べるだけでなく、
**説明が実物と合っているかを人が見ること。**
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


# ツリーの見出し。**変えるならここ 1 箇所。**
TREE_HEADING = "## ファイル構成"

# ツリーに載せなくてよいもの。**除外は理由を書くこと**——
# 「載せ忘れ」と「意図して載せない」を区別できなくなる。
TREE_EXEMPT = {
    "__init__": "パッケージの印。中身が無い",
}


def src_modules() -> set[str]:
    """`src/*.py` のモジュール名（拡張子なし）。"""
    return {p.stem for p in (ROOT / "src").glob("*.py")} - set(TREE_EXEMPT)


def tree_block(text: str) -> str:
    """README のファイル構成のツリー（``` で囲まれた部分）を切り出す。"""
    if TREE_HEADING not in text:
        raise AssertionError(
            f"README に見出し {TREE_HEADING!r} が見つかりません。"
            "見出しを変えたなら TREE_HEADING も直してください。")
    after = text.split(TREE_HEADING, 1)[1]
    m = re.search(r"```[^\n]*\n(.*?)```", after, re.S)
    if not m:
        raise AssertionError(
            f"{TREE_HEADING} の直後に ``` で囲まれたツリーが見つかりません。")
    return m.group(1)


def tree_modules(block: str) -> set[str]:
    return {m for m in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\.py", block)}


def test_tree_lists_every_module() -> None:
    """**src に足したモジュールが README のツリーに載っているか。**

    載っていないと、README を読んだ人が構成から機能を推し量る経路が塞がる。
    実際 v2.1.0 の直前まで、**話者分離の中核 diarize.py が載っていなかった。**
    """
    block = tree_block((ROOT / "README.md").read_text(encoding="utf-8"))
    missing = sorted(src_modules() - tree_modules(block))
    assert not missing, (
        "README のファイル構成のツリーに載っていないモジュールがあります: "
        + "、".join(f"src/{m}.py" for m in missing)
        + "\n  → README.md の「## ファイル構成」に足してください。"
        "\n  **名前を並べるだけでなく、一行説明が実物と合っているかも見ること**"
        "（この検査は説明文までは守れません）。"
        "\n  意図して載せないなら、tests/test_docs_sync.py の TREE_EXEMPT に"
        "理由つきで足してください。")


def test_tree_has_no_ghost_module() -> None:
    """**消したモジュールがツリーに残っていないか。**

    残っていると、無いものを探す人が出る。足す側だけ縛ると片手落ちになる。
    """
    block = tree_block((ROOT / "README.md").read_text(encoding="utf-8"))
    real = {p.stem for p in (ROOT / "src").glob("*.py")}
    ghosts = sorted(m for m in tree_modules(block) if m not in real)
    assert not ghosts, (
        "README のツリーに、src に無いモジュールが載っています: "
        + "、".join(f"{m}.py" for m in ghosts)
        + "\n  → 消したなら README からも消してください。")


def test_exempt_entries_are_real() -> None:
    """**除外リストが古くなっていないか。**

    実在しないものを除外し続けると、除外の意味が読めなくなる。
    """
    real = {p.stem for p in (ROOT / "src").glob("*.py")}
    stale = sorted(set(TREE_EXEMPT) - real)
    assert not stale, (
        "TREE_EXEMPT に、もう存在しないモジュールがあります: "
        + "、".join(stale) + "\n  → tests/test_docs_sync.py から消してください。")


def test_guard_actually_fires() -> None:
    """**この検査が本当に落ちるかを確かめる。**

    このリポジトリの約束（HANDOFF.md「だから: ガードを足したら、発火する
    ことを実測すること」）。**落ちない検査は、守られているという誤解を足す。**
    壊した README を組み立てて、上の 2 つが本当に落ちることを見る。
    """
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    block = tree_block(text)
    real = {p.stem for p in (ROOT / "src").glob("*.py")}

    # **いまの状態を基準にして、壊したぶんだけ増えるかを見る。**
    # 「ツリーが揃っている」を前提にすると、既にずれているときに
    # この検査まで巻き添えで落ち、しかも「削っても検出できません」という
    # **嘘の文言**を出す(実際に踏んだ)。落ちるべきは上の 2 つだけ。
    missing_before = src_modules() - tree_modules(block)
    ghosts_before = {m for m in tree_modules(block) if m not in real}

    # 1) モジュールを 1 つ削ったツリー → 欠落が 1 つ増えるはず
    victim = sorted(src_modules() & tree_modules(block))[0]
    broken = re.sub(rf"^.*\b{re.escape(victim)}\.py.*\n", "", block, flags=re.M)
    assert victim not in tree_modules(broken), "検査用のツリーを壊せていません"
    missing_after = src_modules() - tree_modules(broken)
    assert missing_after - missing_before == {victim}, (
        f"ツリーから {victim}.py を削っても、欠落として検出できません"
        f"（増えたのは {sorted(missing_after - missing_before)}）")

    # 2) 実在しないモジュールを足したツリー → 幽霊が 1 つ増えるはず
    added = block + "│  ├─ nonexistent_module.py ... 検査用\n"
    ghosts_after = {m for m in tree_modules(added) if m not in real}
    assert ghosts_after - ghosts_before == {"nonexistent_module"}, (
        f"実在しないモジュールを足しても検出できません"
        f"（増えたのは {sorted(ghosts_after - ghosts_before)}）")


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
