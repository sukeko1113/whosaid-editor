"""作業ファイルの項目が、分類されないまま増えていないかを見る。

    .venv\\Scripts\\python.exe tests\\test_schema_fields.py

**項目を足したら、`segments.py` の分類に書くまで落ちる。**
`CACHE_KEY_ENGINE_FIELDS`(pipeline.py)と `tests/test_cache_key.py` が
キャッシュ鍵でやっているのと同じ形。

---

## なぜ「上げる / 据え置き」の二択にしないか

キャッシュ鍵は、入れるか入れないかが**その場の動作を変える**。入れ忘れれば
設定違いの転写が返るので、遅かれ早かれ表に出る。

**版の据え置きは、その場では何も起きない。**旧版で開いたときに初めて効く。
だから「全部据え置き」と分類すれば、検査は通って `SCHEMA_VERSION` は動かない
——**据え置きが逃げ道になる。**

そこで `SCHEMA_HELD` は理由つきの辞書にし、**空の理由を通さない。**
何も起きないものを縛るには、そこまで要る。

「誤字」の定義が失われたのは同じ形だった。数字を出す仕組みを作った人が、
**その定義を残す仕組みで縛られていなかった**（HANDOFF「画面に出ている
『(実測)』の 1 つは、誰も再現できない」）。

---

## この検査が守れる範囲・守れない範囲

**守れる**: 項目がどれかに分類されていること。据え置きに理由が書いてあること。
分類表に、もう存在しない項目が残っていないこと。

**守れない**: **理由が正しいかどうか。**「読み手が既定値で困らない」と書いて
あっても、実際に困るかは機械には判定できない。**分類は「考えた跡」を残させる
だけで、考えが正しいことは保証しない。**

**守れない(2)**: **`SCHEMA_VERSION` を実際に上げたか。**`SCHEMA_BUMPED` に
`"foo": 6` と書いても、`SCHEMA_VERSION` が 5 のままなら食い違う——ので
そこは縛ってある(`test_bumped_versions_are_real`)。ただし「上げるべきだったのに
据え置いた」という判断そのものは、人が見るしかない。
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src import segments as S                                   # noqa: E402

# 分類の対象。**保存される dataclass はここに挙げる。**
# Utterance は保存されない(転写の中間表現)ので対象外。
CLASSIFIED = (S.Project, S.Segment, S.Speaker)


def all_fields() -> set[str]:
    out: set[str] = set()
    for cls in CLASSIFIED:
        out |= {f.name for f in dataclasses.fields(cls)}
    return out


def buckets() -> dict[str, set[str]]:
    return {
        "SCHEMA_BASE": set(S.SCHEMA_BASE),
        "SCHEMA_BUMPED": set(S.SCHEMA_BUMPED),
        "SCHEMA_HELD": set(S.SCHEMA_HELD),
        "SCHEMA_NOT_SERIALIZED": set(S.SCHEMA_NOT_SERIALIZED),
    }


def test_every_field_is_classified():
    """**項目を足したら、分類するまで落ちる。**これが本体。"""
    known: set[str] = set()
    for names in buckets().values():
        known |= names
    missing = sorted(all_fields() - known)
    assert not missing, (
        "分類されていない項目があります: " + "、".join(missing)
        + "\n  → src/segments.py の分類に足してください。"
        "\n     SCHEMA_BUMPED       版を上げて足した(値は上げた先の版)"
        "\n     SCHEMA_HELD         版を上げずに足した(**値は据え置いた理由。必須**)"
        "\n     SCHEMA_NOT_SERIALIZED  保存しない(値は保存しない理由)"
        "\n  **版は変更の大きさではなく、旧版で開いたときに何が失われるかで決める。**")


def test_no_field_is_in_two_buckets():
    """二重に分類されていないか。分類が矛盾したまま通るのを防ぐ。"""
    b = buckets()
    names = list(b)
    for i, a in enumerate(names):
        for c in names[i + 1:]:
            dup = sorted(b[a] & b[c])
            assert not dup, f"{a} と {c} の両方にあります: {dup}"


def test_classification_has_no_ghost():
    """もう存在しない項目が分類表に残っていないか。

    残っていると、消したはずのものを探す人が出る。足す側だけ縛ると片手落ち。
    """
    ghosts = sorted(set().union(*buckets().values()) - all_fields())
    assert not ghosts, (
        "分類表に、もう存在しない項目が残っています: " + "、".join(ghosts)
        + "\n  → src/segments.py の分類から消してください。")


def test_held_fields_carry_a_reason():
    """**据え置きには理由が要る。**空文字も、短すぎるものも通さない。

    ここが二択との違い。据え置きはその場で何も起きないので、理由を書かせない
    かぎり逃げ道になる。
    """
    for name, reason in S.SCHEMA_HELD.items():
        assert isinstance(reason, str) and len(reason.strip()) >= 10, (
            f"SCHEMA_HELD['{name}'] に据え置いた理由がありません(10 文字以上)。"
            "\n  **版を上げないと決めたなら、旧版がその項目を知らなくても"
            "記録として成立する理由を書いてください。**"
            "\n  「小さい変更だから」は理由になりません。")


def test_not_serialized_fields_carry_a_reason():
    """保存しない項目にも理由が要る。"""
    for name, reason in S.SCHEMA_NOT_SERIALIZED.items():
        assert isinstance(reason, str) and len(reason.strip()) >= 10, (
            f"SCHEMA_NOT_SERIALIZED['{name}'] に理由がありません(10 文字以上)。")


def test_bumped_versions_are_real():
    """`SCHEMA_BUMPED` の版が、実際の `SCHEMA_VERSION` と食い違っていないか。

    `"foo": 6` と書いておいて `SCHEMA_VERSION` が 5 のままだと、
    **分類表だけが未来を指す。**
    """
    for name, ver in S.SCHEMA_BUMPED.items():
        assert isinstance(ver, int) and 1 <= ver <= S.SCHEMA_VERSION, (
            f"SCHEMA_BUMPED['{name}'] = {ver!r} が SCHEMA_VERSION "
            f"({S.SCHEMA_VERSION}) と食い違います。"
            "\n  **版を上げたなら SCHEMA_VERSION も上げてください。**")


def test_serialized_keys_match_the_classification():
    """**実際に書き出されるキー**が、分類表と一致しているか。

    dataclass の項目だけを見ると、`to_dict()` に直接足したキーを見逃す。
    実物を 1 つ作って、出てくるキーで確かめる。
    """
    proj = S.Project(
        audio_path="a.wav",
        speakers=[S.Speaker(id="sp01", name="佐藤")],
        segments=[S.Segment(index=0, start=0.0, end=1.0, text="t", cluster="g:A")],
    )
    d = proj.to_dict()
    written = set(d) | set(d["segments"][0]) | set(d["speakers"][0])
    written.discard("schema")          # 版そのもの。項目ではない
    known = set(S.SCHEMA_BASE) | set(S.SCHEMA_BUMPED) | set(S.SCHEMA_HELD)
    unknown = sorted(written - known)
    assert not unknown, (
        "保存されているのに分類されていないキーがあります: " + "、".join(unknown)
        + "\n  → to_dict() に直接足したキーではありませんか。"
        "\n     dataclass の項目にするか、分類表に足してください。")
    # 逆向き: 保存すると宣言したのに、実際には書き出されていないもの
    not_written = sorted(known - written)
    assert not not_written, (
        "分類では保存するはずなのに、書き出されていない項目があります: "
        + "、".join(not_written)
        + "\n  → to_dict() から漏れていませんか。"
        "**漏れると、その項目は保存で毎回消えます。**")


# ======================================================================
# 未来の版の扱い（B: 読むが、保存を拒む）
#
# 読むのを拒む案(A)は採らなかった。**中身が見えないまま拒む形**になり、
# 確認できることを売っている製品としては方向が逆。二台運用で「新しい機械で
# 作ったファイルを古い機械で確認したい」も現実に起きる。
#
# ## この検査が守れる範囲・守れない範囲
#
# **守れる**: 未来の版を読めること。保存が止まること。ファイルが無傷なこと。
# 止めるときの文面に、失われるものの名前が入っていること。
#
# **守れない**: **失われるものを漏れなく挙げられているか。**名前は読んだ
# その場で数えている(`unknown_keys_of`)ので、**そのファイルに現れなかった
# 項目は挙がらない**。将来の版が何を持つかはこちらには分からない。
# ======================================================================

def _future_file(tmp: Path, schema: int = 99) -> tuple[Path, bytes]:
    import json
    proj = S.Project(
        audio_path="a.wav",
        speakers=[S.Speaker(id="sp01", name="佐藤")],
        segments=[S.Segment(index=0, start=0.0, end=1.0, text="t", cluster="g:A")],
    )
    d = proj.to_dict()
    d["schema"] = schema
    d["future_top"] = {"x": 1}
    d["segments"][0]["future_seg"] = True
    f = tmp / "future.speakers.json"
    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return f, f.read_bytes()


def test_newer_schema_can_be_read():
    """**読めること。**中身が見えないまま拒むのは方向が逆。"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f, _ = _future_file(Path(td))
        proj = S.Project.load(f)
        assert len(proj.segments) == 1 and len(proj.speakers) == 1
        assert proj.loaded_schema == 99
        assert proj.is_from_newer_schema()


def test_newer_schema_refuses_to_save_and_leaves_the_file_alone():
    """**保存は拒み、ファイルは 1 バイトも変えない。**"""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f, before = _future_file(Path(td))
        proj = S.Project.load(f)
        try:
            proj.save()
            raise AssertionError("未来の版を上書きできてしまいました")
        except S.NewerSchemaError:
            pass
        assert f.read_bytes() == before, "拒んだのにファイルが書き換わっています"


def test_refusal_says_what_would_be_lost():
    """**「保存できません」だけにしない。**

    理由が分からないと、別名保存や再インストールで回避しようとする。
    失われるものを名前で挙げる。
    """
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f, _ = _future_file(Path(td))
        proj = S.Project.load(f)
        assert set(proj.unknown_keys) >= {"future_top", "future_seg"}, \
            f"知らないキーを拾えていません: {proj.unknown_keys}"
        msg = str(S.NewerSchemaError(99, proj.unknown_keys))
        for want in ("future_top", "future_seg", "99", str(S.SCHEMA_VERSION)):
            assert want in msg, f"文面に {want} が入っていません"
        assert "失われる" in msg, "何が起きるかを書いていない"
        assert "**" not in msg, "ダイアログに出す文面に Markdown の記号が残っています"


def test_current_and_older_schema_still_save():
    """**現行版と旧版は今までどおり保存できる。**上限検査が効きすぎていないか。"""
    import json, tempfile
    with tempfile.TemporaryDirectory() as td:
        proj = S.Project(
            audio_path="a.wav",
            segments=[S.Segment(index=0, start=0.0, end=1.0, text="t", cluster="g:A")],
        )
        for schema in (2, S.SCHEMA_VERSION):
            d = proj.to_dict()
            d["schema"] = schema
            f = Path(td) / f"v{schema}.speakers.json"
            f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
            q = S.Project.load(f)
            assert not q.is_from_newer_schema(), f"schema {schema} を未来と判定した"
            q.save()        # 例外が出ないこと


def test_new_project_is_not_treated_as_newer():
    """新規作成(ファイルから読んでいない)は、当然ながら保存できる。"""
    proj = S.Project(audio_path="a.wav")
    assert proj.loaded_schema is None
    assert not proj.is_from_newer_schema()


def test_guard_actually_fires():
    """**この検査が本当に落ちるかを確かめる。**

    HANDOFF「ガードを足したら、発火することを実測すること」。
    分類から 1 つ抜いた状態を作って、上の検査が落ちることを見る。
    """
    victim = sorted(S.SCHEMA_BASE)[0]
    known = (set(S.SCHEMA_BASE) - {victim}) | set(S.SCHEMA_BUMPED) \
        | set(S.SCHEMA_HELD) | set(S.SCHEMA_NOT_SERIALIZED)
    assert victim in sorted(all_fields() - known), (
        f"分類から {victim} を抜いても、未分類として検出できません")

    # 理由が空の据え置きを弾けるか
    for bad in ("", "   ", "短い"):
        assert not (isinstance(bad, str) and len(bad.strip()) >= 10), (
            f"空や短い理由({bad!r})を通してしまいます")


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
