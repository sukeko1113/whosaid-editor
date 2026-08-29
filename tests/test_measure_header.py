"""集計 CSV の見出し検査(tools/measure/check_header.py)の検査。

    .venv\\Scripts\\python.exe tests\\test_measure_header.py

見出しが揃っているかを見る道具なので、**揃っていないものを本当に落とすか**を
確かめる。通ることだけを見ると、何も検査していない道具でも通ってしまう。
"""
from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "measure"))
import check_header as ch                                   # noqa: E402


TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "tools" / "measure"


def header(kind: str) -> list[str]:
    return ch.header_of(TEMPLATE_DIR / ch.TEMPLATES[kind])


def write_csv(path: Path, cols: list[str], rows: list[dict] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows or []:
            w.writerow({c: r.get(c, "") for c in cols})


def full_row(kind: str, **over) -> dict:
    """必須列が全部埋まった行を作る。"""
    row = {c: "x" for c in header(kind)}
    row.update(over)
    return row


# ======================================================================
# 雛形そのもの
# ======================================================================

def test_templates_pass_self_check():
    assert ch.main([]) == 0


def test_every_template_exists_and_has_no_duplicate_columns():
    for kind, name in ch.TEMPLATES.items():
        cols = ch.header_of(TEMPLATE_DIR / name)
        assert cols, kind
        assert len(cols) == len(set(cols)), f"{kind} に重複した列がある"


def test_required_columns_are_actually_in_the_templates():
    """必須指定した列が雛形に無ければ、検査は永久に空振りする。"""
    for kind, cols in ch.REQUIRED.items():
        have = header(kind)
        for c in cols:
            assert c in have, f"{kind}.{c} が雛形に無い"


def test_segment_metrics_are_in_the_truth_template():
    have = header("truth")
    for c in ch.SEGMENT_METRICS:
        assert c in have, c


# ======================================================================
# 落ちるべきものが落ちるか
# ======================================================================

def test_missing_column_is_caught():
    cols = header("runs")[:-1]              # 末尾を 1 つ落とす
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "runs.csv"
        write_csv(p, cols)
        problems = ch.compare("runs", p)
    assert problems and any("足りない列" in x for x in problems), problems


def test_extra_column_is_caught():
    cols = header("runs") + ["余計な列"]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "runs.csv"
        write_csv(p, cols)
        problems = ch.compare("runs", p)
    assert problems and any("余分な列" in x for x in problems), problems


def test_reordered_columns_are_caught():
    """列が揃っていても順序が違えば落とす(貼り合わせで崩れる)。"""
    cols = header("stages")
    cols[1], cols[2] = cols[2], cols[1]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "stages.csv"
        write_csv(p, cols)
        problems = ch.compare("stages", p)
    assert problems and any("順序" in x for x in problems), problems


def test_empty_required_cell_is_caught():
    """min_block が空の行を通すと、どの閾値で測ったか復元できなくなる。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "runs.csv"
        write_csv(p, header("runs"), [full_row("runs", min_block="")])
        problems = ch.check_rows("runs", p)
    assert problems and any("min_block" in x for x in problems), problems


def test_filled_optional_cell_is_not_caught():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "runs.csv"
        write_csv(p, header("runs"), [full_row("runs", notes="")])
        assert ch.check_rows("runs", p) == []


def test_text_only_row_must_leave_segment_metrics_blank():
    """演説帯で区間単位の指標に 0 を書くと、測っていないことが消える。"""
    row = full_row("truth", metrics_scope="text_only")
    row["short_hit"] = "0"                  # 0 と書いてしまった
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "truth.csv"
        write_csv(p, header("truth"), [row])
        problems = ch.check_rows("truth", p)
    assert problems and any("text_only" in x for x in problems), problems


def test_text_only_row_with_blanks_passes():
    row = full_row("truth", metrics_scope="text_only")
    for c in ch.SEGMENT_METRICS:
        row[c] = ""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "truth.csv"
        write_csv(p, header("truth"), [row])
        assert ch.check_rows("truth", p) == []


def test_full_scope_row_may_fill_segment_metrics():
    row = full_row("truth", metrics_scope="full")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "truth.csv"
        write_csv(p, header("truth"), [row])
        assert ch.check_rows("truth", p) == []


def test_unknown_file_name_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "nazono.csv"
        write_csv(p, header("runs"))
        assert ch.kind_of(p) is None
        assert ch.main([str(p)]) == 1


def test_missing_file_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        assert ch.main([str(Path(d) / "runs.csv")]) == 1


# ======================================================================
# --init
# ======================================================================

def test_init_creates_the_four_files():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "measure"
        assert ch.main(["--init", str(out)]) == 0
        made = sorted(p.name for p in out.glob("*.csv"))
        assert made == ["runs.csv", "sources.csv", "stages.csv", "truth.csv"], made
        assert ch.main([str(out / n) for n in made]) == 0


def test_init_does_not_overwrite_existing_data():
    """実データの入ったファイルを潰さない。"""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "measure"
        out.mkdir()
        keep = out / "runs.csv"
        write_csv(keep, header("runs"), [full_row("runs")])
        before = keep.read_text(encoding="utf-8")
        ch.main(["--init", str(out)])
        assert keep.read_text(encoding="utf-8") == before


def test_init_without_destination_fails():
    assert ch.main(["--init"]) == 2


# ======================================================================
# 帯の指定
# ======================================================================

def test_band_rule_notation_is_documented():
    """README に標準の 3 帯が書いてあること(規則を口伝にしない)。"""
    readme = (TEMPLATE_DIR / "README.md").read_text(encoding="utf-8")
    for token in ("HEnDA12:S02@0-120", "HEnDA12:S08@0-120", "HEnDA12:S01@0-120"):
        assert token in readme, token
    # 12 セクションが全部書いてあること
    for n in range(1, 13):
        assert f"S{n:02d}" in readme, f"S{n:02d}"


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
