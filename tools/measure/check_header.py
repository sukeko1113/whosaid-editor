"""集計 CSV の見出しが雛形と一致するかを確かめる。

    python tools/measure/check_header.py                      # 雛形の自己検査
    python tools/measure/check_header.py <csv> [<csv>...]     # 実データを検査
    python tools/measure/check_header.py --init <出力先>       # 雛形をコピー

1 本目を流す前に列を確定させる決まりにしてある(引き継ぎ資料 §5 作業1)。
途中で列を足すと前半と後半が揃わず、10 本を並べられなくなる。
人が気づくのは全部流し終えたあとなので、機械で止める。
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent

# 雛形の名前 → 実データの名前
TEMPLATES = {
    "sources": "sources.header.csv",
    "runs": "runs.header.csv",
    "stages": "stages.header.csv",
    "truth": "truth.header.csv",
}

# 空欄を許さない列。ここが空だと、あとから何を測ったのか復元できない。
REQUIRED = {
    "sources": ("source_id", "kind", "sha256", "duration_sec",
                "channels", "channel_note", "ac1_gain_db"),
    "runs": ("source_id", "engine", "model", "run_date",
             "min_block", "min_coverage"),
    "stages": ("source_id", "engine", "stages_detected", "start_cue"),
    "truth": ("source_id", "engine", "band_id", "band_rule",
              "truth_id", "metrics_scope"),
}

# metrics_scope=text_only の行で空欄にしておく列(区間単位の指標)。
# 0 と書いてはいけない。測っていないことと 0 だったことは違う。
SEGMENT_METRICS = (
    "backchannel_retention", "backchannel_hit", "backchannel_total",
    "short_utterance_rate", "short_hit", "short_total",
    "missed_recovery", "missed_hit", "missed_total",
)


def header_of(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        row = next(csv.reader(f), [])
    return [c.strip() for c in row]


def kind_of(path: Path) -> str | None:
    """ファイル名から、どの雛形と比べるかを決める。"""
    stem = path.name.lower()
    for kind in TEMPLATES:
        if stem.startswith(kind):
            return kind
    return None


def compare(kind: str, path: Path) -> list[str]:
    want = header_of(HERE / TEMPLATES[kind])
    got = header_of(path)
    if got == want:
        return []
    problems = []
    missing = [c for c in want if c not in got]
    extra = [c for c in got if c not in want]
    if missing:
        problems.append(f"足りない列: {', '.join(missing)}")
    if extra:
        problems.append(f"余分な列: {', '.join(extra)}")
    if not missing and not extra:
        problems.append("列の順序が違う")
        problems.append(f"  雛形: {', '.join(want)}")
        problems.append(f"  実物: {', '.join(got)}")
    return problems


def check_rows(kind: str, path: Path) -> list[str]:
    """必須列の空欄と、text_only 行の書き方を見る。"""
    problems: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            for col in REQUIRED.get(kind, ()):
                if col in row and not (row.get(col) or "").strip():
                    problems.append(f"{i} 行目: {col} が空です")
            if kind == "truth" and (row.get("metrics_scope") or "").strip() == "text_only":
                filled = [c for c in SEGMENT_METRICS if (row.get(c) or "").strip()]
                if filled:
                    problems.append(
                        f"{i} 行目: metrics_scope=text_only なのに区間単位の指標が"
                        f"入っています({', '.join(filled)})。測っていない列は空欄にします")
    return problems


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--init":
        if len(argv) < 2:
            print("--init のあとに出力先フォルダを指定してください。")
            return 2
        out = Path(argv[1])
        out.mkdir(parents=True, exist_ok=True)
        for kind, name in TEMPLATES.items():
            dst = out / f"{kind}.csv"
            if dst.exists():
                print(f"  そのまま: {dst}(既にあります)")
                continue
            shutil.copyfile(HERE / name, dst)
            print(f"  作成:     {dst}")
        return 0

    targets = [Path(a) for a in argv] if argv else \
        [HERE / n for n in TEMPLATES.values()]

    failures = 0
    for path in targets:
        if not path.exists():
            print(f"FAIL {path} — ファイルがありません")
            failures += 1
            continue
        kind = kind_of(path)
        if kind is None:
            print(f"FAIL {path} — どの雛形と比べるか決められません"
                  f"(名前を {'/'.join(TEMPLATES)} で始めてください)")
            failures += 1
            continue
        problems = compare(kind, path) + check_rows(kind, path)
        if problems:
            print(f"FAIL {path.name}({kind})")
            for p in problems:
                print(f"     {p}")
            failures += 1
        else:
            print(f"ok   {path.name}({kind})")

    print(f"\n{'FAILED' if failures else 'ALL PASSED'}({failures} 件)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
