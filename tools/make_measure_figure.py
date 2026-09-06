# -*- coding: utf-8 -*-
r"""7 回の測定を 1 枚の図にする（note 記事・報告用）。

    .venv\Scripts\python.exe tools\make_measure_figure.py [出力先.png]
    省略したときの出力先は _tmp\測定7回.png（.gitignore 済み）。

    必要: Pillow（requirements.txt には入れない。配布物に不要なため）

**見せたいのは 1 つだけ。**「本文は揺れるが、声のまとまりは動かない」。
そのために、区間数の列（揺れる）とまとまりの列（動かない）を隣に並べる。

数字は `.work_<音声名>\timings.jsonl` と `run-*.log` から読む（推測しない）。
DIRS のパスと .work_01+02edited は 1 号機の測定データ専用（他の環境では動かない）。
"""
import io
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DIRS = [
    ("G ドライブ", r"G:\マイドライブ\2026年度\アプリ開発\26-08-01逐語反訳アプリ開発"
                   r"\テストデータ\測定 1（1号機 small  話者分離 ON）"),
    ("ローカル", r"C:\dev\whosaid-editor-testdata\測定 2（ローカル small  話者分離 ON）"),
    ("ローカル", r"C:\dev\whosaid-editor-testdata\測定 3（large-v3  話者分離 ON）"),
]
BUILD = "2026-08-31T12:09"

F = r"C:\Windows\Fonts"
def font(name, size):
    return ImageFont.truetype(str(Path(F) / name), size)

REG, BOLD = "YuGothM.ttc", "YuGothB.ttc"
INK, SUB, LINE = (26, 26, 26), (110, 110, 110), (208, 208, 208)
MOVE, STAY = (183, 28, 28), (21, 101, 192)      # 揺れる / 動かない


def collect():
    rows = []
    for place, base in DIRS:
        w = Path(base) / ".work_01+02edited"
        tj = w / "timings.jsonl"
        if not tj.is_file():
            continue
        for ln in tj.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r["started_at"] < BUILD or not r.get("log"):
                continue        # 旧ビルドの回は混ぜない
            body = (w / r["log"]).read_text(encoding="utf-8", errors="replace")
            m = re.search(r"話者の分離が終わりました\((\d+) 区間", body)
            c = re.search(r"声のまとまりを (\d+) 種類", body)
            r["_place"] = place
            for key, name, hit in (
                    ("_diar", "話者の分離が終わりました(N 区間", m),
                    ("_clusters", "声のまとまりを N 種類", c)):
                if hit is None:
                    # 0 を入れると「0 区間・0 まとまり」の図が黙って出る
                    print(f"  ログの文言が変わりました: {w / r['log']}")
                    print(f"  この形が見つかりません: {name}")
                    sys.exit(1)
                r[key] = int(hit.group(1))
            rows.append(r)
    rows.sort(key=lambda r: r["started_at"])
    return rows


def hms(sec):
    m, s = divmod(int(round(sec)), 60)
    return f"{m}分{s:02d}秒"


def main() -> int:
    rows = collect()
    if len(rows) != 7:
        print(f"  7 回ぶん見つかりません（{len(rows)} 回）。測定データを確認してください")
        return 1

    # 図は「7 回とも同じ」と言い切る。本当に同じかを確かめる（推測しない）
    for key, label in (("_diar", "話者分離の区間"), ("_clusters", "まとまり")):
        if len({r[key] for r in rows}) > 1:
            print(f"  7 回で{label}が一致しません。図の主張が成り立ちません")
            base = rows[0][key]
            for i, r in enumerate(rows, 1):
                mark = "" if r[key] == base else "  ← 1 回目と違う"
                print(f"    {i} 回目 {r['started_at']} "
                      f"{r['log']}: {r[key]}{mark}")
            return 1

    # 振れ幅はモデルで絞る。測定順に頼ると、並びが変わったとき
    # 高精度（large-v3）の回が標準の振れ幅に混ざる
    std = [r for r in rows if r["engine"]["model"] == "small"]
    if not std:
        print("  標準（small）の回が 1 つもありません。振れ幅を出せません")
        return 1

    W, H = 1240, 760
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    d.text((56, 40), "同じ音声を 7 回、起こし直しました",
           font=font(BOLD, 40), fill=INK)
    d.text((56, 96),
           "67 分の会議／毎回キャッシュを削除し、アプリも起動し直し",
           font=font(REG, 21), fill=SUB)

    # (x, 見出し, 寄せ, 色の役割)。**色は列で決める。**値で決めると
    # 「6 回目」の 6 が「まとまり 6」と同じ色になった（実際になった）
    cols = [(56, "回", "l", None), (110, "出力先", "l", None),
            (268, "モデル", "l", None), (500, "終わるまで", "r", None),
            (700, "本文の区間数", "r", MOVE),
            (950, "話者分離の区間", "r", STAY),
            (1184, "まとまり", "r", STAY)]
    y = 164
    for x, label, al, role in cols:
        f = font(BOLD, 20)
        w = d.textlength(label, font=f)
        d.text((x - w if al == "r" else x, y), label, font=f,
               fill=role or SUB)
    y += 34
    d.line([(56, y), (W - 56, y)], fill=INK, width=2)

    y += 14
    for i, r in enumerate(rows, 1):
        big = r["engine"]["model"] != "small"
        vals = [str(i), r["_place"], "高精度" if big else "標準",
                hms(r["total_seconds"]), f"{r['segments']:,}",
                f"{r['_diar']:,}", str(r["_clusters"])]
        for (x, _, al, role), v in zip(cols, vals):
            f = font(BOLD if role else REG, 22)
            w = d.textlength(v, font=f)
            d.text((x - w if al == "r" else x, y), v, font=f, fill=role or INK)
        y += 44
        if i < 7:
            d.line([(56, y - 10), (W - 56, y - 10)], fill=LINE, width=1)

    y += 10
    d.line([(56, y), (W - 56, y)], fill=INK, width=2)
    y += 26

    lo = min(r["segments"] for r in std)
    hi = max(r["segments"] for r in std)
    d.text((56, y), "本文は揺れました。", font=font(BOLD, 27), fill=MOVE)
    d.text((56 + d.textlength("本文は揺れました。", font=font(BOLD, 27)), y + 4),
           f"　標準の {len(std)} 回では、区間の数が "
           f"{lo} から {hi} まで動きます。",
           font=font(REG, 23), fill=INK)
    y += 46
    d.text((56, y), "声のまとまりは動きませんでした。",
           font=font(BOLD, 27), fill=STAY)
    d.text((56 + d.textlength("声のまとまりは動きませんでした。",
                              font=font(BOLD, 27)), y + 4),
           f"　7 回とも {rows[0]['_diar']} 区間・"
           f"{rows[0]['_clusters']} つのまとまり。",
           font=font(REG, 23), fill=INK)
    y += 54
    d.text((56, y),
           "生データが残っていた 3 回を突き合わせると、区切りの時刻も話者の番号も、"
           "小数点の全桁まで同じでした。",
           font=font(REG, 20), fill=SUB)
    y += 30
    # 内訳は実データから。日付と機械は timings.jsonl に無いので直書きのまま
    big_n = len(rows) - len(std)
    breakdown = f"標準 {len(std)} 回"
    if big_n:
        breakdown += f"・高精度 {big_n} 回"
    d.text((56, y),
           f"※ 実測は音源 1 本・{len(rows)} 回（2026-08-31）。{breakdown}・"
           "機械 1 台（GTX 1660 SUPER）。",
           font=font(REG, 18), fill=SUB)

    if len(sys.argv) > 1:
        out = Path(sys.argv[1])
    else:
        # 既定はリポジトリの _tmp/（.gitignore 済み）。直下に置くと
        # git status に出てきて、毎回消す手間になる
        out = Path(__file__).resolve().parent.parent / "_tmp" / "測定7回.png"
        out.parent.mkdir(exist_ok=True)
    img.save(out)
    print(f"  書きました: {out}  ({img.width}x{img.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
