"""同じ 8 分に対して、複数のモデルを同じ条件で回して SRT にする（段階 2 の比較）。

    python tools\\bench_models.py <音声.wav> <出す先> [モデル名 ...]

    例:
      python tools\\bench_models.py C:\\dev\\01\\test-audio\\01+02edited.wav ^
        C:\\dev\\01\\test-audio\\bench small large-v3-turbo distil-large-v3

正解がある 4 帯（tools/verbatim_truth.py の BANDS）だけを切り出して起こす。
67 分の全長を回すと 1 モデルあたり数十分かかるが、測るのは 8 分だけなので
そこだけで足りる。

**small も必ず回すこと。**製品の `local` は 67 分を一括で転写したもので、
2 分の切り出しとは前後の文脈が違う。切り出しの small を対照に置かないと、
「モデルを大きくした効果」と「切り出しの効果」が混ざって読めなくなる。

出す SRT の時刻は**元の音声の絶対時刻**にする（帯の開始秒を足す）。
report_verbatim.py は帯 [開始, 終わり] で突き合わせるので、相対時刻のままだと
どの帯にも当たらない。

音声は外に出ない。faster-whisper は端末内で動く。
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.local_asr import LocalTranscriber  # noqa: E402
from verbatim_truth import BANDS  # noqa: E402


def cut(audio: Path, out: Path, start: float, dur: float) -> Path:
    """帯を切り出す。16kHz モノラル（whisper が内部で使う形）。"""
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
         "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(audio),
         "-ac", "1", "-ar", "16000", str(out)],
        check=True)
    return out


def ts(x: float) -> str:
    h, r = divmod(max(0.0, x), 3600)
    m, s = divmod(r, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    audio = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    models = sys.argv[3:] or ["small", "large-v3-turbo", "distil-large-v3"]
    if not audio.exists():
        print(f"音声が見つかりません: {audio}")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)

    bands = sorted(BANDS.items(), key=lambda kv: kv[1][0])
    print(f"帯 {len(bands)} 本 / 合計 "
          f"{sum(hi - lo for _, (lo, hi) in bands) / 60:.0f} 分")

    cuts = []
    for name, (lo, hi) in bands:
        p = cut(audio, outdir / f"band.{name}.wav", lo, hi - lo)
        cuts.append((name, lo, p))
        print(f"  切り出し: {name} ({lo:.0f}〜{hi:.0f} 秒)")

    for model in models:
        # HF の repo ID は「/」を含む。そのまま使うと下位フォルダに割れて
        # 書き出しで落ちる(kotoba-tech/... で実際に落ちた)。
        srt = outdir / (re.sub(r"[\\/:]+", "-", model) + ".srt")
        if srt.exists():
            print(f"\n[{model}] 既にあります: {srt.name}")
            continue
        # 「small+nocond」= 直前の本文を引き継がない
        # 「…+nowords」 = 単語時刻を取らない(CT2 変換が不完全なモデルは
        #                 単語時刻でネイティブ層ごと落ちる。kotoba で実測)
        parts = model.split("+")
        name_only, opts = parts[0], set(parts[1:])
        cond = "nocond" not in opts
        words = "nowords" not in opts
        print(f"\n[{model}] 読み込み中…"
              + ("" if cond else "（直前の本文を引き継がない）")
              + ("" if words else "（単語時刻なし）"))
        t0 = time.monotonic()
        try:
            tr = LocalTranscriber(model=name_only,
                                  condition_on_previous_text=cond)
        except Exception as e:                     # モデル名の間違い・取得失敗
            print(f"  使えません: {e}")
            continue

        rows: list[tuple[float, float, str]] = []
        spoken = 0.0
        for name, lo, path in cuts:
            print(f"  {name} …", end="", flush=True)
            t1 = time.monotonic()
            res = tr.transcribe(path, word_timestamps=words)
            if res is None:
                print(" 中止")
                continue
            # **帯の開始秒を足して絶対時刻にする。**相対のままだと
            # report_verbatim.py がどの帯にも当てられない。
            for u in res.utterances:
                rows.append((lo + u.rel_start, lo + u.rel_end, u.text))
            spoken += res.duration
            print(f" {len(res.utterances)} 区間 / {time.monotonic()-t1:.0f} 秒")

        rows.sort(key=lambda r: (r[0], r[1]))
        srt.write_text(
            "\n".join(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n"
                      for i, (a, b, t) in enumerate(rows, 1)),
            encoding="utf-8")
        took = time.monotonic() - t0
        print(f"  → {srt.name}（{len(rows)} 区間 / {took:.0f} 秒 / "
              f"実時間比 {took/spoken:.2f}）" if spoken else f"  → {srt.name}")

    print(f"\n次:\n  python tools\\report_verbatim.py <truth> "
          + " ".join(
              f"{re.sub(r'[^A-Za-z0-9]+', '_', m)}="
              f"{outdir / (re.sub(r'[\\\\/:]+', '-', m) + '.srt')}"
              for m in models))
    return 0


if __name__ == "__main__":
    sys.exit(main())
