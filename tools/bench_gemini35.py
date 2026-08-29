"""B（gemini-3.5-transcribe）を Interactions API で起こす（評価スパイク・段階2）。

    python tools\\bench_gemini35.py band <出す先>          帯 4 本（判定1・判定2）
    python tools\\bench_gemini35.py full <出す先> <音声>    全長 3 分割（判定3・判定4）
    python tools\\bench_gemini35.py determinism <出す先>    決定性の実測（帯 1 本 × 2 回）

**`tools/bench_cloud.py` の写しだが、API の呼び方と読み方だけが別物である。**
あちらは `generateContent` にプロンプトを投げてテキスト行を正規表現で読む。
こちらは **Interactions API** に `transcription_config` を投げ、
**構造化された `word_info` を読む**——テキスト行を経由しないので、時刻が秒に丸められず、
`redistribute_times()` の文字数按分もかからない。**これが B の価値そのものである。**

**録音は Gemini に送られる。** 利用者が明示的に選んだときだけ動かすこと
（CLAUDE.md の原則）。この道具は測定用で、製品からは呼ばない。

---

## 投げる形（v4.4 で実測して確定した。ドキュメントからの転記ではない）

```python
"transcription_config": {
    "language_codes": ["ja-JP"],
    "diarization_mode": "speaker",          # ← transcription_config の直下
    "timestamp_granularities": ["word"],    # ← 同じく直下
}
```

**`mode` の入れ子を作らないこと。**ドキュメントは
`mode: {type: "verbatim", diarization_mode: ..., timestamp_granularities: ...}`
と書いているが、**それは通るが黙って無視される**——`status: completed` が返り、
本文も返り、フィラーも残るのに、**`word_info` が 0 件・話者ラベルが空**になる。
**エラーにならないので、書いても動かないことに気づけない。**（指示書 §4-1 の警告）

**`custom_vocabulary` を渡さない。** 単語時刻と併用不可（実測）。
**`verbatim` は指定できない。** SDK の型に無く、入れ子は無視される。
したがって**逐語性はモードで担保できず、出力を見て判定する**（指示書 §2-3）。

## 出すもの

    gemini-3.5-transcribe.srt              判定1（report_verbatim.py に載せる。昇順）
    gemini-3.5-transcribe.words.json       **一次資料。**判定2・判定3・判定4 が読む
    gemini-3.5-transcribe.raw.<帯>.json    生レスポンス（§7-3 の必須要件）

**SRT は派生物である。**単語時刻も話者も捨てている。
`report_verbatim.load_engine()` が読み込み時にソートするので、SRT は昇順で書いてよい。
**`.words.json` は生成順を保存する**——並べ替えると単調性の破れが消える。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src import transcribe                       # noqa: E402  classify_api_error
from src.audio import find_ffmpeg, probe_duration  # noqa: E402
from src.config import load_config               # noqa: E402
from src.pipeline import _make_client            # noqa: E402
from bench_models import ts                      # noqa: E402
from verbatim_truth import BANDS                 # noqa: E402
import report_timing as rt                       # noqa: E402

NAME = "gemini-3.5-transcribe"

# **版を固定した識別子を使う。**エイリアスを使わない——提供側が中身を
# 差し替えたとき、同じ鍵で別物の転写が返る（PR #11 が 4 回起こした事故と同じ形）。
# **実行時に models 一覧で実在を確かめ、meta に記録する。**
MODEL = "gemini-3.5-transcribe"

# 全長モードの分割。**30 分基準で 3 分割に固定**（指示書 §7-3）。
# 実測は 40:47 まで通ったが、**緩い方向の実測は設計の根拠にしない**（§1-1）。
CHUNK_SECONDS = 30 * 60

# **決定性の制御は `seed`。`temperature` は存在しない**（SDK の型で確認）。
# **既定まかせにしない。**値は meta に残す。
# **`seed` があることは決定性を保証しないので、実測で確かめる**（determinism モード）。
SEED = 20260830

# **`max_output_tokens` は既定のまま。**（指示書 §7-3）
# 上げれば切れにくくなるが、**どこまで上げれば足りるかは日本語の逐語＋単語時刻の
# 出力量を測らないと分からない。**測る前に上げると、上げた値が適切だったかを
# 検証できない。**まず既定で測り、切れたらその値を根拠に決める。**
MAX_OUTPUT_TOKENS = None


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def build_body(uri: str, mime: str) -> dict:
    """投げる body。**この 1 箇所だけで組み立てる。**"""
    gc: dict = {
        "transcription_config": {
            "language_codes": ["ja-JP"],
            "diarization_mode": "speaker",
            "timestamp_granularities": ["word"],
            # custom_vocabulary は渡さない（単語時刻と併用不可）
        },
        "seed": SEED,
    }
    if MAX_OUTPUT_TOKENS is not None:
        gc["max_output_tokens"] = MAX_OUTPUT_TOKENS
    return {"model": MODEL,
            "input": [{"type": "audio", "uri": uri, "mime_type": mime}],
            "generation_config": gc}


def parse_offset(v) -> float:
    """`"0.900s"` を秒にする。**時刻は文字列で返る**（実測・指示書 §4-1）。"""
    if v is None:
        return 0.0
    s = str(v).strip()
    return float(s[:-1]) if s.endswith("s") else float(s)


def read_words(interaction, offset: float) -> list[dict]:
    """`word_info` を生成順のまま読み、元音声の絶対秒にする。

    **並べ替えない。**`.words.json` の `order` が生成順を持つ。
    """
    out: list[dict] = []
    for step in (getattr(interaction, "steps", None) or []):
        for content in (getattr(step, "content", None) or []):
            for a in (getattr(content, "annotations", None) or []):
                if getattr(a, "type", "") != "word_info":
                    continue
                out.append({
                    "order": len(out),
                    "text": getattr(a, "text", "") or "",
                    "speaker": getattr(a, "speaker", None),
                    "start": offset + parse_offset(getattr(a, "start_offset", None)),
                    "end": offset + parse_offset(getattr(a, "end_offset", None)),
                    # **出所のまま残す。**output_text 内の位置と思われるが未確認
                    "start_index": getattr(a, "start_index", None),
                    "end_index": getattr(a, "end_index", None),
                })
    return out


def group_segments(words: list[dict], request: int) -> list[dict]:
    """話者が切り替わる境界で束ねるだけ。**整形しない。**

    区間は**導出物**である。`word_from` / `word_to` を持たせるので、
    束ね方を変えたくなったら `words` から作り直せる。
    """
    segs: list[dict] = []
    for i, w in enumerate(words):
        if segs and segs[-1]["speaker"] == w["speaker"]:
            segs[-1]["text"] += w["text"]
            segs[-1]["end"] = w["end"]
            segs[-1]["word_to"] = i + 1
            continue
        segs.append({"order": len(segs), "speaker": w["speaker"],
                     "start": w["start"], "end": w["end"], "text": w["text"],
                     "word_from": i, "word_to": i + 1, "request": request})
    return segs


def transcribe_one(client, path: Path, offset: float, request: int,
                   outdir: Path, tag: str) -> tuple[list[dict], dict]:
    """1 リクエスト。**応答直後に打ち切りを判定する。**

    **切れたものを持って先へ進まない。**中断してもよいように、
    生レスポンスだけは必ず先に保存する（課金済みなので捨てない）。
    """
    print(f"  [{tag}] アップロード …", end="", flush=True)
    f = client.files.upload(file=str(path))
    print(" 完了", end="", flush=True)

    body = build_body(f.uri, getattr(f, "mime_type", "audio/wav"))
    t0 = time.monotonic()
    try:
        it = client.interactions.create(**body)
    except Exception as e:
        # **即座に中断。リトライで誤魔化さない**（v2.0.2 の再発防止）。
        why = transcribe.classify_api_error(e)
        print(f"\n  [{tag}] 失敗: {type(e).__name__}\n  {e}")
        if why:
            print(f"  → {why}")
        raise SystemExit(1)
    elapsed = time.monotonic() - t0

    raw_path = outdir / f"{NAME}.raw.{tag}.json"
    raw_path.write_text(json.dumps(
        it.model_dump(mode="json") if hasattr(it, "model_dump") else {},
        ensure_ascii=False, indent=2), encoding="utf-8")

    words = read_words(it, offset)
    segs = group_segments(words, request)
    audio_seconds = probe_duration(path)

    # --- 打ち切りの検出（§7-3・必須） ---
    # **report_timing.coverage() を呼ぶ。**同じ計算を 2 箇所に書かない。
    spans = [(s["start"] - offset, s["end"] - offset, s["text"]) for s in segs]
    cov = rt.coverage(spans, audio_seconds)
    print(f" / 転写 {elapsed:.0f}s / 単語 {len(words)} / 区間 {len(segs)}"
          f" / カバー率 {cov.ratio:.3f} / 最大ギャップ {cov.max_gap:.1f}s")

    rec = {
        "index": request, "tag": tag, "audio": path.name,
        "audio_seconds": audio_seconds, "offset": offset,
        "last_end": cov.last_end, "coverage": cov.ratio,
        "max_gap": cov.max_gap, "truncated": cov.truncated,
        "elapsed_sec": round(elapsed, 1),
        "words": len(words), "segments": len(segs),
        "usage": (it.model_dump(mode="json").get("usage")
                  if hasattr(it, "model_dump") else None),
        "raw": raw_path.name,
    }

    if cov.truncated:
        print(f"\n  [{tag}] **打ち切りの疑い。中断する。**")
        print(f"      カバー率 {cov.ratio:.3f} < {rt.COVERAGE_FLOOR}")
        print(f"      最終区間の終了 {cov.last_end:.1f}s / 入力音声 {audio_seconds:.1f}s")
        print(f"      不足 {cov.missing_seconds:.1f}s")
        print(f"      生レスポンスは残してある: {raw_path.name}")
        print("      **リトライしない。**分割を細かくするかは人間が決める。")
        (outdir / f"{NAME}.truncated.{tag}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(2)

    # **中抜けは自動で中断しない**（無音は実在する）。値を出して人が見る。
    return words, rec | {"segments_data": segs}


# ---------------------------------------------------------------------------
# 入力の用意
# ---------------------------------------------------------------------------

def slice_audio(src: Path, dst: Path, start: float, seconds: float) -> Path:
    """時間で切り出す。**再エンコードしない。**

    `src.audio.split_audio()` を使わない——あれは**無条件に 16kHz モノラル
    64kbps AAC へ再エンコードする**ので、帯 WAV と音質条件が揃わなくなる。
    """
    if not dst.exists():
        cmd = [find_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
               "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}",
               "-i", str(src), "-c", "copy", str(dst)]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0 or not dst.exists():
            raise SystemExit(f"切り出しに失敗: {r.stderr}")

    # **切り出した長さを必ず確かめる。**`-c copy` は容器によっては
    # キーフレーム境界に寄って要求と違う長さになる。**ここがずれると
    # 打ち切り検出の分母がずれ、切れていないのに切れたと判定する。**
    got = probe_duration(dst)
    if abs(got - seconds) > 1.0:
        raise SystemExit(
            f"切り出しの長さが合いません: {dst.name}\n"
            f"  要求 {seconds:.2f}s / 実際 {got:.2f}s（差 {got - seconds:+.2f}s）\n"
            "  **打ち切り検出の分母がずれるので進めない。**"
            "再エンコードで切り直すか、切り出しの方法を見直すこと。")
    return dst


# ---------------------------------------------------------------------------
# 出力
# ---------------------------------------------------------------------------

def write_outputs(outdir: Path, mode: str, audio_name: str,
                  words: list[dict], segs: list[dict], requests: list[dict],
                  model_note: dict) -> None:
    """`.words.json`（一次資料）と `.srt`（派生物）を書く。"""
    doc = {
        "schema": 1,
        "meta": {
            "backend": NAME, "model": MODEL, "api": "interactions",
            "run_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "mode": mode, "audio": audio_name,
            "language_codes": ["ja-JP"],
            "features": {
                "diarization_mode": "speaker",
                "timestamp_granularities": ["word"],
                "custom_vocabulary": None,
                "verbatim": None,
            },
            "verbatim_note": (
                "この API から verbatim / smart を指定する手段が無い"
                "（SDK の型に無く、mode の入れ子は通るが無視される）。"
                "逐語性は出力を見て判定すること（指示書 §2-3）。"),
            "seed": SEED,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "max_output_tokens_note": (
                "既定のまま。切れた場合にその値を根拠に調整する（§7-3）。"),
            "model_check": model_note,
            "chunk_seconds": CHUNK_SECONDS if mode == "full" else None,
            "requests": requests,
        },
        "words": words,        # ← API が返した順のまま
        "segments": segs,      # ← 生成順に order を振ってある
    }
    (outdir / f"{NAME}.words.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # SRT は派生物。**昇順で書いてよい**（report_verbatim が読み込み時にソートする）。
    rows = sorted(((s["start"], s["end"], s["text"]) for s in segs),
                  key=lambda r: (r[0], r[1]))
    (outdir / f"{NAME}.srt").write_text(
        "\n".join(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n"
                  for i, (a, b, t) in enumerate(rows, 1)), encoding="utf-8")
    print(f"\n  → {NAME}.words.json（単語 {len(words)} / 区間 {len(segs)}）")
    print(f"  → {NAME}.srt（{len(rows)} 区間・昇順）")


def check_model(client) -> dict:
    """**モデル識別子が実在するかを API に問い合わせる。推測で選ばない。**

    エイリアスと版固定の両方が見えたら、**版固定のほうを使う。**
    確認結果はそのまま meta に残す（後から「なぜこの文字列か」を追える）。
    """
    try:
        names = [getattr(m, "name", "") for m in client.models.list()]
    except Exception as e:
        return {"checked": False, "error": f"{type(e).__name__}: {e}"}
    hit = [n for n in names if MODEL in n]
    return {"checked": True, "requested": MODEL, "matched": hit,
            "note": "エイリアスは使わない。版固定の識別子を使う"}


# ---------------------------------------------------------------------------
# モード
# ---------------------------------------------------------------------------

def run_bands(client, outdir: Path, model_note: dict) -> int:
    """帯 4 本。判定1・判定2 用。"""
    bands = sorted(BANDS.items(), key=lambda kv: kv[1][0])
    for name, (lo, _hi) in bands:
        if not (outdir / f"band.{name}.wav").exists():
            print(f"切り出しがありません: band.{name}.wav\n"
                  "先に tools\\bench_models.py を流してください。")
            return 1

    all_words: list[dict] = []
    all_segs: list[dict] = []
    reqs: list[dict] = []
    for i, (name, (lo, _hi)) in enumerate(bands):
        w, rec = transcribe_one(client, outdir / f"band.{name}.wav",
                                lo, i, outdir, name)
        segs = rec.pop("segments_data")
        for x in w:
            x["order"] = len(all_words); x["request"] = i; all_words.append(x)
        for s in segs:
            s["order"] = len(all_segs); all_segs.append(s)
        reqs.append(rec)
    write_outputs(outdir, "band", "band.*.wav", all_words, all_segs,
                  reqs, model_note)
    return 0


def run_full(client, outdir: Path, audio: Path, model_note: dict) -> int:
    """全長。**30 分基準で明示的に 3 分割する。**判定3・判定4 用。"""
    total = probe_duration(audio)
    n = int(total // CHUNK_SECONDS) + (1 if total % CHUNK_SECONDS else 0)
    print(f"音声 {total:.1f}s を {CHUNK_SECONDS}s ごとに **{n} 分割**します"
          "（30 分基準。緩い方向の実測は設計の根拠にしない）。")

    all_words: list[dict] = []
    all_segs: list[dict] = []
    reqs: list[dict] = []
    for i in range(n):
        start = i * CHUNK_SECONDS
        part = slice_audio(audio, outdir / f"{NAME}.part{i}{audio.suffix}",
                           start, min(CHUNK_SECONDS, total - start))
        w, rec = transcribe_one(client, part, start, i, outdir, f"part{i}")
        segs = rec.pop("segments_data")
        for x in w:
            x["order"] = len(all_words); x["request"] = i; all_words.append(x)
        for s in segs:
            s["order"] = len(all_segs); all_segs.append(s)
        reqs.append(rec)

    # **結合後の被覆は補助。**一次判定はリクエストごと（上）。
    spans = [(s["start"], s["end"], s["text"]) for s in all_segs]
    cov = rt.coverage(spans, total)
    print(f"\n  結合後（**補助**）: カバー率 {cov.ratio:.3f} / "
          f"最大ギャップ {cov.max_gap:.1f}s")
    print("    ※ 一次判定はリクエストごと。**結合後だけでは中抜けが素通りする**")
    reqs.append({"index": "combined", "coverage": cov.ratio,
                 "max_gap": cov.max_gap, "audio_seconds": total,
                 "note": "補助。一次判定ではない"})
    write_outputs(outdir, "full", audio.name, all_words, all_segs,
                  reqs, model_note)
    return 0


def run_determinism(client, outdir: Path, model_note: dict) -> int:
    """帯 1 本を 2 回投げて、出力が一致するかを見る。

    **`seed` があることは決定性を保証しない。**提供側が明言していない以上、
    型を読むだけでは分からない。**実測で確かめる。**

    一致しなければ**どこがどう違ったかを残す。**単語の境界だけがずれるのか、
    本文そのものが変わるのか、話者ラベルが入れ替わるのかで意味が違う。
    全体が違えば設計の前提が崩れるが、末尾の 1 語だけなら許容できるかもしれない。
    **「不一致」とだけ報告しても判断できない。**
    """
    name, (lo, _hi) = sorted(BANDS.items(), key=lambda kv: kv[1][0])[0]
    band = outdir / f"band.{name}.wav"
    if not band.exists():
        print(f"切り出しがありません: {band.name}")
        return 1

    runs = []
    for k in (1, 2):
        w, rec = transcribe_one(client, band, lo, k - 1, outdir,
                                f"{name}.run{k}")
        rec.pop("segments_data")
        runs.append(w)

    a, b = runs
    diff = {
        "identical": a == b,
        "word_count": [len(a), len(b)],
        "text_identical": ("".join(x["text"] for x in a)
                           == "".join(x["text"] for x in b)),
        "speakers": [sorted({x["speaker"] for x in a if x["speaker"]}),
                     sorted({x["speaker"] for x in b if x["speaker"]})],
    }
    # **どこがどう違ったかを残す。**先頭 20 件まで、種類がわかる形で。
    mismatches = []
    for i, (x, y) in enumerate(zip(a, b)):
        kinds = [k for k in ("text", "speaker", "start", "end") if x[k] != y[k]]
        if kinds:
            mismatches.append({"order": i, "kinds": kinds, "a": x, "b": y})
        if len(mismatches) >= 20:
            break
    diff["first_mismatches"] = mismatches
    diff["mismatch_kinds"] = sorted({k for m in mismatches for k in m["kinds"]})

    (outdir / f"{NAME}.determinism.json").write_text(
        json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  決定性: {'一致' if diff['identical'] else '**不一致**'}")
    print(f"    単語数 {diff['word_count']} / 本文一致 {diff['text_identical']}")
    print(f"    話者 {diff['speakers']}")
    if not diff["identical"]:
        print(f"    違いの種類: {diff['mismatch_kinds']}")
        print(f"    最初の不一致（先頭 {len(mismatches)} 件）を "
              f"{NAME}.determinism.json に保存")
        print("\n  **一致しないなら、キャッシュと比較の前提が崩れる。**")
        print("  生レスポンスを保存して解析し直す設計も、投げ直しが再現しない")
        print("  なら意味が薄れる。**報告して停止する。**")
        return 3
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    mode, outdir = sys.argv[1], Path(sys.argv[2])
    if mode not in ("band", "full", "determinism"):
        print(__doc__)
        return 2
    outdir.mkdir(parents=True, exist_ok=True)

    key = (load_config().get("api_key") or os.environ.get("GOOGLE_API_KEY")
           or os.environ.get("GEMINI_API_KEY") or "")
    if not key:
        print("API キーがありません（設定にも環境変数にも）。")
        return 1
    client = _make_client(key)

    note = check_model(client)
    print(f"モデルの確認: {note}")
    print(f"seed={SEED} / max_output_tokens={MAX_OUTPUT_TOKENS}（既定のまま）")
    print("**録音は Gemini に送られる。**\n")

    if mode == "band":
        return run_bands(client, outdir, note)
    if mode == "determinism":
        return run_determinism(client, outdir, note)
    if len(sys.argv) < 4:
        print("full モードには音声のパスが要ります。")
        return 2
    return run_full(client, outdir, Path(sys.argv[3]), note)


if __name__ == "__main__":
    raise SystemExit(main())
