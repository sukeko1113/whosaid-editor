"""話者分離の実測(sherpa-onnx + pyannote 分割 + TitaNet 埋め込み)。

    python tools\\diarize_poc.py <音声> [切り出す秒数(0=全長)] [話者数(-1=自動)]

製品には含めない(実測のための道具)。技術方針書 §3.4 / ローカル転写設計書 §6 の
裏付けを取るために書いた。2026-08-15 に 67 分実会議で実測し、RTF 0.153・
区間への割当 97%・まとまり 6 個という結果を得ている。

必要なもの:
  pip install sherpa-onnx
  モデル(既定 C:\\dev\\01\\diarize-models。環境変数 DIARIZE_MODELS で変更可)
    - sherpa-onnx-pyannote-segmentation-3-0/model.onnx  … MIT
    - nemo_en_titanet_small.onnx                        … CC-BY-4.0
  どちらも sherpa-onnx の GitHub Releases から取得できる。
  ライセンスの確認は技術方針書 §2 を参照(TitaNet はクレジット表記が要る)。

音声は外へ出さない。すべて端末内で動く。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from collections import Counter
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

MODELS = Path(os.environ.get("DIARIZE_MODELS", r"C:\dev\01\diarize-models"))
SEG = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB = MODELS / "nemo_en_titanet_small.onnx"


def load_16k_mono(path: Path, seconds: float | None):
    """ffmpeg で 16kHz モノラルに落として読む(sherpa-onnx の要求)。"""
    import numpy as np

    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "x.wav"
        cmd = ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le"]
        if seconds:
            cmd += ["-t", str(seconds)]
        cmd += [str(wav), "-loglevel", "error"]
        subprocess.run(cmd, check=True)
        with wave.open(str(wav), "rb") as w:
            raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    try:
        import sherpa_onnx as so
    except ImportError:
        print("sherpa-onnx が要ります:  pip install sherpa-onnx")
        return 1
    for p in (SEG, EMB):
        if not p.exists():
            print(f"モデルがありません: {p}")
            print("DIARIZE_MODELS で置き場を指定できます。")
            return 1

    audio = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else None
    num_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else -1

    print(f"音声を読み込んでいます: {audio.name}"
          f"{f' (先頭 {seconds:.0f} 秒)' if seconds else ' (全長)'}")
    t0 = time.time()
    samples = load_16k_mono(audio, seconds)
    dur = len(samples) / 16000.0
    print(f"  {dur/60:.1f} 分 / 読み込み {time.time()-t0:.1f} 秒")

    config = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEG)),
            num_threads=1,
        ),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=str(EMB), num_threads=1),
        # 名簿の人数を渡せるのが本製品の強み(技術方針書 §3.4)。
        # -1 なら自動。自動だと 1 区間だけの幽霊話者が出るが、区間へ割り当てる
        # 段階で消える(重なりの多数決を取れないため)。
        clustering=so.FastClusteringConfig(num_clusters=num_clusters),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        print("設定が不正です(モデルの置き場を確認)")
        return 1

    sd = so.OfflineSpeakerDiarization(config)
    print(f"話者分離を開始します"
          f"(話者数 {'自動' if num_clusters < 0 else num_clusters} / 1 スレッド)...")
    last = [0]

    def progress(done: int, total: int) -> int:
        pct = done * 100 // max(1, total)
        if pct >= last[0] + 20:
            last[0] = pct
            print(f"  {pct}%", flush=True)
        return 0

    t1 = time.time()
    result = sd.process(samples, callback=progress)
    took = time.time() - t1
    turns = result.sort_by_start_time()

    print(f"\n所要 {took/60:.1f} 分 (音声の {took/dur:.3f} 倍 = RTF)")
    print(f"話者区間 {len(turns)} 件")
    counts = Counter(t.speaker for t in turns)
    talk: Counter = Counter()
    for t in turns:
        talk[t.speaker] += t.end - t.start
    print(f"話者 {len(counts)} 人ぶん検出")
    for spk, n in sorted(counts.items()):
        print(f"  話者 {spk}: {n:>4} 区間 / 発話 {talk[spk]/60:>6.1f} 分 "
              f"({talk[spk]/dur*100:>4.1f}%)")

    tag = f"n{num_clusters}" if num_clusters > 0 else "auto"
    out = audio.with_suffix(f".diarize.{tag}.{int(dur)}s.json")
    out.write_text(json.dumps({
        "audio": str(audio),
        "seconds": dur,
        "rtf": took / dur,
        "num_clusters_requested": num_clusters,
        "turns": [{"start": t.start, "end": t.end, "speaker": t.speaker}
                  for t in turns],
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\n書き出し: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
