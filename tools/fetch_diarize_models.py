"""話者分離のモデルを取ってきて models/diarize に置く。

    python tools\\fetch_diarize_models.py [置き場]

CI（`.github/workflows/build.yml`）と手作業の両方がこれを使う。
**URL を workflow と README に別々に書くと、片方だけ古くなる。**

`models/` は `.gitignore` で除外してある（合わせて 44MB あり、
ライセンスも別々なので、リポジトリに入れずに取得する形にした）。
そのため取得を忘れやすく、`build.spec` は無ければビルドを止める。

置き場と名前は `src/diarize.py` の SEG_NAME / EMB_NAME に合わせる。
ここで直書きすると、片方だけ直したときに「置いたのに見つからない」になる。

標準ライブラリだけで動く。CI では依存を入れる前に走らせられる。
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from src.diarize import EMB_NAME, SEG_NAME  # noqa: E402

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

# **SHA-256 で留める。**配布元の資産が差し替わったら、黙って別物を配るのではなく
# 止まってほしい。ここに書いてあるのは、いま手元で動作を確認した実体の値。
#
# **転写のモデル（src/asr_fetch.py）は固定していない。片手落ちではない。**
# あちらは Hugging Face Hub から取り、**LFS プロトコル自体がハッシュを照合する。**
# こちらは GitHub Releases の生 URL で、**配布側に照合の仕組みが無い**ため、
# 固定値を持つのがここだけの必要になる。理由の全文は asr_fetch.py の
# NEEDED の直前にある。
MODELS = (
    {
        "name": "pyannote segmentation-3.0（声の切れ目・MIT）",
        # tag の綴りは配布元のまま（"recongition" も同様に配布元の綴り）
        "url": f"{BASE}/speaker-segmentation-models"
               "/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        "member": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
        "dest": SEG_NAME,
        "sha256": "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    },
    {
        "name": "NeMo TitaNet small（声の埋め込み・CC-BY-4.0）",
        "url": f"{BASE}/speaker-recongition-models/nemo_en_titanet_small.onnx",
        "member": None,                     # 単体のファイル
        "dest": EMB_NAME,
        "sha256": "ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
    },
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, out: Path, tries: int = 3) -> None:
    """落とす。ネットワークの一時的な失敗で CI を落とさない程度には粘る。"""
    import time
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, \
                    out.open("wb") as f:
                shutil.copyfileobj(r, f)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"    失敗（{attempt}/{tries}）: {e}")
            if attempt == tries:
                raise
            time.sleep(15 * attempt)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "models" / "diarize"
    root.mkdir(parents=True, exist_ok=True)

    for m in MODELS:
        dest = root / m["dest"]
        if dest.exists() and sha256(dest) == m["sha256"]:
            print(f"  そのまま使えます: {m['dest']}")
            continue

        print(f"  取得: {m['name']}")
        print(f"    {m['url']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / "download"
            download(m["url"], tmp)
            if m["member"]:
                with tarfile.open(tmp, "r:bz2") as tar:
                    src = tar.extractfile(m["member"])
                    if src is None:
                        print(f"    書庫に {m['member']} がありません")
                        return 1
                    with dest.open("wb") as f:
                        shutil.copyfileobj(src, f)
            else:
                shutil.copyfile(tmp, dest)

        got = sha256(dest)
        if got != m["sha256"]:
            # **合わなければ消して止める。**中途半端なものを残すと、次回
            # 「ある」と判断されて壊れたまま配ることになる
            dest.unlink(missing_ok=True)
            print(f"    SHA-256 が合いません:\n      期待 {m['sha256']}\n"
                  f"      実際 {got}\n"
                  "    配布元の資産が差し替わった可能性があります。"
                  "中身を確かめてから tools/fetch_diarize_models.py を直してください。")
            return 1
        print(f"    → {dest}（{dest.stat().st_size/1048576:.1f} MB・照合済み）")

    print(f"\n置き場: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
