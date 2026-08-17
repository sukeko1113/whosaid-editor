"""正解のある 4 帯を AmiVoice API で起こして SRT にする（段階 2 の比較）。

    set AMIVOICE_API_KEY=...   (PowerShell: $env:AMIVOICE_API_KEY="...")
    python tools\\bench_amivoice.py <出す先> [エンジン名]

    例:
      python tools\\bench_amivoice.py C:\\dev\\01\\test-audio\\bench

- 既定エンジンは -a-general（汎用）。
- **keepFillerToken=1 を必ず付ける。**AmiVoice は既定でフィラーを落とす
  （ScribeAssist の「言い淀み自動削除」と同じ挙動が API の既定）。
  これを付けずに測ると、逐語の物差しでは不当に悪く出る。
- **ログ保存なしのエンドポイント（/v1/nolog/）を使う。**録音は機微情報。
  送信は利用者が明示して実行したときだけ（この道具は測定用。製品からは呼ばない）。
- 毎月 60 分の無料枠がある。4 帯 8 分はその中に収まる。

音声は tools/bench_models.py が切り出した band.<帯>.wav を使う。
キーは環境変数からだけ読む。**引数やファイルには書かない**（履歴に残るため）。
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from bench_models import ts  # noqa: E402
from verbatim_truth import BANDS  # noqa: E402

NOLOG_URL = "https://acp-api.amivoice.com/v1/nolog/recognize"


def recognize(key: str, wav: Path, engine: str) -> dict:
    """1 ファイルを同期 API で起こす。返り値は API の JSON。"""
    boundary = uuid.uuid4().hex
    d_value = f"grammarFileNames={engine} keepFillerToken=1"

    def part(name: str, value: bytes, filename: str = "",
             ctype: str = "") -> bytes:
        head = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
        if filename:
            head += f'; filename="{filename}"'
        if ctype:
            head += f"\r\nContent-Type: {ctype}"
        return head.encode() + b"\r\n\r\n" + value + b"\r\n"

    body = (part("u", key.encode())
            + part("d", d_value.encode())
            + part("a", wav.read_bytes(), wav.name, "application/octet-stream")
            + f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        NOLOG_URL, data=body,
        headers={"Content-Type":
                 f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def to_rows(resp: dict, offset: float) -> list[tuple[float, float, str]]:
    """API の結果を (絶対開始, 絶対終了, 本文) に直す。

    フィラーは written が「%えー%」のように % で囲まれて返ることがある。
    **語そのものは残し、印だけ剥がす**（落とせば keepFillerToken の意味が無い）。
    """
    rows = []
    for res in resp.get("results", []):
        tokens = res.get("tokens") or []
        if not tokens:
            continue
        text = "".join((t.get("written") or "").replace("%", "")
                       for t in tokens).strip()
        if not text:
            continue
        start = float(tokens[0].get("starttime", 0)) / 1000.0
        end = float(tokens[-1].get("endtime", 0)) / 1000.0
        if end <= start:
            end = start + 0.5
        rows.append((offset + start, offset + end, text))
    return rows


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    outdir = Path(sys.argv[1])
    engine = sys.argv[2] if len(sys.argv) > 2 else "-a-general"

    key = os.environ.get("AMIVOICE_API_KEY", "")
    if not key:
        print("環境変数 AMIVOICE_API_KEY がありません。\n"
              '  PowerShell:  $env:AMIVOICE_API_KEY="<APPKEY>"\n'
              "キーはこの道具に引数で渡さないでください(履歴に残ります)。")
        return 1

    bands = sorted(BANDS.items(), key=lambda kv: kv[1][0])
    cuts = []
    for name, (lo, hi) in bands:
        p = outdir / f"band.{name}.wav"
        if not p.exists():
            print(f"切り出しがありません: {p}\n"
                  "先に tools\\bench_models.py を流してください。")
            return 1
        cuts.append((name, lo, p))

    tag = f"amivoice{engine.replace('-', '.')}"
    srt = outdir / f"{tag}.srt"
    if srt.exists():
        print(f"既にあります: {srt.name}(消してから再実行してください)")
        return 0

    print(f"[AmiVoice {engine}] ログ保存なし・keepFillerToken=1")
    rows: list[tuple[float, float, str]] = []
    t0 = time.monotonic()
    for name, lo, path in cuts:
        print(f"  {name} …", end="", flush=True)
        t1 = time.monotonic()
        try:
            resp = recognize(key, path, engine)
        except urllib.error.HTTPError as e:
            print(f" 失敗: HTTP {e.code} {e.read().decode('utf-8')[:200]}")
            continue
        except Exception as e:
            print(f" 失敗: {e}")
            continue
        got = to_rows(resp, lo)
        rows += got
        print(f" {len(got)} 区間 / {time.monotonic()-t1:.0f} 秒")

    if not rows:
        print("結果が取れませんでした。SRT は作りません。")
        return 1
    rows.sort(key=lambda r: (r[0], r[1]))
    srt.write_text("\n".join(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n"
                             for i, (a, b, t) in enumerate(rows, 1)),
                   encoding="utf-8")
    print(f"  → {srt.name}({len(rows)} 区間 / {time.monotonic()-t0:.0f} 秒)")
    print(f"\n次:\n  python tools\\report_verbatim.py <truth> {tag}={srt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
