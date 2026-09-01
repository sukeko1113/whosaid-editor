# -*- coding: utf-8 -*-
r"""CI（GPU 無しの機械）を手元で模して検査を回す。

    .venv\Scripts\python.exe tools\run_as_ci.py tests\test_gui_smoke.py

`align.pick_device()` を CPU 固定にしてから検査を走らせる。

## なぜ要るか

**「GPU のある機械でだけ通る検査」を炙り出すため。**
2026-09-01 に、CI が 19 回連続で落ちていたことに気づいた（`c2d4bf4` 以降）。
手元では通るので分からなかった——`_local_model_choices()` が
`pick_device()` を見て「高精度」を候補に出すか決めており、GPU のある機械では
出て、CI では出ない。**機械の状態が検査に漏れていた。**

`tests/test_core.py` の `test_device_choice_pairs_with_the_model` にも
同じ型の記録がある（「機械の状態が漏れていただけだった」）。

## 何を差し替えるか

**`pick_device` だけ。** `default_model` や `suggest_gpu_model` まで差し替えると、
それ自体を検査している `test_core.py` が落ちる（模擬のやりすぎ）。どちらも内部で
`pick_device` を見るので、これだけで CI と同じ振る舞いになる。

## 使いどころ

**GPU や装置に触る変更をしたら、両方で回すこと。**

    .venv\Scripts\python.exe tests\test_gui_smoke.py              ← この機械
    .venv\Scripts\python.exe tools\run_as_ci.py tests\test_gui_smoke.py  ← CI 相当
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import align                      # noqa: E402

align.pick_device = lambda prefer_gpu=True: (align.DEVICE, align.COMPUTE_TYPE)
sys.stderr.write("[CI 模擬] GPU 無しとして実行します\n")

if len(sys.argv) < 2:
    sys.stderr.write(__doc__.splitlines()[2] + "\n")
    raise SystemExit(2)

target = Path(sys.argv[1])
if not target.is_absolute():
    target = ROOT / target
if not target.is_file():
    sys.stderr.write(f"見つかりません: {target}\n")
    raise SystemExit(2)

sys.argv = [str(target)] + sys.argv[2:]
runpy.run_path(str(target), run_name="__main__")
