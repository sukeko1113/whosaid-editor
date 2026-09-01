# -*- coding: utf-8 -*-
"""CI（GPU 無し）を手元で模して検査を回す。

    python run_as_ci.py tests\test_gui_smoke.py

`align.pick_device` を CPU 固定にしてから検査を走らせる。
**「GPU のある機械でだけ通る検査」を炙り出すため。**CI で 19 回連続で
落ちていたのに、手元では通っていた（2026-09-01）。
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(r"C:\dev\01\whosaid-editor")
sys.path.insert(0, str(ROOT))

from src import align                      # noqa: E402

align.pick_device = lambda prefer_gpu=True: (align.DEVICE, align.COMPUTE_TYPE)
# **pick_device だけを差し替える。**default_model や suggest_gpu_model まで
# 差し替えると、それ自体を検査している test_core が落ちる（模擬のやりすぎ）。
# どちらも内部で pick_device を見るので、これだけで CI と同じ振る舞いになる。
sys.stderr.write("[CI 模擬] GPU 無しとして実行します\n")

target = ROOT / sys.argv[1]
sys.argv = [str(target)] + sys.argv[2:]
runpy.run_path(str(target), run_name="__main__")
