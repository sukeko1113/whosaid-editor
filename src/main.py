"""Whosaid 反訳エディタ - エントリーポイント"""
import sys
import traceback
import tkinter as tk
from tkinter import messagebox


def main() -> None:
    # **画面を作る前に見る。**話者分離は sherpa-onnx が GIL を握るため、
    # 同じプロセスで回すと画面がまったく描けない(実測: 7 分の音声で 17 秒)。
    # 親が自分自身をこの形で呼び直し、計算だけを子に任せる。
    if "--diarize" in sys.argv[1:]:
        try:
            from src.diarize import child_main
        except ImportError:
            from diarize import child_main  # type: ignore
        i = sys.argv.index("--diarize")
        sys.exit(child_main(sys.argv[i + 1:]))

    try:
        from src.gui import App
    except ImportError:
        # PyInstaller でフリーズされた場合のフォールバック
        from gui import App  # type: ignore
    try:
        app = App()
        app.mainloop()
    except Exception:
        # GUI 起動前の致命的エラーをユーザに見せる
        err = traceback.format_exc()
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("起動エラー", err)
        except Exception:
            print(err, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
