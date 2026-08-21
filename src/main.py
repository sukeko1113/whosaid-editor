"""Whosaid 反訳エディタ - エントリーポイント"""
import sys
import traceback
import tkinter as tk
from tkinter import messagebox


def main() -> None:
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
