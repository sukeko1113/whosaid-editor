@echo off
REM ============================================================
REM Gemini Transcriber - Build Script (Windows)
REM ------------------------------------------------------------
REM 必要環境:
REM   - Python 3.11 以上(PATH 通っていること)
REM   - ffmpeg/ffmpeg.exe を配置(後述)
REM   - Inno Setup 6 をインストール、iscc.exe へパス通すか
REM     C:\Program Files (x86)\Inno Setup 6\ISCC.exe を呼ぶ
REM ============================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

echo === Gemini Transcriber Build ===
echo.

REM ----- 0) ffmpeg 確認 -----
if not exist "ffmpeg\ffmpeg.exe" (
    echo [ERROR] ffmpeg\ffmpeg.exe が見つかりません。
    echo.
    echo  https://www.gyan.dev/ffmpeg/builds/  から
    echo  "ffmpeg-release-essentials.zip" をダウンロードし、
    echo  解凍して bin\ffmpeg.exe をこのプロジェクトの ffmpeg\ に置いてください。
    echo.
    pause
    exit /b 1
)

if not exist "ffmpeg\ffplay.exe" (
    echo [WARN] ffmpeg\ffplay.exe が見つかりません。
    echo        話者割当画面の区間再生が簡易モード ^(winsound^) になります。
    echo        同じ zip の bin\ffplay.exe を ffmpeg\ にコピーすることを推奨します。
    echo.
)

REM ----- 1) Python 仮想環境 -----
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] 仮想環境を作成しています...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] 仮想環境の作成に失敗しました。Python が PATH にあるか確認してください。
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"

REM ----- 2) 依存パッケージ -----
echo [2/4] 依存パッケージをインストールしています...
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] パッケージインストールに失敗しました。
    pause
    exit /b 1
)
python -m pip install pyinstaller
if errorlevel 1 (
    echo [ERROR] PyInstaller のインストールに失敗しました。
    pause
    exit /b 1
)

REM ----- 3) PyInstaller -----
echo.
echo [3/4] PyInstaller でバンドル中...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist
pyinstaller --clean -y build.spec
if errorlevel 1 (
    echo [ERROR] PyInstaller によるバンドルに失敗しました。
    pause
    exit /b 1
)

REM ----- 4) Inno Setup -----
echo.
echo [4/4] Inno Setup でインストーラを作成中...

set "ISCC="
where iscc >nul 2>&1
if not errorlevel 1 set "ISCC=iscc"
if "%ISCC%"=="" if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if "%ISCC%"=="" if exist "C:\Program Files\Inno Setup 6\ISCC.exe"       set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"

if "%ISCC%"=="" (
    echo [WARN] Inno Setup ^(ISCC^) が見つかりません。
    echo  https://jrsoftware.org/isdl.php からインストール後に再実行してください。
    echo.
    echo  ※ 実行可能ファイルだけは dist\GeminiTranscriber\GeminiTranscriber.exe に出力済みです。
    pause
    exit /b 0
)

"%ISCC%" installer.iss
if errorlevel 1 (
    echo [ERROR] Inno Setup によるビルドに失敗しました。
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  完了!
echo  インストーラ: Output\GeminiTranscriberSetup-2.0.0.exe
echo ============================================================
pause
endlocal
