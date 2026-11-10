@echo off
rem ============================================================
rem  CallQA - install missing extras (ffmpeg, GPU, diarization, keys)
rem  Thin launcher; all logic lives in install_extras.py
rem  ASCII only: cmd.exe mis-parses .bat files with non-ASCII text.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   [ERROR] Run install.bat first.
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python" "%~dp0install_extras.py" %*
echo.
pause
