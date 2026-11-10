@echo off
rem ASCII only: cmd.exe mis-parses .bat files containing non-ASCII text.
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo   [ERROR] Run install.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python" -m callqa.cli web %*
pause
