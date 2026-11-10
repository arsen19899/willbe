@echo off
rem ============================================================
rem  CallQA installer - thin launcher.
rem  All logic lives in install.py so it can be tested properly.
rem  ASCII only: cmd.exe mis-parses .bat files with non-ASCII text.
rem ============================================================
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PROBE=%TEMP%\callqa_boot.txt"
set "BOOT="

rem  Any Python 3.8+ can run the installer - including 3.14.
rem  Verified by output, not by exit code.
for %%C in (python py python3) do call :probe %%C

if not defined BOOT (
  echo.
  echo   [ERROR] Python not found on this computer.
  echo.
  echo   Install Python 3.12 and run install.bat again:
  echo     https://www.python.org/downloads/release/python-31210/
  echo     pick "Windows installer (64-bit)"
  echo     tick "Add python.exe to PATH" during setup
  echo.
  pause
  exit /b 1
)

!BOOT! "%~dp0install.py" %*
set "RC=!ERRORLEVEL!"
echo.
pause
exit /b %RC%

:probe
if defined BOOT goto :eof
if exist "%PROBE%" del "%PROBE%" >nul 2>&1
%1 -c "print('BOOTOK')" > "%PROBE%" 2>nul
if not exist "%PROBE%" goto :eof
set "OUT="
set /p OUT=<"%PROBE%"
if "!OUT!"=="BOOTOK" set "BOOT=%1"
goto :eof
