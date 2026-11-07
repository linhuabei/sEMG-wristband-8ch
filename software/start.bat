@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"
title Node Yushi 8CH sEMG Launcher

rem ===== Node Yushi host launcher (ASCII only + CRLF) =====
rem Entry file names live inside run.py, never inside this .bat,
rem so the console code page can never break this script.

set "LAUNCH=%~dp0run.py"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "VENV_PYW=%~dp0.venv\Scripts\pythonw.exe"
set "PY="
set "PYW="

if /i "%~1"=="check" goto :check
if /i "%~1"=="debug" goto :debug

rem ---------------- normal launch (double-click) ----------------
if exist "%VENV_PYW%" (
    set "PYW=%VENV_PYW%"
) else (
    where pythonw >nul 2>&1 && set "PYW=pythonw"
)
if not defined PYW (
    echo.
    echo [ERROR] pythonw.exe not found. Install Python or create .venv first.
    echo.
    pause
    exit /b 1
)
if not exist "%LAUNCH%" (
    echo [ERROR] run.py not found. Please run this from the app root folder.
    pause
    exit /b 1
)
start "" "%PYW%" "%LAUNCH%"
exit /b 0

:debug
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
) else (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo.
    echo [ERROR] python.exe not found.
    echo.
    pause
    exit /b 1
)
echo Launching in debug mode...
"%PY%" -X utf8 "%LAUNCH%"
echo.
echo Program exited. Press any key to close this window...
pause >nul
exit /b 0

:check
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
) else (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo.
    echo [ERROR] python.exe not found.
    echo.
    pause
    exit /b 1
)
"%PY%" -X utf8 "%LAUNCH%" --check
set "EC=!errorlevel!"
echo.
pause
exit /b %EC%