@echo off
REM Launch Claude Agent Server in background (no console window).
REM Prefers the official py launcher (pyw -3) to avoid PATH ambiguity when
REM multiple Python installs exist; falls back to pythonw.exe in PATH.
where pyw >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    start "" /B pyw -3 "%~dp0server.py" %*
) else (
    start "" /B pythonw.exe "%~dp0server.py" %*
)
