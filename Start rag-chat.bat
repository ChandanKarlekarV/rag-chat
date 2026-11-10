@echo off
rem Double-click to start rag-chat in the background and open it in your browser.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\start.ps1"
timeout /t 5 >nul
