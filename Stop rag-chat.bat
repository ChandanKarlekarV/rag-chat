@echo off
rem Double-click to stop rag-chat completely (frees CPU, RAM and GPU memory).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\stop.ps1"
timeout /t 5 >nul
