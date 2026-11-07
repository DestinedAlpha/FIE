@echo off
if exist "%~dp0.fie-update-in-progress" (
  echo An application update was interrupted. Re-run Update existing.cmd to recover before starting.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1"
if errorlevel 1 pause
