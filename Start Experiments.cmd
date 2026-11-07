@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" experiment_gui.py
) else (
  python experiment_gui.py
)
if errorlevel 1 pause
