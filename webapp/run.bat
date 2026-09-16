@echo off
REM Launch VisionPass using the project virtual environment.
cd /d "%~dp0"
"..\.venv\Scripts\python.exe" app.py
pause
