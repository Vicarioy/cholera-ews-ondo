@echo off
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python was not found. Install Python 3.13 from python.org,
  echo select "Add Python to PATH", then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the isolated Python environment...
  python -m venv .venv
  if errorlevel 1 goto :failed
)

echo Installing the required libraries...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo SETUP COMPLETE.
echo Next, double-click run_dashboard.bat.
pause
exit /b 0

:failed
echo.
echo SETUP FAILED. Read the error above, check your internet connection,
echo and run setup_windows.bat again.
pause
exit /b 1
