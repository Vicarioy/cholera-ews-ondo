@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: The local environment is missing.
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)

echo Starting the Cholera Early Warning System...
echo Leave this window open while using the dashboard.
echo Open http://localhost:8501 if the browser does not open automatically.
".venv\Scripts\python.exe" -m streamlit run "dashboard\app.py"
pause
