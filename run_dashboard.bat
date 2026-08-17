@echo off
setlocal
cd /d "%~dp0"

set "LOG_FILE=%~dp0dashboard_startup.log"

if not exist ".venv\Scripts\python.exe" (
  echo [%date% %time%] ERROR: .venv\Scripts\python.exe was not found.> "%LOG_FILE%"
  echo ERROR: The local environment is missing.
  echo Run setup_windows.bat from this project folder first.
  echo.
  echo Project folder: %CD%
  echo Details were saved to dashboard_startup.log.
  pause
  exit /b 1
)

echo Starting the Cholera Early Warning System...
echo Leave this window open while using the dashboard.
echo Open http://localhost:8501 if the browser does not open automatically.
echo [%date% %time%] Starting dashboard from %CD%> "%LOG_FILE%"

start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:8501'"
".venv\Scripts\python.exe" -m streamlit run "dashboard\app.py" --server.address localhost --server.port 8501 --server.headless true >> "%LOG_FILE%" 2>&1

set "DASHBOARD_EXIT=%ERRORLEVEL%"
echo.
if not "%DASHBOARD_EXIT%"=="0" (
  echo ERROR: The dashboard stopped with exit code %DASHBOARD_EXIT%.
  echo Read dashboard_startup.log in this folder for the full error.
) else (
  echo The dashboard has stopped normally.
)
pause
exit /b %DASHBOARD_EXIT%
