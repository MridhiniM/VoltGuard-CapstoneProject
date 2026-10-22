@echo off
setlocal
cd /d "%~dp0"
echo ========================================
echo VoltGuard - integrated demo setup
echo ========================================
where py >nul 2>nul
if errorlevel 1 (echo Python launcher not found. Install Python 3.10-3.12 and enable Add to PATH.& pause & exit /b 1)
where node >nul 2>nul
if errorlevel 1 (echo Node.js not found. Install Node.js 18 or newer, then rerun this file.& pause & exit /b 1)
if not exist "voltguard-edge\ml\.venv\Scripts\python.exe" (
  echo Creating Python environment...
  py -3 -m venv "voltguard-edge\ml\.venv"
  if errorlevel 1 (echo Could not create Python environment.& pause & exit /b 1)
)
echo Installing/checking Python packages...
"voltguard-edge\ml\.venv\Scripts\python.exe" -m pip install --upgrade pip
"voltguard-edge\ml\.venv\Scripts\python.exe" -m pip install -r "voltguard-edge\ml\requirements-demo.txt"
if errorlevel 1 (echo Python package installation failed.& pause & exit /b 1)
if not exist "voltguard\node_modules" (
 echo Installing frontend packages...
 cd /d "%~dp0voltguard"
 call npm install
 if errorlevel 1 (echo npm install failed.& pause & exit /b 1)
 cd /d "%~dp0"
)
echo Starting ML gateway...
start "VoltGuard ML Gateway" cmd /k "cd /d ""%~dp0voltguard-edge\ml"" && "".venv\Scripts\python.exe"" gateway.py"
timeout /t 3 /nobreak >nul
echo Starting dashboard...
start "VoltGuard Dashboard" cmd /k "cd /d ""%~dp0voltguard"" && npm run dev -- --host 127.0.0.1"
timeout /t 4 /nobreak >nul
start "" http://localhost:5173
echo.
echo Dashboard: http://localhost:5173
echo ML gateway health: http://localhost:5001/health
echo This is simulation mode. Do NOT connect it to mains or rely on it for protection.
echo Close the two terminal windows to stop the demo.
pause
