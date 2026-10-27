@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul || (echo Python launcher missing & pause & exit /b 1)
where npm >nul 2>nul || (echo Node.js / npm missing & pause & exit /b 1)
if not exist "voltguard-edge\ml\.venv\Scripts\python.exe" (
  py -3 -m venv "voltguard-edge\ml\.venv"
  if errorlevel 1 (pause & exit /b 1)
)
"voltguard-edge\ml\.venv\Scripts\python.exe" -m pip install -r "voltguard-edge\ml\requirements-demo.txt"
if errorlevel 1 (pause & exit /b 1)
if not exist "voltguard-edge\ml\data\processed\pzem_readings.csv" (
  echo Preparing the recorded PZEM dataset for the simulator replay...
  pushd voltguard-edge\ml
  ".venv\Scripts\python.exe" prepare_data.py
  if errorlevel 1 (popd & pause & exit /b 1)
  popd
)
if not exist "voltguard\node_modules" (
  pushd voltguard
  call npm install
  if errorlevel 1 (pause & exit /b 1)
  popd
)
if not exist "node-red-runtime\node_modules\node-red" (
  pushd node-red-runtime
  call npm install
  if errorlevel 1 (pause & exit /b 1)
  popd
)
echo Starting local MQTT broker on 1883...
start "VoltGuard MQTT Broker" cmd /k "cd /d ""%~dp0node-red-runtime"" && node broker.js"
timeout /t 2 /nobreak >nul
echo Starting Flask ML service on 5001...
start "VoltGuard Flask ML" cmd /k "cd /d ""%~dp0voltguard-edge\ml"" && "".venv\Scripts\python.exe"" gateway.py"
timeout /t 4 /nobreak >nul
echo Starting Node-RED on 1880 (dedicated user directory)...
if not exist "node-red-runtime\userdata" mkdir "node-red-runtime\userdata"
copy /Y "voltguard-edge\nodered\demo-flow.json" "node-red-runtime\userdata\flows.json" >nul
start "VoltGuard Node-RED" cmd /k "cd /d ""%~dp0node-red-runtime"" && node_modules\.bin\node-red.cmd --userDir ""%~dp0node-red-runtime\userdata"" --port 1880"
timeout /t 6 /nobreak >nul
echo Starting React dashboard on 5173...
start "VoltGuard React" cmd /k "cd /d ""%~dp0voltguard"" && npm run dev -- --host 127.0.0.1"
timeout /t 3 /nobreak >nul
start "" http://localhost:1880
start "" http://localhost:5173
echo.
echo Dashboard: http://localhost:5173  (Live: System settings, Node-RED URL http://localhost:1880; weather forecast on the Forecast (ML) page)
echo Readings: http://localhost:1880/api/readings/latest   Outlook: http://127.0.0.1:5001/api/predict/weekly-outlook
echo Default input is SIMULATION. Switch Node-RED input mode to hardware after flashing ESP32.
echo No physical relay is controlled. Do not use as mains protection.
pause
