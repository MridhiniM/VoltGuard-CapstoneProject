# VoltGuard Integrated Demo (ESP32 not required yet)

This folder combines the existing React dashboard (`voltguard/`) and edge ML package (`voltguard-edge/`). `START-DEMO.bat` starts a local telemetry simulator, the trained RandomForest model, a Flask gateway, and the dashboard. No Node-RED or MQTT broker is required for the pre-hardware demo.

## Start
1. Install Python 3.10–3.12 and Node.js 18+ on Windows. Keep internet available for first-time package installation.
2. Double-click `START-DEMO.bat`.
3. Dashboard: http://localhost:5173. Gateway health: http://localhost:5001/health.
4. Open **System settings** and confirm **Live (VoltGuard gateway / Node-RED)** with base URL `http://localhost:5001`, latest path `/api/readings/latest`, transport HTTP, polling 1000–2000 ms. Save if needed.
5. The dashboard receives simulator telemetry from the gateway and the ML model prediction. Readings are explicitly marked simulation; relay state is simulated only.

## Demo fault scenarios
Open PowerShell or a terminal and send one of these while the app is running:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:5001/api/demo/scenario -ContentType 'application/json' -Body '{"scenario":"overcurrent","duration_seconds":25}'
Invoke-RestMethod -Method Post -Uri http://localhost:5001/api/demo/scenario -ContentType 'application/json' -Body '{"scenario":"voltage_sag","duration_seconds":25}'
Invoke-RestMethod -Method Post -Uri http://localhost:5001/api/demo/scenario -ContentType 'application/json' -Body '{"scenario":"thermal_anomaly","duration_seconds":25}'
Invoke-RestMethod -Method Post -Uri http://localhost:5001/api/demo/scenario -ContentType 'application/json' -Body '{"scenario":"normal","duration_seconds":25}'
```

Available names: `normal`, `voltage_sag`, `voltage_swell`, `overcurrent`, `thermal_anomaly`, `driver_instability`. API docs: http://localhost:5001/api/demo/scenarios.

## What is working before hardware arrives
- Dashboard pages, charting, history and settings from the supplied frontend.
- Gateway-generated telemetry and API endpoint.
- Real inference using the included `model_rf.joblib`, with the supplied feature engineering.
- Scenario injection to demonstrate changing readings, ML labels, threshold alerts and simulated relay state.

## When ESP32 arrives
The current gateway is deliberately simulator-only. The existing ESP32 firmware and Node-RED flow are preserved in `voltguard-edge/firmware/` and `voltguard-edge/nodered/`. To switch to hardware, install/configure Mosquitto and Node-RED, import `nodered/flow.json`, configure the ESP32 firmware, then set the dashboard to the Node-RED URL (typically `http://localhost:1880`) and use its `/api/readings/latest` endpoint. Verify the imported flow's endpoints and MQTT broker settings before relying on it. This package does **not** automatically enable real relay control; test with a low-voltage isolated setup first.

## Important safety / accuracy notes
- This is a demonstration, not a certified electrical protection device. Do not connect the software directly to live 230 V wiring or depend on it to trip a real relay.
- The included dataset is synthetic and the model's thermal-fault class has limited event coverage, as noted in the original edge README. Displayed predictions are not proof of real-world accuracy.
- Do not claim 5G or sub-100-ms physical isolation from this laptop demo; it uses local HTTP polling and simulated relay state.
