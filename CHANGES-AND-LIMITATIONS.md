# VoltGuard demo corrections

- Simulator normal current changed from ~1.7 A to ~0.04 A, matching the low-current training dataset. Normal temperature adjusted near the dataset median (~42.3 C).
- Frontend now treats ML-only anomalies as advisory warnings, not confirmed critical faults. Safety threshold breaches remain independent.
- Overcurrent remains a threshold-rule scenario, **not** an ML class. The Random Forest has five labels: normal, driver_instability, supply_sag, supply_swell, thermal_anomaly.
- The original model is retained unchanged. No new accuracy claims are made.
- LIVE in the dashboard means connected to the Flask gateway; the gateway itself is still SIMULATED. No real Node-RED runtime or ESP32 connection is required by START-DEMO.bat.
- A fault trip is only simulated. This project is not a certified electrical safety controller.

## Start
Extract to a new folder. Close previous VoltGuard terminals (especially the gateway on port 5001) before running START-DEMO.bat. If the browser is stale, hard refresh Ctrl+Shift+R.
