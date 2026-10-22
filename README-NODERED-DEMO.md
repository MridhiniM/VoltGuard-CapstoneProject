# VoltGuard: Node-RED in the real demo pipeline

## Start on Windows

1. Close old VoltGuard Flask, dashboard, and Node-RED processes (ports 5001, 5173, 1880).
2. Extract the ZIP to a new folder. Double-click **START-NODERED-DEMO.bat**. First run requires internet to install packages.
3. Confirm http://localhost:5001/health responds and http://localhost:1880/api/readings/latest returns a processed JSON object (allow several seconds).
4. Open http://localhost:1880 to inspect the Node-RED flow.
5. In dashboard at http://localhost:5173 open System settings: Data source **Live**, transport **HTTP polling**, base URL **http://localhost:1880**, latest path **/api/readings/latest**, polling **2000 ms**. Clear optional prediction endpoint to avoid a second model call. Use **Test connection**.
6. Trigger faults using the same Flask POST endpoint as before, e.g. `Invoke-RestMethod -Method Post -Uri http://localhost:5001/api/demo/scenario -ContentType 'application/json' -Body '{"scenario":"overcurrent","duration_seconds":25}'`.

## Data flow

Flask raw simulator `/api/demo/raw` → Node-RED polling → Flask `/api/edge/predict` (Random Forest + independent threshold) → Node-RED `/api/readings/latest` → React dashboard.

The simulator is still inside Flask for convenience, but **the dashboard only receives the Node-RED processed result** when configured as above. The header may say LIVE, which means a live HTTP connection, **not live physical sensors**. The returned JSON has `source: SIMULATION VIA NODE-RED` and `sim: true`.

The existing ESP32 MQTT Node-RED flow is preserved separately in `voltguard-edge/nodered/flow.json`; it is NOT used in this pre-hardware demo. ESP32 MQTT ingestion and actual relay control still require hardware integration and safety validation. No physical relay is controlled here.

If the Node-RED endpoint responds with `waiting for first processed reading`, check the Node-RED console and wait a few seconds. If port 1880 is occupied, stop the old Node-RED process. The launcher's `flows.json` is copied on every run; changes made in the Node-RED editor may be overwritten. This is a demo, not an electrical protection system.
