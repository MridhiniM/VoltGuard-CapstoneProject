# VoltGuard hardware-ready integrated project

## What is and is not connected
Default pipeline: Flask simulated readings -> Node-RED -> Flask Random Forest -> Node-RED HTTP -> React. The **hardware pipeline is implemented but unverified** until your actual ESP32 and sensors are connected. The header LIVE means live HTTP polling, not verified physical hardware.

## Start on Windows
1. Install Python 3, Node.js and npm. Extract this archive. Close previous instances on ports 1883, 1880, 5001, 5173.
2. Double click `START-NODERED-DEMO.bat`. First run installs dependencies and requires internet. It launches MQTT broker 1883, Flask 5001, Node-RED 1880, React 5173.
3. In dashboard settings choose Live, HTTP polling, base URL `http://localhost:1880`, latest path `/api/readings/latest`, clear prediction endpoint.
4. Test `http://localhost:1880/api/readings/latest` and `http://localhost:1880/api/input/mode`. The default mode is simulation.

## Switching to actual ESP32 (after SAFE hardware validation)
1. Configure Wi-Fi SSID/password and MQTT_HOST (Windows laptop LAN IPv4) in `voltguard-edge/firmware/voltguard_esp32/voltguard_esp32.ino`. Flash ESP32 with Arduino IDE and listed libraries. ESP32 publishes JSON to `voltguard/telemetry` on TCP 1883. Allow broker access in Windows firewall on **trusted private network only**.
2. Test **only isolated low-voltage sensor/relay setups** with a qualified supervisor. The PZEM-004T mains connection and RCBO are hazardous; a certified electrician must design and inspect mains connections. Never treat this software as certified protective equipment. Firmware starts with relay OFF (startup lockout), and remote RESET is disabled; any physical energization requires a separate reviewed manual-reset design.
3. Check broker console for MQTT client `voltguard-esp32`, and Node-RED editor MQTT node status. Verify JSON includes numeric voltage/current/temperature and `sensor_ok:true`.
4. Switch to hardware mode via PowerShell:
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:1880/api/input/mode -ContentType 'application/json' -Body '{"mode":"hardware"}'
```
5. Open `http://localhost:1880/api/readings/latest`. It should show `source: ESP32 VIA NODE-RED`, `sim:false`, and `input_mode:hardware`. It returns HTTP 503 if no fresh valid hardware telemetry arrives within 12 seconds; it never silently substitutes simulator readings.
6. Return to simulator mode via the same command with `"mode":"simulation"`. This endpoint has no authentication: use only on a trusted local computer/network.

## Important limitations
- Actual ESP32, sensor pin mapping, Wi-Fi, MQTT, power supply, ML accuracy, and physical relay operation have NOT been validated.
- The relay is NOT remotely operated by Flask or Node-RED in this package; the firmware contains separate local trip logic. Never assume a reported relay state proves physical contact position.
- `I_MAX` in firmware is 0.50 A; the simulator demo's hard limit is 5 A. **These intentionally differ** and must be engineered to match the chosen load and hardware rating before a hardware demo.
- Hardware input is accepted only when `sensor_ok:true`; stale or invalid data makes the dashboard connection unavailable.
- Model labels do not include `overcurrent`; threshold faults and ML predictions are distinct.
- Local MQTT broker is unauthenticated, not suitable for public networks or production.
- Do not expose the demo HTTP endpoints outside your trusted LAN.
