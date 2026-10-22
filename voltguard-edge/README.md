# VoltGuard: edge + laptop ML

```
ESP32 --MQTT voltguard/telemetry--> Node-RED "Fault Detection" --> Laptop ML (Flask :5000)
  ^                                        |  Fuse + Decide
  +------MQTT voltguard/cmd (TRIP/RESET)---+--> /ws/readings + /api/readings/latest --> VoltGuard dashboard
```

| Layer | Runs | Does |
|---|---|---|
| ESP32 | `firmware/voltguard_esp32` | Reads PZEM (UART2) + DS18B20 (GPIO4) every 5 s. **Hard rules** (V>253, V<200, I>0.5 A, T>60 °C; 2 readings) always trip the relay (GPIO25) locally. **Edge ML** (depth-7 decision tree in `edge_model.h`) runs on every reading; if MQTT is down it trips after 6 consecutive faults (30 s). |
| Laptop | `ml/service.py` | RandomForest (300 trees) on the same 8 features, keeps its own 5-min window. |
| Node-RED | `nodered/flow.json` | Calls the service, fuses: both agree = 2 pts, laptop ML ≥ 0.6 alone = 1 pt, edge alone (service down) = 0.5 pt; **3 pts → TRIP**. Feeds the dashboard. Inject buttons for RESET / manual TRIP. |
| Dashboard | your VoltGuard app, unchanged | Settings → Live, WebSocket `ws://localhost:1880/ws/readings` (or HTTP `/api/readings/latest`). |

## Run (laptop)
1. Mosquitto: install and start with a listener on all interfaces (`listener 1883` + `allow_anonymous true` in `mosquitto.conf`), so the ESP32 can connect.
2. ML:
   ```
   cd ml
   python -m venv .venv && .venv\Scripts\activate      # Windows  (Linux/mac: source .venv/bin/activate)
   pip install -r requirements.txt
   python train.py        # retrains with YOUR sklearn version, regenerates firmware/.../edge_model.h
   python service.py      # http://localhost:5000/health
   ```
3. Node-RED: Menu → Import → `nodered/flow.json` → Deploy.
4. ESP32 (Arduino IDE, board "ESP32 Dev Module"): install libraries **PZEM004Tv30, OneWire, DallasTemperature, PubSubClient, ArduinoJson**. Set Wi-Fi, `MQTT_HOST` = laptop IP, check `RELAY_ACTIVE_LOW`, `I_MAX`. Upload. Serial monitor at 115200 shows each JSON reading.
5. Dashboard: `npm run dev` → Settings → Live → WebSocket → Test connection.

## Dataset and model
- **Training data:** `bulb_10000_plausible_anomalies` (10,000 rows at 5 s, the only labelled file). Classes: normal, supply_sag, supply_swell, driver_instability, thermal_anomaly.
- **Features** (identical in Python and C, verified to match on all 10,000 rows): V, I, P, T, voltage deviation from 1-min mean, 1-min std of power and of current, 5-min temperature rise.
- **Evaluation:** 5-fold grouped CV on ~21-min time blocks (no neighbouring-window leakage). See `ml/metrics.json`.

| Class (events in data) | Laptop RF F1 | Edge tree F1 |
|---|---|---|
| normal | 0.99 | 0.99 |
| supply_sag (3) | 0.96 | 0.94 |
| driver_instability (3) | 0.89 | 0.82 |
| supply_swell (2) | 0.36 | 0.83 |
| thermal_anomaly (1) | 0.00 | 0.00 |

- **Real bulb check** (`real_values_or.csv`, 278 energized rows, no faults): 97 % predicted normal. The scattered 3 % are why tripping needs several consecutive readings.
- **Limits, say these in the review:** The dataset is synthetic. Thermal has only one event, so it can't be validated when that event is held out (hence 0.00). Swell has two events. Overheating is therefore protected by the T > 60 °C hard rule, not by ML. The final models are trained on all events. Retrain when you have real recorded faults.

## Wiring notes from your diagram
- UART must cross: ESP32 **TX2 (GPIO17) → PZEM RX**, **RX2 (GPIO16) → PZEM TX**. In the drawing TX2 looks like it goes to PZEM Tx, so check this.
- PZEM is powered from 5 V, so its TX line is 5 V into a 3.3 V pin. Most builds survive this, but a 1 k series resistor (or level shifter) on GPIO16 is safer.
- PZEM measures supply voltage before the relay, so after a trip the voltage is still present but current is 0. The firmware and Node-RED skip ML while the relay is TRIPPED to avoid false labels.
