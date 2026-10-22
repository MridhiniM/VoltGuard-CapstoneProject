# MQTT → Node-RED → backend ingestion

## What exists (inspected) and what does not
The repo has **no LSTM, forecasting, database, or auth** (`voltguard-edge/ml` holds a scikit-learn RandomForest only). This change adds the missing ingestion layer to the existing Flask service `ml/service.py`; it does not add an LSTM or forecasts. When the LSTM exists, plug it into `Ingest.decide()` in `ml/ingest.py` (training feature order, window and cadence must be checked first).

```
ESP32 ─MQTT voltguard/telemetry→ Mosquitto 192.168.29.46:1883 ─→ Node-RED (nodered/ingest-flow.json)
   parse+validate → POST /api/telemetry (X-API-Key, Idempotency-Key; 3 bounded retries, backoff 0.5/1/2 s)
        ↓
ml/service.py: validate → SQLite (readings, predictions, ingest_failures) → queue → RF worker thread
        ↓
GET /api/readings/latest  ←polled (2 s)─  voltguard dashboard      GET /api/status (per-stage counters)
```
Node-RED status: `GET :1880/voltguard/ingest-status` (`mqtt_received` = broker messages only; `delivered` = backend-confirmed).

## Request schema `POST /api/telemetry`
Headers: `X-API-Key: $VOLTGUARD_INGEST_TOKEN`, `Idempotency-Key: <8-200 chars>` (fixed per message, reused by retries). Body (all finite numbers): `voltage, current, power, powerFactor, energy, frequency, apparentPower, reactivePower, uptime`; `temperature` number **or null** (null is stored as NULL, never 0). Responses: 201 stored · 200 `duplicate:true` · 400 missing key · 401 bad token · 422 invalid (`details`) · 503 DB down / token unset. The server adds `received_at` (UTC) and `device_id`.

## Inference contract (RandomForest, 5-class)
Features/order/window from `features.py` (12-sample window; trained at 5 s spacing). Telemetry is ~2 s, so a sample enters the window only if ≥4 s after the last (window spans ~48 s, not 60 s: **approximation, documented, unvalidated**). Window resets after a >15 s gap or supply-off. `inference_state` ∈ `ok | insufficient_data | sensor_error | off | inference_error`; `prediction` is only set when `ok`. Comms/sensor/model problems never become fault labels. Model version = hash of `model_rf.joblib`. Classes are fault *detections*; there is no forecast or time-to-failure.

## LSTM anomaly detector (`ml/train_lstm.py`, artifacts in `ml/lstm_ae/`)
Trained on the two PZEM logs (25,307 rows, 2026-10-19 21:58 → 2026-10-20 13:32, ~2 s cadence). **Both files are normal-only, no fault labels**, so this is an LSTM autoencoder: it scores how unlike normal a 60 s window (30 samples) is. It does not classify the fault type, forecast, or estimate time-to-failure.
Features/order: voltage, current, power, powerFactor, frequency, temperature (z-scored with train mean and scale floors in `meta.json`). Windows never span a >5 s gap or a 30-min split block; split 70/15/15 by block. Threshold = p99.5 of validation error; `alert` needs 3 consecutive over-threshold windows. Output on `/api/readings/latest` → `anomaly: {state, score, threshold, is_anomaly, alert, model_version}`; states `ok | insufficient_data | sensor_error | off | unavailable | inference_error`. Missing TensorFlow/model only disables this field.
Retrain: `pip install -r requirements-lstm.txt && python train_lstm.py <files>` (~2 min). Results (`lstm_ae/metrics_lstm.json`): false alarms on clean held-out windows 1.3% single / 1.0% three-consecutive; forward test (train on first 80% in time, test last 20%, where mean voltage drifts 248→241 V) 1.4% / 1.2%. Synthetic perturbations (sag, swell, 20x current, temperature ramp/step, instability) are all detected, but they are large and artificial: **a sanity check, not validation on real faults**. Limits: one resistive ~8 W load, 27-34 °C, voltage 230-256 V; other loads or seasons will need retraining; the existing hardware rules (V>253 etc.) sit inside this data's normal range (up to 255.9 V).

## Configure / deploy (Debian)
```
cd voltguard-edge/ml && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-demo.txt
export VOLTGUARD_INGEST_TOKEN=$(openssl rand -hex 24)   # keep out of git
VOLTGUARD_PORT=5000 python service.py
```
Node-RED service env: `VOLTGUARD_BACKEND_URL=http://127.0.0.1:5000/api/telemetry`, `VOLTGUARD_INGEST_TOKEN` (same), optional `VOLTGUARD_DEVICE_ID`. Import `nodered/ingest-flow.json` (use it instead of the old flows that share topic `voltguard/telemetry`; it uses a different MQTT client id) and Deploy. Dashboard: `VITE_NODERED_URL=http://<server>:5000`. Use HTTPS (reverse proxy) if traffic leaves the LAN. Anonymous Mosquitto is a dev setup. `/api/readings/latest` and `/api/status` are unauthenticated read-only (no auth exists in the app).

## Tests
`cd ml && pytest test_ingest.py` → 29 passed (adds LSTM window/gap/streak logic and a real-model check) (validation, auth, idempotency, null vs 0, window/gap/decimation, sensor_error/off, model outage, inference exception, DB outage + recovery, queue full, stale latest). Node-RED function code was also run against the live Flask service over HTTP with mocks of the http/delay/catch nodes (valid, null temp, bad JSON, missing/wrong-type/out-of-range, 401 no-retry, duplicate key, backend down → 3 retries → give up). **Not tested:** a real Node-RED runtime, the real ESP32, frontend live update, DB failure on real disks.

## Troubleshooting (check in order)
1. **MQTT**: `mosquitto_sub -h 192.168.29.46 -t voltguard/telemetry -v`; Node-RED `mqtt_connected`/`mqtt_last_age_s` in ingest-status.
2. **Validation**: `invalid`/`last_error` in ingest-status, debug sidebar "ingest problems", table `ingest_failures`.
3. **Backend delivery**: `delivered`, `retries`, `delivery_failed`, `rejected_by_backend`; `curl -H "X-API-Key: …" -H "Idempotency-Key: test-key-0001" -d @sample.json :5000/api/telemetry`.
4. **Persistence**: `/api/status` → `database_ok`, `counters.stored/db_errors`; `sqlite3 ml/voltguard.db 'select count(*) from readings'`.
5. **Inference**: `/api/status` → `inference.model_loaded`, `counters.inference_errors`, `window`; `inference_state` on latest.
6. **Frontend**: `curl :5000/api/readings/latest` (503 = no/stale data, comms issue not a fault); dashboard base URL and CORS; null temperature makes the dashboard drop the reading (unchanged frontend behaviour).
