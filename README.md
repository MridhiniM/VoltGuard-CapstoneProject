# VoltGuard

Electrical monitoring for one ESP32 + PZEM-004T + DS18B20 node. The existing dashboard is unchanged: the Simulation / Live switch, all pages, and the temperature limit of 15–30 °C that shows *Warning* / *Fault detected* when the load goes above 30 °C. This release **adds**:

| Addition | What it is | Where |
|---|---|---|
| **Forecast (ML)** page, plus two panels at the bottom of **Overview** | *Today:* hour-by-hour power-outage risk from weather hazards, and when the load is predicted to exceed 30 °C. *Next 7 days:* forecast ambient, predicted load peak, and outage risk per day | `voltguard/src/forecast.tsx` |
| Backend rule result | Every reading also carries `rule_based` (`temperature > 30 °C`, exactly 30 is not an alert) | `voltguard-edge/ml/rules.py` |
| Backend ML result | Every reading also carries `ml`: a trained anomaly detector and a 30-min load-temperature nowcast | `voltguard-edge/ml/ml_runtime.py` |
| Outage forecast | LSTM trained on real Bengaluru outages (ESMI 2014–2018) and weather; rule-based fallback | `voltguard-edge/ml/outage_runtime.py`, `train_outage_lstm.py` |

The rule and the ML results are computed independently. Neither reads the other's output.

> Demonstration / advisory system. It is not certified protective equipment. See [Limitations](#limitations).

## Architecture

```
ESP32 ─MQTT voltguard/telemetry─► Node-RED (demo-flow.json) ◄─poll /api/demo/raw── Flask simulator
                                        │                                   (replays the recorded PZEM log)
                                        ▼
                          Flask gateway POST /api/edge/predict
                           ├─ rules.temperature_alert()        → rule_based
                           ├─ MLEngine.analyse()               → ml.anomaly, ml.temperature_forecast
                           │      ▲ Open-Meteo forecast (cached) │
                           ├─ legacy RandomForest              → prediction   (unchanged, synthetic-data model)
                           └─ safety limits                    → hard_rule_fault
                                        ▼
            Node-RED GET /api/readings/latest ──► React dashboard (Vite, :5173)
            Flask   GET /api/predict/weekly-outlook ─┘   (7-day outlook, demo controls)

Offline: PZEM logs (.xlsx) + Open-Meteo history ─► prepare_data.py ─► train_ml.py ─► models/ + reports/
```

Ports: MQTT 1883, Node-RED 1880, Flask gateway 5001, dashboard 5173.

## Quick start (Windows, no hardware needed)

Double-click `START-NODERED-DEMO.bat`. The first run installs dependencies. It starts the broker, the gateway, Node-RED and the dashboard, then opens http://localhost:5173.

Manual start, from the repository root:

```powershell
cd voltguard-edge\ml
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.venv\Scripts\python.exe prepare_data.py          # builds data/processed/ (needed for the simulator replay)
.venv\Scripts\python.exe gateway.py               # http://127.0.0.1:5001/health

cd ..\..\node-red-runtime ; npm install
copy ..\voltguard-edge\nodered\demo-flow.json userdata\flows.json
node node_modules\node-red\red.js --userDir userdata --port 1880

cd ..\voltguard ; npm install ; npm run dev       # http://localhost:5173
```

`START-NODERED-DEMO.bat` also starts `node-red-runtime/broker.js` (MQTT on 1883). If Mosquitto is already running on 1883, that broker exits and Mosquitto is used instead.

## Using the dashboard

- **Simulation / Live:** use the header button as before. For Live, open **System settings** and set *Node-RED base URL* to `http://localhost:1880`, *Latest-readings path* to `/api/readings/latest`, and *HTTP polling*.
- **Temperature fault:** this is unchanged. Above 30 °C the temperature channel shows *Warning*, and above about 32 °C *Fault detected*. The range comes from *Operating limits* in System settings.
- **Forecasts:** use the **Forecast (ML)** page, or scroll to the bottom of **Overview**. These panels call the gateway at `VITE_GATEWAY_URL` (default `http://127.0.0.1:5001`) and refresh every 10 minutes.

To test the 30 °C behaviour in Live mode without hardware, set the gateway simulator's temperature. It flows through Node-RED like a real reading:

```powershell
$g='http://127.0.0.1:5001'
Invoke-RestMethod -Method Post "$g/api/demo/temperature" -ContentType 'application/json' -Body '{"temperature":32}'   # 28 / 30 / 32 ...
Invoke-RestMethod -Method Post "$g/api/demo/temperature" -ContentType 'application/json' -Body '{"temperature":null}' # back to recorded values
```

If the weather API or a model is unavailable, the forecast panels say *Unavailable* and show no values. The rest of the dashboard is unaffected.

## Power-outage forecast (LSTM)

**Question answered:** how likely is it that a power outage starts in each upcoming hour, for a typical household in Bengaluru, given the weather? The dashboard turns this into windows such as "15:00–17:00 due to heavy rain (9.1 mm/h)".

**Data (all real):**
- **ESMI:** the Electricity Supply Monitoring Initiative by Prayas Energy Group, CC0, [Harvard Dataverse doi:10.7910/DVN/CLLZZM](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CLLZZM).
  - Minute-wise supply voltage from **13 Bengaluru household sites, Nov 2014 – Dec 2018**.
  - An outage is ≥ 5 consecutive minutes at 0 V after supply was present. Zero runs over 12 h are excluded as monitor faults.
  - Result: **9,390 outages over 240,550 site-hours**. A median outage lasts 25 min, and an outage starts in 3.7% of household-hours.
- **Open-Meteo Historical Weather API:** hourly Bengaluru temperature, humidity, precipitation, weather code (thunderstorm), wind, gusts, cloud cover and solar radiation for the same period.

**Model:** a PyTorch LSTM (32 units) over the previous 24 hours of weather plus hour-of-day, season and Sunday. Its output is calibrated with Platt scaling on the validation data.
- **Split (chronological):** train before Jul 2017, validate Jul–Dec 2017, test on **2018** (never used for fitting or tuning).
- **Live input:** the Open-Meteo forecast, with `past_days=1` so the model has its 24-hour history.

**Results on the 2018 test year** (full table in [`reports/OUTAGE_EVALUATION.md`](voltguard-edge/ml/reports/OUTAGE_EVALUATION.md)):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Usual rate for hour × month (baseline) | 0.651 | 0.051 |
| Logistic regression, current hour | 0.658 | 0.057 |
| LSTM without weather | 0.661 | 0.057 |
| **LSTM with weather (deployed)** | **0.668** | **0.060** |

The base rate is 3.5%, so a random model would score PR-AUC ≈ 0.035.

**How the risk band holds up in 2018:**
- **Elevated hours** (probability ≥ 7.2%, threshold set on validation data) had **7.2% outage starts, vs 3.2% for other hours**.
- **Weather-driven hours** are hours where the weather model is ≥ 1.25× the no-weather model. They had 5.0% outage starts, against 3.7% expected without weather.

In the raw data, outages start in 3.4% of dry hours and **12.2% of heavy-rain hours**.

**Explanations:** the model is re-run with rain/thunder removed, with temperature set to its typical value, and with wind set to its typical value. A factor is named only if removing it lowers the probability by ≥ 20%; otherwise the window is labelled "the usual outage pattern at this time of day". The panel also shows the **last recorded outage in Bengaluru** from ESMI (historical; the dataset ends in 2018).

**What it can and cannot do:**
- **Weak at exact hours:** it ranks hours better than chance (ROC-AUC 0.67), and flagged hours really do have about 2× more outages. But most outages are not predictable from weather.
- **Planned shutdowns are included:** ESMI cannot separate them from faults.
- **Whole-city weather:** one weather grid cell covers the city, and the forecast has its own error.
- **Not your feeder:** it is not specific to your feeder or to the VoltGuard load.

**Why "accuracy" is not the headline metric:** outages start in only 3.5% of home-hours, so always predicting "no outage" already scores 96.5%. The model also scores 96.5%. The dashboard's *Outage model performance* card reports this openly. Its headline numbers are the ones that measure real skill on 2018:
- the 6% of hours it flags had **2.2× more outages**;
- heavy-rain hours had 3.6× more outages;
- ROC-AUC is 0.668, vs 0.651 for the usual-rate baseline.

Reformulations were also tested (3-hour city blocks, whole days). None beat simply guessing the majority answer on accuracy, so none is presented as "90%+".

**Your site's own outage history:** `LiveOutageLog` records every real supply loss seen by the ESP32 to `data/live/site_outages.csv` (git-ignored). Simulated readings are never logged. A supply loss is voltage < 100 V (`VOLTGUARD_OUTAGE_V`); a telemetry gap > 60 s is logged as `no_telemetry`. The panel shows the last one, and the log is the data needed for a site-specific model later.

Open a page directly with `?page=forecast` (or `overview`, `live`, ...).

**Fallback:** if `models/outage_lstm.pt` is missing, `outage_risk.py` (fixed weather-hazard rules) is used, and the panel says so.

**Reproduce:**
```powershell
cd voltguard-edge\ml
# ESMI download needs a Dataverse guestbook: put your answers in data\raw\esmi\gb.json (git-ignored) or set ESMI_NAME/EMAIL/INSTITUTION/POSITION/PURPOSE
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe download_esmi.py          # ~1.7 GB streamed, Bengaluru rows kept (~30 min)
.venv\Scripts\python.exe prepare_outage.py         # labels + weather join -> data\processed\outage_hourly.csv
.venv\Scripts\python.exe train_outage_lstm.py      # -> models\outage_lstm.pt, reports\OUTAGE_EVALUATION.md (~5 min)
```
The Bengaluru weather archive (Open-Meteo, 2014-11-01 to 2018-12-31, hourly) is stored in `data/raw/weather/`.

## API (Flask gateway, 127.0.0.1:5001)

| Method + path | Purpose |
|---|---|
| `GET /health` | Model load status/versions, rule threshold, weather state |
| `GET /api/readings/latest` | Gateway's own latest simulated reading (the dashboard normally uses Node-RED's) |
| `POST /api/edge/predict` | Node-RED → rule + ML + legacy RF + safety limits for one reading |
| `POST /api/detect/rule-based` | `{"temperature": 32}` → rule result |
| `POST /api/predict/anomaly` | `{"reading": {...}, "history": [...]}` → anomaly result (stateless) |
| `POST /api/predict/temperature` | `{"history": [... ≥10 min, timestamped ...]}` → 30-min nowcast (422 if too short) |
| `GET /api/weather/forecast` | Cached Open-Meteo 7-day hourly forecast (`?refresh=1` forces a fetch; 503 if unavailable) |
| `GET /api/predict/weekly-outlook` | 7-day outlook: ambient, predicted load peak, outage risk per day (503 if no forecast) |
| `GET /api/predict/today` | Rest of today hour by hour: LSTM outage probability/windows with reasons, and hours with load predicted > 30 °C |
| `GET /api/outage/last` | Most recent real outage in the ESMI Bengaluru data (historical), plus `this_site`: the last supply loss logged from your ESP32 |
| `GET /api/outage/model-card` | Validated outage-model numbers shown on the dashboard (2018 test year) |
| `POST /api/demo/temperature`, `POST /api/demo/scenario` | Simulator controls (simulated data only) |

Node-RED keeps serving `GET :1880/api/readings/latest` and `GET/POST :1880/api/input/mode` (`simulation` | `hardware`).

The reading payload adds:
```json
"rule_based": {"name":"Rule-Based Demo — Temperature Threshold Alert","rule":"temperature > 30 °C","threshold_c":30,"active":true,"state":"ok", ...},
"ml": {"anomaly": {"state":"ok","algorithm":"robust_covariance","score":3.1,"threshold":118.0,"score_ratio":0.03,"is_anomaly":false,
                   "score_meaning":"unusualness relative to recorded normal operation; not a probability", ...},
       "temperature_forecast": {"state":"ok","horizon_min":30,"predicted_c":31.2,"persistence_c":31.0,"uses_weather":true,"test_mae_c":0.593, ...},
       "insight": null}
```
`state` can be `ok`, `warming_up`, `insufficient_data`, `unavailable`, `invalid_input` or `inference_error`. No value is filled in when a state is not `ok`.

## Data

| | Electrical (primary) | Weather |
|---|---|---|
| Name | VoltGuard PZEM-004T + DS18B20 logs: `pzem_temperature_measurements_new_YESENEW.xlsx`, `pzem_serial_diagnostics.xlsx` (repo root) | Open-Meteo Historical Weather API (reanalysis) |
| Source | Recorded by this project's ESP32 with `pzem_logger_serial.py` | https://archive-api.open-meteo.com/v1/archive |
| Licence | Project's own data | Open-Meteo data CC BY 4.0 (attribution: Open-Meteo.com) |
| Records / span | 25,307 rows (25,306 after cleaning), 2026-10-19 21:58 → 2026-10-20 13:32 IST | 48 hourly rows, Bengaluru (12.9716, 77.5946), Asia/Kolkata |
| Sampling | ~2 s; one 25.5-min gap between files | hourly |
| Features | V, I, P, PF, Hz, equipment temperature | temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation, shortwave_radiation |
| Fault labels | **None**: normal operation of one ~8 W load only | n/a |
| Quality | 1 load-off row removed; V 230–256 V, I 0.025–0.036 A, T 27.9–34.3 °C | City-scale, not at the equipment |

**Public datasets considered and rejected.** None matches a single ~8 W AC load with an equipment temperature sensor and real fault labels:
- [Crompton Greaves three-phase pump](https://ieee-dataport.org/documents/crompton-greeves-three-phase-pump-predictive-maintenance-dataset): synthetic, three-phase motor.
- [Brushed DC motor faults (Zenodo 4314249)](https://zenodo.org/records/4314249): DC motor, no power channel.
- [CREAM coffeemakers](https://mediatum.ub.tum.de/1554766): 103 GB, high-rate waveforms.
- [UCI household power consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption): whole-house aggregate in France, no equipment temperature or faults.
- [NASA PCoE IGBT ageing](https://data.nasa.gov/dataset/insulated-gate-bipolar-transistor-igbt-accelerated-aging): power semiconductor stress tests.

Combining any of these with the VoltGuard load would not be technically defensible. The repository's `voltguard-edge/ml/dataset.csv` is synthetic and is used only by the legacy Random Forest.

## Training and evaluation

```powershell
cd voltguard-edge\ml
.venv\Scripts\python.exe prepare_data.py     # validate + clean logs, download/align weather → data/processed/, data/raw/weather/
.venv\Scripts\python.exe train_ml.py         # train + evaluate → models/*.joblib, reports/EVALUATION.md, reports/evaluation.json
```
`prepare_data.py --offline` reuses the stored weather download. Training takes about 3 minutes. Models are loaded at gateway start and are never retrained by the server.

The full results are in [`voltguard-edge/ml/reports/EVALUATION.md`](voltguard-edge/ml/reports/EVALUATION.md). Summary:

- **A. Anomaly detector.** Isolation Forest and robust covariance (Mahalanobis) are both trained. Robust covariance was selected on validation-only injections.
  - **Split:** interleaved 30-min blocks (5:1:1) with a purge between blocks.
  - **Threshold:** the 99.5th percentile of validation scores.
  - **False alarms on normal data:** 0.5% validation, 1.1% test, 0.7% forward-in-time test.
  - **Synthetic injections:** sag, overcurrent and unstable load are detected in 100% of blocks, thermal runaway in 95%, swell in 100%. These are a sanity check only.
  - **Baseline:** a simple z>4 rule performs comparably on these large artificial faults. **No real faults exist to measure precision/recall.**
- **B1. 30-min temperature nowcast.**
  - **Selected models:** gradient boosting with weather (test MAE 0.593 °C), electrical-only random forest (0.611 °C).
  - **Persistence baseline:** 0.630 °C.
  - The gain is small, and the weather vs electrical-only difference is not meaningful.
- **B2. Equipment temperature vs 3-h ambient (outlook).**
  - **Model:** linear, test MAE 0.44 °C vs 2.84 °C for the mean baseline.
  - **Caveat:** it did *not* beat the baseline on validation, so treat it as low-confidence.
- **C. Supervised fault prediction: not trained.** There are no confirmed, timestamped fault events. It would need weeks or months of logged real faults (type + time) from this equipment.

## Seven-day outlook: how the weather is used

1. **Forecast fetch:** the gateway fetches the Open-Meteo forecast (`api.open-meteo.com/v1/forecast`, 7 days, hourly, configured timezone). It is cached 30 min in memory and in `data/cache/forecast.json`. After 6 h without a successful fetch the outlook becomes *unavailable*, and invented values are never shown.
2. **Ambient feature:** the 3-hour trailing mean of forecast ambient temperature is fed to model B2.
3. **Future load assumption:** future load is **not forecast**. The outlook assumes the median recorded load (8.5 W) runs continuously, and says so.
4. **Daily output:** each day shows forecast ambient max/min and the predicted equipment peak.
   - **elevated:** the predicted peak exceeds the 99th percentile of recorded equipment temperatures.
   - **low conf.:** the forecast ambient is outside the training range.
   - **unavailable:** the day's forecast is incomplete.

   There are no fault probabilities. Electrical anomalies cannot be forecast from weather.

The live nowcast also uses the forecast for ambient now and in 30 min. If the forecast is unavailable, it falls back to the electrical-only model.

## Configuration

Environment variables (see `voltguard-edge/ml/.env.example`):
- `VOLTGUARD_LOCATION`, `VOLTGUARD_LAT`, `VOLTGUARD_LON`, `VOLTGUARD_TZ`: default Bengaluru / Asia/Kolkata.
- `VOLTGUARD_TEMP_THRESHOLD_C`: default **30**.
- `VOLTGUARD_GATEWAY_HOST`, `VOLTGUARD_GATEWAY_PORT`.
- `VOLTGUARD_SIM_TICK_S`: default 2.

Dashboard (see `voltguard/.env.example`):
- `VITE_NODERED_URL`: default `http://localhost:1880`.
- `VITE_GATEWAY_URL`: default `http://127.0.0.1:5001` (used by Homes and the forecast panels). Use 127.0.0.1, because `localhost` can stall on IPv6 for Flask on Windows.
- Your `voltguard/.env` is unchanged (`VITE_NODERED_URL=http://localhost:5050`). Node-RED in this repo runs on 1880, so set that in System settings or `.env`.

No API keys are needed.

## Hardware: ESP32 → MQTT → Node-RED

`VoltGuard_ESP32_Sensor_MQTT.ino` publishes JSON (`voltage, current, power, powerFactor, frequency, …, temperature`) to `voltguard/telemetry` every 2 s.

Switch Node-RED to hardware:
```powershell
Invoke-RestMethod -Method Post http://localhost:1880/api/input/mode -ContentType 'application/json' -Body '{"mode":"hardware"}'
```
- **Accepted readings:** readings with numeric V/I/temperature are forwarded to `/api/edge/predict` and shown as **LIVE HARDWARE**. A `temperature: null` (DS18B20 failure), out-of-range values, or `sensor_ok:false` are rejected.
- **Stale data:** stale data produces HTTP 503 and the dashboard shows *No data*. It never falls back to simulation.
- **Fix in this change:** the hardware validation previously required a `sensor_ok:true` field that the deployed sketch never sends, so every real reading was dropped. It is now accepted when the field is absent.
- **Broker address:** the sketch's broker IP and Wi-Fi settings are hard-coded in the `.ino`; set them for your network.

The separate authenticated ingest path (`service.py` + `ingest.py` + `nodered/ingest-flow.json`, LSTM autoencoder) is unchanged. Its LSTM needs TensorFlow 2.16, which does not support Python 3.13.

## Tests

```powershell
cd voltguard-edge\ml ; .venv\Scripts\python.exe -m pytest -q     # 77 passed, 1 skipped (old TensorFlow LSTM test)
cd voltguard ; npm run build                                       # typecheck + production build
```
`test_ml.py` covers:
- **Rule boundaries and sequences:** 28 / 30 / 30.1 / 32 / 28.
- **ML independence:** ML never calls the rule, and there is no discontinuity at 30 °C.
- **Input handling:** invalid inputs and missing model artifacts.
- **Weather and outlook:** parsing, location/timezone, failure/cache/expiry, outlook values recomputed from the model, partial days, outage-risk levels and windows, today endpoint.
- **Gateway endpoints**, including separation of the `rule_based` and `ml` fields and simulated vs hardware labelling.

The frontend changes are limited to `forecast.tsx` plus 8 lines in `App.tsx` / `pages.tsx`; `npm run build` passes.

The Node-RED path (the demo sequence, plus hardware MQTT messages in the real sketch's format) was verified manually against the running stack.

## Limitations

- **Data:** one ~8 W resistive load, one site, ~15.5 hours (one partial diurnal cycle). Other loads, seasons or sites need retraining.
- **Anomaly detection:**
  - It is checked only on normal data and artificial injections. No real faults exist to measure it against.
  - It flags *unusual*, not *faulty*. Mains voltage outside 230–256 V (seen in the recording) will be flagged.
- **Outlook:**
  - It rests on one day of ambient/equipment co-variation. The typical forecast nights (~17 °C) are below the training range and are marked low-confidence.
  - Future load is assumed, not forecast.
- **Not implemented:** supervised fault prediction (no labelled faults exist).
- **Simulator:** the gateway simulator (Live mode via Node-RED) now replays your recorded PZEM log at 2 s cadence instead of the old synthetic 42 °C generator. Scenario injections are artificial. The browser "Switch to simulation" mode is unchanged.
- **Outage forecast:** modest skill (ROC-AUC 0.67 on 2018). It flags hours with about 2× more outages but cannot pinpoint individual outages.
- **Separate thresholds:**
  - The 30 °C demo rule and the safety limits (V < 200, V > 253, I > 5 A, T > 60 °C) are separate, and neither controls a physical relay in this package.
  - `I_MAX` in the firmware differs from the simulator's limit.
- **Repository hygiene:** `node_modules/` directories are committed to git, which caused the incomplete-install failure. Consider `git rm -r --cached voltguard/node_modules node-red-runtime/node_modules`.
- **Credentials:** `VoltGuard_ESP32_Sensor_MQTT.ino` contains Wi-Fi credentials.

Older notes: `README-FINAL-SETUP.md`, `README-NODERED-DEMO.md`, `README-INTEGRATED.md`, `voltguard-edge/README.md`, `voltguard-edge/INGEST.md`.
