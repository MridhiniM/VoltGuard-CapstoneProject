"""Integrated VoltGuard demo gateway: simulator -> Node-RED -> rule check + trained ML -> dashboard API.

Two independent detection paths are computed for every reading and returned as separate fields:
  rule_based  rules.temperature_alert()  - Rule-Based Demo, temperature > 30 °C (configurable)
  ml          ml_runtime.MLEngine         - anomaly detector + 30-min temperature nowcast (trained models)
Legacy fields (prediction = synthetic-data Random Forest, hard_rule_fault = safety limits) are kept
unchanged for existing consumers. This is a DEMO, not a safety controller.

Simulator: replays the recorded real PZEM log (data/processed/pzem_readings.csv) in real time, labelled
SIMULATION. Demo controls can override the temperature or inject scenarios; nothing pretends to be hardware.
"""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import math, os, threading, time, random
import joblib, pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from features import add_features, TW
import rules, weather
from ml_runtime import MLEngine, validate_reading
from outage_runtime import LiveOutageLog, OutagePredictor, last_recorded_outage, model_card

ROOT = Path(__file__).resolve().parent
TICK_S = float(os.environ.get("VOLTGUARD_SIM_TICK_S", "2"))   # matches the ~2 s cadence of the real logger
bundle = joblib.load(ROOT / 'model_rf.joblib')
model, FEATS = bundle['model'], bundle['features']
engine = MLEngine(tz=weather.location()['timezone'])
outage_model = OutagePredictor()
site_outages = LiveOutageLog()       # real hardware readings only
forecast_cache = weather.ForecastCache()
window = deque(maxlen=TW + 1)
edge_window = deque(maxlen=TW + 1)
edge_lock = threading.Lock()
lock = threading.Lock()
latest = {"timestamp": datetime.now(timezone.utc).isoformat(), "voltage": 230.0, "current": 0.0,
          "temperature": None, "prediction": {"label": "starting", "confidence": 0.0}, "sim": True,
          "relay": "CLOSED", "source": "SIMULATION"}
scenario = {"name": "normal", "until": 0}
demo = {"temperature": None}          # manual temperature override for the Rule-Based Demo
app = Flask(__name__); CORS(app)
rng = random.Random(42)

REPLAY_FILE = ROOT / 'data' / 'processed' / 'pzem_readings.csv'
try:
    REPLAY = pd.read_csv(REPLAY_FILE)[['voltage', 'current', 'power', 'temperature']].to_dict('records')
except (OSError, KeyError, ValueError):
    REPLAY = None                       # falls back to the older synthetic generator


def current_weather():
    """Forecast for ML features; never blocks a reading for long and never invents values."""
    try:
        f = forecast_cache.get()
        return f if f.get('state') == 'ok' else None
    except Exception:
        return None


def legacy_rf(win, row):
    win.append(row)
    frame = add_features(pd.DataFrame(list(win)))
    probs = model.predict_proba(frame.iloc[[-1]])[0]
    classes = list(model.classes_)
    idx = int(probs.argmax())
    return {"label": str(classes[idx]), "confidence": round(float(probs[idx]), 3),
            "probs": {str(c): round(float(p), 3) for c, p in zip(classes, probs)},
            "model": "legacy RandomForest trained on synthetic dataset.csv"}


def detections(stream, row, ts):
    """The two independent paths. Neither reads the other's output."""
    return {"rule_based": rules.temperature_alert(row['temperature']),
            "ml": engine.analyse(stream, row, timestamp=ts, weather=current_weather())}


def sample_loop():
    global latest
    tick = 0
    while True:
        tick += 1
        now = time.time()
        with lock:
            name = scenario['name'] if now < scenario['until'] else 'normal'
            t_override = demo['temperature']
        if REPLAY:
            r = REPLAY[tick % len(REPLAY)]
            v, i, t = r['voltage'], r['current'], r['temperature']
            src_note = 'replay of recorded PZEM log'
        else:
            v = 230 + 1.4 * math.sin(tick / 9) + rng.uniform(-0.7, 0.7)
            i = max(0.035, 0.040 + 0.002 * math.sin(tick / 6) + rng.uniform(-0.001, 0.001))
            t = 30.0 + 0.3 * math.sin(tick / 30) + rng.uniform(-0.1, 0.1)
            src_note = 'synthetic generator'
        if name == 'voltage_sag': v = 175 + rng.uniform(-3, 3)
        elif name == 'voltage_swell': v = 270 + rng.uniform(-3, 3)
        elif name == 'overcurrent': i = 7.0 + rng.uniform(-0.4, 0.4)
        elif name == 'thermal_anomaly': t = 68 + rng.uniform(-1.5, 1.5)
        elif name == 'driver_instability': v += 14 * math.sin(tick * 2.7); i += 1.4 * abs(math.sin(tick * 2.1))
        if t_override is not None: t = t_override
        row = {"voltage": round(v, 2), "current": round(i, 4), "power": round(v * i, 2), "temperature": round(t, 2)}
        ts = datetime.now(timezone.utc).isoformat()
        pred = legacy_rf(window, dict(row))
        critical = row['voltage'] > 253 or row['voltage'] < 200 or row['current'] > 5 or row['temperature'] > 60
        reading = {"timestamp": ts, **row, "prediction": pred,
                   "relay": "TRIPPED (SIMULATED)" if critical else "CLOSED (SIMULATED)",
                   "source": "SIMULATION", "sim": True, "sim_detail": src_note, "scenario": name,
                   "temperature_override": t_override, "hard_rule_fault": critical,
                   **detections('gateway', row, ts)}
        with lock: latest = reading
        time.sleep(max(0.0, TICK_S - (time.time() - now)))


@app.get('/health')
def health():
    f = forecast_cache.get()
    return jsonify(ok=True, service='VoltGuard integrated demo gateway', mode='simulation',
                   simulator='replay of recorded PZEM log' if REPLAY else 'synthetic generator',
                   rule_based={"rule": rules.temperature_alert(0)['rule'], "threshold_c": rules.threshold_c()},
                   ml=engine.status() | {'outage_lstm': outage_model.status()}, legacy_model='RandomForest', classes=[str(c) for c in model.classes_],
                   weather={"state": f['state'], "location": f['location'], "fetched_at": f.get('fetched_at'), "error": f.get('error')})

@app.get('/api/demo/raw')
def demo_raw():
    # Unprocessed simulated telemetry, supplied to Node-RED as an ESP32 stand-in.
    with lock:
        r = dict(latest)
    return jsonify({k: r[k] for k in ('timestamp','voltage','current','power','temperature','scenario','temperature_override') if k in r} | {'source':'SIMULATED_ESP32','sim':True})

@app.post('/api/edge/predict')
def edge_predict():
    # Node-RED sends telemetry here. Rule check, ML and safety limits are applied to this incoming payload.
    data = request.get_json(silent=True) or {}
    try:
        v, i, t = (float(data[k]) for k in ('voltage','current','temperature'))
        if not all(math.isfinite(x) for x in (v,i,t)):
            raise ValueError('non-finite value')
    except (ValueError, KeyError, TypeError):
        return jsonify(error='voltage, current, temperature must be finite numbers'), 400
    try:
        p = float(data.get('power')); p = p if math.isfinite(p) else v * i
    except (TypeError, ValueError):
        p = v * i
    hw = data.get('source') == 'ESP32'
    row = {'voltage': v, 'current': i, 'temperature': t, 'power': p}
    ts = data.get('timestamp') or datetime.now(timezone.utc).isoformat()
    if hw:
        site_outages.on_reading(ts, v)
    with edge_lock:
        pred = legacy_rf(edge_window, dict(row))
    hard = v < 200 or v > 253 or i > 5 or t > 60
    return jsonify(timestamp=ts, voltage=v, current=i, temperature=t, power=round(p, 2), prediction=pred,
        hard_rule_fault=hard, relay=(data.get('hardware_relay','UNKNOWN') if hw else ('TRIPPED (SIMULATED)' if hard else 'CLOSED (SIMULATED)')),
        scenario=data.get('scenario','normal'), source='ESP32 VIA NODE-RED' if hw else 'SIMULATION VIA NODE-RED', sim=not hw,
        temperature_override=None if hw else data.get('temperature_override'),
        hardware_relay=data.get('hardware_relay') if hw else None, trip_reason=data.get('trip_reason') if hw else None,
        edge_label=data.get('edge_label') if hw else None,
        **detections('esp32' if hw else 'nodered-sim', row, ts))

@app.get('/api/readings/latest')
def get_latest():
    with lock: return jsonify(dict(latest))

@app.post('/api/detect/rule-based')
def detect_rule_based():
    data = request.get_json(silent=True) or {}
    if 'temperature' not in data:
        return jsonify(error='temperature is required'), 400
    res = rules.temperature_alert(data['temperature'])
    return (jsonify(res), 200) if res['state'] == 'ok' else (jsonify(res | {"error": "temperature must be a finite number"}), 400)

@app.post('/api/predict/anomaly')
def predict_anomaly():
    """Stateless: body {reading:{...}, history:[{timestamp,voltage,current,power,temperature}, ...]} or a bare reading."""
    data = request.get_json(silent=True) or {}
    reading = data.get('reading', data)
    hist = data.get('history') or []
    if not isinstance(hist, list):
        return jsonify(error='history must be a list'), 400
    rows = []
    for k, r in enumerate(hist + [reading]):
        row, err = validate_reading(r)
        if err:
            return jsonify(error=f"{'reading' if k == len(hist) else f'history[{k}]'}: {err}"), 400
        rows.append((r.get('timestamp') if isinstance(r, dict) else None, row))
    stream = f"api-{threading.get_ident()}-{time.time_ns()}"
    try:
        for ts, row in rows[:-1]:
            engine.analyse(stream, row, timestamp=ts)
        res = engine.analyse(stream, rows[-1][1], timestamp=rows[-1][0])
    finally:
        engine.reset(stream)
    a = res['anomaly']
    code = 503 if a['state'] == 'unavailable' else 200
    return jsonify(anomaly=a, note='Anomaly score is unusualness relative to recorded normal data; it is not a fault probability.'), code

@app.post('/api/predict/temperature')
def predict_temperature():
    """Body {history:[{timestamp,voltage,current,power,temperature}, ...]} covering >= 10 minutes, newest last."""
    data = request.get_json(silent=True) or {}
    hist = data.get('history')
    if not isinstance(hist, list) or not hist:
        return jsonify(error='history (list of readings, >= 10 minutes, newest last) is required'), 400
    stream = f"api-{threading.get_ident()}-{time.time_ns()}"
    try:
        res = None
        for k, r in enumerate(hist):
            row, err = validate_reading(r)
            if err:
                return jsonify(error=f'history[{k}]: {err}'), 400
            if not r.get('timestamp'):
                return jsonify(error=f'history[{k}]: timestamp is required'), 400
            res = engine.analyse(stream, row, timestamp=r['timestamp'], weather=current_weather() if data.get('use_weather', True) else None)
    finally:
        engine.reset(stream)
    tf = res['temperature_forecast']
    return jsonify(temperature_forecast=tf), (503 if tf['state'] == 'unavailable' else 422 if tf['state'] != 'ok' else 200)

@app.get('/api/weather/forecast')
def weather_forecast():
    f = forecast_cache.get(force=request.args.get('refresh') == '1')
    return jsonify(f), (200 if f['state'] == 'ok' else 503)

@app.get('/api/predict/weekly-outlook')
def weekly_outlook():
    out = engine.weekly_outlook(forecast_cache.get(), outage=outage_model)
    return jsonify(out), (200 if out['state'] == 'ok' else 503)

@app.get('/api/predict/today')
def today_outlook():
    """Rest of today: LSTM outage-risk windows (weather-driven) + ML load temperature vs the 30 °C threshold."""
    out = engine.today_outlook(forecast_cache.get(), rules.threshold_c(), outage=outage_model)
    return jsonify(out), (200 if out['state'] == 'ok' else 503)

@app.get('/api/outage/last')
def outage_last():
    """Most recent real outage recorded in Bengaluru in the ESMI training data (historical, ends 2018)."""
    e = last_recorded_outage()
    site = site_outages.last()
    if not e and not site:
        return jsonify(state='unavailable', error='run prepare_outage.py'), 503
    return jsonify(state='ok', **(e or {}), this_site=site)

@app.get('/api/outage/model-card')
def outage_model_card():
    c = model_card()
    return (jsonify(state='ok', **c), 200) if c else (jsonify(state='unavailable', error='run train_outage_lstm.py'), 503)

@app.post('/api/demo/temperature')
def set_demo_temperature():
    """Rule-Based Demo control: {"temperature": 32} overrides the simulated temperature; {"temperature": null} clears it.
    The value flows through the normal pipeline (simulator -> Node-RED -> /api/edge/predict -> dashboard)."""
    data = request.get_json(silent=True) or {}
    t = data.get('temperature')
    if t is not None:
        try:
            t = float(t)
        except (TypeError, ValueError):
            return jsonify(error='temperature must be a number or null'), 400
        if not math.isfinite(t) or not -55 <= t <= 125:
            return jsonify(error='temperature must be between -55 and 125 °C (DS18B20 range)'), 400
    with lock: demo['temperature'] = t
    return jsonify(ok=True, temperature_override=t, note='Simulated reading only; not hardware.')

@app.post('/api/demo/scenario')
def set_scenario():
    global scenario
    data = request.get_json(silent=True) or {}
    name = str(data.get('scenario', 'normal'))
    allowed = {'normal','voltage_sag','voltage_swell','overcurrent','thermal_anomaly','driver_instability'}
    if name not in allowed: return jsonify(error='scenario must be one of '+', '.join(sorted(allowed))), 400
    try: duration = max(5, min(120, int(data.get('duration_seconds', 25))))
    except (TypeError, ValueError): return jsonify(error='duration_seconds must be an integer'), 400
    with lock: scenario = {'name': name, 'until': time.time() + duration}
    return jsonify(ok=True, scenario=name, duration_seconds=duration, note='Simulated readings only; no physical relay is controlled.')

@app.get('/api/demo/scenarios')
def scenarios():
    return jsonify(scenarios=['normal','voltage_sag','voltage_swell','overcurrent','thermal_anomaly','driver_instability'],
                   trigger='POST /api/demo/scenario with JSON {"scenario":"overcurrent","duration_seconds":25}',
                   temperature='POST /api/demo/temperature with JSON {"temperature":32} (null clears)')


# ---- Multi-home fleet: 10 simulated homes, each with its own window, scenario and fault detection ----
import math
N_HOMES = 10
FAULTS = {'normal','voltage_sag','voltage_swell','overcurrent','thermal_anomaly','driver_instability'}
homes_lock = threading.Lock()
homes = {}
for n in range(1, N_HOMES + 1):
    hr = random.Random(1000 + n)
    homes[n] = {"id": n, "name": f"Home {n}", "base_v": 230 + hr.uniform(-1.5, 1.5), "base_i": 0.040 + hr.uniform(-0.003, 0.003),
                "base_t": 42.3 + hr.uniform(-1.0, 1.0), "phase": hr.uniform(0, 6.28), "rng": hr,
                "window": deque(maxlen=TW + 1), "scenario": "normal", "until": 0, "tick": 0,
                "latest": None, "fault_since": None, "detections": 0}

def home_step(h, now):
    h['tick'] += 1
    k = h['tick'] + h['phase'] * 10
    name = h['scenario'] if now < h['until'] else 'normal'
    r = h['rng']
    v = h['base_v'] + 1.4 * math.sin(k / 9) + r.uniform(-0.7, 0.7)
    i = max(0.035, h['base_i'] + 0.002 * math.sin(k / 6) + r.uniform(-0.001, 0.001))
    t = h['base_t'] + 0.3 * math.sin(k / 30) + r.uniform(-0.1, 0.1)
    if name == 'voltage_sag': v = 175 + r.uniform(-3, 3)
    elif name == 'voltage_swell': v = 270 + r.uniform(-3, 3)
    elif name == 'overcurrent': i = 7.0 + r.uniform(-0.4, 0.4)
    elif name == 'thermal_anomaly': t = 68 + r.uniform(-1.5, 1.5)
    elif name == 'driver_instability': v += 14 * math.sin(k * 2.7); i += 1.4 * abs(math.sin(k * 2.1))
    row = {"voltage": round(v, 2), "current": round(i, 3), "power": round(v * i, 2), "temperature": round(t, 2)}
    h['window'].append(row)
    frame = add_features(pd.DataFrame(list(h['window'])))
    probs = model.predict_proba(frame.iloc[[-1]])[0]
    classes = list(model.classes_)
    idx = int(probs.argmax())
    label, conf = str(classes[idx]), round(float(probs[idx]), 3)
    rules = [m for m, bad in (('voltage', row['voltage'] > 253 or row['voltage'] < 207), ('current', row['current'] > 5),
                              ('temperature', row['temperature'] > 60)) if bad]
    hard = row['voltage'] > 253 or row['voltage'] < 200 or row['current'] > 5 or row['temperature'] > 60
    ml_fault = label != 'normal' and conf >= 0.5
    status = 'fault' if hard else 'warning' if (rules or ml_fault) else 'normal'
    if status != 'normal' and h['fault_since'] is None:
        h['fault_since'] = now; h['detections'] += 1
    elif status == 'normal':
        h['fault_since'] = None
    detected_in = round(now - h['fault_since'], 1) if h['fault_since'] else None
    h['latest'] = {"id": h['id'], "name": h['name'], "timestamp": datetime.now(timezone.utc).isoformat(), **row,
                   "prediction": {"label": label, "confidence": conf}, "rule_breaches": rules, "status": status,
                   "relay": "TRIPPED (SIMULATED)" if hard else "CLOSED (SIMULATED)", "injected_fault": name,
                   "detected": status != 'normal', "fault_active_s": detected_in, "detections": h['detections'],
                   "source": "SIMULATION", "sim": True}

def homes_loop():
    while True:
        now = time.time()
        with homes_lock:
            for h in homes.values(): home_step(h, now)
        time.sleep(1)

@app.get('/api/homes')
def get_homes():
    with homes_lock:
        return jsonify(homes=[h['latest'] for h in homes.values() if h['latest']], count=N_HOMES)

@app.post('/api/homes/<int:hid>/fault')
def inject_home_fault(hid):
    data = request.get_json(silent=True) or {}
    name = str(data.get('scenario', 'normal'))
    if hid not in homes: return jsonify(error=f'home must be 1-{N_HOMES}'), 404
    if name not in FAULTS: return jsonify(error='scenario must be one of ' + ', '.join(sorted(FAULTS))), 400
    try: duration = max(5, min(300, int(data.get('duration_seconds', 30))))
    except (TypeError, ValueError): return jsonify(error='duration_seconds must be an integer'), 400
    with homes_lock:
        homes[hid]['scenario'] = name; homes[hid]['until'] = time.time() + duration
    return jsonify(ok=True, home=hid, scenario=name, duration_seconds=duration, note='Simulated only.')

@app.post('/api/homes/reset')
def reset_homes():
    with homes_lock:
        for h in homes.values(): h['scenario'] = 'normal'; h['until'] = 0
    return jsonify(ok=True)

if __name__ == '__main__':
    threading.Thread(target=sample_loop, daemon=True).start()
    threading.Thread(target=homes_loop, daemon=True).start()
    threading.Thread(target=forecast_cache.get, daemon=True).start()     # warm the weather cache
    port = int(os.environ.get('VOLTGUARD_GATEWAY_PORT', '5001'))
    print(f'VoltGuard integrated demo gateway: http://localhost:{port}')
    print(f'Health: http://localhost:{port}/health | outlook: http://localhost:{port}/api/predict/weekly-outlook')
    print('ML models:', {k: v.get('loaded') for k, v in engine.status().items()})
    app.run(host=os.environ.get('VOLTGUARD_GATEWAY_HOST', '127.0.0.1'), port=port, threaded=True, use_reloader=False)
