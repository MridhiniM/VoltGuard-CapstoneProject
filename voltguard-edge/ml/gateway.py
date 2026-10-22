"""Integrated VoltGuard demo gateway: simulator -> feature engineering -> trained RF -> dashboard API.
This is a DEMO simulator, not a safety controller. Real hardware mode is intentionally not enabled here.
"""
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import threading, time, random
import joblib, pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from features import add_features, TW

ROOT = Path(__file__).resolve().parent
bundle = joblib.load(ROOT / 'model_rf.joblib')
model, FEATS = bundle['model'], bundle['features']
window = deque(maxlen=TW + 1)
edge_window = deque(maxlen=TW + 1)
edge_lock = threading.Lock()
lock = threading.Lock()
latest = {"timestamp": datetime.now(timezone.utc).isoformat(), "voltage": 230.0, "current": 0.0,
          "temperature": 30.0, "prediction": {"label": "starting", "confidence": 0.0}, "sim": True,
          "relay": "CLOSED", "source": "SIMULATION"}
scenario = {"name": "normal", "until": 0}
app = Flask(__name__); CORS(app)
rng = random.Random(42)

def sample_loop():
    global latest
    tick = 0
    while True:
        tick += 1
        now = time.time()
        with lock:
            name = scenario['name'] if now < scenario['until'] else 'normal'
        v = 230 + 1.4 * __import__('math').sin(tick / 9) + rng.uniform(-0.7, 0.7)
        i = max(0.035, 0.040 + 0.002 * __import__('math').sin(tick / 6) + rng.uniform(-0.001, 0.001))
        t = 42.3 + 0.3 * __import__('math').sin(tick / 30) + rng.uniform(-0.1, 0.1)
        if name == 'voltage_sag': v = 175 + rng.uniform(-3, 3)
        elif name == 'voltage_swell': v = 270 + rng.uniform(-3, 3)
        elif name == 'overcurrent': i = 7.0 + rng.uniform(-0.4, 0.4)
        elif name == 'thermal_anomaly': t = 68 + rng.uniform(-1.5, 1.5)
        elif name == 'driver_instability': v += 14 * __import__('math').sin(tick * 2.7); i += 1.4 * abs(__import__('math').sin(tick * 2.1))
        row = {"voltage": round(v, 2), "current": round(i, 3), "power": round(v*i, 2), "temperature": round(t, 2)}
        if row['voltage'] == 0 and row['current'] == 0: window.clear()
        window.append(row)
        frame = add_features(pd.DataFrame(list(window)))
        probs = model.predict_proba(frame.iloc[[-1]])[0]
        classes = list(model.classes_)
        idx = int(probs.argmax()); label = str(classes[idx]); confidence = round(float(probs[idx]), 3)
        # Independent hard safety rules are displayed separately from the ML label.
        critical = row['voltage'] > 253 or row['voltage'] < 200 or row['current'] > 5 or row['temperature'] > 60
        reading = {"timestamp": datetime.now(timezone.utc).isoformat(), "voltage": row['voltage'],
                   "current": row['current'], "temperature": row['temperature'],
                   "power": row['power'], "prediction": {"label": label, "confidence": confidence, "probs": {str(c): round(float(p), 3) for c,p in zip(classes, probs)}},
                   "relay": "TRIPPED (SIMULATED)" if critical else "CLOSED (SIMULATED)",
                   "source": "SIMULATION", "sim": True, "scenario": name,
                   "hard_rule_fault": critical}
        with lock: latest = reading
        time.sleep(1)

@app.get('/health')
def health():
    return jsonify(ok=True, service='VoltGuard integrated demo gateway', model='RandomForest', classes=[str(c) for c in model.classes_], mode='simulation')

@app.get('/api/demo/raw')
def demo_raw():
    # Unprocessed simulated telemetry, supplied to Node-RED as an ESP32 stand-in.
    with lock:
        r = dict(latest)
    return jsonify({k: r[k] for k in ('timestamp','voltage','current','power','temperature','scenario') if k in r} | {'source':'SIMULATED_ESP32','sim':True})

@app.post('/api/edge/predict')
def edge_predict():
    # Node-RED sends telemetry here. ML and safety rules are applied to this incoming payload.
    data = request.get_json(silent=True) or {}
    try:
        v, i, t = (float(data[k]) for k in ('voltage','current','temperature'))
        if not all(__import__('math').isfinite(x) for x in (v,i,t)):
            raise ValueError('non-finite value')
    except (ValueError, KeyError, TypeError):
        return jsonify(error='voltage, current, temperature must be finite numbers'), 400
    row = {'voltage':v,'current':i,'temperature':t,'power':v*i}
    with edge_lock:
        edge_window.append(row)
        frame = add_features(pd.DataFrame(list(edge_window)))
        probs = model.predict_proba(frame.iloc[[-1]])[0]
    classes = list(model.classes_)
    idx = int(probs.argmax())
    hard = v < 200 or v > 253 or i > 5 or t > 60
    return jsonify(timestamp=data.get('timestamp') or datetime.now(timezone.utc).isoformat(),
        voltage=v,current=i,temperature=t,power=round(v*i,2),
        prediction={'label':str(classes[idx]),'confidence':round(float(probs[idx]),3),
                    'probs':{str(c):round(float(p),3) for c,p in zip(classes,probs)}},
        hard_rule_fault=hard,relay=(data.get('hardware_relay','UNKNOWN') if data.get('source')=='ESP32' else ('TRIPPED (SIMULATED)' if hard else 'CLOSED (SIMULATED)')),
        scenario=data.get('scenario','normal'),source='ESP32 VIA NODE-RED' if data.get('source')=='ESP32' else 'SIMULATION VIA NODE-RED',sim=data.get('source')!='ESP32',hardware_relay=data.get('hardware_relay') if data.get('source')=='ESP32' else None,trip_reason=data.get('trip_reason') if data.get('source')=='ESP32' else None,edge_label=data.get('edge_label') if data.get('source')=='ESP32' else None)

@app.get('/api/readings/latest')
def get_latest():
    with lock: return jsonify(dict(latest))

@app.post('/api/demo/scenario')
def set_scenario():
    global scenario
    data = request.get_json(silent=True) or {}
    name = str(data.get('scenario', 'normal'))
    allowed = {'normal','voltage_sag','voltage_swell','overcurrent','thermal_anomaly','driver_instability'}
    if name not in allowed: return jsonify(error='scenario must be one of '+', '.join(sorted(allowed))), 400
    duration = max(5, min(120, int(data.get('duration_seconds', 25))))
    with lock: scenario = {'name': name, 'until': time.time() + duration}
    return jsonify(ok=True, scenario=name, duration_seconds=duration, note='Simulated readings only; no physical relay is controlled.')

@app.get('/api/demo/scenarios')
def scenarios():
    return jsonify(scenarios=['normal','voltage_sag','voltage_swell','overcurrent','thermal_anomaly','driver_instability'],
                   trigger='POST /api/demo/scenario with JSON {"scenario":"overcurrent","duration_seconds":25}')


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
    print('VoltGuard integrated demo gateway: http://localhost:5001')
    print('Health: http://localhost:5001/health | scenarios: http://localhost:5001/api/demo/scenarios')
    app.run(host='127.0.0.1', port=5001, threaded=True, use_reloader=False)
