"""Authenticated, idempotent telemetry ingestion for VoltGuard (Node-RED -> this service).

POST /api/telemetry   X-API-Key + Idempotency-Key headers, JSON body (see validate()).
GET  /api/readings/latest   latest reading + RandomForest result (dashboard contract).
GET  /api/status            per-stage counters, so "MQTT ok" is never mistaken for "stored".

Ingestion only validates and persists; RandomForest inference runs on a worker thread so slow or
failing inference can never block or fail ingestion. Communication/sensor/inference problems are
reported as `inference_state`, never as an electrical fault label.
"""
import hashlib, hmac, json, math, os, queue, sqlite3, threading, time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib, pandas as pd
from flask import Blueprint, jsonify, request
from features import add_features, W, TW

ROOT = Path(__file__).resolve().parent
LSTM_DIR = ROOT / "lstm_ae"
NUM = ("voltage", "current", "power", "powerFactor", "energy", "frequency",
       "apparentPower", "uptime")
NULLABLE = ("reactivePower",)   # PZEM often cannot compute it; the ESP32 then sends NaN -> null
LIMITS = {"voltage": (0, 500), "current": (0, 100), "power": (-30000, 30000), "powerFactor": (-1, 1),
          "energy": (0, 1e9), "frequency": (0, 100), "apparentPower": (0, 30000),
          "reactivePower": (-30000, 30000), "uptime": (0, 2 ** 40)}
TEMP_RANGE = (-55, 125)
GAP_S = 15.0        # no accepted sample for this long = reconnect/outage: restart the feature window
MIN_STEP_S = 4.0    # model was trained at 5 s spacing; telemetry is ~2 s, so feed every >=4 s
STALE_S = 12.0


def utcnow():
    return datetime.now(timezone.utc)


def validate(b):
    """Return (clean, errors). null temperature is preserved; every other field must be a finite number."""
    if not isinstance(b, dict):
        return None, ["body must be a JSON object"]
    errs, out = [], {}
    for k in NUM:
        v = b.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            errs.append(f"{k}: required finite number")
        elif not LIMITS[k][0] <= v <= LIMITS[k][1]:
            errs.append(f"{k}: out of range {LIMITS[k]}")
        else:
            out[k] = v
    for k in NULLABLE:
        v = b.get(k, "missing")
        if v is None:
            out[k] = None
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            errs.append(f"{k}: number or null required")
        elif not LIMITS[k][0] <= v <= LIMITS[k][1]:
            errs.append(f"{k}: out of range {LIMITS[k]}")
        else:
            out[k] = v
    t = b.get("temperature", "missing")
    if t is None:
        out["temperature"] = None
    elif isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
        errs.append("temperature: number or null required")
    elif not TEMP_RANGE[0] <= t <= TEMP_RANGE[1]:
        errs.append(f"temperature: out of range {TEMP_RANGE}")
    else:
        out["temperature"] = t
    return (None, errs) if errs else (out, [])


class Ingest:
    def __init__(self, db_path, token, device_id, model_path):
        self.db_path, self.token, self.device_id, self.model_path = db_path, token, device_id, Path(model_path)
        self.lock = threading.Lock()
        self.q = queue.Queue(maxsize=500)
        self.c = dict(received=0, stored=0, duplicates=0, rejected=0, unauthorized=0, db_errors=0,
                      queue_dropped=0, inferred=0, inference_errors=0)
        self.last = dict(received_at=None, stored_at=None, inference_at=None, db_error=None, inference_error=None)
        self.latest = None
        self.window, self.last_fed, self.last_result = deque(maxlen=TW + 1), None, None
        self.model = self.feats = None
        self.model_version = None
        self.ae = self.ae_meta = None            # LSTM autoencoder anomaly detector (optional)
        self.ae_window, self.ae_last_ts, self.ae_streak = deque(), None, 0
        self.ae_error = None
        self._init_db()

    # ---- storage -------------------------------------------------------------------------
    def conn(self):
        c = sqlite3.connect(self.db_path, timeout=5)
        c.row_factory = sqlite3.Row
        return c

    def _init_db(self):
        with self.conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS readings(
              id INTEGER PRIMARY KEY AUTOINCREMENT, device_id TEXT NOT NULL, idem_key TEXT NOT NULL UNIQUE,
              received_at TEXT NOT NULL, voltage REAL, current REAL, power REAL, power_factor REAL,
              energy REAL, frequency REAL, apparent_power REAL, reactive_power REAL,
              device_uptime_ms INTEGER, temperature REAL);
            CREATE INDEX IF NOT EXISTS ix_readings_time ON readings(device_id, received_at);
            CREATE TABLE IF NOT EXISTS predictions(
              reading_id INTEGER PRIMARY KEY REFERENCES readings(id), model TEXT, model_version TEXT,
              state TEXT NOT NULL, label TEXT, confidence REAL, probs TEXT, window_len INTEGER, created_at TEXT);
            CREATE TABLE IF NOT EXISTS anomalies(
              reading_id INTEGER PRIMARY KEY REFERENCES readings(id), model_version TEXT, state TEXT NOT NULL,
              score REAL, threshold REAL, is_anomaly INTEGER, alert INTEGER, created_at TEXT);
            CREATE TABLE IF NOT EXISTS ingest_failures(
              id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, reason TEXT, detail TEXT);""")

    def fail(self, reason, detail):
        try:
            with self.conn() as c:
                c.execute("INSERT INTO ingest_failures(at,reason,detail) VALUES(?,?,?)",
                          (utcnow().isoformat(), reason, str(detail)[:500]))
        except sqlite3.Error:
            pass

    def store(self, key, d):
        now = utcnow().isoformat()
        with self.conn() as c:
            try:
                cur = c.execute(
                    "INSERT INTO readings(device_id,idem_key,received_at,voltage,current,power,power_factor,energy,"
                    "frequency,apparent_power,reactive_power,device_uptime_ms,temperature) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self.device_id, key, now, d["voltage"], d["current"], d["power"], d["powerFactor"], d["energy"],
                     d["frequency"], d["apparentPower"], d["reactivePower"], int(d["uptime"]), d["temperature"]))
                return cur.lastrowid, now, False
            except sqlite3.IntegrityError:
                r = c.execute("SELECT id,received_at FROM readings WHERE idem_key=?", (key,)).fetchone()
                return r["id"], r["received_at"], True

    # ---- inference worker ----------------------------------------------------------------
    def load_model(self):
        try:
            b = joblib.load(self.model_path)
            self.model, self.feats = b["model"], b["features"]
            self.model_version = "RandomForest-" + hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:12]
        except Exception as e:  # ingestion must survive a broken model file
            self.model = None
            self.last["inference_error"] = f"model load failed: {e}"

    def load_lstm(self):
        """Optional: absent TensorFlow or model files only disable anomaly scoring, nothing else."""
        try:
            meta = json.loads((LSTM_DIR / "meta.json").read_text())
            import tensorflow as tf
            self.ae = tf.keras.models.load_model(LSTM_DIR / "model.keras", compile=False)
            self.ae_meta = meta
            self.ae_window = deque(maxlen=meta["window"])
        except Exception as e:
            self.ae = None
            self.ae_error = f"lstm unavailable: {e}"[:300]

    def reset_ae(self):
        self.ae_window.clear(); self.ae_last_ts = None; self.ae_streak = 0

    def decide_anomaly(self, d, ts):
        """LSTM-AE reconstruction error over the last `window` consecutive ~2 s samples (training order/scaling)."""
        m = self.ae_meta
        if self.ae is None or m is None:
            return dict(state="unavailable", reason=self.ae_error or "model not loaded")
        if d["temperature"] is None:
            self.reset_ae(); return dict(state="sensor_error", reason="temperature unavailable")
        if d["voltage"] == 0 and d["current"] == 0:
            self.reset_ae(); return dict(state="off", reason="supply off")
        if self.ae_last_ts is not None:
            if ts - self.ae_last_ts > m["gap_s"]:
                self.reset_ae()                      # outage/reconnect: never bridge a gap inside a window
            elif ts - self.ae_last_ts < 0.5 * m["step_s"]:
                return self.last_anomaly or dict(state="insufficient_data", window=len(self.ae_window))
        self.ae_window.append([d[k] for k in m["features"]]); self.ae_last_ts = ts
        if len(self.ae_window) < m["window"]:
            return dict(state="insufficient_data", window=len(self.ae_window), needed=m["window"])
        import numpy as np
        x = (np.array([list(self.ae_window)], dtype="float32") - np.array(m["mean"])) / np.array(m["scale"])
        score = float(np.mean((self.ae.predict(x, verbose=0) - x) ** 2))
        flag = score > m["threshold"]
        self.ae_streak = self.ae_streak + 1 if flag else 0
        return dict(state="ok", score=round(score, 5), threshold=round(m["threshold"], 5), is_anomaly=bool(flag),
                    alert=self.ae_streak >= m["consecutive"], window=len(self.ae_window))

    last_anomaly = None

    def decide(self, rid, d, ts):
        """-> dict(state,label,confidence,probs,window). Pure w.r.t. DB; mutates the feature window."""
        if d["temperature"] is None:
            return dict(state="sensor_error", reason="temperature unavailable")
        if d["voltage"] == 0 and d["current"] == 0:
            self.window.clear(); self.last_fed = None
            return dict(state="off", reason="supply off")
        if self.last_fed is not None and ts - self.last_fed > GAP_S:
            self.window.clear(); self.last_fed = None
        if self.last_fed is not None and ts - self.last_fed < MIN_STEP_S:
            return self.last_result or dict(state="insufficient_data", window=len(self.window))
        self.window.append({k: d[k] for k in ("voltage", "current", "power", "temperature")})
        self.last_fed = ts
        if len(self.window) < W + 1:
            return dict(state="insufficient_data", window=len(self.window), needed=W + 1)
        if self.model is None:
            return dict(state="inference_error", reason="model unavailable")
        x = add_features(pd.DataFrame(list(self.window))).iloc[[-1]][self.feats]
        p = self.model.predict_proba(x)[0]
        probs = {str(c): round(float(v), 3) for c, v in zip(self.model.classes_, p)}
        label = max(probs, key=probs.get)
        return dict(state="ok", label=label, confidence=probs[label], probs=probs, window=len(self.window))

    def process(self, rid, d, received_at):
        try:
            res = self.decide(rid, d, datetime.fromisoformat(received_at).timestamp())
            if res["state"] in ("ok", "insufficient_data"):
                self.last_result = res    # only window-based results may be reused by decimation
            self.c["inferred"] += 1
            self.last["inference_at"] = utcnow().isoformat()
        except Exception as e:
            res = dict(state="inference_error", reason=str(e)[:200])
            self.c["inference_errors"] += 1
            self.last["inference_error"] = res["reason"]
        try:
            an = self.decide_anomaly(d, datetime.fromisoformat(received_at).timestamp())
            if an["state"] in ("ok", "insufficient_data"):
                self.last_anomaly = an
        except Exception as e:
            an = dict(state="inference_error", reason=str(e)[:200])
            self.c["inference_errors"] += 1; self.last["inference_error"] = an["reason"]
        an_version = self.ae_meta["version"] if self.ae_meta else None
        try:
            with self.conn() as c:
                c.execute("INSERT OR REPLACE INTO anomalies VALUES(?,?,?,?,?,?,?,?)",
                          (rid, an_version, an["state"], an.get("score"), an.get("threshold"),
                           None if an.get("is_anomaly") is None else int(an["is_anomaly"]),
                           None if an.get("alert") is None else int(an["alert"]), utcnow().isoformat()))
                c.execute("INSERT OR REPLACE INTO predictions VALUES(?,?,?,?,?,?,?,?,?)",
                          (rid, "RandomForest", self.model_version, res["state"], res.get("label"),
                           res.get("confidence"), json.dumps(res.get("probs")), res.get("window"),
                           utcnow().isoformat()))
        except sqlite3.Error as e:
            self.c["db_errors"] += 1; self.last["db_error"] = str(e)
        pred = {"label": res["label"], "confidence": res["confidence"], "probs": res["probs"]} \
            if res["state"] == "ok" else None
        with self.lock:
            self.latest = dict(timestamp=received_at, device_id=self.device_id, **d, prediction=pred,
                               inference_state=res["state"], model_version=self.model_version,
                               anomaly=dict(an, model_version=an_version),
                               source="ESP32", sim=False)

    def worker(self):
        self.load_model()
        self.load_lstm()
        while True:
            rid, d, at = self.q.get()
            self.process(rid, d, at)


def init_ingest(app, model_path=None, db_path=None, start_worker=True):
    from flask_cors import CORS
    ing = Ingest(db_path or os.environ.get("VOLTGUARD_DB", str(ROOT / "voltguard.db")),
                 os.environ.get("VOLTGUARD_INGEST_TOKEN", ""),
                 os.environ.get("VOLTGUARD_DEVICE_ID", "voltguard-esp32-01"),
                 model_path or ROOT / "model_rf.joblib")
    CORS(app, resources={r"/api/readings/*": {"origins": "*"}, r"/api/status": {"origins": "*"}})
    bp = Blueprint("ingest", __name__)

    @bp.post("/api/telemetry")
    def telemetry():
        if not ing.token:
            return jsonify(ok=False, error="ingestion not configured (VOLTGUARD_INGEST_TOKEN unset)"), 503
        if not hmac.compare_digest(request.headers.get("X-API-Key", ""), ing.token):
            ing.c["unauthorized"] += 1
            return jsonify(ok=False, error="unauthorized"), 401
        key = request.headers.get("Idempotency-Key", "")
        if not 8 <= len(key) <= 200:
            ing.c["rejected"] += 1
            return jsonify(ok=False, error="Idempotency-Key header (8-200 chars) required"), 400
        ing.c["received"] += 1
        ing.last["received_at"] = utcnow().isoformat()
        d, errs = validate(request.get_json(silent=True))
        if errs:
            ing.c["rejected"] += 1; ing.fail("validation", errs)
            return jsonify(ok=False, error="invalid telemetry", details=errs), 422
        try:
            rid, at, dup = ing.store(key, d)
        except sqlite3.Error as e:
            ing.c["db_errors"] += 1; ing.last["db_error"] = str(e)
            return jsonify(ok=False, error="database unavailable"), 503
        if dup:
            ing.c["duplicates"] += 1
            return jsonify(ok=True, duplicate=True, id=rid, received_at=at), 200
        ing.c["stored"] += 1; ing.last["stored_at"] = at
        try:
            ing.q.put_nowait((rid, d, at))
        except queue.Full:
            ing.c["queue_dropped"] += 1   # reading is stored; only its inference is skipped
        return jsonify(ok=True, duplicate=False, id=rid, received_at=at), 201

    @bp.get("/api/readings/latest")
    def latest():
        with ing.lock:
            r = dict(ing.latest) if ing.latest else None
        if not r:
            return jsonify(error="waiting for first reading"), 503
        age = time.time() - datetime.fromisoformat(r["timestamp"]).timestamp()
        if age > STALE_S:
            return jsonify(error="stale telemetry", age_s=round(age, 1)), 503
        return jsonify(r)

    @bp.get("/api/status")
    def status():
        db_ok = True
        try:
            with ing.conn() as c:
                c.execute("SELECT 1")
        except sqlite3.Error:
            db_ok = False
        return jsonify(ingestion_configured=bool(ing.token), database_ok=db_ok, counters=ing.c, last=ing.last,
                       inference=dict(model_loaded=ing.model is not None, model_version=ing.model_version,
                                      lstm_loaded=ing.ae is not None, lstm_version=(ing.ae_meta or {}).get("version"),
                                      lstm_error=ing.ae_error, lstm_window=len(ing.ae_window),
                                      queue_depth=ing.q.qsize(), window=len(ing.window)),
                       server_time=utcnow().isoformat())

    app.register_blueprint(bp)
    if start_worker:
        threading.Thread(target=ing.worker, daemon=True).start()
    return ing
