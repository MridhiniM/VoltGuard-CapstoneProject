"""Laptop ML service. Node-RED's Fault Detection node POSTs each reading here.
POST /predict  {"voltage":230.1,"current":0.04,"power":8.3,"temperature":42.6}
-> {"label":"normal","confidence":0.98,"probs":{...}}
POST /reset    clears the rolling window (call after a trip/reset or power-off)."""
from collections import deque
import joblib, pandas as pd
from flask import Flask, request, jsonify
from features import add_features, TW
import os
from ingest import init_ingest

bundle = joblib.load("model_rf.joblib")
model, FEATS = bundle["model"], bundle["features"]
buf = deque(maxlen=TW + 1)
app = Flask(__name__)
ingest = init_ingest(app)

@app.post("/predict")
def predict():
    r = request.get_json(force=True)
    try:
        row = {k: float(r[k]) for k in ("voltage", "current", "power", "temperature")}
    except (KeyError, TypeError, ValueError):
        return jsonify(error="need numeric voltage, current, power, temperature"), 400
    if row["voltage"] == 0 and row["current"] == 0:          # supply off: not a fault
        buf.clear()
        return jsonify(label="off", confidence=None, probs={})
    buf.append(row)
    x = add_features(pd.DataFrame(buf)).iloc[[-1]]
    p = model.predict_proba(x)[0]
    probs = {c: round(float(v), 3) for c, v in zip(model.classes_, p)}
    label = max(probs, key=probs.get)
    return jsonify(label=label, confidence=probs[label], probs=probs, window=len(buf))

@app.post("/reset")
def reset():
    buf.clear(); return jsonify(ok=True)

@app.get("/health")
def health():
    return jsonify(ok=True, classes=list(model.classes_), window=len(buf))

if __name__ == "__main__":
    app.run(host=os.environ.get("VOLTGUARD_HOST", "0.0.0.0"), port=int(os.environ.get("VOLTGUARD_PORT", "5000")))
