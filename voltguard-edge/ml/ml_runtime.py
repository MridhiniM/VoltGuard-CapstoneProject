"""Live ML inference for VoltGuard: loads the artifacts written by train_ml.py (never retrains).

Independent of rules.py by design: nothing here reads the rule-based temperature alert.
Every output carries an explicit `state`; a missing model or input yields `unavailable` /
`insufficient_data`, never a substitute value.
"""
import math, threading
from collections import deque
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib, numpy as np, pandas as pd

import ml_features as F
import outage_risk
import outage_runtime

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
LABELS = {"voltage": "voltage", "current": "current", "power": "power", "temperature": "temperature",
          "v_dev": "voltage step", "i_std": "current fluctuation", "p_std": "power fluctuation", "t_rate": "temperature rise (5 min)"}
UNITS = {"voltage": "V", "current": "A", "power": "W", "temperature": "°C", "v_dev": "V", "i_std": "A", "p_std": "W", "t_rate": "°C"}
WARMUP_S = 60
BUFFER_S = (F.HIST_MIN + 2) * 60


def _load(name):
    p = MODELS / name
    try:
        return joblib.load(p), None
    except FileNotFoundError:
        return None, f"{name} not found - run prepare_data.py and train_ml.py"
    except Exception as e:                       # corrupt / incompatible artifact
        return None, f"{name} failed to load: {type(e).__name__}: {e}"


def explain(x, z, stats):
    """One short, data-backed sentence: prefer a measured channel outside its recorded range."""
    lo, hi = stats["min"], stats["max"]
    out = [f for f in ("voltage", "current", "power", "temperature") if not lo[f] <= float(x[f].iloc[0]) <= hi[f]]
    if out:
        f = max(out, key=lambda k: abs(z[k]))
        v = float(x[f].iloc[0])
        return (f"{LABELS[f].capitalize()} {v:.3g} {UNITS[f]} is {'above' if v > hi[f] else 'below'} the recorded normal range "
                f"({lo[f]:.3g}–{hi[f]:.3g} {UNITS[f]})")
    f = max(z, key=lambda k: abs(z[k]))
    v = float(x[f].iloc[0])
    return (f"Unusual combination: {LABELS[f]} {v:.3g} {UNITS[f]} is {abs(z[f]):.0f}σ "
            f"{'above' if z[f] > 0 else 'below'} its recorded normal, with the other channels in range")


def to_local(ts, tz):
    """ISO string / datetime -> tz-naive local time (the training data is naive IST)."""
    t = pd.Timestamp(ts) if ts is not None else pd.Timestamp.now(tz="UTC")
    if t.tzinfo is None:
        return t
    return t.tz_convert(ZoneInfo(tz)).tz_localize(None)


def validate_reading(r):
    """Return (row, error). voltage/current/temperature required; power derived when missing."""
    if not isinstance(r, dict):
        return None, "reading must be a JSON object"
    out = {}
    for k in ("voltage", "current", "temperature"):
        try:
            v = float(r.get(k))
        except (TypeError, ValueError):
            return None, f"{k} must be a number"
        if not math.isfinite(v):
            return None, f"{k} must be finite"
        out[k] = v
    p = r.get("power")
    try:
        out["power"] = float(p) if p is not None and math.isfinite(float(p)) else out["voltage"] * out["current"]
    except (TypeError, ValueError):
        out["power"] = out["voltage"] * out["current"]
    return out, None


class MLEngine:
    def __init__(self, tz="Asia/Kolkata"):
        self.tz = tz
        self.anom, self.anom_err = _load("anomaly_detector.joblib")
        self.now, self.now_err = _load("temp_nowcast.joblib")
        self.amb, self.amb_err = _load("temp_vs_ambient.joblib")
        self._buf, self._lock = {}, threading.Lock()

    def status(self):
        s = lambda b, e, extra: ({"loaded": True, "version": b["version"], "trained_at": b["trained_at"], **extra(b)} if b else {"loaded": False, "error": e})
        return {"anomaly_detector": s(self.anom, self.anom_err, lambda b: {"algorithm": b["algorithm"], "threshold": b["threshold"]}),
                "temperature_nowcast": s(self.now, self.now_err, lambda b: {"horizon_min": b["horizon_min"], "variants": list(b["variants"])}),
                "temperature_vs_ambient": s(self.amb, self.amb_err, lambda b: {"deployable": b["deployable"]})}

    # ---------------------------------------------------------------- streaming entry point
    def analyse(self, stream, reading, timestamp=None, weather=None):
        """Append a validated reading to `stream`'s history and return the ML block."""
        row, err = validate_reading(reading)
        if err:
            return {"kind": "ml", "anomaly": {"state": "invalid_input", "error": err},
                    "temperature_forecast": {"state": "invalid_input", "error": err}, "insight": None}
        ts = to_local(timestamp, self.tz)
        with self._lock:
            b = self._buf.setdefault(stream, deque())
            if b and ts <= b[-1]["timestamp"]:
                ts = b[-1]["timestamp"] + pd.Timedelta(milliseconds=1)
            if b and (ts - b[-1]["timestamp"]).total_seconds() > 15:       # outage: restart history
                b.clear()
            b.append({"timestamp": ts, **row})
            while (ts - b[0]["timestamp"]).total_seconds() > BUFFER_S:
                b.popleft()
            hist = pd.DataFrame(list(b))
        anomaly = self.score_anomaly(hist)
        return {"kind": "ml", "anomaly": anomaly, "temperature_forecast": self.forecast_temperature(hist, weather),
                "insight": anomaly.get("explanation") if anomaly.get("is_anomaly") else None}

    def reset(self, stream=None):
        with self._lock:
            self._buf.clear() if stream is None else self._buf.pop(stream, None)

    # ---------------------------------------------------------------- Model A
    def score_anomaly(self, hist):
        b = self.anom
        if not b:
            return {"state": "unavailable", "error": self.anom_err}
        span = (hist.timestamp.iloc[-1] - hist.timestamp.iloc[0]).total_seconds()
        if span < WARMUP_S:
            return {"state": "warming_up", "history_s": round(span), "needed_s": WARMUP_S, "model_version": b["version"]}
        try:
            x = F.anomaly_frame(hist).iloc[[-1]]
            score = float(-b["model"].score_samples(x[b["features"]].to_numpy())[0])
        except Exception as e:
            return {"state": "inference_error", "error": f"{type(e).__name__}: {e}"}
        thr = b["threshold"]
        mu, sd = b["train_stats"]["mean"], b["train_stats"]["std"]
        z = {f: (float(x[f].iloc[0]) - mu[f]) / sd[f] for f in b["features"]}
        top = max(z, key=lambda f: abs(z[f]))
        is_anom = score > thr
        expl = None
        if is_anom:
            expl = explain(x, z, b["train_stats"])
        return {"state": "ok", "algorithm": b["algorithm"], "model_version": b["version"],
                "score": round(score, 4), "threshold": round(thr, 4), "score_ratio": round(score / thr, 3),
                "is_anomaly": is_anom, "score_meaning": "unusualness relative to recorded normal operation; not a probability",
                "top_feature": top, "top_feature_z": round(z[top], 1), "explanation": expl,
                "temperature_trend_ready": span >= F.T_LAG_S}

    # ---------------------------------------------------------------- Model B1
    def forecast_temperature(self, hist, weather=None):
        b = self.now
        if not b:
            return {"state": "unavailable", "error": self.now_err}
        mins = F.minute_frame(hist)
        if len(mins) < F.HIST_MIN + 1 or mins.iloc[-(F.HIST_MIN + 1):].isna().any().any():
            have = int(mins.dropna().shape[0])
            return {"state": "insufficient_data", "history_min": max(0, have - 1), "needed_min": F.HIST_MIN}
        w = _weather_frame(weather)
        variant = "electrical+weather" if w is not None and "electrical+weather" in b["variants"] else "electrical"
        f = F.nowcast_frame(mins, w if variant != "electrical" else None).iloc[[-1]]
        v = b["variants"][variant]
        if f[v["features"]].isna().any(axis=1).iloc[0] and variant != "electrical":
            variant, v = "electrical", b["variants"]["electrical"]
            f = F.nowcast_frame(mins, None).iloc[[-1]]
        if f[v["features"]].isna().any(axis=1).iloc[0]:
            return {"state": "insufficient_data", "history_min": F.HIST_MIN, "needed_min": F.HIST_MIN}
        now_c = float(f["temp_now"].iloc[0])
        pred = now_c + float(v["model"].predict(f[v["features"]])[0])
        return {"state": "ok", "horizon_min": b["horizon_min"], "predicted_c": round(pred, 2),
                "persistence_c": round(now_c, 2), "model": v["key"], "uses_weather": variant != "electrical",
                "test_mae_c": round(v["test"]["mae"], 3), "model_version": b["version"],
                "for_time": str(mins.index[-1] + pd.Timedelta(minutes=b["horizon_min"]))}

    # ---------------------------------------------------------------- Model B2 + forecast -> 7-day outlook
    def weekly_outlook(self, forecast, outage=None):
        if not forecast or forecast.get("state") != "ok":
            return {"state": "unavailable", "reason": "weather forecast unavailable", "error": (forecast or {}).get("error"), "days": []}
        w = _weather_frame(forecast)
        b = self.amb
        model_ok = bool(b and b.get("deployable"))
        if model_ok:
            feats = F.steady_frame(w)
            X = pd.DataFrame({"amb_3h": feats["amb_3h"], "power": b["median_power_w"]}, index=w.index)[b["features"]]
            ok = X.notna().all(axis=1)
            pred = pd.Series(np.nan, index=w.index)
            pred[ok] = b["model"].predict(X[ok])
            lo, hi = b["amb_range"]
            extrap = (feats["amb_3h"] < lo - 0.5) | (feats["amb_3h"] > hi + 0.5)
        today = pd.Timestamp.now(tz=ZoneInfo(self.tz)).date()
        op = outage.predict(w) if outage is not None and outage.b else None
        days = []
        for day, g in w[w.index.date >= today].groupby(w[w.index.date >= today].index.date):
            amb = g["temperature_2m"].dropna()
            e = {"date": str(day), "weekday": pd.Timestamp(day).strftime("%a"), "hours": int(len(amb)),
                 "ambient_max_c": round(float(amb.max()), 1) if len(amb) else None,
                 "ambient_min_c": round(float(amb.min()), 1) if len(amb) else None,
                 "humidity_mean_pct": round(float(g["relative_humidity_2m"].mean()), 0) if g["relative_humidity_2m"].notna().any() else None}
            od = op[op.index.date == day] if op is not None else None
            if od is not None and len(od):
                lv = "high" if (od.level == "high").any() else "elevated" if (od.level == "elevated").any() else "normal"
                wins = outage_runtime.windows(od)
                top = max(wins, key=lambda x: x["peak_p"]) if wins else None
                e.update(outage_source="lstm", outage_risk=lv, outage_window=top, outage_reasons=(top or {}).get("reasons", []),
                         outage_peak_p=round(float(od.p.max()), 4), outage_typical_p=round(float(od.typical[od.p.idxmax()]), 4))
            elif op is not None:
                e.update(outage_source="lstm", outage_risk="unknown", outage_reasons=[])
            else:
                risks = [outage_risk.hour_risk(r) for r in g.reset_index().to_dict("records")]
                known = [r for r in risks if r[0] != "unknown"]
                e["outage_source"] = "rules"
                if known:
                    lvl = max((r[0] for r in known), key=outage_risk.LEVELS.index)
                    e.update(outage_risk=lvl, outage_reasons=sorted({x for r in known if r[0] == lvl for x in r[1]})[:3] if lvl != "low" else [])
                else:
                    e.update(outage_risk="unknown", outage_reasons=[])
            if not model_ok:
                e.update(equipment_max_c=None, indicator="unavailable", note="temperature model not validated")
            elif len(amb) < 18:
                e.update(equipment_max_c=None, indicator="unavailable", note="incomplete forecast for this day")
            else:
                p = pred[g.index].dropna()
                mx = float(p.max())
                ex = bool(extrap[g.index].any())
                ind = "elevated" if mx > b["temp_p99"] else "normal"
                e.update(equipment_max_c=round(mx, 1), equipment_mean_c=round(float(p.mean()), 1), indicator=ind,
                         extrapolated=ex,
                         note=(f"Predicted peak above {b['temp_p99']:.1f} °C, the top 1% of recorded equipment temperatures" if ind == "elevated" else None)
                              or ("Forecast ambient outside the training range; low confidence" if ex else None))
            days.append(e)
        return {"state": "ok", "location": forecast["location"], "forecast_fetched_at": forecast["fetched_at"],
                "forecast_stale": forecast.get("stale", False), "source": forecast.get("source"),
                "equipment_model": ({"version": b["version"], "key": b["key"], "deployable": b["deployable"]} if b else {"loaded": False, "error": self.amb_err}),
                "assumptions": [f"Load operates continuously at the median recorded power ({b['median_power_w']:.1f} W); future load is not forecast." if b else "No equipment model loaded.",
                                "Equipment temperature is predicted from the 3-hour mean forecast ambient temperature.",
                                "Indicator is a thermal outlook, not a fault probability. Electrical anomalies cannot be forecast from weather."],
                "days": days[:7]}

    def today_outlook(self, forecast, threshold_c, now=None, outage=None):
        """Rest of today, hour by hour: weather-hazard outage indicator (rule-based) and the ML-predicted
        load temperature with the hours it is expected to exceed threshold_c."""
        if not forecast or forecast.get("state") != "ok":
            return {"state": "unavailable", "reason": "weather forecast unavailable", "error": (forecast or {}).get("error"), "hours": []}
        w = _weather_frame(forecast)
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=ZoneInfo(self.tz)).tz_localize(None)
        start = now.floor("h")
        today = w[(w.index >= start) & (w.index.date == start.date())]
        b = self.amb
        model_ok = bool(b and b.get("deployable"))
        pred = None
        if model_ok:
            feats = F.steady_frame(w)                     # 3-h trailing mean uses the hours before `start` too
            X = pd.DataFrame({"amb_3h": feats["amb_3h"], "power": b["median_power_w"]}, index=w.index)[b["features"]]
            pred = pd.Series(np.nan, index=w.index)
            ok = X.notna().all(axis=1)
            pred[ok] = b["model"].predict(X[ok])
            lo, hi = b["amb_range"]
        op = outage.predict(w) if outage is not None and outage.b else None
        hours, prev = [], None
        for t, r in today.iterrows():
            if op is not None:
                if t in op.index:
                    o = op.loc[t]
                    lvl, why, pr, ty = o.level, list(o.reasons), round(float(o.p), 4), round(float(o.typical), 4)
                else:
                    lvl, why, pr, ty = "unknown", [], None, None
            else:
                (lvl, why), pr, ty = outage_risk.hour_risk({**r.to_dict(), "time": t}), None, None
            eq = float(pred[t]) if pred is not None and pd.notna(pred[t]) else None
            amb3 = float(F.steady_frame(w)["amb_3h"][t]) if model_ok else None
            hours.append({"time": t.strftime("%H:%M"), "end": (t + pd.Timedelta(hours=1)).strftime("%H:%M"),
                          "prev_contiguous": prev is not None and t - prev == pd.Timedelta(hours=1),
                          "ambient_c": _r(r.get("temperature_2m")), "precip_mm": _r(r.get("precipitation")),
                          "precip_prob_pct": _r(r.get("precipitation_probability"), 0), "gust_kmh": _r(r.get("wind_gusts_10m"), 0),
                          "weather_code": None if pd.isna(r.get("weather_code", np.nan)) else int(r["weather_code"]),
                          "outage_risk": lvl, "reasons": why, "outage_p": pr, "outage_typical_p": ty,
                          "load_temp_c": None if eq is None else round(eq, 1),
                          "load_over_threshold": None if eq is None else eq > threshold_c,
                          "low_confidence": None if amb3 is None else not (lo - 0.5 <= amb3 <= hi + 0.5)})
            prev = t
        if op is not None:
            outage_w = outage_runtime.windows(op[(op.index >= start) & (op.index.date == start.date())])
        else:
            outage_w = outage_risk.windows(hours)
        load_w = [{"start": x["start"], "end": x["end"]} for x in outage_risk.windows(hours, key="load_over_threshold", active=lambda v: v is True)]
        for h in hours:
            h.pop("prev_contiguous", None)
        return {"state": "ok", "date": str(start.date()), "location": forecast["location"], "forecast_fetched_at": forecast["fetched_at"],
                "forecast_stale": forecast.get("stale", False), "threshold_c": threshold_c, "hours": hours,
                "outage_windows": outage_w, "load_over_threshold_windows": load_w,
                "outage_source": "lstm" if op is not None else "rules",
                "outage_model": outage.status() if outage is not None else {"loaded": False},
                "load_model": ({"version": b["version"], "deployable": b["deployable"]} if b else {"loaded": False, "error": self.amb_err}),
                "notes": [("Outage risk: LSTM trained on real Bengaluru outages (ESMI 2014-2018) and Open-Meteo weather. p = probability that an "
                           "outage starts in that hour at a typical monitored Bengaluru household; 'typical' = usual rate for that hour and month.")
                          if op is not None else
                          "Outage risk is a rule-based weather-hazard indicator (outage model not loaded); not a probability.",
                          f"Load temperature is an ML estimate from forecast ambient temperature, assuming the load runs at "
                          f"{b['median_power_w']:.1f} W; trained on one day of data, so treat as low-confidence." if b else "Load temperature model not loaded."]}


def _weather_frame(forecast):
    """Forecast dict (weather.ForecastCache.get) -> hourly DataFrame indexed by naive local time."""
    if not forecast or forecast.get("state") != "ok":
        return None
    w = pd.DataFrame(forecast["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    w = w.set_index("time")
    for c in ("temperature_2m", "relative_humidity_2m"):
        if c not in w:
            w[c] = np.nan
        w[c] = pd.to_numeric(w[c], errors="coerce")
    return w


def _r(x, nd=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if x != x else round(x, nd) if nd else int(round(x))
