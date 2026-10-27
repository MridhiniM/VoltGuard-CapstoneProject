"""Live inference for the weather -> outage LSTM (artifact from train_outage_lstm.py; never retrains).

Explanations are counterfactual: the model is re-run with one factor set to a typical value
(rain/thunder -> none, temperature -> training mean for that hour and month, wind -> training median).
A factor is named only if removing it lowers the predicted probability by at least 20%.
"""
from pathlib import Path

import numpy as np, pandas as pd

import outage_features as OF

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "models" / "outage_lstm.pt"
EVENTS = ROOT / "data" / "processed" / "outage_events.csv"
HOURLY = ROOT / "data" / "processed" / "outage_hourly.csv"


class OutagePredictor:
    def __init__(self, path=ARTIFACT):
        self.err, self.b, self.model = None, None, None
        try:
            import torch
            from train_outage_lstm import OutageLSTM
            b = torch.load(path, map_location="cpu", weights_only=False)
            m = OutageLSTM(b["n_in"]); m.load_state_dict(b["state_dict"]); m.eval()
            self.b, self.model, self._torch = b, m, torch
        except FileNotFoundError:
            self.err = "outage_lstm.pt not found - run prepare_outage.py and train_outage_lstm.py"
        except Exception as e:
            self.err = f"outage model failed to load: {type(e).__name__}: {e}"

    def status(self):
        if not self.b:
            return {"loaded": False, "error": self.err}
        return {"loaded": True, "version": self.b["version"], "trained_at": self.b["trained_at"],
                "test_roc_auc": round(self.b["test_metrics"]["roc_auc"], 3), "test_pr_auc": round(self.b["test_metrics"]["pr_auc"], 4),
                "elevated_threshold": round(self.b["risk_bands"]["elevated"], 4), "test_base_rate": round(self.b["test_metrics"]["base_rate"], 4)}

    def _prob(self, raw):
        f = OF.feature_frame(raw)
        Z = OF.scale(f, self.b["scaler"])
        idx, X = OF.windows(Z, Z.index)
        if not len(idx):
            return pd.Series(dtype=float)
        with self._torch.no_grad():
            z = self.model(self._torch.tensor(X, dtype=self._torch.float32)).numpy()
        a, c = self.b["platt"]
        return pd.Series(1 / (1 + np.exp(-(a * z + c))), index=idx)

    def predict(self, w):
        """w: hourly weather frame (naive local time) covering >= 24 h before the hours of interest.
        Returns a DataFrame per hour: p, typical, ratio, level, reasons."""
        if not self.b:
            return None
        if any(c not in w for c in OF.REQUIRED):           # incomplete forecast: no prediction rather than a guess
            return _empty()
        p = self._prob(w)
        if p.empty:
            return _empty()
        bm = self.b["baseline_means"]
        cf = {}
        dry = w.copy(); dry["precipitation"] = 0.0; dry["weather_code"] = dry["weather_code"].where(~dry["weather_code"].isin([95, 96, 99]), 3)
        cf["rain"] = self._prob(dry)
        cool = w.copy(); cool["temperature_2m"] = [bm["temperature_by_hour_month"].get(f"{t.hour}-{t.month}", v) for t, v in zip(w.index, w["temperature_2m"])]
        cf["temperature"] = self._prob(cool)
        calm = w.copy(); calm["wind_speed_10m"] = np.minimum(calm["wind_speed_10m"], bm["wind_speed_10m"]); calm["wind_gusts_10m"] = np.minimum(calm["wind_gusts_10m"], bm["wind_gusts_10m"])
        cf["wind"] = self._prob(calm)
        clim, prior = self.b["climatology"], self.b["prior"]
        typical = pd.Series([clim.get(f"{t.hour}-{t.month}", prior) for t in p.index], index=p.index)
        ratio = p / typical
        hi, el = self.b["risk_bands"]["high"], self.b["risk_bands"]["elevated"]      # absolute probabilities, set on validation data
        rows = []
        for t in p.index:
            lvl = "high" if p[t] >= hi else "elevated" if p[t] >= el else "normal"
            reasons = []
            if lvl != "normal":
                win = w.loc[t - pd.Timedelta(hours=3):t]
                for k, s in cf.items():
                    if t in s.index and p[t] - s[t] >= 0.2 * p[t]:
                        reasons.append(_describe(k, win))
                if not reasons:
                    reasons.append("the usual outage pattern at this time of day")
            rows.append({"time": t, "p": float(p[t]), "typical": float(typical[t]), "ratio": float(ratio[t]), "level": lvl, "reasons": reasons})
        return pd.DataFrame(rows).set_index("time")


def _empty():
    return pd.DataFrame(columns=["p", "typical", "ratio", "level", "reasons"], index=pd.DatetimeIndex([], name="time"))


def _describe(kind, win):
    if kind == "rain":
        if win["weather_code"].isin([95, 96, 99]).any():
            return "thunderstorm"
        mm = float(win["precipitation"].max())
        return f"{'heavy' if mm > 7.6 else 'moderate' if mm >= 2.5 else 'light'} rain ({mm:.1f} mm/h)"
    if kind == "temperature":
        return f"high temperature ({float(win['temperature_2m'].max()):.0f} °C)"
    return f"strong wind (gusts {float(win['wind_gusts_10m'].max()):.0f} km/h)"


def windows(df, start_fmt="%H:%M"):
    """Consecutive elevated/high hours -> [{start, end, level, reasons, peak_p, typical_p}]."""
    out, cur, prev = [], None, None
    for t, r in df.iterrows():
        if r.level in ("elevated", "high"):
            if cur and prev is not None and t - prev == pd.Timedelta(hours=1):
                cur["end"] = (t + pd.Timedelta(hours=1)).strftime(start_fmt)
                if r.level == "high": cur["level"] = "high"
                cur["reasons"] += [x for x in r.reasons if x not in cur["reasons"]]
                if r.p > cur["peak_p"]: cur["peak_p"], cur["typical_p"] = round(r.p, 4), round(r.typical, 4)
            else:
                cur = {"start": t.strftime(start_fmt), "end": (t + pd.Timedelta(hours=1)).strftime(start_fmt), "level": r.level,
                       "reasons": list(r.reasons), "peak_p": round(r.p, 4), "typical_p": round(r.typical, 4)}
                out.append(cur)
        else:
            cur = None
        prev = t
    for c in out:   # "usual pattern" only if nothing weather-related explains the window
        if len(c["reasons"]) > 1 and "the usual outage pattern at this time of day" in c["reasons"]:
            c["reasons"].remove("the usual outage pattern at this time of day")
    return out


def last_recorded_outage():
    """Most recent outage in the ESMI Bengaluru data, with the observed weather at that hour."""
    try:
        ev = pd.read_csv(EVENTS, parse_dates=["start", "end"])
        h = pd.read_csv(HOURLY, parse_dates=["hour"]).set_index("hour")
    except (OSError, ValueError):
        return None
    if ev.empty:
        return None
    e = ev.sort_values("start").iloc[-1]
    wx = h.loc[e.start.floor("h")] if e.start.floor("h") in h.index else None
    return {"site": e.site.replace(" [Offline]", ""), "start": e.start.isoformat(), "end": e.end.isoformat(), "minutes": int(e.minutes),
            "weather": None if wx is None else {"temperature_c": round(float(wx.temperature_2m), 1), "precip_mm": round(float(wx.precipitation), 1)},
            "source": "ESMI (Prayas Energy Group) household voltage monitor; historical data ends 2018",
            "events_in_dataset": int(len(ev))}


LIVE_LOG = ROOT / "data" / "live" / "site_outages.csv"


class LiveOutageLog:
    """Records real supply interruptions seen by the VoltGuard hardware (ESP32 readings only, never simulation).
    voltage below VOLTGUARD_OUTAGE_V (default 100 V) = supply lost; a telemetry gap > 60 s is logged separately
    as 'no_telemetry' because it may be an outage or only a Wi-Fi/MQTT drop. Builds the site's own outage history."""

    def __init__(self, path=LIVE_LOG, threshold_v=None, gap_s=60):
        import os, threading
        self.path, self.gap_s = Path(path), gap_s
        self.threshold_v = threshold_v if threshold_v is not None else float(os.environ.get("VOLTGUARD_OUTAGE_V", "100"))
        self.active, self.last_ts, self._lock = None, None, threading.Lock()

    def _write(self, row):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", encoding="utf-8") as f:
            if new:
                f.write("start,end,minutes,kind,min_voltage\n")
            f.write(f"{row['start']},{row['end']},{row['minutes']},{row['kind']},{row['min_voltage']}\n")

    def on_reading(self, ts, voltage):
        ts = pd.Timestamp(ts)
        with self._lock:
            if self.last_ts is not None and (ts - self.last_ts).total_seconds() > self.gap_s and self.active is None:
                self._write({"start": self.last_ts.isoformat(), "end": ts.isoformat(),
                             "minutes": round((ts - self.last_ts).total_seconds() / 60, 1), "kind": "no_telemetry", "min_voltage": ""})
            self.last_ts = ts
            if voltage < self.threshold_v:
                if self.active is None:
                    self.active = {"start": ts, "min_voltage": voltage}
                self.active["min_voltage"] = min(self.active["min_voltage"], voltage)
            elif self.active is not None:
                a, self.active = self.active, None
                self._write({"start": a["start"].isoformat(), "end": ts.isoformat(),
                             "minutes": round((ts - a["start"]).total_seconds() / 60, 1), "kind": "supply_lost", "min_voltage": round(a["min_voltage"], 1)})

    def last(self):
        with self._lock:
            if self.active is not None:
                return {"start": self.active["start"].isoformat(), "end": None, "minutes": None, "kind": "supply_lost", "ongoing": True}
        try:
            d = pd.read_csv(self.path)
        except (OSError, ValueError):
            return None
        if d.empty:
            return None
        r = d.iloc[-1]
        return {"start": r.start, "end": r.end, "minutes": float(r.minutes), "kind": r.kind, "ongoing": False, "logged_events": int(len(d))}


def model_card(report=ROOT / "reports" / "outage_evaluation.json"):
    """Headline, validated numbers for the dashboard (all from the untouched 2018 test year)."""
    import json
    try:
        r = json.loads(Path(report).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    t, b, rain = r["results"]["test"], r["test_risk_bands"], r["observed_onset_rate_by_rain"]
    return {"trained_on": {"outages": r["data"]["events"], "sites": len(r["data"]["sites"]), "span": r["data"]["span"],
                           "site_hours": r["data"]["site_hours_at_risk"], "source": "ESMI (Prayas Energy Group) + Open-Meteo, Bengaluru"},
            "test_period": r["splits"]["test"],
            "roc_auc": {"lstm_weather": t["LSTM weather + calendar"]["roc_auc"], "lstm_no_weather": t["LSTM calendar only (no weather)"]["roc_auc"],
                        "usual_rate_baseline": t["climatology (hour x month)"]["roc_auc"]},
            "flagged": b.get("flagged_test"), "accuracy": b.get("accuracy_test"),
            "rain_effect": {"dry": rain["dry (0 mm)"]["onset_rate"], "heavy": rain["heavy (>7.6)"]["onset_rate"],
                            "moderate": rain["moderate (2.5-7.6)"]["onset_rate"], "light": rain["light (0.1-2.5)"]["onset_rate"]}}
