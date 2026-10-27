"""Feature engineering shared by train_ml.py and ml_runtime.py (single source of truth).

All windows are time-based, so training (≈2 s PZEM log) and live data (any cadence) see the
same definitions. Inputs: a time-ordered DataFrame with a tz-naive local `timestamp` column.
"""
import numpy as np, pandas as pd

ANOMALY_FEATURES = ["voltage", "current", "power", "temperature", "v_dev", "i_std", "p_std", "t_rate"]
ROLL = "60s"          # electrical short-term window
T_LAG_S = 300         # temperature change over 5 minutes
T_LAG_TOL_S = 120

HORIZON_MIN = 30      # temperature nowcast horizon
HIST_MIN = 10         # history needed by the nowcast
NOWCAST_ELEC = ["temp_now", "temp_d10", "power_10", "voltage_10"]
NOWCAST_WEATHER = ["amb_now", "amb_target", "rh_now"]
STEADY_FEATURES = ["amb_3h", "power"]


def anomaly_frame(df):
    """Per-reading features for the Isolation Forest."""
    d = df[["timestamp", "voltage", "current", "power", "temperature"]].copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    s = d.set_index("timestamp")
    prev_mean = s["voltage"].shift(1).rolling(ROLL, min_periods=1).mean()
    d["v_dev"] = (s["voltage"] - prev_mean).fillna(0).to_numpy()
    d["i_std"] = s["current"].rolling(ROLL, min_periods=2).std().fillna(0).to_numpy()
    d["p_std"] = s["power"].rolling(ROLL, min_periods=2).std().fillna(0).to_numpy()
    lag = pd.DataFrame({"timestamp": d["timestamp"] - pd.Timedelta(seconds=T_LAG_S)})
    past = pd.merge_asof(lag, d[["timestamp", "temperature"]], on="timestamp", direction="backward",
                         tolerance=pd.Timedelta(seconds=T_LAG_TOL_S))
    d["t_rate"] = (d["temperature"] - past["temperature"]).fillna(0).to_numpy()
    return d


def minute_frame(df):
    """1-minute means of the electrical/temperature channels (gaps stay NaN)."""
    s = df.set_index(pd.to_datetime(df["timestamp"]))[["voltage", "current", "power", "temperature"]]
    return s.resample("1min").mean()


def nowcast_frame(minutes, weather=None):
    """Nowcast features at each minute t; target is temperature at t+HORIZON_MIN.
    weather: hourly DataFrame indexed by local time with temperature_2m, relative_humidity_2m."""
    m = minutes
    f = pd.DataFrame(index=m.index)
    f["temp_now"] = m["temperature"]
    f["temp_d10"] = m["temperature"] - m["temperature"].shift(HIST_MIN)
    f["power_10"] = m["power"].rolling(HIST_MIN, min_periods=HIST_MIN).mean()
    f["voltage_10"] = m["voltage"].rolling(HIST_MIN, min_periods=HIST_MIN).mean()
    if weather is not None:
        w = weather_minutely(weather, m.index.min(), m.index.max() + pd.Timedelta(minutes=HORIZON_MIN))
        f["amb_now"] = w["temperature_2m"].reindex(f.index).to_numpy()
        f["amb_target"] = w["temperature_2m"].reindex(f.index + pd.Timedelta(minutes=HORIZON_MIN)).to_numpy()
        f["rh_now"] = w["relative_humidity_2m"].reindex(f.index).to_numpy()
    f["target"] = m["temperature"].shift(-HORIZON_MIN)
    return f


def weather_minutely(weather, start, end):
    """Linear interpolation of hourly weather onto a 1-minute grid (local time)."""
    idx = pd.date_range(start.floor("h"), end.ceil("h"), freq="1min")
    w = weather[["temperature_2m", "relative_humidity_2m"]].astype(float)
    return w.reindex(w.index.union(idx)).interpolate(method="time", limit_area="inside").reindex(idx)


def steady_frame(weather_hourly):
    """Ambient features for the equipment-temperature level model: 3-hour trailing ambient mean."""
    w = weather_hourly[["temperature_2m"]].astype(float)
    return pd.DataFrame({"amb_3h": w["temperature_2m"].rolling(3, min_periods=1).mean()}, index=w.index)
