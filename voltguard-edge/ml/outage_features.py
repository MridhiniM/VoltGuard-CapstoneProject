"""Features for the weather -> outage LSTM, shared by train_outage_lstm.py and outage_runtime.py."""
import numpy as np, pandas as pd

WINDOW_H = 24
WEATHER = ["temperature_2m", "relative_humidity_2m", "precip_log", "wind_speed_10m", "wind_gusts_10m",
           "cloud_cover", "shortwave_radiation", "thunder"]
CALENDAR = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "sunday"]
FEATURES = WEATHER + CALENDAR
REQUIRED = ["temperature_2m", "relative_humidity_2m", "precipitation", "weather_code", "wind_speed_10m",
            "wind_gusts_10m", "cloud_cover", "shortwave_radiation"]


def feature_frame(w):
    """w: hourly frame indexed by local time with Open-Meteo columns (precipitation, weather_code, ...)."""
    f = pd.DataFrame(index=w.index)
    for c in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gusts_10m", "cloud_cover", "shortwave_radiation"):
        f[c] = pd.to_numeric(w[c], errors="coerce") if c in w else np.nan
    col = lambda c: pd.to_numeric(w[c], errors="coerce") if c in w else pd.Series(np.nan, index=w.index)
    f["precip_log"] = np.log1p(col("precipitation").clip(lower=0))
    code = col("weather_code")
    f["thunder"] = code.isin([95, 96, 99]).astype(float).where(code.notna())     # missing stays NaN -> window skipped
    hr, doy = w.index.hour + w.index.minute / 60, w.index.dayofyear
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    f["sunday"] = (w.index.day_of_week == 6).astype(float)
    return f[FEATURES]


def fit_scaler(f):
    return {"mean": f.mean().to_dict(), "std": f.std().replace(0, 1).to_dict()}


def scale(f, s):
    return (f - pd.Series(s["mean"])) / pd.Series(s["std"])


def windows(Z, targets):
    """Return (target index, X[n, WINDOW_H, F]) for targets whose full window exists and is complete."""
    arr = Z.to_numpy(dtype=np.float32)
    pos = pd.Series(np.arange(len(Z)), index=Z.index)
    keep, X = [], []
    for t in targets:
        i = pos.get(t)
        if i is None or i < WINDOW_H - 1:
            continue
        x = arr[i - WINDOW_H + 1:i + 1]
        if np.isnan(x).any():
            continue
        keep.append(t); X.append(x)
    return pd.DatetimeIndex(keep), np.stack(X) if X else np.zeros((0, WINDOW_H, Z.shape[1]), np.float32)


def baseline_means(w):
    """Typical (training-period) values used as the counterfactual when explaining a prediction."""
    f = feature_frame(w)
    by = f.groupby([f.index.hour, f.index.month])["temperature_2m"].mean()
    return {"temperature_by_hour_month": {f"{h}-{m}": float(v) for (h, m), v in by.items()},
            "wind_speed_10m": float(f.wind_speed_10m.median()), "wind_gusts_10m": float(f.wind_gusts_10m.median())}
