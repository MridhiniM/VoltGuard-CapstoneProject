"""Feature engineering shared by training and the live service.
The ESP32 computes the SAME features in C (firmware/edge_model.h) over the same window."""
import numpy as np, pandas as pd
W = 12            # 12 samples x 5 s = 1 minute window (voltage/power/current)
TW = 60           # 60 samples x 5 s = 5 minute window (temperature rise)
FEATURES = ["voltage", "current", "power", "temperature",
            "v_dev", "p_std", "i_std", "t_rise"]

def add_features(df):
    """df columns: voltage,current,power,temperature (time-ordered)."""
    out = df[["voltage", "current", "power", "temperature"]].copy()
    prev = out.shift(1).rolling(W, min_periods=1)                # previous W samples
    out["v_dev"]  = out.voltage - prev.voltage.mean().fillna(out.voltage)
    roll = out.rolling(W, min_periods=2)                          # incl. current sample
    out["p_std"]  = roll.power.std().fillna(0)
    out["i_std"]  = roll.current.std().fillna(0)
    out["t_rise"] = out.temperature - out.temperature.shift(TW).fillna(out.temperature.iloc[0])
    return out[FEATURES]
