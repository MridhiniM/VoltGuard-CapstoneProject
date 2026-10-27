"""Rule-Based Demo: the fixed temperature threshold alert used for the capstone demonstration.

Deliberately independent of every ML component: it reads only the measured temperature.
Alert is active iff temperature > threshold (default 30 °C; exactly 30 °C is NOT an alert).
Stateless, so an alert clears on the first reading at or below the threshold.
"""
import math, os

NAME = "Rule-Based Demo — Temperature Threshold Alert"


def threshold_c():
    """Env VOLTGUARD_TEMP_THRESHOLD_C overrides the default 30 °C."""
    try:
        v = float(os.environ.get("VOLTGUARD_TEMP_THRESHOLD_C", "30"))
        return v if math.isfinite(v) else 30.0
    except ValueError:
        return 30.0


def temperature_alert(temperature, threshold=None):
    t = threshold_c() if threshold is None else float(threshold)
    try:
        x = float(temperature)
    except (TypeError, ValueError):
        x = math.nan
    if not math.isfinite(x):
        return {"name": NAME, "kind": "rule", "rule": f"temperature > {t:g} °C", "threshold_c": t,
                "temperature_c": None, "state": "unavailable", "active": False,
                "message": "Temperature unavailable"}
    active = x > t
    return {"name": NAME, "kind": "rule", "rule": f"temperature > {t:g} °C", "threshold_c": t,
            "temperature_c": round(x, 2), "state": "ok", "active": active,
            "message": "Temperature Threshold Alert" if active else "Normal"}
