"""Weather-hazard indicator for possible power outages, from the Open-Meteo hourly forecast.

This is a transparent rule-based heuristic, NOT a trained model: no outage history exists for this site,
so no outage probability can be learned or claimed. Each flag names the forecast condition behind it.
Intensity bands follow common meteorological conventions (hourly rain: moderate 2.5-7.6 mm, heavy > 7.6 mm;
WMO weather codes 95/96/99 = thunderstorm). Thresholds are configurable via environment variables.
"""
import os

LEVELS = ["low", "elevated", "high"]
THUNDER = {95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"}


def _env(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


def thresholds():
    return {"rain_moderate_mm": _env("VOLTGUARD_RAIN_MODERATE_MM", 2.5), "rain_heavy_mm": _env("VOLTGUARD_RAIN_HEAVY_MM", 7.6),
            "gust_elevated_kmh": _env("VOLTGUARD_GUST_ELEVATED_KMH", 40), "gust_high_kmh": _env("VOLTGUARD_GUST_HIGH_KMH", 60),
            "heat_elevated_c": _env("VOLTGUARD_HEAT_ELEVATED_C", 35)}


def _num(x):
    return x if isinstance(x, (int, float)) and x == x else None


def hour_risk(h, th=None):
    """h: one hourly forecast row. Returns (level, [reasons]); 'unknown' if no hazard variables are present."""
    th = th or thresholds()
    code, rain, gust, temp = _num(h.get("weather_code")), _num(h.get("precipitation")), _num(h.get("wind_gusts_10m")), _num(h.get("temperature_2m"))
    if code is None and rain is None and gust is None and temp is None:
        return "unknown", []
    lvl, why = 0, []
    if code is not None and int(code) in THUNDER:
        lvl = 2; why.append(THUNDER[int(code)])
    if rain is not None:
        if rain > th["rain_heavy_mm"]:
            lvl = 2; why.append(f"heavy rain {rain:.1f} mm/h")
        elif rain >= th["rain_moderate_mm"]:
            lvl = max(lvl, 1); why.append(f"moderate rain {rain:.1f} mm/h")
    if gust is not None:
        if gust >= th["gust_high_kmh"]:
            lvl = 2; why.append(f"wind gusts {gust:.0f} km/h")
        elif gust >= th["gust_elevated_kmh"]:
            lvl = max(lvl, 1); why.append(f"wind gusts {gust:.0f} km/h")
    if temp is not None and temp >= th["heat_elevated_c"]:
        lvl = max(lvl, 1); why.append(f"heat {temp:.0f} °C (high grid demand)")
    return LEVELS[lvl], why


def windows(hours, key="outage_risk", active=lambda v: v in ("elevated", "high")):
    """Group consecutive flagged hours into windows {start, end, level, reasons}."""
    out, cur = [], None
    for h in hours:
        if active(h.get(key)):
            if cur and h["prev_contiguous"]:
                cur["end"] = h["end"]
                if h[key] in LEVELS:
                    cur["level"] = max(cur["level"], h[key], key=LEVELS.index)
                cur["reasons"] += [r for r in h.get("reasons", []) if r not in cur["reasons"]]
            else:
                cur = {"start": h["time"], "end": h["end"], "level": h[key], "reasons": list(h.get("reasons", []))}
                out.append(cur)
        else:
            cur = None
    return out
