"""Build the processed training dataset from the real PZEM-004T + DS18B20 logs and Open-Meteo history.

    python prepare_data.py                       # default: the two logs in the repository root
    python prepare_data.py a.xlsx b.xlsx --offline   # reuse cached weather in data/raw/weather/

Raw inputs are never modified. Outputs:
    data/raw/weather/open_meteo_archive_<lat>_<lon>_<start>_<end>.json   (downloaded, untouched)
    data/processed/pzem_readings.csv     cleaned ~2 s readings, local time (Asia/Kolkata)
    data/processed/weather_hourly.csv    hourly Open-Meteo reanalysis aligned to the same timezone
    data/processed/manifest.json         counts, ranges, cleaning log
"""
import argparse, json, sys
from pathlib import Path

import pandas as pd
import weather

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
DEFAULT_LOGS = [REPO / "pzem_temperature_measurements_new_YESENEW.xlsx", REPO / "pzem_serial_diagnostics.xlsx"]
RAW_W, PROC = ROOT / "data" / "raw" / "weather", ROOT / "data" / "processed"
COLS = {"Timestamp (IST)": "timestamp", "Voltage (V)": "voltage", "Current (A)": "current",
        "Active Power (W)": "power", "Power Factor": "power_factor", "Frequency (Hz)": "frequency",
        "Temperature (°C)": "temperature"}
RANGES = {"voltage": (80, 300), "current": (0, 100), "power": (0, 25000), "power_factor": (0, 1),
          "frequency": (45, 65), "temperature": (-55, 125)}


def load_logs(paths):
    frames, log = [], []
    for p in paths:
        d = pd.read_excel(p)
        missing = [c for c in COLS if c not in d.columns]
        if missing:
            sys.exit(f"{p.name}: missing columns {missing} (expected PZEM logger export)")
        d = d[list(COLS)].rename(columns=COLS)
        d["source_file"] = p.name
        log.append({"file": p.name, "rows": len(d)})
        frames.append(d)
    return pd.concat(frames, ignore_index=True), log


def clean(d):
    steps = {}
    n = len(d)
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp", "voltage", "current", "power", "temperature"])
    steps["missing_core_values"] = n - len(d); n = len(d)
    for c, (lo, hi) in RANGES.items():
        d = d[d[c].isna() | d[c].between(lo, hi)] if c in ("power_factor", "frequency") else d[d[c].between(lo, hi)]
    steps["out_of_physical_range"] = n - len(d); n = len(d)
    d = d.sort_values("timestamp").drop_duplicates("timestamp")
    steps["duplicate_timestamps"] = n - len(d); n = len(d)
    off = (d["current"] <= 0.001) | (d["power"] < 2)       # supply/load off: not normal operation
    d = d[~off]
    steps["load_off_rows"] = int(off.sum())
    return d.reset_index(drop=True), steps


def fetch_weather(start, end, offline):
    loc = weather.location()
    RAW_W.mkdir(parents=True, exist_ok=True)
    f = RAW_W / f"open_meteo_archive_{loc['latitude']}_{loc['longitude']}_{start}_{end}.json"
    if not f.exists():
        if offline:
            sys.exit(f"--offline but {f} is missing")
        f.write_text(json.dumps(weather.fetch_archive(start, end, loc)), encoding="utf-8")
    payload = json.loads(f.read_text(encoding="utf-8"))
    w = pd.DataFrame(weather.parse_hourly(payload))
    w["time"] = pd.to_datetime(w["time"])
    return w.rename(columns={"time": "timestamp"}), f, loc, payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", type=Path, default=DEFAULT_LOGS)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    raw, files = load_logs(a.logs)
    d, steps = clean(raw)
    start, end = d.timestamp.min().date().isoformat(), d.timestamp.max().date().isoformat()
    w, wfile, loc, payload = fetch_weather(start, end, a.offline)
    if payload.get("timezone") != loc["timezone"]:
        sys.exit(f"weather timezone {payload.get('timezone')} != {loc['timezone']}")
    PROC.mkdir(parents=True, exist_ok=True)
    d.to_csv(PROC / "pzem_readings.csv", index=False)
    w.to_csv(PROC / "weather_hourly.csv", index=False)
    gaps = d.timestamp.diff().dt.total_seconds()
    manifest = {
        "electrical_dataset": {"files": files, "rows_raw": len(raw), "rows_clean": len(d), "cleaning": steps,
                               "start": str(d.timestamp.min()), "end": str(d.timestamp.max()),
                               "timezone": "Asia/Kolkata (logger local time, IST)",
                               "median_interval_s": float(gaps.median()), "gaps_over_15s": int((gaps > 15).sum()),
                               "largest_gap_min": round(float(gaps.max()) / 60, 1),
                               "fault_labels": "none (all rows are normal operation of one ~8 W load)",
                               "ranges": {c: [float(d[c].min()), float(d[c].max())] for c in ("voltage", "current", "power", "temperature")}},
        "weather_dataset": {"source": "Open-Meteo Historical Weather API (reanalysis)", "url": weather.ARCHIVE_URL,
                            "raw_file": str(wfile.relative_to(ROOT)), "location": loc, "rows": len(w),
                            "variables": [c for c in w.columns if c != "timestamp"]},
    }
    (PROC / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=1, default=str))


if __name__ == "__main__":
    main()
