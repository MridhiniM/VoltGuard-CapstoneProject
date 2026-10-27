"""Build the Bengaluru outage dataset: ESMI minute-wise voltage (real) + Open-Meteo hourly weather (real).

    python download_esmi.py && python prepare_outage.py

Label: an outage ONSET = supply voltage reads 0 for >= MIN_OUTAGE_MIN consecutive minutes after supply
was present. Zero runs longer than MAX_OUTAGE_H are treated as monitor problems and excluded.
For every site-hour we record whether the site was supplied at the start (at risk) and whether an
outage began in that hour. Sites share one weather grid cell, so rows are aggregated per city-hour:
    at_risk = sites supplied & reporting,  onsets = sites where an outage began.
Outputs data/processed/outage_hourly.csv and outage_events.csv (one row per outage event).
"""
import glob, json
from pathlib import Path

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
MIN_OUTAGE_MIN = 5
MAX_OUTAGE_H = 12
WEATHER = RAW / "weather" / "open_meteo_archive_bengaluru_2014-11-01_2018-12-31.json"


def minute_series(files):
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["Location name"] = d["Location name"].str.strip()
    d["start"] = pd.to_datetime(d["Date"], format="%d-%m-%Y") + pd.to_timedelta(d["Hour"], unit="h")
    d = d.drop_duplicates(["Location name", "start"]).sort_values(["Location name", "start"])
    mins = [f"Min {i}" for i in range(60)]
    out = {}
    for site, g in d.groupby("Location name"):
        v = g[mins].to_numpy(dtype=float).ravel()
        t = (g["start"].to_numpy()[:, None] + (np.arange(60) * np.timedelta64(1, "m"))[None, :]).ravel()
        out[site] = pd.Series(v, index=pd.DatetimeIndex(t)).sort_index()
    return out


def events_for(site, s):
    """Zero-voltage runs on a contiguous minute grid; returns events and a per-hour table."""
    full = s.reindex(pd.date_range(s.index.min().floor("h"), s.index.max().ceil("h") - pd.Timedelta(minutes=1), freq="1min"))
    reported = full.notna().to_numpy()
    zero = (full == 0).to_numpy()
    ev, i, n = [], 0, len(full)
    excluded = np.zeros(n, bool)
    while i < n:
        if zero[i]:
            j = i
            while j < n and zero[j]:
                j += 1
            dur = j - i
            prev_ok = i > 0 and reported[i - 1] and not zero[i - 1]     # supply present just before
            if dur > MAX_OUTAGE_H * 60:
                excluded[i:j] = True
            elif dur >= MIN_OUTAGE_MIN and prev_ok:
                ev.append({"site": site, "start": full.index[i], "end": full.index[j - 1] + pd.Timedelta(minutes=1), "minutes": dur})
            i = j
        else:
            i += 1
    idx = full.index
    hours = pd.DataFrame({"hour": idx.floor("h"), "reported": reported, "supplied": reported & ~zero, "excluded": excluded})
    h = hours.groupby("hour").agg(reported=("reported", "sum"), first_supplied=("supplied", "first"), excluded=("excluded", "any"))
    h["at_risk"] = (h.reported >= 50) & h.first_supplied & ~h.excluded       # monitor reporting and supply on at hour start
    h["onset"] = False
    if ev:
        on = pd.DatetimeIndex([e["start"] for e in ev]).floor("h")
        h.loc[h.index.intersection(on), "onset"] = True
        h.loc[h.onset, "at_risk"] = ~h.loc[h.onset, "excluded"]                # an onset hour was supplied before the cut
    h["site"] = site
    return ev, h.reset_index()


def main():
    files = sorted(glob.glob(str(RAW / "esmi" / "bengaluru_*.csv")))
    if not files:
        raise SystemExit("run download_esmi.py first")
    series = minute_series(files)
    all_ev, all_h = [], []
    for site, s in series.items():
        ev, h = events_for(site, s)
        all_ev += ev; all_h.append(h)
        print(f"{site}: {len(ev)} outages, {int(h.at_risk.sum())} at-risk hours")
    ev = pd.DataFrame(all_ev).sort_values("start")
    h = pd.concat(all_h)
    city = h[h.at_risk].groupby("hour").agg(at_risk=("at_risk", "sum"), onsets=("onset", "sum")).astype(int)
    wj = json.loads(WEATHER.read_text(encoding="utf-8"))
    w = pd.DataFrame(wj["hourly"]); w["time"] = pd.to_datetime(w["time"]); w = w.set_index("time")
    full = w.join(city, how="left").fillna({"at_risk": 0, "onsets": 0})
    full[["at_risk", "onsets"]] = full[["at_risk", "onsets"]].astype(int)
    PROC.mkdir(parents=True, exist_ok=True)
    full.rename_axis("hour").to_csv(PROC / "outage_hourly.csv")
    ev.to_csv(PROC / "outage_events.csv", index=False)
    lab = full[full.at_risk > 0]
    summary = {"sites": sorted(series), "files": [Path(f).name for f in files], "events": len(ev),
               "event_minutes_median": float(ev.minutes.median()), "event_minutes_p90": float(ev.minutes.quantile(.9)),
               "labelled_hours": len(lab), "site_hours_at_risk": int(lab.at_risk.sum()), "onsets": int(lab.onsets.sum()),
               "onset_rate_per_site_hour": float(lab.onsets.sum() / lab.at_risk.sum()),
               "span": [str(lab.index.min()), str(lab.index.max())],
               "definition": f"onset = >= {MIN_OUTAGE_MIN} consecutive minutes at 0 V after supply; runs > {MAX_OUTAGE_H} h excluded as monitor faults"}
    (PROC / "outage_manifest.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
