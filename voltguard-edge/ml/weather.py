"""Open-Meteo client: historical weather (training) and 7-day forecast (live outlook).

Docs: https://open-meteo.com/en/docs (forecast), /en/docs/historical-weather-api (archive).
No API key is required. Forecasts are cached in memory and on disk; an API failure never
produces invented values: callers get `None` / an `unavailable` state instead.
"""
import json, os, threading, time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "shortwave_radiation"]
# Forecast-only extras for the outage-hazard indicator (not available from the archive API)
FORECAST_HOURLY = HOURLY + ["weather_code", "precipitation_probability", "wind_gusts_10m", "cloud_cover"]


def location():
    """Configured site (default Bengaluru). Env: VOLTGUARD_LAT, VOLTGUARD_LON, VOLTGUARD_LOCATION, VOLTGUARD_TZ."""
    return {"name": os.environ.get("VOLTGUARD_LOCATION", "Bengaluru"),
            "latitude": float(os.environ.get("VOLTGUARD_LAT", "12.9716")),
            "longitude": float(os.environ.get("VOLTGUARD_LON", "77.5946")),
            "timezone": os.environ.get("VOLTGUARD_TZ", "Asia/Kolkata")}


def _get(url, params, timeout):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise ValueError(j.get("reason", "Open-Meteo error"))
    return j


def fetch_archive(start_date, end_date, loc=None, timeout=30):
    """Hourly reanalysis weather for [start_date, end_date] (YYYY-MM-DD, local timezone)."""
    loc = loc or location()
    return _get(ARCHIVE_URL, {"latitude": loc["latitude"], "longitude": loc["longitude"],
                              "start_date": start_date, "end_date": end_date,
                              "hourly": ",".join(HOURLY), "timezone": loc["timezone"]}, timeout)


def parse_hourly(payload):
    """Open-Meteo hourly block -> list of dicts. Missing values stay None. Raises ValueError if malformed."""
    h = (payload or {}).get("hourly")
    if not isinstance(h, dict) or not isinstance(h.get("time"), list) or not h["time"]:
        raise ValueError("forecast payload has no hourly data")
    n = len(h["time"])
    keys = [k for k in FORECAST_HOURLY if isinstance(h.get(k), list) and len(h[k]) == n]
    if "temperature_2m" not in keys:
        raise ValueError("forecast payload has no temperature_2m")
    return [{"time": h["time"][i], **{k: h[k][i] for k in keys}} for i in range(n)]


class ForecastCache:
    """7-day hourly forecast with TTL refresh, disk persistence and an explicit staleness flag."""

    def __init__(self, ttl_s=1800, max_age_s=6 * 3600, timeout=10, path=None, fetch=None):
        self.ttl_s, self.max_age_s, self.timeout = ttl_s, max_age_s, timeout
        self.path = Path(path) if path else ROOT / "data" / "cache" / "forecast.json"
        self._fetch = fetch or self._fetch_live
        self._lock = threading.Lock()
        self._data = None          # {"fetched_at": epoch, "location": {...}, "payload": {...}}
        self._error = None
        self._load_disk()

    def _fetch_live(self, loc):
        return _get(FORECAST_URL, {"latitude": loc["latitude"], "longitude": loc["longitude"],
                                   "hourly": ",".join(FORECAST_HOURLY), "forecast_days": 7, "past_days": 1,
                                   "timezone": loc["timezone"]}, self.timeout)

    def _load_disk(self):
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            if d.get("location") == location() and "cloud_cover" in d.get("payload", {}).get("hourly", {}):
                self._data = d
        except (OSError, ValueError):
            pass

    def get(self, force=False):
        """Return {state, location, fetched_at, stale, hourly[]} or {state:'unavailable', error}."""
        loc = location()
        with self._lock:
            fresh = self._data and self._data.get("location") == loc and time.time() - self._data["fetched_at"] < self.ttl_s
            if force or not fresh:
                try:
                    payload = self._fetch(loc)
                    parse_hourly(payload)              # validate before replacing a good cache
                    self._data = {"fetched_at": time.time(), "location": loc, "payload": payload}
                    self._error = None
                    try:
                        self.path.parent.mkdir(parents=True, exist_ok=True)
                        self.path.write_text(json.dumps(self._data), encoding="utf-8")
                    except OSError:
                        pass
                except (requests.RequestException, ValueError) as e:
                    self._error = f"{type(e).__name__}: {e}"
            d = self._data if self._data and self._data.get("location") == loc else None
            if not d or time.time() - d["fetched_at"] > self.max_age_s:
                return {"state": "unavailable", "location": loc, "error": self._error or "no forecast fetched yet"}
            return {"state": "ok", "location": loc, "source": "Open-Meteo forecast API",
                    "fetched_at": datetime.fromtimestamp(d["fetched_at"], timezone.utc).isoformat(),
                    "stale": time.time() - d["fetched_at"] > self.ttl_s, "last_error": self._error,
                    "timezone": d["payload"].get("timezone", loc["timezone"]),
                    "utc_offset_seconds": d["payload"].get("utc_offset_seconds", 0),
                    "hourly": parse_hourly(d["payload"])}
