"""Tests for the Rule-Based Demo, trained ML models, weather handling and gateway API.
Run: python -m pytest -q   (models must exist: python prepare_data.py && python train_ml.py)
Network is never used: the forecast fetcher is replaced with fakes."""
import json, math, time
from datetime import datetime, timedelta, timezone

import numpy as np, pandas as pd, pytest

import ml_features as F
import ml_runtime, rules, weather

MODELS_OK = (ml_runtime.MODELS / "anomaly_detector.joblib").exists()
needs_models = pytest.mark.skipif(not MODELS_OK, reason="run prepare_data.py && train_ml.py first")


def fake_forecast(start=None, days=7, temp=lambda h: 24 + 4 * math.sin((h - 9) / 24 * 2 * math.pi), rh=70):
    start = start or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    times = [(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(days * 24)]
    return {"timezone": "Asia/Kolkata", "utc_offset_seconds": 19800,
            "hourly": {"time": times, "temperature_2m": [round(temp(h), 1) for h in range(days * 24)],
                       "relative_humidity_2m": [rh] * (days * 24)}}


def readings(n, start=None, step_s=2.0, **over):
    """Normal readings resembling the recorded load, oldest first, with UTC ISO timestamps."""
    start = start or datetime.now(timezone.utc) - timedelta(seconds=n * step_s)
    rng = np.random.default_rng(0)
    out = []
    for k in range(n):
        r = {"timestamp": (start + timedelta(seconds=k * step_s)).isoformat(), "voltage": 246 + rng.normal(0, 0.3),
             "current": 0.034 + rng.normal(0, 0.0003), "temperature": 30.5 + rng.normal(0, 0.02)}
        r["power"] = r["voltage"] * r["current"] * 0.99
        r.update(over)
        out.append(r)
    return out


# ------------------------------------------------------------------ Rule-Based Demo
@pytest.mark.parametrize("t,active", [(28, False), (30, False), (30.0, False), (30.1, True), (32, True), (29.99, False)])
def test_rule_threshold(t, active):
    r = rules.temperature_alert(t)
    assert r["active"] is active and r["state"] == "ok" and r["threshold_c"] == 30
    assert r["message"] == ("Temperature Threshold Alert" if active else "Normal")
    assert r["name"].startswith("Rule-Based Demo")


def test_rule_sequence_clears_and_reactivates():
    seq = [28, 30, 32, 28, 31, 30, 30.1, 25]
    assert [rules.temperature_alert(t)["active"] for t in seq] == [False, False, True, False, True, False, True, False]


def test_rule_invalid_and_configurable(monkeypatch):
    for bad in (None, "x", float("nan"), float("inf")):
        r = rules.temperature_alert(bad)
        assert r["state"] == "unavailable" and r["active"] is False
    monkeypatch.setenv("VOLTGUARD_TEMP_THRESHOLD_C", "35")
    assert rules.temperature_alert(34)["active"] is False and rules.temperature_alert(35.5)["active"] is True
    monkeypatch.delenv("VOLTGUARD_TEMP_THRESHOLD_C")
    assert rules.threshold_c() == 30


# ------------------------------------------------------------------ ML
@pytest.fixture(scope="module")
def engine():
    return ml_runtime.MLEngine()


@needs_models
def test_models_load(engine):
    s = engine.status()
    assert all(v["loaded"] for v in s.values()), s
    assert s["anomaly_detector"]["algorithm"] in ("isolation_forest", "robust_covariance")


@needs_models
def test_normal_readings_not_flagged_and_sag_flagged(engine):
    rs = readings(60)
    for r in rs:
        res = engine.analyse("t-normal", r, timestamp=r["timestamp"])
    a = res["anomaly"]
    assert a["state"] == "ok" and isinstance(a["score"], float) and a["is_anomaly"] is False
    sag = dict(rs[-1], timestamp=(datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat(), voltage=176.0)
    a2 = engine.analyse("t-normal", sag, timestamp=sag["timestamp"])["anomaly"]
    assert a2["is_anomaly"] is True and "voltage" in a2["explanation"].lower()
    engine.reset("t-normal")


@needs_models
def test_anomaly_is_not_a_probability(engine):
    for r in readings(40):
        res = engine.analyse("t-prob", r, timestamp=r["timestamp"])
    a = res["anomaly"]
    assert "not a probability" in a["score_meaning"]
    assert not any(k in json.dumps(res).lower() for k in ("probability\":", "confidence", "fault_probability"))
    engine.reset("t-prob")


def test_invalid_inputs(engine):
    for bad in ({}, {"voltage": "x", "current": 0.03, "temperature": 30}, {"voltage": 240, "current": float("nan"), "temperature": 30}, None):
        res = engine.analyse("t-bad", bad)
        assert res["anomaly"]["state"] == "invalid_input" and res["temperature_forecast"]["state"] == "invalid_input"


def test_missing_artifacts_report_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_runtime, "MODELS", tmp_path)
    e = ml_runtime.MLEngine()
    assert not any(v["loaded"] for v in e.status().values())
    r = readings(1)[0]
    res = e.analyse("x", r, timestamp=r["timestamp"])
    assert res["anomaly"]["state"] == "unavailable" and "train_ml.py" in res["anomaly"]["error"]
    assert res["temperature_forecast"]["state"] == "unavailable"
    o = e.weekly_outlook({"state": "ok", "location": weather.location(), "fetched_at": "now", "hourly": weather.parse_hourly(fake_forecast())})
    assert all(d["indicator"] == "unavailable" and d["equipment_max_c"] is None for d in o["days"])


@needs_models
def test_ml_never_calls_rule(engine, monkeypatch):
    def boom(*a, **k): raise AssertionError("ML path called the rule")
    monkeypatch.setattr(rules, "temperature_alert", boom)
    for r in readings(40):
        res = engine.analyse("t-indep", r, timestamp=r["timestamp"])
    assert res["anomaly"]["state"] == "ok"
    engine.reset("t-indep")


@needs_models
def test_ml_has_no_30c_discontinuity(engine):
    """The ML score must not jump at the demo threshold (it would if it reproduced the rule)."""
    base = readings(40)
    scores = {}
    for t in (29.9, 30.0, 30.1, 30.2):
        s = f"t-disc-{t}"
        for r in base:
            res = engine.analyse(s, dict(r, temperature=t), timestamp=r["timestamp"])
        scores[t] = res["anomaly"]["score"]
        assert res["anomaly"]["is_anomaly"] is False
        engine.reset(s)
    assert abs(scores[30.1] - scores[30.0]) <= abs(scores[30.2] - scores[30.1]) * 3 + 1.0


@needs_models
def test_temperature_nowcast(engine):
    short = readings(150)                                       # 5 minutes
    for r in short:
        res = engine.analyse("t-now", r, timestamp=r["timestamp"])
    assert res["temperature_forecast"]["state"] == "insufficient_data"
    engine.reset("t-now")
    long = readings(400)                                        # >13 minutes
    for r in long:
        res = engine.analyse("t-now", r, timestamp=r["timestamp"], weather=None)
    t = res["temperature_forecast"]
    assert t["state"] == "ok" and t["uses_weather"] is False and 25 < t["predicted_c"] < 40 and t["horizon_min"] == F.HORIZON_MIN
    engine.reset("t-now")


# ------------------------------------------------------------------ weather
def test_parse_hourly():
    rows = weather.parse_hourly(fake_forecast(days=1))
    assert len(rows) == 24 and set(rows[0]) >= {"time", "temperature_2m", "relative_humidity_2m"}
    with pytest.raises(ValueError):
        weather.parse_hourly({"hourly": {"time": []}})
    with pytest.raises(ValueError):
        weather.parse_hourly({"hourly": {"time": ["2026-01-01T00:00"]}})


def test_forecast_respects_location_and_timezone(tmp_path, monkeypatch):
    monkeypatch.setenv("VOLTGUARD_LAT", "19.07"); monkeypatch.setenv("VOLTGUARD_LON", "72.88")
    monkeypatch.setenv("VOLTGUARD_LOCATION", "Mumbai"); monkeypatch.setenv("VOLTGUARD_TZ", "Asia/Kolkata")
    seen = []
    c = weather.ForecastCache(path=tmp_path / "f.json", fetch=lambda loc: seen.append(loc) or fake_forecast())
    f = c.get()
    assert f["state"] == "ok" and seen[0]["latitude"] == 19.07 and seen[0]["timezone"] == "Asia/Kolkata"
    assert f["location"]["name"] == "Mumbai" and f["timezone"] == "Asia/Kolkata"


def test_forecast_failure_never_fabricates(tmp_path):
    def fail(loc): raise weather.requests.ConnectionError("offline")
    c = weather.ForecastCache(path=tmp_path / "f.json", fetch=fail)
    f = c.get()
    assert f["state"] == "unavailable" and "offline" in f["error"] and "hourly" not in f


def test_forecast_cache_fallback_and_expiry(tmp_path):
    calls = {"n": 0}
    def flaky(loc):
        calls["n"] += 1
        if calls["n"] > 1: raise weather.requests.Timeout("timeout")
        return fake_forecast()
    c = weather.ForecastCache(ttl_s=0, max_age_s=3600, path=tmp_path / "f.json", fetch=flaky)
    assert c.get()["state"] == "ok"
    f2 = c.get()                                  # refresh fails -> last good forecast, flagged
    assert f2["state"] == "ok" and f2["stale"] is True and "timeout" in f2["last_error"]
    c._data["fetched_at"] = time.time() - 7200    # older than max_age -> unavailable
    assert c.get()["state"] == "unavailable"


@needs_models
def test_weekly_outlook_uses_model(engine):
    fc = {"state": "ok", "location": weather.location(), "fetched_at": "2026-10-27T00:00:00+00:00",
          "hourly": weather.parse_hourly(fake_forecast())}
    o = engine.weekly_outlook(fc)
    assert o["state"] == "ok" and len(o["days"]) == 7
    b = engine.amb
    w = pd.DataFrame(fc["hourly"]).assign(time=lambda d: pd.to_datetime(d.time)).set_index("time")
    amb = F.steady_frame(w)
    X = pd.DataFrame({"amb_3h": amb["amb_3h"], "power": b["median_power_w"]}, index=w.index)[b["features"]]
    day0 = w.index.date[0]
    expected = float(b["model"].predict(X[w.index.date == day0]).max())
    assert o["days"][0]["equipment_max_c"] == round(expected, 1)
    assert all(d["indicator"] in ("normal", "elevated") for d in o["days"])
    assert not any("probab" in json.dumps(d).lower() for d in o["days"])


def test_outlook_unavailable_and_partial(engine):
    o = engine.weekly_outlook({"state": "unavailable", "error": "x"})
    assert o["state"] == "unavailable" and o["days"] == []
    fc = fake_forecast(days=2)
    fc["hourly"]["temperature_2m"][24:] = [None] * 24        # second day missing
    o2 = engine.weekly_outlook({"state": "ok", "location": weather.location(), "fetched_at": "t", "hourly": weather.parse_hourly(fc)})
    assert o2["days"][1]["indicator"] == "unavailable" and o2["days"][1]["equipment_max_c"] is None


# ------------------------------------------------------------------ gateway integration
@pytest.fixture(scope="module")
def client():
    import gateway
    gateway.forecast_cache._fetch = lambda loc: fake_forecast()
    gateway.forecast_cache._data = None
    gateway.app.testing = True
    return gateway.app.test_client()


def test_health(client):
    j = client.get("/health").get_json()
    assert j["ok"] and j["rule_based"]["threshold_c"] == 30 and "anomaly_detector" in j["ml"]


def test_edge_predict_separates_paths(client):
    for t, active in ((28, False), (30, False), (32, True), (28, False)):
        j = client.post("/api/edge/predict", json={"voltage": 246, "current": 0.034, "temperature": t, "source": "SIMULATED_ESP32"}).get_json()
        assert j["rule_based"]["active"] is active and j["rule_based"]["kind"] == "rule"
        assert j["ml"]["kind"] == "ml" and "active" not in j["ml"]
        assert j["sim"] is True and j["source"] == "SIMULATION VIA NODE-RED"
    hw = client.post("/api/edge/predict", json={"voltage": 246, "current": 0.034, "temperature": 29, "source": "ESP32"}).get_json()
    assert hw["sim"] is False and hw["source"] == "ESP32 VIA NODE-RED"
    assert client.post("/api/edge/predict", json={"voltage": 246}).status_code == 400


def test_rule_endpoint(client):
    assert client.post("/api/detect/rule-based", json={"temperature": 30}).get_json()["active"] is False
    assert client.post("/api/detect/rule-based", json={"temperature": 30.1}).get_json()["active"] is True
    assert client.post("/api/detect/rule-based", json={}).status_code == 400
    assert client.post("/api/detect/rule-based", json={"temperature": "hot"}).status_code == 400


@needs_models
def test_predict_endpoints(client):
    h = readings(60)
    j = client.post("/api/predict/anomaly", json={"reading": h[-1], "history": h[:-1]}).get_json()
    assert j["anomaly"]["state"] == "ok" and "not a fault probability" in j["note"]
    assert client.post("/api/predict/anomaly", json={"reading": {"voltage": 1}}).status_code == 400
    r = client.post("/api/predict/temperature", json={"history": readings(400)})
    assert r.status_code == 200 and r.get_json()["temperature_forecast"]["state"] == "ok"
    assert client.post("/api/predict/temperature", json={"history": readings(30)}).status_code == 422


def test_weather_and_outlook_endpoints(client):
    import gateway
    assert client.get("/api/weather/forecast").get_json()["state"] == "ok"
    o = client.get("/api/predict/weekly-outlook")
    assert o.status_code == 200 and len(o.get_json()["days"]) == 7
    def fail(loc): raise weather.requests.ConnectionError("down")
    gateway.forecast_cache._fetch, gateway.forecast_cache._data = fail, None
    assert client.get("/api/predict/weekly-outlook").status_code == 503
    gateway.forecast_cache._fetch = lambda loc: fake_forecast()


def test_demo_temperature_endpoint(client):
    import gateway
    assert client.post("/api/demo/temperature", json={"temperature": 32}).get_json()["temperature_override"] == 32
    assert client.post("/api/demo/temperature", json={"temperature": "x"}).status_code == 400
    assert client.post("/api/demo/temperature", json={"temperature": 500}).status_code == 400
    assert client.post("/api/demo/temperature", json={"temperature": None}).get_json()["temperature_override"] is None
    assert gateway.demo["temperature"] is None


# ------------------------------------------------------------------ outage-risk indicator + today outlook
import outage_risk


@pytest.mark.parametrize("h,level", [
    ({"weather_code": 3, "precipitation": 0, "wind_gusts_10m": 20, "temperature_2m": 27}, "low"),
    ({"weather_code": 95, "precipitation": 0.5}, "high"),
    ({"precipitation": 3.0}, "elevated"),
    ({"precipitation": 9.0}, "high"),
    ({"wind_gusts_10m": 45}, "elevated"),
    ({"wind_gusts_10m": 65}, "high"),
    ({"temperature_2m": 36}, "elevated"),
    ({}, "unknown"),
])
def test_hour_risk(h, level):
    lvl, why = outage_risk.hour_risk(h)
    assert lvl == level and (bool(why) == (level in ("elevated", "high")))


def storm_forecast():
    fc = fake_forecast(days=2)
    n = len(fc["hourly"]["time"])
    fc["hourly"]["precipitation"] = [0.0] * n
    fc["hourly"]["weather_code"] = [3] * n
    fc["hourly"]["wind_gusts_10m"] = [10] * n
    for k in (15, 16, 17):                      # 15:00-18:00 thunderstorm on day 1
        fc["hourly"]["weather_code"][k] = 95; fc["hourly"]["precipitation"][k] = 8.0
    return fc


def test_today_outlook_windows(engine):
    fc = storm_forecast()
    day0 = pd.Timestamp(fc["hourly"]["time"][0])
    o = engine.today_outlook({"state": "ok", "location": weather.location(), "fetched_at": "t",
                              "hourly": weather.parse_hourly(fc)}, 30.0, now=day0 + pd.Timedelta(hours=9, minutes=20))
    assert o["state"] == "ok" and o["hours"][0]["time"] == "09:00" and o["hours"][-1]["time"] == "23:00"
    assert o["outage_windows"] == [{"start": "15:00", "end": "18:00", "level": "high",
                                    "reasons": ["thunderstorm", "heavy rain 8.0 mm/h"]}]
    assert "not a probability" in o["notes"][0]
    if engine.amb and engine.amb["deployable"]:
        for h in o["hours"]:
            assert h["load_over_threshold"] == (h["load_temp_c"] > 30) or abs(h["load_temp_c"] - 30) < 0.06


def test_today_outlook_unavailable(engine):
    assert engine.today_outlook({"state": "unavailable", "error": "down"}, 30)["state"] == "unavailable"


def test_today_endpoint(client):
    r = client.get("/api/predict/today")
    assert r.status_code == 200 and r.get_json()["threshold_c"] == 30


# ------------------------------------------------------------------ weather -> outage LSTM
import outage_runtime

OUTAGE_OK = outage_runtime.ARTIFACT.exists()
needs_outage = pytest.mark.skipif(not OUTAGE_OK, reason="run download_esmi.py, prepare_outage.py, train_outage_lstm.py")


def wx_frame(rain_hours=(), mm=12.0, thunder=False, days=2):
    start = pd.Timestamp("2026-06-10 00:00")
    idx = pd.date_range(start, periods=days * 24, freq="h")
    w = pd.DataFrame({"temperature_2m": 26.0, "relative_humidity_2m": 70.0, "precipitation": 0.0, "weather_code": 3,
                      "wind_speed_10m": 8.0, "wind_gusts_10m": 18.0, "cloud_cover": 60.0, "shortwave_radiation": 300.0}, index=idx)
    for h in rain_hours:
        w.iloc[24 + h, w.columns.get_loc("precipitation")] = mm
        if thunder: w.iloc[24 + h, w.columns.get_loc("weather_code")] = 95
    return w


@pytest.fixture(scope="module")
def outage():
    return outage_runtime.OutagePredictor()


@needs_outage
def test_outage_model_loads(outage):
    s = outage.status()
    assert s["loaded"] and 0.5 < s["test_roc_auc"] < 1


@needs_outage
def test_outage_rain_raises_risk_and_is_explained(outage):
    dry = outage.predict(wx_frame())
    wet = outage.predict(wx_frame(rain_hours=(14, 15, 16), mm=12.0, thunder=True))
    t = pd.Timestamp("2026-06-11 16:00")
    assert wet.loc[t, "p"] > dry.loc[t, "p"]
    if wet.loc[t, "level"] != "normal":
        assert any("rain" in r or "thunder" in r for r in wet.loc[t, "reasons"])
    assert 0 < wet.p.min() and wet.p.max() < 1


@needs_outage
def test_outage_windows_format(outage):
    df = pd.DataFrame({"p": [0.01, 0.09, 0.08, 0.01], "typical": 0.04, "ratio": 1.0,
                       "level": ["normal", "elevated", "elevated", "normal"],
                       "reasons": [[], ["heavy rain (9.0 mm/h)"], ["thunderstorm"], []]},
                      index=pd.date_range("2026-06-11 14:00", periods=4, freq="h"))
    w = outage_runtime.windows(df)
    assert w == [{"start": "15:00", "end": "17:00", "level": "elevated", "reasons": ["heavy rain (9.0 mm/h)", "thunderstorm"],
                  "peak_p": 0.09, "typical_p": 0.04}]


@needs_outage
def test_outage_needs_24h_history(outage):
    assert outage.predict(wx_frame(days=1).iloc[:10]).empty


def test_outage_missing_artifact(tmp_path):
    o = outage_runtime.OutagePredictor(tmp_path / "none.pt")
    assert o.status()["loaded"] is False and o.predict(wx_frame()) is None


def test_last_outage_endpoint(client):
    r = client.get("/api/outage/last")
    assert r.status_code in (200, 503)
    if r.status_code == 200:
        j = r.get_json(); assert j["minutes"] >= 5 and "ESMI" in j["source"]


def test_live_outage_log(tmp_path):
    log = outage_runtime.LiveOutageLog(tmp_path / "o.csv", threshold_v=100, gap_s=60)
    t0 = pd.Timestamp("2026-10-27 10:00:00")
    assert log.last() is None
    for k, v in enumerate([240, 241, 0, 0, 0, 239]):            # 2 s readings, supply lost for ~6 s
        log.on_reading(t0 + pd.Timedelta(seconds=2 * k), v)
    e = log.last()
    assert e["kind"] == "supply_lost" and e["ongoing"] is False and e["logged_events"] == 1
    log.on_reading(t0 + pd.Timedelta(seconds=300), 240)          # 288 s without telemetry
    assert log.last()["kind"] == "no_telemetry"
    log.on_reading(t0 + pd.Timedelta(seconds=302), 10)
    assert log.last()["ongoing"] is True


def test_simulated_readings_never_logged_as_site_outages(client):
    import gateway
    before = gateway.site_outages.last()
    client.post("/api/edge/predict", json={"voltage": 0, "current": 0, "temperature": 30, "source": "SIMULATED_ESP32"})
    client.post("/api/edge/predict", json={"voltage": 240, "current": 0.03, "temperature": 30, "source": "SIMULATED_ESP32"})
    assert gateway.site_outages.last() == before


@needs_outage
def test_model_card(client):
    j = client.get("/api/outage/model-card").get_json()
    assert j["state"] == "ok" and j["flagged"]["lift"] > 1 and 0.5 < j["roc_auc"]["lstm_weather"] < 1
    assert abs(j["accuracy"]["model_at_0.5"] - j["accuracy"]["always_no_outage_baseline"]) < 0.01   # reported with its baseline
