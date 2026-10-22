import os, sqlite3, pytest
from datetime import datetime, timedelta, timezone
from flask import Flask
import ingest as I

TOKEN = "test-token-123"
GOOD = dict(voltage=230.1, current=0.04, power=8.2, powerFactor=0.9, energy=120, frequency=50.0,
            apparentPower=9.1, reactivePower=3.9, uptime=123456, temperature=42.5)
H = {"X-API-Key": TOKEN, "Idempotency-Key": "dev:1:1000"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VOLTGUARD_INGEST_TOKEN", TOKEN)
    app = Flask(__name__)
    ing = I.init_ingest(app, db_path=str(tmp_path / "t.db"), start_worker=False)
    ing.load_model()
    return app.test_client(), ing


def post(c, body=GOOD, headers=H):
    return c.post("/api/telemetry", json=body, headers=headers)


def rows(ing, t="readings"):
    with ing.conn() as c:
        return c.execute(f"SELECT * FROM {t}").fetchall()


def test_valid_stored_once_and_idempotent(env):
    c, ing = env
    r = post(c); assert r.status_code == 201 and r.json["duplicate"] is False
    r2 = post(c); assert r2.status_code == 200 and r2.json["duplicate"] is True and r2.json["id"] == r.json["id"]
    assert len(rows(ing)) == 1 and ing.c["duplicates"] == 1


def test_auth(env):
    c, ing = env
    assert post(c, headers={"Idempotency-Key": "abcdefgh1"}).status_code == 401
    assert post(c, headers={"X-API-Key": "wrong", "Idempotency-Key": "abcdefgh1"}).status_code == 401
    assert len(rows(ing)) == 0


def test_fails_closed_without_token(tmp_path, monkeypatch):
    monkeypatch.delenv("VOLTGUARD_INGEST_TOKEN", raising=False)
    app = Flask(__name__); I.init_ingest(app, db_path=str(tmp_path / "t.db"), start_worker=False)
    assert post(app.test_client()).status_code == 503


def test_missing_idempotency_key(env):
    assert post(env[0], headers={"X-API-Key": TOKEN}).status_code == 400


@pytest.mark.parametrize("mut", [{"voltage": "230"}, {"current": None}, {"power": float("inf")},
                                 {"voltage": 9999}, {"temperature": "hot"}, {"uptime": True}, {"frequency": -1}])
def test_invalid_rejected(env, mut):
    c, ing = env
    r = post(c, {**GOOD, **mut}, {**H, "Idempotency-Key": "k" + str(abs(hash(str(mut))))})
    assert r.status_code == 422 and len(rows(ing)) == 0 and len(rows(ing, "ingest_failures")) == 1


def test_missing_field_and_non_object(env):
    c, ing = env
    b = dict(GOOD); del b["frequency"]
    assert post(c, b).status_code == 422
    assert post(c, [1, 2]).status_code == 422


def test_null_temperature_preserved_zero_is_not_null(env):
    c, ing = env
    assert post(c, {**GOOD, "temperature": None}, {**H, "Idempotency-Key": "null-temp-1"}).status_code == 201
    assert post(c, {**GOOD, "temperature": 0}, {**H, "Idempotency-Key": "zero-temp-1"}).status_code == 201
    t = [r["temperature"] for r in rows(ing)]
    assert t == [None, 0.0]


def feed(ing, n, start=None, step=4.0, **over):
    t0 = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    res = None
    for k in range(n):
        d = {**GOOD, **over}
        res = ing.decide(k, d, (t0 + timedelta(seconds=k * step)).timestamp())
    return res, t0 + timedelta(seconds=n * step)


def test_insufficient_then_ok(env):
    _, ing = env
    assert feed(ing, 5)[0]["state"] == "insufficient_data"
    ing.window.clear(); ing.last_fed = None
    r, _ = feed(ing, I.W + 1)
    assert r["state"] == "ok" and r["label"] in ing.model.classes_ and r["window"] == I.W + 1


def test_decimation_of_2s_telemetry(env):
    _, ing = env
    feed(ing, 10, step=2.0)
    assert len(ing.window) == 5          # every >=4 s sample only


def test_gap_resets_window(env):
    _, ing = env
    _, nxt = feed(ing, I.W + 1)
    r = ing.decide(0, GOOD, (nxt + timedelta(seconds=60)).timestamp())
    assert r["state"] == "insufficient_data" and len(ing.window) == 1


def test_sensor_error_and_off_are_not_faults(env):
    _, ing = env
    assert ing.decide(0, {**GOOD, "temperature": None}, 1000.0)["state"] == "sensor_error"
    assert len(ing.window) == 0
    assert ing.decide(0, {**GOOD, "voltage": 0, "current": 0}, 1010.0)["state"] == "off"


def test_model_outage_does_not_break_ingestion(env):
    c, ing = env
    ing.model = None
    feed(ing, I.W)
    assert ing.decide(0, GOOD, 1e9)["state"] in ("insufficient_data", "inference_error")
    assert post(c, headers={**H, "Idempotency-Key": "model-down-1"}).status_code == 201


def test_inference_exception_recorded(env):
    c, ing = env
    rid = c.post("/api/telemetry", json=GOOD, headers={**H, "Idempotency-Key": "infer-exc-1"}).json["id"]
    ing.decide = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    ing.process(rid, GOOD, datetime.now(timezone.utc).isoformat())
    assert rows(ing, "predictions")[0]["state"] == "inference_error" and ing.c["inference_errors"] == 1
    assert c.get("/api/readings/latest").json["prediction"] is None


def test_database_outage_returns_503_and_recovers(env, monkeypatch):
    c, ing = env
    good = ing.db_path
    ing.db_path = "/nonexistent-dir/x.db"
    r = post(c, headers={**H, "Idempotency-Key": "db-down-0001"})
    assert r.status_code == 503 and ing.c["db_errors"] == 1
    assert c.get("/api/status").json["database_ok"] is False
    ing.db_path = good
    assert post(c, headers={**H, "Idempotency-Key": "db-down-0001"}).status_code == 201


def test_latest_and_status(env):
    c, ing = env
    assert c.get("/api/readings/latest").status_code == 503
    r = post(c)
    ing.process(r.json["id"], GOOD, r.json["received_at"])
    j = c.get("/api/readings/latest").get_json()
    assert j["voltage"] == 230.1 and j["device_id"] == "voltguard-esp32-01" and j["inference_state"] == "insufficient_data"
    s = c.get("/api/status").json
    assert s["counters"]["stored"] == 1 and s["database_ok"] and s["inference"]["model_loaded"]


def test_stale_latest(env):
    c, ing = env
    ing.latest = {**GOOD, "timestamp": (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()}
    assert c.get("/api/readings/latest").status_code == 503


def test_queue_full_still_stores(env):
    c, ing = env
    for _ in range(ing.q.maxsize): ing.q.put_nowait(None)
    assert post(c, headers={**H, "Idempotency-Key": "queue-full-01"}).status_code == 201
    assert ing.c["queue_dropped"] == 1 and len(rows(ing)) == 1


def test_sensor_error_result_not_reused_for_next_good_reading(env):
    _, ing = env
    ing.process(1, {**GOOD, "temperature": None}, "2026-01-01T00:00:00+00:00")
    ing.process(2, GOOD, "2026-01-01T00:00:01+00:00")
    assert ing.latest["inference_state"] == "insufficient_data"


# ---- LSTM autoencoder anomaly detector (stub model for logic; real model test below) ----
import numpy as np

class StubAE:
    def __init__(self, err): self.err = err
    def predict(self, x, verbose=0): return x + self.err


def with_ae(ing, err=0.0):
    ing.ae = StubAE(err)
    ing.ae_meta = dict(version="stub", features=["voltage", "current", "power", "powerFactor", "frequency", "temperature"],
                       window=5, step_s=2.0, gap_s=5.0, consecutive=3, mean=[0] * 6, scale=[1] * 6, threshold=0.5)
    ing.ae_window = __import__("collections").deque(maxlen=5)


def test_ae_unavailable_does_not_affect_rf_or_ingestion(env):
    c, ing = env
    assert ing.decide_anomaly(GOOD, 0.0)["state"] == "unavailable"
    assert post(c).status_code == 201


def test_ae_window_gap_streak_and_states(env):
    _, ing = env
    with_ae(ing, 0.0)
    states = [ing.decide_anomaly(GOOD, 2.0 * k)["state"] for k in range(5)]
    assert states == ["insufficient_data"] * 4 + ["ok"]
    assert ing.decide_anomaly(GOOD, 100.0)["state"] == "insufficient_data"      # gap resets window
    assert ing.decide_anomaly({**GOOD, "temperature": None}, 102.0)["state"] == "sensor_error"
    assert ing.decide_anomaly({**GOOD, "voltage": 0, "current": 0}, 104.0)["state"] == "off"
    with_ae(ing, 2.0)                                                            # error 4.0 > 0.5 threshold
    out = [ing.decide_anomaly(GOOD, 2.0 * k) for k in range(8)]
    assert [o.get("alert") for o in out[4:]] == [False, False, True, True]
    assert out[4]["is_anomaly"] is True


def test_ae_duplicate_sample_within_step_not_double_counted(env):
    _, ing = env
    with_ae(ing)
    ing.decide_anomaly(GOOD, 0.0); ing.decide_anomaly(GOOD, 0.1)
    assert len(ing.ae_window) == 1


def test_ae_result_reaches_latest_and_db(env):
    c, ing = env
    with_ae(ing, 2.0)
    for k in range(6):
        r = c.post("/api/telemetry", json=GOOD, headers={**H, "Idempotency-Key": f"ae-key-{k:04d}"}).json
        ing.process(r["id"], GOOD, (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=2 * k)).isoformat())
    assert rows(ing, "anomalies")[-1]["is_anomaly"] == 1
    ing.latest["timestamp"] = datetime.now(timezone.utc).isoformat()
    assert c.get("/api/readings/latest").json["anomaly"]["state"] == "ok"


def test_real_lstm_model_scores_normal_below_threshold_and_fault_above(env):
    pytest.importorskip("tensorflow")
    _, ing = env
    ing.load_lstm()
    if ing.ae is None: pytest.skip(ing.ae_error)
    m = ing.ae_meta
    normal = dict(voltage=247.0, current=0.033, power=8.4, powerFactor=0.99, frequency=50.0, temperature=30.0, uptime=1)
    for k in range(m["window"]): r = ing.decide_anomaly(normal, 2.0 * k)
    assert r["state"] == "ok" and not r["is_anomaly"]
    ing.reset_ae()
    for k in range(m["window"]): r = ing.decide_anomaly({**normal, "voltage": 180.0, "power": 4.9, "current": 0.026}, 2.0 * k)
    assert r["is_anomaly"]


def test_null_reactive_power_accepted_but_other_nulls_rejected(env):
    c, ing = env
    assert post(c, {**GOOD, "reactivePower": None}, {**H, "Idempotency-Key": "nullreact-01"}).status_code == 201
    assert rows(ing)[0]["reactive_power"] is None
    assert post(c, {**GOOD, "reactivePower": "x"}, {**H, "Idempotency-Key": "badreact-001"}).status_code == 422
    assert post(c, {**GOOD, "apparentPower": None}, {**H, "Idempotency-Key": "nullapp-0001"}).status_code == 422
