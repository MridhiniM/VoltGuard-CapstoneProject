"""Train + evaluate the VoltGuard ML models on the real PZEM log and Open-Meteo history.

    python prepare_data.py && python train_ml.py

Models (artifacts in models/, report in reports/):
  A  Unsupervised anomaly detector: Isolation Forest vs robust covariance (Mahalanobis), chosen on validation.
  B1 30-minute equipment-temperature nowcast (electrical-only and electrical+weather variants).
  B2 Equipment-temperature level vs ambient (used only by the 7-day outlook, if it beats baseline).
  C  Supervised fault prediction: NOT trained - the real data has no fault labels (see report).

All splits are chronological (70/15/15 of the time span). Preprocessing is fitted on training data only.
"""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

import joblib, numpy as np, pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ml_features as F

ROOT = Path(__file__).resolve().parent
PROC, MODELS, REPORTS = ROOT / "data" / "processed", ROOT / "models", ROOT / "reports"
SEED = 42
FPR_TARGET = 99.5          # anomaly threshold = this percentile of validation scores
BLOCK_S, PURGE_S = 1800, 300


def mae(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def split_times(ts):
    t0, t1 = ts.min(), ts.max()
    return t0 + 0.70 * (t1 - t0), t0 + 0.85 * (t1 - t0)


# ------------------------------------------------------------------ A: anomaly detection
INJECTIONS = {
    "voltage_sag": lambda b, r: b.assign(voltage=b.voltage * 0.80, power=b.power * 0.64),
    "voltage_swell": lambda b, r: b.assign(voltage=b.voltage * 1.10, power=b.power * 1.21),
    "overcurrent": lambda b, r: b.assign(current=b.current * 3, power=b.power * 3),
    "thermal_runaway": lambda b, r: b.assign(temperature=b.temperature + np.linspace(0, 8, len(b))),
    "unstable_load": lambda b, r: b.assign(current=b.current * (1 + r.uniform(-0.4, 0.4, len(b))),
                                           power=b.power * (1 + r.uniform(-0.4, 0.4, len(b)))),
}


def block_parts(ts):
    """Interleaved 30-min blocks (5 train : 1 val : 1 test) so every split spans the whole day,
    with the first 5 min of each val/test block purged (rolling features look back into the previous block)."""
    sec = (ts - ts.min()).dt.total_seconds()
    blk, into = (sec // BLOCK_S).astype(int), sec % BLOCK_S
    m = blk % 7
    part = np.where(m == 5, "val", np.where(m == 6, "test", "train"))
    part[(part != "train") & (into < PURGE_S)] = "purged"
    return part


def anomaly_candidates():
    return {"isolation_forest": lambda seed: make_pipeline(StandardScaler(), IsolationForest(n_estimators=300, max_samples=256, random_state=seed)),
            "robust_covariance": lambda seed: make_pipeline(StandardScaler(), EllipticEnvelope(support_fraction=0.99, random_state=seed))}


def inject_eval(model, thr, raw, rng, n_blocks=20, zflag=None):
    """Detection on artificial perturbations of normal data (sanity check, NOT real faults)."""
    out = {}
    for name, fn in INJECTIONS.items():
        hits, rows, zhits = 0, [], 0
        for _ in range(n_blocks):
            a = int(rng.integers(200, len(raw) - 120))
            b = raw.copy()
            blk = slice(a, a + 60)                         # ~2 minutes at 2 s
            cols = ["voltage", "current", "power", "temperature"]
            b.loc[b.index[blk], cols] = fn(b.iloc[blk].copy(), rng)[cols].to_numpy()
            fb = F.anomaly_frame(b).iloc[blk]
            f = -model.score_samples(fb[F.ANOMALY_FEATURES].to_numpy()) > thr
            rows.append(float(f.mean())); hits += bool(f.any())
            if zflag is not None: zhits += bool(zflag(fb).any())
        out[name] = {"blocks": n_blocks, "block_detection_rate": hits / n_blocks, "mean_row_detection_rate": float(np.mean(rows))}
        if zflag is not None: out[name]["block_detection_rate_zscore_baseline"] = zhits / n_blocks
    return out


def train_anomaly(d, val_start, test_start):
    fr = F.anomaly_frame(d)
    part = block_parts(fr.timestamp)
    tr, va, te = fr[part == "train"], fr[part == "val"], fr[part == "test"]
    raw_va, raw_te = d[part == "val"].reset_index(drop=True), d[part == "test"].reset_index(drop=True)
    X = lambda f: f[F.ANOMALY_FEATURES].to_numpy()
    score = lambda m, f: -m.score_samples(X(f))          # higher = more unusual
    ch = ["voltage", "current", "power", "temperature"]
    mu, sd_ = tr[ch].mean(), tr[ch].std()
    zflag = lambda f: (((f[ch] - mu) / sd_).abs() > 4).any(axis=1).to_numpy()

    cands = {}
    for name, mk in anomaly_candidates().items():
        m = mk(SEED).fit(X(tr))
        thr = float(np.percentile(score(m, va), FPR_TARGET))
        sel = inject_eval(m, thr, raw_va, np.random.default_rng(SEED))            # selection uses validation only
        s_te = score(m, te)
        # Forward-in-time diagnostic: fit on the earliest 70 %, threshold on the next 15 %, test on the last 15 %.
        fw_tr, fw_va, fw_te = fr[fr.timestamp < val_start], fr[(fr.timestamp >= val_start) & (fr.timestamp < test_start)], fr[fr.timestamp >= test_start]
        fm = mk(SEED).fit(X(fw_tr)); fthr = float(np.percentile(score(fm, fw_va), FPR_TARGET))
        cands[name] = {"model": m, "thr": thr,
                       "val_selection_score": float(np.mean([v["mean_row_detection_rate"] for v in sel.values()])),
                       "false_alarm_rate": {"train": float(np.mean(score(m, tr) > thr)), "val": float(np.mean(score(m, va) > thr)), "test": float(np.mean(s_te > thr))},
                       "forward_in_time_test_false_alarm_rate": float(np.mean(score(fm, fw_te) > fthr)),
                       "test_injections": inject_eval(m, thr, raw_te, np.random.default_rng(SEED + 1), zflag=zflag),
                       "score_percentiles_test": {p: round(float(np.percentile(s_te, p)), 4) for p in (50, 90, 99, 99.9)}}
    best = max(cands, key=lambda k: cands[k]["val_selection_score"])
    model, thr = cands[best]["model"], cands[best]["thr"]
    flags_te = score(model, te) > thr
    agree = []
    for sd in (1, 2, 3, 4):
        m2 = anomaly_candidates()[best](sd).fit(X(tr))
        t2 = float(np.percentile(score(m2, va), FPR_TARGET))
        agree.append(float(np.mean((score(m2, te) > t2) == flags_te)))
    report = {
        "selected": best, "selection_rule": "highest mean row-detection on synthetic injections into VALIDATION data, at the p99.5 validation threshold",
        "features": F.ANOMALY_FEATURES, "split": "interleaved 30-min blocks 5:1:1 (train:val:test), 5-min purge at val/test block starts",
        "rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "threshold": {"value": thr, "rule": f"p{FPR_TARGET} of validation scores (target ~0.5% false alarms on normal data)"},
        "candidates": {k: {kk: v[kk] for kk in v if kk not in ("model", "thr")} | {"threshold": v["thr"]} for k, v in cands.items()},
        "zscore_baseline_false_alarm_rate": {"val": float(np.mean(zflag(va))), "test": float(np.mean(zflag(te)))},
        "seed_stability_test_flag_agreement": agree,
        "notes": ["No real fault labels exist, so precision/recall on real faults cannot be computed.",
                  "Injection results are a sanity check with artificial perturbations, not validation.",
                  "Forward-in-time false alarms are high because mains voltage drifts below the earlier range (to ~234 V) late in the log: the detector only knows the conditions it has seen.",
                  "Anomaly scores are relative unusualness, not probabilities of failure."],
    }
    stats = {"mean": tr[F.ANOMALY_FEATURES].mean().to_dict(), "std": tr[F.ANOMALY_FEATURES].std().replace(0, 1e-9).to_dict(),
             "min": tr[F.ANOMALY_FEATURES].min().to_dict(), "max": tr[F.ANOMALY_FEATURES].max().to_dict()}
    return model, thr, stats, report


# ------------------------------------------------------------------ B1: 30-min temperature nowcast
def candidates():
    return {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "random_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
            "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED)}


def train_nowcast(d, w, val_start, test_start):
    mins = F.minute_frame(d)
    fr = F.nowcast_frame(mins, w).dropna()
    h = pd.Timedelta(minutes=F.HORIZON_MIN)
    # Purge rows whose target lies in the next split (prevents overlap leakage).
    tr = fr[fr.index + h < val_start]
    va = fr[(fr.index >= val_start) & (fr.index + h < test_start)]
    te = fr[fr.index >= test_start]
    tv = fr[fr.index + h < test_start]
    res, best = {}, {}
    persist = {"val": {"mae": mae(va.target, va.temp_now), "rmse": rmse(va.target, va.temp_now)},
               "test": {"mae": mae(te.target, te.temp_now), "rmse": rmse(te.target, te.temp_now)}}
    for variant, feats in (("electrical", F.NOWCAST_ELEC), ("electrical+weather", F.NOWCAST_ELEC + F.NOWCAST_WEATHER)):
        for name, mk in candidates().items():
            m = mk().fit(tr[feats], tr.target - tr.temp_now)
            pv = va.temp_now + m.predict(va[feats])
            final = mk().fit(tv[feats], tv.target - tv.temp_now)          # refit on train+val, then test once
            pt = te.temp_now + final.predict(te[feats])
            key = f"{variant}:{name}"
            res[key] = {"val": {"mae": mae(va.target, pv), "rmse": rmse(va.target, pv)},
                        "test": {"mae": mae(te.target, pt), "rmse": rmse(te.target, pt)}}
            if variant not in best or res[key]["val"]["mae"] < res[best[variant][0]]["val"]["mae"]:
                best[variant] = (key, final, feats)
    deploy = {}
    for variant, (key, model, feats) in best.items():
        deploy[variant] = {"model": model, "features": feats, "key": key, "test": res[key]["test"],
                           "beats_persistence_on_test": res[key]["test"]["mae"] < persist["test"]["mae"]}
    report = {"target": f"equipment temperature (1-min mean) {F.HORIZON_MIN} min ahead", "history_needed_min": F.HIST_MIN,
              "rows": {"train": len(tr), "val": len(va), "test": len(te)},
              "baseline_persistence": persist, "candidates": res,
              "selected_by_validation_mae": {v: d_["key"] for v, d_ in deploy.items()},
              "selected_test": {v: {"test": d_["test"], "beats_persistence_on_test": d_["beats_persistence_on_test"]} for v, d_ in deploy.items()},
              "weather_helps_on_test": deploy["electrical+weather"]["test"]["mae"] < deploy["electrical"]["test"]["mae"],
              "notes": ["Training uses Open-Meteo reanalysis as a stand-in for the forecast that would be available live.",
                        "Only ~15.5 h of data (one partial day); the test period is midday, warmer than most training data.",
                        "Improvements over persistence are small and come from one ~2-hour test window; a difference of a few hundredths of a degree between the electrical and weather variants is not evidence that weather helps."]}
    return deploy, report


# ------------------------------------------------------------------ B2: equipment temperature vs ambient
def train_steady(d, w, val_start, test_start):
    tenmin = F.minute_frame(d).resample("10min").mean().dropna()
    amb = F.steady_frame(w)
    a10 = amb.reindex(amb.index.union(tenmin.index)).interpolate(method="time", limit_area="inside").reindex(tenmin.index)
    fr = tenmin.join(a10).dropna()
    tr, va, te = fr[fr.index < val_start], fr[(fr.index >= val_start) & (fr.index < test_start)], fr[fr.index >= test_start]
    tv = fr[fr.index < test_start]
    res = {"baseline_train_mean": {}}
    for part, x in (("val", va), ("test", te)):
        ref = (tr if part == "val" else tv).temperature.mean()
        res["baseline_train_mean"][part] = {"mae": mae(x.temperature, np.full(len(x), ref)), "rmse": rmse(x.temperature, np.full(len(x), ref))}
    models = {}
    for key, feats in (("linear:amb_3h", ["amb_3h"]), ("linear:amb_3h+power", F.STEADY_FEATURES)):
        m = LinearRegression().fit(tr[feats], tr.temperature)
        final = LinearRegression().fit(tv[feats], tv.temperature)
        res[key] = {"val": {"mae": mae(va.temperature, m.predict(va[feats])), "rmse": rmse(va.temperature, m.predict(va[feats]))},
                    "test": {"mae": mae(te.temperature, final.predict(te[feats])), "rmse": rmse(te.temperature, final.predict(te[feats]))},
                    "coefficients": dict(zip(feats, map(float, final.coef_))), "intercept": float(final.intercept_)}
        models[key] = (final, feats)
    best = min(models, key=lambda k: res[k]["val"]["mae"])
    beats = res[best]["test"]["mae"] < res["baseline_train_mean"]["test"]["mae"]
    model, feats = models[best]
    report = {"target": "equipment temperature (10-min mean) as a function of 3-h mean ambient temperature and load power",
              "rows": {"train": len(tr), "val": len(va), "test": len(te)}, "results": res, "selected": best,
              "beats_baseline_on_test": beats, "deployed_for_outlook": beats,
              "training_ambient_range_c": [float(fr.amb_3h.min()), float(fr.amb_3h.max())],
              "training_equipment_temp_range_c": [float(fr.temperature.min()), float(fr.temperature.max())],
              "val_beats_baseline": res[best]["val"]["mae"] < res["baseline_train_mean"]["val"]["mae"],
              "notes": ["One diurnal cycle only: ambient is confounded with time-of-day and any slow drift.",
                        "The train-only fit does not beat the mean baseline on the validation period; it beats it on test only after refitting on train+val. Treat outlook temperatures as low-confidence.",
                        "Extrapolation beyond the training ambient range is flagged in the outlook."]}
    return {"model": model, "features": feats, "key": best, "deployable": beats,
            "amb_range": report["training_ambient_range_c"], "temp_range": report["training_equipment_temp_range_c"],
            "temp_p99": float(np.percentile(fr.temperature, 99)), "median_power_w": float(tv.power.median())}, report


def md_report(rep):
    A, B, S = rep["anomaly_detector"], rep["temperature_nowcast"], rep["temperature_vs_ambient"]
    f3 = lambda x: f"{x:.3f}"
    lines = [f"# VoltGuard ML evaluation report", f"Generated {rep['generated_at']} by `train_ml.py`.", "",
             "## Data", f"- Electrical: {rep['data']['electrical_dataset']['rows_clean']} cleaned readings from "
             f"{', '.join(x['file'] for x in rep['data']['electrical_dataset']['files'])} "
             f"({rep['data']['electrical_dataset']['start']} → {rep['data']['electrical_dataset']['end']}, IST, ~2 s).",
             f"- Weather: {rep['data']['weather_dataset']['source']}, {rep['data']['weather_dataset']['location']['name']}, hourly.",
             f"- Splits (chronological): train < {rep['splits']['val_start']} ≤ val < {rep['splits']['test_start']} ≤ test.", "",
             "## A. Anomaly detector (unsupervised)", f"Features: {', '.join(A['features'])}. Split: {A['split']}.",
             f"Selected: **{A['selected']}** ({A['selection_rule']}). Threshold {f3(A['threshold']['value'])} ({A['threshold']['rule']}).", "",
             "| Normal data: false-alarm rate | " + " | ".join(A["candidates"]) + " | z>4 baseline |", "|---|" + "---|" * (len(A["candidates"]) + 1),
             "| validation | " + " | ".join(f"{v['false_alarm_rate']['val']:.2%}" for v in A["candidates"].values()) + f" | {A['zscore_baseline_false_alarm_rate']['val']:.2%} |",
             "| test | " + " | ".join(f"{v['false_alarm_rate']['test']:.2%}" for v in A["candidates"].values()) + f" | {A['zscore_baseline_false_alarm_rate']['test']:.2%} |",
             "| forward-in-time test (fit on earliest 70 %) | " + " | ".join(f"{v['forward_in_time_test_false_alarm_rate']:.2%}" for v in A["candidates"].values()) + " | – |", "",
             "Synthetic injections into held-out test data (20 two-minute blocks each, rows flagged / blocks detected; **sanity check, not real faults**):", "",
             "| Injection | " + " | ".join(A["candidates"]) + " | z>4 baseline blocks |", "|---|" + "---|" * (len(A["candidates"]) + 1)]
    first = next(iter(A["candidates"].values()))["test_injections"]
    for k in first:
        lines.append(f"| {k} | " + " | ".join(f"{v['test_injections'][k]['mean_row_detection_rate']:.0%} / {v['test_injections'][k]['block_detection_rate']:.0%}" for v in A["candidates"].values())
                     + f" | {first[k]['block_detection_rate_zscore_baseline']:.0%} |")
    lines += ["", f"Seed stability (test-flag agreement, 4 other seeds): {', '.join(f'{x:.3f}' for x in A['seed_stability_test_flag_agreement'])}.", "",
              f"## B1. {B['target']}", "", "| Model | val MAE | val RMSE | test MAE | test RMSE |", "|---|---|---|---|---|",
              f"| persistence baseline | {f3(B['baseline_persistence']['val']['mae'])} | {f3(B['baseline_persistence']['val']['rmse'])} | {f3(B['baseline_persistence']['test']['mae'])} | {f3(B['baseline_persistence']['test']['rmse'])} |"]
    for k, v in B["candidates"].items():
        lines.append(f"| {k} | {f3(v['val']['mae'])} | {f3(v['val']['rmse'])} | {f3(v['test']['mae'])} | {f3(v['test']['rmse'])} |")
    lines += ["", f"Selected by validation MAE: {B['selected_by_validation_mae']}. Weather improves selected test MAE: **{B['weather_helps_on_test']}**.",
              f"Selected models beat persistence on test: " + ", ".join(f"{k}: **{v['beats_persistence_on_test']}**" for k, v in B["selected_test"].items()) + ".", "",
              f"## B2. {S['target']}", "", "| Model | val MAE | test MAE | test RMSE |", "|---|---|---|---|"]
    for k, v in S["results"].items():
        lines.append(f"| {k} | {f3(v['val']['mae'])} | {f3(v['test']['mae'])} | {f3(v['test']['rmse'])} |")
    lines += ["", f"Selected: `{S['selected']}`; beats baseline on validation: **{S['val_beats_baseline']}**, on test: **{S['beats_baseline_on_test']}** → used by 7-day outlook: **{S['deployed_for_outlook']}**.",
              *[f"- {n}" for n in S["notes"]], *[f"- B1: {n}" for n in B["notes"]],
              f"Training ambient range {S['training_ambient_range_c']} °C; equipment {S['training_equipment_temp_range_c']} °C.", "",
              "## C. Supervised fault prediction", rep["fault_prediction"]["reason"], "",
              "## Limitations", *[f"- {x}" for x in rep["limitations"]], "",
              "## Deployment suitability", rep["deployment_suitability"], ""]
    return "\n".join(lines)


def main():
    d = pd.read_csv(PROC / "pzem_readings.csv", parse_dates=["timestamp"])
    w = pd.read_csv(PROC / "weather_hourly.csv", parse_dates=["timestamp"]).set_index("timestamp")
    manifest = json.loads((PROC / "manifest.json").read_text(encoding="utf-8"))
    val_start, test_start = split_times(d.timestamp)
    print("splits", val_start, test_start)
    an_model, thr, stats, an_rep = train_anomaly(d, val_start, test_start)
    print("anomaly done")
    nc, nc_rep = train_nowcast(d, w, val_start, test_start)
    print("nowcast done")
    st, st_rep = train_steady(d, w, val_start, test_start)
    data_hash = hashlib.sha256((PROC / "pzem_readings.csv").read_bytes()).hexdigest()[:12]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    common = {"trained_at": now, "data_sha256_12": data_hash, "train_end": str(test_start), "location": manifest["weather_dataset"]["location"]}
    joblib.dump({**common, "version": f"ANOM-{data_hash}", "algorithm": an_rep["selected"], "model": an_model, "features": F.ANOMALY_FEATURES, "threshold": thr, "train_stats": stats}, MODELS / "anomaly_detector.joblib")
    joblib.dump({**common, "version": f"NOW-{data_hash}", "horizon_min": F.HORIZON_MIN, "history_min": F.HIST_MIN,
                 "variants": {k: {kk: v[kk] for kk in ("model", "features", "key", "test", "beats_persistence_on_test")} for k, v in nc.items()}}, MODELS / "temp_nowcast.joblib")
    joblib.dump({**common, "version": f"AMB-{data_hash}", **st}, MODELS / "temp_vs_ambient.joblib")
    rep = {"generated_at": now, "data": manifest, "splits": {"val_start": str(val_start), "test_start": str(test_start)},
           "anomaly_detector": an_rep, "temperature_nowcast": nc_rep, "temperature_vs_ambient": st_rep,
           "fault_prediction": {"trained": False, "reason": "Not possible: neither the real PZEM logs nor any accessible matching public dataset contain confirmed, timestamped fault events for this equipment. The repository's `dataset.csv` labels are synthetic and derived from injected patterns, so a 'fault within 24 h' classifier trained on them would not be evidence of real fault prediction. Required: logged real faults (type + timestamp) from the monitored equipment over weeks/months."},
           "limitations": ["Single ~8 W resistive load, one site, ~15.5 hours (one partial diurnal cycle).",
                           "No real faults observed; anomaly detection is validated only on normal data plus synthetic injections.",
                           "Weather is reanalysis at city scale, not an indoor sensor at the equipment.",
                           "Future electrical load is unknown; the outlook assumes the median observed load operates continuously."],
           "deployment_suitability": "Demonstration / advisory only. Suitable for flagging readings that differ from the recorded normal behaviour of this one load. Not a protective device and not validated for other loads, seasons or sites; retrain when more data (especially real faults) is logged."}
    (REPORTS / "evaluation.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (REPORTS / "EVALUATION.md").write_text(md_report(rep), encoding="utf-8")
    print(md_report(rep))


if __name__ == "__main__":
    main()
