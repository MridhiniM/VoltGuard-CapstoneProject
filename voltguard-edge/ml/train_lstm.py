"""LSTM autoencoder anomaly detector trained on NORMAL-ONLY PZEM telemetry (no fault labels exist).

    python train_lstm.py FILE [FILE ...]      # .numbers / .xlsx / .csv exports of the PZEM logger

It learns what normal 60 s windows (30 samples at 2 s) look like; a large reconstruction error means
"unlike the normal data it was trained on". It does NOT name the fault type and does NOT forecast.
Fault detection on synthetic perturbations is a sanity check only, not validation on real faults.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

OUT = Path(__file__).resolve().parent / "lstm_ae"
FEATURES = ["voltage", "current", "power", "powerFactor", "frequency", "temperature"]
# floor on the scale so sensor quantisation (e.g. current has 0.001 A steps) is not blown up by z-scoring
SCALE_FLOOR = np.array([1.0, 0.002, 0.3, 0.02, 0.1, 0.5])
L, STEP_S, GAP_S, BLOCK_S, K_CONSEC, PCT = 30, 2.0, 5.0, 1800, 3, 99.5
COLS = {"Timestamp (IST)": "ts", "Voltage (V)": "voltage", "Current (A)": "current", "Active Power (W)": "power",
        "Power Factor": "powerFactor", "Frequency (Hz)": "frequency", "Temperature (°C)": "temperature"}


def load(path):
    p = Path(path)
    if p.suffix == ".numbers":
        from numbers_parser import Document
        rows = Document(str(p)).sheets[0].tables[0].rows(values_only=True)
        df = pd.DataFrame(rows[1:], columns=rows[0])
    elif p.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    df = df[list(COLS)].rename(columns=COLS)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def clean(df):
    n0 = len(df)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df.dropna()
    df = df[(df.voltage > 0) | (df.current > 0)]      # drop supply-off rows; they are not "normal operation"
    print(f"rows {n0} -> {len(df)} after dedupe/NaN/off removal")
    return df.reset_index(drop=True)


def windows(df, mask_blocks=None):
    """Windows of L consecutive samples that never span a gap >GAP_S or a split block. -> (X, end_index)."""
    t = df.ts.astype("int64").to_numpy() / 1e9
    blk = ((t - t[0]) // BLOCK_S).astype(int)
    cut = np.r_[True, (np.diff(t) > GAP_S) | (np.diff(blk) != 0)]
    seg = np.cumsum(cut)
    v = df[FEATURES].to_numpy(float)
    X, idx = [], []
    for s in np.unique(seg):
        ii = np.where(seg == s)[0]
        if mask_blocks is not None and blk[ii[0]] not in mask_blocks:
            continue
        for a in range(0, len(ii) - L + 1):
            X.append(v[ii[a:a + L]]); idx.append(ii[a + L - 1])
    return np.array(X), np.array(idx), blk


def build():
    import tensorflow as tf
    from tensorflow.keras import layers as ly
    m = tf.keras.Sequential([ly.Input((L, len(FEATURES))), ly.LSTM(32), ly.RepeatVector(L),
                             ly.LSTM(32, return_sequences=True), ly.TimeDistributed(ly.Dense(len(FEATURES)))])
    m.compile("adam", "mse")
    return m


def scores(m, Xs):
    return np.mean((m.predict(Xs, verbose=0, batch_size=512) - Xs) ** 2, axis=(1, 2))


def fit(Xtr, Xva, mean, scale):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    m = build()
    f = lambda X: (X - mean) / scale
    m.fit(f(Xtr), f(Xtr), validation_data=(f(Xva), f(Xva)), epochs=60, batch_size=128, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)])
    return m, f


def consec(flags, k):
    run = np.convolve(flags.astype(int), np.ones(k, int), "valid") == k
    return float(run.mean()) if len(run) else float("nan")


def synth(X):
    """Synthetic perturbations of whole windows (resistive-load physics for sag/swell)."""
    rng, out, n = np.random.default_rng(0), {}, len(X)
    def mk(f):
        Y = X.copy(); f(Y); return Y
    def sag(Y): Y[:, :, 0] *= .78; Y[:, :, 1] *= .78; Y[:, :, 2] *= .78 ** 2
    def swell(Y): Y[:, :, 0] *= 1.12; Y[:, :, 1] *= 1.12; Y[:, :, 2] *= 1.25
    def over(Y): Y[:, :, 1] *= 20; Y[:, :, 2] = Y[:, :, 0] * Y[:, :, 1] * Y[:, :, 3]
    def therm(Y): Y[:, :, 5] += np.linspace(0, 12, L)
    def tstep(Y): Y[:, L // 2:, 5] += 8
    def unstable(Y):
        Y[:, :, 0] += 12 * np.sin(np.arange(L) * 2.7); Y[:, :, 1] *= 1 + .3 * rng.uniform(-1, 1, (n, L))
        Y[:, :, 2] = Y[:, :, 0] * Y[:, :, 1] * Y[:, :, 3]
    for k, f in dict(voltage_sag_22pct=sag, voltage_swell_12pct=swell, overcurrent_20x=over,
                     temp_ramp_12C=therm, temp_step_8C=tstep, instability=unstable).items():
        out[k] = mk(f)
    return out


def main(paths):
    df = clean(pd.concat([load(p) for p in paths], ignore_index=True))
    X, idx, blk = windows(df)
    ub = np.unique(blk); pos = {b: i for i, b in enumerate(ub)}
    part = lambda b: "train" if pos[b] % 20 < 14 else "val" if pos[b] % 20 < 17 else "test"
    sets = {k: [b for b in ub if part(b) == k] for k in ("train", "val", "test")}
    W = {k: windows(df, set(v))[0] for k, v in sets.items()}
    print({k: len(v) for k, v in W.items()}, "windows (30-min block split, no window crosses a block)")
    rows = np.vstack([df[FEATURES].to_numpy(float)[np.isin(blk, sets["train"])]])
    mean, std = rows.mean(0), rows.std(0)
    scale = np.maximum(std, SCALE_FLOOR)
    m, f = fit(W["train"], W["val"], mean, scale)
    sv = scores(m, f(W["val"])); thr = float(np.percentile(sv, PCT))
    st = scores(m, f(W["test"]))
    rep = dict(rows=len(df), span=[str(df.ts.min()), str(df.ts.max())], windows={k: len(v) for k, v in W.items()},
               threshold=thr, threshold_rule=f"p{PCT} of validation window MSE (scaled space)",
               test_false_alarm_single_window=float((st > thr).mean()),
               test_false_alarm_3_consecutive=consec(st > thr, K_CONSEC),
               test_score_percentiles={p: float(np.percentile(st, p)) for p in (50, 95, 99, 100)})
    syn = {}
    for k, Y in synth(W["test"]).items():
        s = scores(m, f(Y)); syn[k] = dict(detected_single_window=float((s > thr).mean()),
                                           median_score_over_threshold=float(np.median(s) / thr))
    rep["synthetic_fault_detection_SANITY_CHECK_ONLY"] = syn
    # forward test: train on the first 80% in time, test on the last 20% (does drift cause false alarms?)
    cut = df.ts.iloc[int(len(df) * .8)]
    a, b = df[df.ts < cut].reset_index(drop=True), df[df.ts >= cut].reset_index(drop=True)
    Xa, Xb = windows(a)[0], windows(b)[0]
    nv = int(len(Xa) * .85); mu, sd = a[FEATURES].to_numpy(float).mean(0), a[FEATURES].to_numpy(float).std(0)
    sc2 = np.maximum(sd, SCALE_FLOOR); m2, f2 = fit(Xa[:nv - L], Xa[nv:], mu, sc2)
    t2 = np.percentile(scores(m2, f2(Xa[nv:])), PCT); sb = scores(m2, f2(Xb))
    rep["forward_test_last_20pct_in_time"] = dict(false_alarm_single_window=float((sb > t2).mean()),
                                                  false_alarm_3_consecutive=consec(sb > t2, K_CONSEC),
                                                  train_voltage_mean=float(mu[0]), test_voltage_mean=float(b.voltage.mean()))
    OUT.mkdir(exist_ok=True)
    m.save(OUT / "model.keras")
    ver = "LSTM-AE-" + hashlib.sha256((OUT / "model.keras").read_bytes()).hexdigest()[:12]
    json.dump(dict(version=ver, features=FEATURES, window=L, step_s=STEP_S, gap_s=GAP_S, consecutive=K_CONSEC,
                   mean=mean.tolist(), scale=scale.tolist(), threshold=thr, report=rep),
              open(OUT / "meta.json", "w"), indent=1)
    json.dump(rep, open(OUT / "metrics_lstm.json", "w"), indent=1)
    print(json.dumps(rep, indent=1)); print("saved", OUT, ver)


if __name__ == "__main__":
    main(sys.argv[1:])
