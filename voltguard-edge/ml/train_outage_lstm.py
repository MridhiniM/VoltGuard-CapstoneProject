"""Train the weather -> power-outage LSTM on real Bengaluru data (ESMI outages + Open-Meteo weather).

    python prepare_outage.py && python train_outage_lstm.py

Target: probability that a power outage (>= 5 min at 0 V) STARTS in hour t at a monitored Bengaluru
household, given the 24 hourly weather observations up to and including hour t plus calendar features.
Chronological split: train < 2017-07-01 <= val < 2018-01-01 <= test (2018, never used for fitting or tuning).
Baselines: hour x month climatology, logistic regression on the current hour, and the same LSTM
without weather (calendar only) - the last one measures what weather actually adds.
"""
import json, math, os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn

import outage_features as OF

ROOT = Path(__file__).resolve().parent
PROC, MODELS, REPORTS = ROOT / "data" / "processed", ROOT / "models", ROOT / "reports"
VAL_START = pd.Timestamp(os.environ.get("VOLTGUARD_OUTAGE_VAL_START", "2017-07-01"))
TEST_START = pd.Timestamp(os.environ.get("VOLTGUARD_OUTAGE_TEST_START", "2018-01-01"))
SEED = 7


class OutageLSTM(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def expand(p, onsets, at_risk):
    """City-hour aggregates -> site-level samples via weights (positives = onsets, negatives = the rest)."""
    y = np.r_[np.ones(len(p)), np.zeros(len(p))]
    w = np.r_[onsets, at_risk - onsets].astype(float)
    pp = np.r_[p, p]
    m = w > 0
    return y[m], pp[m], w[m]


def metrics(p, df):
    y, pp, w = expand(np.clip(p, 1e-6, 1 - 1e-6), df.onsets.to_numpy(), df.at_risk.to_numpy())
    return {"log_loss": float(log_loss(y, pp, sample_weight=w)), "brier": float(brier_score_loss(y, pp, sample_weight=w)),
            "roc_auc": float(roc_auc_score(y, pp, sample_weight=w)), "pr_auc": float(average_precision_score(y, pp, sample_weight=w)),
            "base_rate": float(df.onsets.sum() / df.at_risk.sum())}


def fit_lstm(Xtr, ytr, wtr, Xva, yva, wva, epochs=40, patience=6):
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = OutageLSTM(Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    Xtr_, ytr_, wtr_, Xva_, yva_, wva_ = map(T, (Xtr, ytr, wtr, Xva, yva, wva))
    best, best_state, bad = math.inf, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr_))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = (bce(m(Xtr_[b]), ytr_[b]) * wtr_[b]).sum() / wtr_[b].sum()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float((bce(m(Xva_), yva_) * wva_).sum() / wva_.sum())
        if vl < best - 1e-5:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, best, ep + 1


def logits(m, X):
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32)).numpy()


def main():
    h = pd.read_csv(PROC / "outage_hourly.csv", parse_dates=["hour"]).set_index("hour")
    feats = OF.feature_frame(h)
    tr_mask = (h.index < VAL_START)
    scaler = OF.fit_scaler(feats[tr_mask])
    Z = OF.scale(feats, scaler)
    idx, X = OF.windows(Z, h.index[h.at_risk > 0])
    lab = h.loc[idx]
    y = (lab.onsets / lab.at_risk).to_numpy(); w = lab.at_risk.to_numpy(dtype=float)
    split = np.where(idx < VAL_START, "train", np.where(idx < TEST_START, "val", "test"))
    S = {k: split == k for k in ("train", "val", "test")}
    res = {}

    # Baseline 1: climatology (hour of day x month) from training data.
    tr = lab[S["train"]]
    clim = tr.groupby([tr.index.hour, tr.index.month])[["onsets", "at_risk"]].sum()
    prior = tr.onsets.sum() / tr.at_risk.sum()
    clim_rate = ((clim.onsets + 20 * prior) / (clim.at_risk + 20)).to_dict()            # shrunk toward the mean
    p_clim = np.array([clim_rate.get((t.hour, t.month), prior) for t in idx])
    for k in ("val", "test"): res.setdefault(k, {})["climatology (hour x month)"] = metrics(p_clim[S[k]], lab[S[k]])

    # Baseline 2: logistic regression on the current hour only (weather + calendar).
    lr = LogisticRegression(max_iter=2000, C=1.0)
    yy, _, ww = expand(np.zeros(S["train"].sum()), lab.onsets[S["train"]].to_numpy(), lab.at_risk[S["train"]].to_numpy())
    Xcur = X[:, -1, :]
    Xl = np.r_[Xcur[S["train"]], Xcur[S["train"]]][np.r_[lab.onsets[S["train"]].to_numpy(), (lab.at_risk - lab.onsets)[S["train"]].to_numpy()] > 0]
    lr.fit(Xl, yy, sample_weight=ww)
    p_lr = lr.predict_proba(Xcur)[:, 1]
    for k in ("val", "test"): res[k]["logistic regression (current hour)"] = metrics(p_lr[S[k]], lab[S[k]])

    # LSTMs: calendar-only (ablation) and weather + calendar (deployed).
    cal_cols = [i for i, c in enumerate(OF.FEATURES) if c in OF.CALENDAR]
    models = {}
    for name, cols in (("LSTM calendar only (no weather)", cal_cols), ("LSTM weather + calendar", list(range(len(OF.FEATURES))))):
        Xc = X[:, :, cols]
        m, vl, eps = fit_lstm(Xc[S["train"]], y[S["train"]], w[S["train"]], Xc[S["val"]], y[S["val"]], w[S["val"]])
        # Platt calibration on validation (test untouched).
        zv = logits(m, Xc[S["val"]])
        yv, zz, wv = expand(zv, lab.onsets[S["val"]].to_numpy(), lab.at_risk[S["val"]].to_numpy())
        cal = LogisticRegression(C=1e6).fit(zz.reshape(-1, 1), yv, sample_weight=wv)
        p = cal.predict_proba(logits(m, Xc).reshape(-1, 1))[:, 1]
        for k in ("val", "test"): res[k][name] = metrics(p[S[k]], lab[S[k]])
        res.setdefault("training", {})[name] = {"epochs": eps, "best_val_weighted_bce": vl}
        models[name] = (m, cal, cols, p)

    # Risk bands: absolute probability thresholds chosen on VALIDATION (top 10% / top 3% of val hours),
    # then checked on the untouched test year: do flagged hours really have more outages?
    m, cal, cols, p = models["LSTM weather + calendar"]
    pc = models["LSTM calendar only (no weather)"][3]
    va, te = S["val"], S["test"]
    q_el, q_hi = float(np.quantile(p[va], 0.90)), float(np.quantile(p[va], 0.97))
    bands = {"thresholds": {"elevated": q_el, "high": q_hi}}
    for band, lo, hi in (("normal", 0, q_el), ("elevated", q_el, q_hi), ("high", q_hi, 1.01)):
        for k, sel0 in (("val", va), ("test", te)):
            sel = sel0 & (p >= lo) & (p < hi)
            bands.setdefault(band, {})[k] = {"hours": int(sel.sum()), "observed_onset_rate": float(lab.onsets[sel].sum() / max(1, lab.at_risk[sel].sum())),
                                             "model_mean_probability": float(np.average(p[sel], weights=w[sel])) if sel.any() else None}
    bands["all_test_hours_rate"] = float(lab.onsets[te].sum() / lab.at_risk[te].sum())
    # The deployed band: everything >= the elevated threshold (one level), on the test year.
    fl, nf = te & (p >= q_el), te & (p < q_el)
    r_fl, r_nf = float(lab.onsets[fl].sum() / max(1, lab.at_risk[fl].sum())), float(lab.onsets[nf].sum() / max(1, lab.at_risk[nf].sum()))
    yt, pt, wt = expand(p[te], lab.onsets[te].to_numpy(), lab.at_risk[te].to_numpy())
    acc = lambda thr: float(np.average((pt >= thr) == (yt == 1), weights=wt))
    bands["flagged_test"] = {"hours": int(fl.sum()), "share_of_hours": float(fl.sum() / te.sum()), "observed_onset_rate": r_fl,
                             "unflagged_onset_rate": r_nf, "lift": r_fl / r_nf if r_nf else None,
                             "share_of_outages_in_flagged_hours": float(lab.onsets[fl].sum() / lab.onsets[te].sum())}
    bands["accuracy_test"] = {"model_at_0.5": acc(0.5), "model_at_elevated_threshold": acc(q_el),
                              "always_no_outage_baseline": float(1 - lab.onsets[te].sum() / lab.at_risk[te].sum())}
    # Weather-driven hours: the weather model rates the hour >= 25% riskier than the calendar-only model.
    wd = te & (p >= 1.25 * pc)
    bands["weather_driven_test"] = {"hours": int(wd.sum()), "observed_onset_rate": float(lab.onsets[wd].sum() / max(1, lab.at_risk[wd].sum())),
                                    "calendar_model_expected": float(np.average(pc[wd], weights=w[wd])) if wd.any() else None,
                                    "weather_model_expected": float(np.average(p[wd], weights=w[wd])) if wd.any() else None}

    # Data-only check of the rain effect (no model): onset rate by current-hour rain intensity, all labelled hours.
    rain = lab["precipitation"]
    rain_tab = {}
    for nm, lo, hi in (("dry (0 mm)", -1, 0.05), ("light (0.1-2.5)", 0.05, 2.5), ("moderate (2.5-7.6)", 2.5, 7.6), ("heavy (>7.6)", 7.6, 1e9)):
        sel = (rain > lo) & (rain <= hi)
        rain_tab[nm] = {"hours": int(sel.sum()), "onset_rate": float(lab.onsets[sel].sum() / max(1, lab.at_risk[sel].sum()))}

    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    clim_table = {f"{hh}-{mm}": float(v) for (hh, mm), v in clim_rate.items()}
    torch.save({"state_dict": m.state_dict(), "n_in": len(cols), "features": OF.FEATURES, "scaler": scaler,
                "platt": [float(cal.coef_[0, 0]), float(cal.intercept_[0])], "climatology": clim_table, "prior": float(prior),
                "baseline_means": OF.baseline_means(h[tr_mask]),
                "risk_bands": {"elevated": q_el, "high": float("inf")}, "window_h": OF.WINDOW_H,
                "version": "OUT-LSTM-" + datetime.now(timezone.utc).strftime("%Y%m%d"),
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "test_metrics": res["test"]["LSTM weather + calendar"]}, MODELS / "outage_lstm.pt")
    manifest = json.loads((PROC / "outage_manifest.json").read_text(encoding="utf-8"))
    rep = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "data": manifest,
           "splits": {"train": f"< {VAL_START.date()}", "val": f"{VAL_START.date()} .. {TEST_START.date()}", "test": f">= {TEST_START.date()}"},
           "rows": {k: int(S[k].sum()) for k in S}, "site_hours": {k: int(w[S[k]].sum()) for k in S},
           "features": OF.FEATURES, "window_hours": OF.WINDOW_H, "results": res, "test_risk_bands": bands,
           "observed_onset_rate_by_rain": rain_tab}
    (REPORTS / "outage_evaluation.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (REPORTS / "OUTAGE_EVALUATION.md").write_text(md(rep), encoding="utf-8")
    print(md(rep))


def md(r):
    L = ["# Weather → power-outage LSTM: evaluation", f"Generated {r['generated_at']} by `train_outage_lstm.py`.", "",
         "## Data", f"- ESMI (Prayas Energy Group, CC0, Harvard Dataverse doi:10.7910/DVN/CLLZZM): {len(r['data']['sites'])} Bengaluru household sites, "
         f"{r['data']['events']} outage events, span {r['data']['span'][0]} → {r['data']['span'][1]}.",
         f"- Label: {r['data']['definition']}. Base rate {r['data']['onset_rate_per_site_hour']:.2%} of site-hours.",
         "- Weather: Open-Meteo Historical Weather API, Bengaluru (12.97, 77.59), hourly, Asia/Kolkata.",
         f"- Features ({r['window_hours']} h window): {', '.join(r['features'])}.",
         f"- Split: train {r['splits']['train']}, val {r['splits']['val']}, test {r['splits']['test']}; site-hours {r['site_hours']}.", "",
         "## Results (site-level, weighted)", "", "| Model | split | ROC-AUC | PR-AUC | log loss | Brier |", "|---|---|---|---|---|---|"]
    for split in ("val", "test"):
        for k, v in r["results"][split].items():
            L.append(f"| {k} | {split} | {v['roc_auc']:.3f} | {v['pr_auc']:.4f} | {v['log_loss']:.4f} | {v['brier']:.4f} |")
    L += ["", f"Test base rate: {r['results']['test']['climatology (hour x month)']['base_rate']:.2%} per site-hour (PR-AUC of a random model ≈ base rate).", "",
          f"## Risk bands (probability thresholds from validation: elevated ≥ {r['test_risk_bands']['thresholds']['elevated']:.2%}, high ≥ {r['test_risk_bands']['thresholds']['high']:.2%})", "",
          "| Band | split | hours | model mean p | observed onset rate |", "|---|---|---|---|---|"]
    for k in ("normal", "elevated", "high"):
        for sp in ("val", "test"):
            b = r["test_risk_bands"][k][sp]
            L.append(f"| {k} | {sp} | {b['hours']} | {'' if b['model_mean_probability'] is None else f'{b['model_mean_probability']:.2%}'} | {b['observed_onset_rate']:.2%} |")
    wdr = r["test_risk_bands"]["weather_driven_test"]
    L.append(f"| weather-driven hours (weather model ≥ 1.25× calendar-only) | test | {wdr['hours']} | {'' if wdr['weather_model_expected'] is None else f'{wdr['weather_model_expected']:.2%} (calendar-only: {wdr['calendar_model_expected']:.2%})'} | {wdr['observed_onset_rate']:.2%} |")
    L += [f"| all test hours | test | | | {r['test_risk_bands']['all_test_hours_rate']:.2%} |", "", "## Observed outage-onset rate by rain in the same hour (data only, no model)", "",
          "| Rain | hours | onset rate |", "|---|---|---|"]
    for k, v in r["observed_onset_rate_by_rain"].items():
        L.append(f"| {k} | {v['hours']} | {v['onset_rate']:.2%} |")
    L += ["", "## Limitations",
          "- Outage onsets are inferred from household voltage (0 V ≥ 5 min); planned shutdowns are included and cannot be separated from faults.",
          "- One weather grid cell for the whole city; local storms may be missed. Monitors cover 13 Bengaluru sites, 2014–2018.",
          "- Live predictions use Open-Meteo forecasts, which have their own error; the model was trained on reanalysis (observed) weather.",
          "- Probabilities are per household-hour at the monitored sites, not for a specific feeder or for the VoltGuard load."]
    return "\n".join(L)


if __name__ == "__main__":
    main()
