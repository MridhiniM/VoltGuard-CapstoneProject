# VoltGuard ML evaluation report
Generated 2026-10-27T05:58:16+00:00 by `train_ml.py`.

## Data
- Electrical: 25306 cleaned readings from pzem_temperature_measurements_new_YESENEW.xlsx, pzem_serial_diagnostics.xlsx (2026-10-19 21:58:51.008000 → 2026-10-20 13:32:14.692000, IST, ~2 s).
- Weather: Open-Meteo Historical Weather API (reanalysis), Bengaluru, hourly.
- Splits (chronological): train < 2026-10-20 08:52:13.586800 ≤ val < 2026-10-20 11:12:14.139400 ≤ test.

## A. Anomaly detector (unsupervised)
Features: voltage, current, power, temperature, v_dev, i_std, p_std, t_rate. Split: interleaved 30-min blocks 5:1:1 (train:val:test), 5-min purge at val/test block starts.
Selected: **robust_covariance** (highest mean row-detection on synthetic injections into VALIDATION data, at the p99.5 validation threshold). Threshold 118.034 (p99.5 of validation scores (target ~0.5% false alarms on normal data)).

| Normal data: false-alarm rate | isolation_forest | robust_covariance | z>4 baseline |
|---|---|---|---|
| validation | 0.51% | 0.51% | 0.00% |
| test | 0.15% | 1.11% | 0.04% |
| forward-in-time test (fit on earliest 70 %) | 0.00% | 0.72% | – |

Synthetic injections into held-out test data (20 two-minute blocks each, rows flagged / blocks detected; **sanity check, not real faults**):

| Injection | isolation_forest | robust_covariance | z>4 baseline blocks |
|---|---|---|---|
| voltage_sag | 54% / 100% | 100% / 100% | 100% |
| voltage_swell | 23% / 90% | 46% / 100% | 60% |
| overcurrent | 83% / 100% | 100% / 100% | 100% |
| thermal_runaway | 3% / 60% | 56% / 95% | 100% |
| unstable_load | 76% / 100% | 100% / 100% | 100% |

Seed stability (test-flag agreement, 4 other seeds): 1.000, 1.000, 1.000, 1.000.

## B1. equipment temperature (1-min mean) 30 min ahead

| Model | val MAE | val RMSE | test MAE | test RMSE |
|---|---|---|---|---|
| persistence baseline | 0.337 | 0.450 | 0.630 | 0.756 |
| electrical:ridge | 2.321 | 2.349 | 0.719 | 0.862 |
| electrical:random_forest | 0.907 | 0.975 | 0.611 | 0.722 |
| electrical:gradient_boosting | 0.953 | 1.027 | 0.604 | 0.713 |
| electrical+weather:ridge | 1.926 | 1.947 | 0.728 | 0.902 |
| electrical+weather:random_forest | 1.427 | 1.464 | 0.608 | 0.743 |
| electrical+weather:gradient_boosting | 0.266 | 0.300 | 0.593 | 0.723 |

Selected by validation MAE: {'electrical': 'electrical:random_forest', 'electrical+weather': 'electrical+weather:gradient_boosting'}. Weather improves selected test MAE: **True**.
Selected models beat persistence on test: electrical: **True**, electrical+weather: **True**.

## B2. equipment temperature (10-min mean) as a function of 3-h mean ambient temperature and load power

| Model | val MAE | test MAE | test RMSE |
|---|---|---|---|
| baseline_train_mean | 2.167 | 2.838 | 2.869 |
| linear:amb_3h | 2.312 | 0.440 | 0.563 |
| linear:amb_3h+power | 2.387 | 0.343 | 0.432 |

Selected: `linear:amb_3h`; beats baseline on validation: **False**, on test: **True** → used by 7-day outlook: **True**.
- One diurnal cycle only: ambient is confounded with time-of-day and any slow drift.
- The train-only fit does not beat the mean baseline on the validation period; it beats it on test only after refitting on train+val. Treat outlook temperatures as low-confidence.
- Extrapolation beyond the training ambient range is flagged in the outlook.
- B1: Training uses Open-Meteo reanalysis as a stand-in for the forecast that would be available live.
- B1: Only ~15.5 h of data (one partial day); the test period is midday, warmer than most training data.
- B1: Improvements over persistence are small and come from one ~2-hour test window; a difference of a few hundredths of a degree between the electrical and weather variants is not evidence that weather helps.
Training ambient range [20.866666666666664, 28.1] °C; equipment [27.905000000000005, 33.975100000000005] °C.

## C. Supervised fault prediction
Not possible: neither the real PZEM logs nor any accessible matching public dataset contain confirmed, timestamped fault events for this equipment. The repository's `dataset.csv` labels are synthetic and derived from injected patterns, so a 'fault within 24 h' classifier trained on them would not be evidence of real fault prediction. Required: logged real faults (type + timestamp) from the monitored equipment over weeks/months.

## Limitations
- Single ~8 W resistive load, one site, ~15.5 hours (one partial diurnal cycle).
- No real faults observed; anomaly detection is validated only on normal data plus synthetic injections.
- Weather is reanalysis at city scale, not an indoor sensor at the equipment.
- Future electrical load is unknown; the outlook assumes the median observed load operates continuously.

## Deployment suitability
Demonstration / advisory only. Suitable for flagging readings that differ from the recorded normal behaviour of this one load. Not a protective device and not validated for other loads, seasons or sites; retrain when more data (especially real faults) is logged.
