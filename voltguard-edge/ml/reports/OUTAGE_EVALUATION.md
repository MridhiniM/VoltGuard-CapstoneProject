# Weather → power-outage LSTM: evaluation
Generated 2026-10-27T07:08:06+00:00 by `train_outage_lstm.py`.

## Data
- ESMI (Prayas Energy Group, CC0, Harvard Dataverse doi:10.7910/DVN/CLLZZM): 13 Bengaluru household sites, 9390 outage events, span 2014-11-05 17:00:00 → 2018-12-31 23:00:00.
- Label: onset = >= 5 consecutive minutes at 0 V after supply; runs > 12 h excluded as monitor faults. Base rate 3.71% of site-hours.
- Weather: Open-Meteo Historical Weather API, Bengaluru (12.97, 77.59), hourly, Asia/Kolkata.
- Features (24 h window): temperature_2m, relative_humidity_2m, precip_log, wind_speed_10m, wind_gusts_10m, cloud_cover, shortwave_radiation, thunder, hour_sin, hour_cos, doy_sin, doy_cos, sunday.
- Split: train < 2017-07-01, val 2017-07-01 .. 2018-01-01, test >= 2018-01-01; site-hours {'train': 175526, 'val': 25986, 'test': 39038}.

## Results (site-level, weighted)

| Model | split | ROC-AUC | PR-AUC | log loss | Brier |
|---|---|---|---|---|---|
| climatology (hour x month) | val | 0.634 | 0.0562 | 0.1644 | 0.0380 |
| logistic regression (current hour) | val | 0.671 | 0.0670 | 0.1606 | 0.0377 |
| LSTM calendar only (no weather) | val | 0.666 | 0.0680 | 0.1606 | 0.0377 |
| LSTM weather + calendar | val | 0.669 | 0.0665 | 0.1604 | 0.0377 |
| climatology (hour x month) | test | 0.651 | 0.0508 | 0.1464 | 0.0332 |
| logistic regression (current hour) | test | 0.658 | 0.0572 | 0.1454 | 0.0331 |
| LSTM calendar only (no weather) | test | 0.661 | 0.0566 | 0.1450 | 0.0330 |
| LSTM weather + calendar | test | 0.668 | 0.0598 | 0.1445 | 0.0330 |

Test base rate: 3.46% per site-hour (PR-AUC of a random model ≈ base rate).

## Risk bands (probability thresholds from validation: elevated ≥ 7.15%, high ≥ 9.16%)

| Band | split | hours | model mean p | observed onset rate |
|---|---|---|---|---|
| normal | val | 3974 | 3.47% | 3.52% |
| normal | test | 8222 | 3.28% | 3.23% |
| elevated | val | 309 | 7.92% | 8.33% |
| elevated | test | 432 | 7.90% | 7.20% |
| high | val | 133 | 10.61% | 7.98% |
| high | test | 106 | 10.76% | 7.37% |
| weather-driven hours (weather model ≥ 1.25× calendar-only) | test | 809 | 5.31% (calendar-only: 3.73%) | 5.01% |
| all test hours | test | | | 3.46% |

## Observed outage-onset rate by rain in the same hour (data only, no model)

| Rain | hours | onset rate |
|---|---|---|
| dry (0 mm) | 30694 | 3.43% |
| light (0.1-2.5) | 5267 | 5.22% |
| moderate (2.5-7.6) | 325 | 7.64% |
| heavy (>7.6) | 27 | 12.18% |

## Limitations
- Outage onsets are inferred from household voltage (0 V ≥ 5 min); planned shutdowns are included and cannot be separated from faults.
- One weather grid cell for the whole city; local storms may be missed. Monitors cover 13 Bengaluru sites, 2014–2018.
- Live predictions use Open-Meteo forecasts, which have their own error; the model was trained on reanalysis (observed) weather.
- Probabilities are per household-hour at the monitored sites, not for a specific feeder or for the VoltGuard load.