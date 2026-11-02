# ADR 0004: Detection — rules baseline plus a calibrated, capacity-aware ML scorer

- Status: accepted
- Date: 2026-10-20

## Context
Banks already run rule scenarios, and most of their alerts are false positives. A model is only
worth deploying if it beats the rules **at the same investigator capacity** and if every score
can be explained to an investigator, a validator and a regulator.

## Decisions
1. **Rules stay, as configuration** (`detection/rules.yaml`, 6 scenarios). They are the baseline
   every model result is compared against, and they still raise low-priority alerts.
2. **LightGBM** on point-in-time features, **time-based split** 60/20/20 (never random: a random
   split leaks the future into training). Class imbalance via `scale_pos_weight`; early
   stopping on the validation period.
3. **Isotonic calibration** on the validation period, stored as two arrays in `bundle.json`
   and applied with `np.interp`. Calibrated scores are used as probabilities for routing.
4. **Threshold from capacity, not from 0.5**: alerts/day investigators can review (default 100)
   sets the threshold on the validation period.
5. **Reasons per alert** from TreeSHAP (`pred_contrib=True`), translated to plain language.
6. **Versioned bundle** `scorer-YYYYMMDD-<datahash>` (model.txt + bundle.json + CURRENT pointer),
   logged to MLflow (falls back to local SQLite when the server is down).

## Result (medium synthetic, 6-day test period)
| | Alerts | Precision | Recall | Investigator hours |
|---|---|---|---|---|
| Rules (any scenario) | 4,185 | 7.8% | 82.6% | 2,092.5 |
| Model at 100 alerts/day | 457 | 75.3% | 86.9% | 228.5 |

ROC-AUC 0.9955 · PR-AUC 0.886 · ECE 0.0021.

## Consequences
- Synthetic data flatters the model; the numbers prove the method, not real performance.
- Score values are not rounded anywhere (rounding pushed scores below the threshold — a bug we hit).
