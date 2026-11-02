# Model card: AML transaction risk scorer

| | |
|---|---|
| Model id | `aml-risk-scorer`, versions `scorer-YYYYMMDD-<datahash>` (MLflow registry) |
| Type | Gradient-boosted trees (LightGBM), isotonic calibration |
| Owner | AML Analytics (model owner) · validated by Model Risk (independent) |
| Tier | High: drives which transactions investigators review |
| Decision role | **Prioritisation only.** It never files or closes a case; a human decides |

## Intended use
Rank transactions by the probability that they belong to a money-laundering pattern, so that
investigators' fixed capacity (alerts/day) is spent on the riskiest ones. Scores at or above
the capacity threshold open a high-priority case; rule hits below it open a low-priority case.

**Out of scope:** decisions about customers (exits, account freezes), credit, sanctions
screening (handled by exact list matching), any use without human review.

## Data
- Transactions: IBM AML synthetic simulation or the project's synthetic generator, with
  generated KYC profiles. **No real customer data.**
- Labels: simulation ground truth (`is_laundering`). In a bank, labels are investigator
  outcomes (STR filed / closed), which are delayed, noisy and biased toward what was alerted.
- Features (55): windowed account state (1h/24h/7d/30d counts, amounts, distinct counterparties,
  cash near threshold), daily graph features (degrees, 3-cycles, known-bad neighbours with a
  lag), KYC (segment, declared volume, risk rating). All point-in-time: a leakage test proves
  no feature uses information after the transaction.

## Training and selection
Time split 60/20/20; early stopping and calibration on the validation period; threshold from
capacity on the validation period; final numbers on the untouched test period.

## Performance (medium synthetic, test period of 6.1 days, 100 alerts/day capacity)
| Metric | Value |
|---|---|
| ROC-AUC | 0.9955 |
| PR-AUC | 0.886 |
| Expected calibration error | 0.0021 |
| Alerts at threshold | 457 (≈ 75/day) |
| Precision at threshold | 75.3% |
| Recall at threshold | 86.9% |
| Investigator hours at threshold | 228.5 vs 2,092.5 for the rule set |
| Rules at the same budget | precision 7.8%, recall 82.6% |

Eval-harness world (seed 2024, 60 alerts/day): PR-AUC 0.835, precision 25.5%, recall 92.6%,
ECE 0.0013 — lower precision because the eval world has more decoys per pattern.

## Explanations
Per-alert top drivers from TreeSHAP contributions, in plain language
(e.g. "4 cash payments near the reporting threshold in 24 hours").

## Limitations and risks
- **Synthetic data**: patterns are cleaner than reality; real precision will be far lower.
- **Label bias**: trained on what was labelled; novel typologies are missed until labelled.
- **Window warm-up**: 30-day features are not stationary in a 30-day dataset (PSI flags them);
  production uses a steady-state rolling reference.
- **Segment effects**: business customers have different baselines; monitor alert rates by
  segment for unfair concentration.

## Monitoring
PSI on features and scores (watch 0.10, act 0.25), alerts/day vs capacity, precision on decided
cases (investigator feedback), shadow comparison before any new version is promoted.
