# Performance and completion correction

The 0.6 software release was published with passing implementation checks. Calling the user's entire requested system fully complete overstated the evidence. Functional checks, exhaustive feature acceptance, end-to-end integration and competitive football performance are separate requirements.

## Executed GW1–8 comparison

| Policy | Actual points |
|---|---:|
| Reference model replay | 472 |
| Archived availability-adjusted ownership | 512 |
| Archived official expected points | 406 |
| Archived availability-adjusted points per game | 392 |

All four policies receive independent weekly £100m squads, the same archived registered market, prices, positional and club limits, no chips, an additive squad/XI/captain MIP with requested zero gap, and the same actual scoring rules. All 24 control solves finished with zero reported additive gap. Baseline squads and lineups were frozen before target scoring and use only archived published signals. No live ingestion or model retraining was performed for this audit.

The reference retains its own world-optimized bench and vice policy. Controls order their bench and choose captain/vice using their own published signal. This is a comparison of declared policies; it does not isolate forecasting from every bench-policy difference.

The model trails ownership by 40 points and beats it in only two of eight weeks. The descriptive two-week block bootstrap interval for the model-minus-ownership weekly difference is [-15.875, 4.0]; this short, already-inspected sample does not establish a general performance ordering. All controls are reported rather than selecting a convenient weaker comparator. The published weekly official averages sum to 461, but that is context only because ordinary managers face transfer constraints and may use chips.

The full decisions, source identities, solver results and scores are in [weekly-baseline-comparison.json](../artifacts/evaluations/weekly-baseline-comparison.json). Run `uv run --extra neural fpl benchmark-weekly` against the prepared data to reproduce the controls.

## What has and has not been established

- The reported 472 tests the refitted reference forecaster and weekly team selection. It does **not** test the complete integrated live system, top-manager response models, source-supported recourse and season-long transfer/chip policy together.
- Neural fusion, decision critics and learned continuation are implemented and trained. Their replacement gates fail; they are not demonstrated improvements to the primary policy. Training losses or model size do not change this conclusion.
- The live multiweek planner returns feasible, independently replayed candidates. Its selected 0.6 result timed out at a 2.31% additive MIP gap. Its 12-point-hit route's 4.3-point modeled advantage is not a proven advantage over the best possible no-hit policy, and the Monte Carlo interval excludes model and solver uncertainty.
- Optional source adapters exist, but coverage is incomplete: the free odds feed supplied no compatible Premier League markets, and shot enrichment retained cached observations after refresh failures. Those observations cannot be represented as complete live coverage.
- The 65-item catalog links selected implementation symbols, passing regressions and runtime artifacts. It does not prove exhaustive acceptance of every supplied-document and user requirement.
- A current-release continuous-account, full-pipeline benchmark with transfers/chips and matched ownership/expert controls remains outstanding. Future competitive superiority also requires genuinely prospective frozen decisions and subsequent outcomes.

The benchmark weeks have now been inspected. They must not be reused for tuning and then described as an untouched test proving improvement. Current models and the user's account remain unchanged by this diagnostic comparison.
