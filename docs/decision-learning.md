# Decision learning

Forecasting and transfer choice have separate objectives. Calibrated player distributions drive legal candidate generation and multiweek HiGHS planning. A boosted-tree critic learns four-week return residuals from legal actions on evolving accounts; a separate continuation model values squad, bank, FT and chips. Manager behavior is a conditional rival model, not an optimal-transfer oracle.

## Outcome learning without hindsight leakage

Training starts with sampled legal squads and observed population accounts. Each candidate is applied through the exact account engine, including sale-profit sharing, position and club limits, free transfers, hits, chip windows and Free Hit restoration. Later actions use that later deadline's out-of-fold forecasts. Outcomes are read only after choices are frozen. Independently replayed actual net returns and hindsight best-searched returns are labels and regret diagnostics, never privileged test inputs.

Whole overlapping four-week windows are purged between training, calibration and later tests. Residual influence is selected on earlier calibration accounts. Positive later mean gain alone cannot promote a critic whose block interval includes zero. Realized hindsight teams contain luck and cannot be copied as knowable targets. Training against all actions found by the search also does not prove a globally optimal action was found.

The completed expanded historical experiment contains 936 accounts and 9,706 candidate trajectories over 39 starting deadlines, with completed four-week labels through current GW9. After retiring the third forecast, the retained outcomes are migrated to the two-source action contract and the tree critic is refitted. Calibration chooses zero residual influence; the primary legal multiweek policy remains active. The continuation replacement also lacks a positive full policy gate.

## Current learning and validation

Ordinary retraining learns finalized outcomes only and refreshes the role-aware immediate participation model and accepted future expert. Unfinalized upcoming matches cannot become labels. The new minutes model has separate 2024/25 acceptance and 2025/26 replication; its probability and point-error improvements do not retroactively change earlier frozen policy returns. Prospective decisions remain sealed until official finalization.

`uv run fpl train-complete` runs the connected CPU workflow. `uv run fpl strategy-analysis` compares current legal candidates; `uv run fpl refine-frontier` retains stronger verified feasible incumbents. Search bounds, hit payback, official forecast disagreement and paired common-world outcomes remain visible. Model complexity is not an acceptance criterion.
