# Current model card · 0.8 · 4 November 2026

The current prepared build uses the validated `learned-points-v1` reference, calibrated recovery and accepted joint-DGW minute transitions. Training and calibration include finalized current-season GW1–9 in their chronological roles. GW10 has no outcome label. The six-block 326,671-parameter temporal neural challenger finishes training and improves matched point RMSE, but fails the complete captain/policy acceptance gate; it remains a connected comparison rather than replacing the reference.

Current results and individual adoption decisions are in [the release audit](../artifacts/evaluations/release_0_8.json), [football model methods](football-models.md) and [atomic implementation coverage](completion-matrix.md). Historical weekly teams are reconstructed independently with prior-only fits and calibration; they are not a prospective record. Older numerical results below describe earlier executed model versions.

The full production multiweek policy scores 536 net points in current GW1–9, versus 561 for ownership with free transfers. In earlier-season GW20–38 it scores 1,098, versus 1,154 for the same-forecast one-week policy and 1,022 for ownership. Its gain interval over ownership includes zero. The completed replay verifies 28 frozen deadlines and 84 continuous actions; it does not establish reliable superiority or consistent 550+ returns. Full results are in the [performance audit](performance-audit.md). New synchronized prospective accounts start at GW10 with equal hypothetical £100m resources and exact purchase lots; their finalized outcomes are pending.

The earlier 0.5 shadow decision frozen on 30 October for GW9 scored **48 net points after an 8-point hit**. The user's actual revealed account scored **52 with no hit**, using a different transfer. The shadow policy did not beat the account that week. This negative prospective result is retained; the 0.6 GW10 shadow decision awaits finalized future results.

The active transfer forecaster below remains the reference architecture. Version 0.4 adds independently trained multi-source/neural analysis, manager behavior and four-week counterfactual learning. Its executed results and promotion decisions are documented in [source-fusion.md](source-fusion.md); new point-error gains are not presented as a validated improvement in full transfer policies.

# Model card · 0.3 · executed 29 October 2026

## The validated production path

`learned-points-v1` is the active decision forecaster. The 0.2 component simulator's ranking was not validated by the separately scored direct challenger; that readiness gap is corrected. The old component simulator is research-only and its numerical results cannot be labeled execution-ready.

The shared `context_vector` transformation consumes strictly earlier deadlines and stable player codes across 2022/23, 2023/24, 2024/25, 2025/26 and completed 2026/27 fixtures. Both fixtures of a double share a cutoff. Inputs include lag-only xG/xA, actual scoring/assist rates, appearances, role/minutes, recent points, bonus, defensive-action coverage, last observed price, venue and lagged team/opponent strength. Future rest uses known preceding scheduled fixtures, rather than an ever-growing time since today's last match. All fixture updates have deterministic order.

Current injuries/news and future prices do not leak into historical folds. The most recent archived price is used in historical budget tests; archive capture/corrections are retrospective. Early 2022 fixtures with an all-zero starts field have missing start labels, rather than 22 false substitute labels. Missing defensive actions remain missing, never observed zeros.

Training separates the probability of four minutes bands and conditional non-defensive-contribution points. Observed minutes bands are legitimate conditional training inputs, but test/live points marginalize over **predicted bands**, never test actual minutes. Defensive contribution is separated because earlier seasons did not award it; current thresholds are applied to exposure-conditioned action distributions. Goalkeeper goal scoring is harmonized to current reward where applicable. Complete historic bonus formulas cannot be reconstructed, and the limitation is retained.

Every four-deadline test block trains on earlier records and fits isotonic probability calibration on eight separate earlier deadlines. The final production base fit uses 113,260 rows; 5,220 recent rows are held out for probability calibration/residual tails. Base fitting precedes `2026/27-GW01`. Model text files have SHA-256 integrity checks and no executable pickle.

## Point accuracy and neural comparison

34,967 held-out player-fixtures:

| Forecast | RMSE | MAE | Mean bias |
|---|---:|---:|---:|
| Rolling shrinkage | 2.078634 | 1.287294 | 0.206890 |
| Learned production mixture | 1.975220 | 1.036404 | 0.017001 |
| Neural challenger | 1.972572 | 1.034164 | 0.014253 |

Paired deadline-block bootstrap RMSE difference, production versus prior: [-0.11506358259867933, -0.09158032323105651]. Likely-starter RMSE: 3.124704 → 3.033483, on 8,527 rows. Promotion requires both the declared overall bootstrap gate and starter improvement.

A standardized two-layer MLP (32,16), alpha 10, 100 iterations, trains only inside earlier folds. Its paired RMSE difference versus trees is [-0.005641918722270772, 0.00011298252437268989]. The interval includes zero. A more complex model has not established an improvement here; it is not silently promoted based on architecture or a tiny point estimate.

## Decision returns and hindsight

Captain/vice bonuses on 46 held-out deadlines: learned 323; highest-price comparator 274; rolling-points comparator 283. The learned model selected a defender on 1 deadline. No player-name rule, defender ban or forced premium captain exists.

Single-week matched-budget team totals: learned 1791; rolling prior 1681; actual-outcome hindsight optimum 4638. The hindsight oracle uses the same budget, composition and club constraints, with zero bench-value heuristic and a requested zero MIP gap; all executed oracle solves reported gap zero. This ceiling deliberately knows match outcomes. It diagnoses missed opportunities and unavoidable outcome variance; oracle winners are never test-week training inputs.

Continuous 2025/26 GW9–38 replay starts all policies from the same initial squad and bankroll, uses earlier-fold forecasts, actual points, exact captain fallback/autosubs, integer quotes, FT transitions and hits. Learned transfer policy: 1708 net points, 4 hit points, 31 transfers. Rolling transfer policy: 1527 net points. Hold policies score 1359 / 1352. Every exported action is legal. Chips are excluded to isolate forecast/transfer effects. These are historical statistical policy results; small forecast differences can compound through later account decisions, so they are not a guaranteed gain.

## Distribution and dependency checks

The **same integer outcome sampler** runs in production and earlier-fold uncertainty evaluation. It preserves nonappearance zero points, empirical conditional residual skew/tails, calibrated minutes-band probabilities and exact FPL autosubs/captain fallback downstream. Reported player means and planned world arrays are checked for equality. The residual distribution is centered so the tested point expectation is not replaced by an untested simulator expectation.

On likely starters, observed central 80% coverage is 87.886%; central 90% coverage 94.136%; CRPS 1.5144; mean sampling bias -0.0020 points. Discrete central intervals are conservative and coverage is reported rather than described as exact. Evaluation uses 512 marginal worlds per row; the prepared live forecast uses 4,000.

Within-team defensive/attacking role factors are estimated from earlier calibration residuals. Cross-team, player competition and long-horizon dependencies remain approximate. These statistical worlds do not claim finite football XIs, exact scorer allocation or exact official BPS reconstruction. Marginal interval coverage cannot validate joint rival-win probabilities. The physical component simulator remains a distinct research comparator.

## Bias and manager evidence

Bias is not assumed absent. On likely starters, the current budget/midprice/premium mean biases are 0.027 / 0.173 / 0.443 points, with sample sizes 6172 / 2160 / 195. Premium forecasts remain somewhat high in this holdout; no correction is fitted to these held-out errors and then advertised as independent validation. Position-specific metrics are retained in the scorecard.

50 actual released manager squads were compared using the identical 15 players and chip. Prior-season histories were independently retrieved; only 1 manager meets two prior top-10,000 finishes in the latest three recorded seasons. The current-rank sample is not a representative elite population. The model's same-squad lineup difference is 0.34 points on one event, with entry-bootstrap interval [-0.78, 1.48]. That is insufficient to establish an advantage over managers. Manager decisions are a comparator, not assumed truth or privileged training targets.

## Limits and operation

Recompute after new finalized outcomes and meaningful availability/fixture changes. Reconcile actual account actions separately; proposed transfers are never treated as submitted. Future purchases and chip timings are conditional, and solver bounds/gaps remain visible. Unknown purchase prices remain null; price scenarios use explicitly labelled hypotheses.

Hosted research is an optional source interpreter, not the numerical training engine. Eligible Gemini/Groq accounts have authenticated and undergone the separate small synthetic extraction benchmark; no perfect real-world extraction claim follows. Independent setup failures cannot stop numerical startup. Free capability expiry, quota reservation and prohibition of paid tools remain enforced.

Pre-deadline forecasts/shadow decisions are immutable; prospective scores require subsequently finalized events. The next deadline has not occurred. No top-1% rank, perfect prediction, bias-free model or zero-error guarantee is established.
