# Current model card · 0.12 · 6 November 2026

The prepared build combines the validated `learned-points-v1` immediate reference with an accepted future-fixture expert. Calibrated recovery and accepted joint-DGW minute transitions remain connected. Current-week probabilities and sampled outcomes are preserved; future fixtures through an 18-week lead use the selected mixture. The expert's final fit uses 747,484 rows and 18,930 reserved calibration rows, with completed labels through GW9. GW10 has no outcome label.

The future expert passes separate prediction, marginal-distribution and full legal-policy gates. Its earlier-season continuous policy scores **1,202 versus 1,098** over 19 weeks. The current-season diagnostic scores **538 versus 536** over GW1–9. The earlier-season per-deadline return-gain block interval is [0.58, 11.05]. Current outcomes do not select the configuration or authorize promotion. These studied historical archives do not establish future top-1% strength or consistent 550+ returns. See the [performance audit](performance-audit.md) and [release audit](../artifacts/evaluations/release_0_9.json).

Locally trained neural components have been removed at the user’s request. The role-aware boosted-tree minutes expert has separate two-season validation. The four-week transfer critic is retrained against the current horizon model; random/evolving-account decision learning, manager scenarios and conditional recourse remain separate, explicitly gated components.

Reference and future-expert replays preserve their models, cutoff proofs, initial states, account transitions and actual outcomes. Prospective synchronized accounts freeze real future decisions with equal hypothetical resources. They do not submit account transfers, and finalized future results remain pending.

The earlier 0.5 shadow decision frozen on 30 October for GW9 scored **48 net points after an 8-point hit**. The user's actual revealed account scored **52 with no hit**, using a different transfer. This negative prospective result remains part of the record. Older numerical results below describe earlier executed versions, not the current release.

# Model card · 0.3 · executed 29 October 2026

## The validated production path

`learned-points-v1` is the active decision forecaster. The 0.2 component simulator's ranking was not validated by the separately scored direct challenger; that readiness gap is corrected. The old component simulator is research-only and its numerical results cannot be labeled execution-ready.

The shared `context_vector` transformation consumes strictly earlier deadlines and stable player codes across 2022/23, 2023/24, 2024/25, 2025/26 and completed 2026/27 fixtures. Both fixtures of a double share a cutoff. Inputs include lag-only xG/xA, actual scoring/assist rates, appearances, role/minutes, recent points, bonus, defensive-action coverage, last observed price, venue and lagged team/opponent strength. Future rest uses known preceding scheduled fixtures, rather than an ever-growing time since today's last match. All fixture updates have deterministic order.

Current injuries/news and future prices do not leak into historical folds. The most recent archived price is used in historical budget tests; archive capture/corrections are retrospective. Early 2022 fixtures with an all-zero starts field have missing start labels, rather than 22 false substitute labels. Missing defensive actions remain missing, never observed zeros.

Training separates the probability of four minutes bands and conditional non-defensive-contribution points. Observed minutes bands are legitimate conditional training inputs, but test/live points marginalize over **predicted bands**, never test actual minutes. Defensive contribution is separated because earlier seasons did not award it; current thresholds are applied to exposure-conditioned action distributions. Goalkeeper goal scoring is harmonized to current reward where applicable. Complete historic bonus formulas cannot be reconstructed, and the limitation is retained.

Every four-deadline test block trains on earlier records and fits isotonic probability calibration on eight separate earlier deadlines. The final production base fit uses 113,260 rows; 5,220 recent rows are held out for probability calibration/residual tails. Base fitting precedes `2026/27-GW01`. Model text files have SHA-256 integrity checks and no executable pickle.

## Base reference point accuracy

34,967 held-out player-fixtures:

| Forecast | RMSE | MAE | Mean bias |
|---|---:|---:|---:|
| Rolling shrinkage | 2.078634 | 1.287294 | 0.206890 |
| Learned production mixture | 1.975220 | 1.036404 | 0.017001 |

Paired deadline-block bootstrap RMSE difference, production versus prior: [-0.11506358259867933, -0.09158032323105651]. Likely-starter RMSE: 3.124704 → 3.033483, on 8,527 rows. Promotion requires both the declared overall bootstrap gate and starter improvement.


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

## Future-fixture expert (0.9)

Three LightGBM heads learn four minute bands, conditional points and starts. Each uses 260 boosting iterations, 24 leaves, learning rate 0.035, minimum leaf sample 100 and L2 regularization 20. Inputs are 26 shared historical features, 47 published player statistics with observation masks, and 18 fixture/horizon/status fields. Model files, feature contract, calibration curves, dataset hash and acceptance IDs are retained and verified before serving.

This expert changes future fixtures through an 18-week lead. The immediate reference and sampled current-week outcomes remain unchanged. Official next-week EP is missing for later weeks. Source injury information enters once; explicit scenario availability and served bans override it coherently. Existing shared residual worlds remain the sampling contract.

Source and target completion cutoffs purge multiweek overlap. Eight earlier source deadlines are reserved for calibration. Final fitting uses completed labels only; reserved calibration observations are not simultaneously base-training observations. Accepted configuration stays fixed while ordinary training refits its weights. Ingestion retains real predeadline registries for subsequent learning, and unchanged historical chunks are cached.

Later 2024/25 and 2025/26 forecast error improves. A separate 2025/26 marginal distribution test improves CRPS and appearance/60-minute Brier, with conservative interval coverage. Full transfer/chip and continuous-account checks govern activation. Historical future-head weights freeze before each evaluation season; the immediate reference refits each deadline. These retrospective policy results do not test prospective repeated weekly future-head refits or guarantee future scores. See the [performance audit](performance-audit.md) for actual measurements and scope.
