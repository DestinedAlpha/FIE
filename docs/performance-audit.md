# Measured performance and acceptance

Consistent 550+ continuous-account performance is not established. Fresh weekly teams and a persistent account measure different problems. Neither guarantees future scores or ranks.

## Independent weekly £100m teams, no chips

| GW | Refitted reference | Ensemble | Ensemble's frozen reference | Ownership |
|---|---:|---:|---:|---:|
| 1 | 47 | 52 | 47 | 54 |
| 2 | 89 | 79 | 90 | 98 |
| 3 | 39 | 43 | 48 | 46 |
| 4 | 64 | 84 | 79 | 85 |
| 5 | 65 | 67 | 70 | 33 |
| 6 | 43 | 68 | 58 | 82 |
| 7 | 61 | 70 | 55 | 74 |
| 8 | 64 | 63 | 45 | 40 |
| 9 | 79 | 60 | 70 | 51 |
| **Total** | **551** | **586** | **562** | **563** |

The original first eight reference results remain 472; GW9 adds 79 without retuning earlier decisions. Official expected-points and points-per-game controls score 474 and 433. All policies use dated registered markets, prices, announced schedules, £100m, 15-player/club constraints, no chips and actual autosubs/captain fallback. Baseline utility is zero in announced blanks; solver gaps are retained.

The 586 experiment freezes its statistical reference and crowd tree before 2025/26 while updating player history at each deadline. The 551 reference refits weekly and uses sampled-world bench/vice optimization; the frozen experiment orders bench/vice by its own signal. Therefore 586 minus 551 is not an isolated gain from three new inputs. Its directly matched comparison is 586 versus 562. Ensemble captains are B. Fernandes in GW1 and Haaland in GW2–9; no premium-player restriction is imposed.

## Selection and temporal boundaries

Shared football/history/news/ownership/workload/enrichment tensors are extended with archived ep_next, points_per_game and form plus missingness masks. Weekly EP is allocated across announced fixtures, never cloned into distant unknown weeks. Historical serving excludes target/future matches and requires a four-hour completion margin. Experts are a prior-qualified, deadline-lagged cohort; current top-1,000 ranks are not backfilled into old deadlines.

Training/calibration end before 2025/26. The statistical reference reserves eight earlier deadlines for calibration; the crowd tree reserves four. Select a crowd weight from {0, .25, .5, .75, 1} by legal-team returns in GW1–19, freeze it before GW20–38, and never select using current GW1–9. The selected weight is .5. Later-season weekly-team gain is 87 points; its two-week block-bootstrap weekly-gain interval is [0.21, 9.11]. The per-deadline MSE-difference interval is [-0.1025, -0.0499]. The declared weekly gates pass.

This is chronological acceptance on a repeatedly studied archive, not a blinded prospective trial. Historic match facts and enrichment were imported retrospectively. Published forecasts are explicit optional inputs; independent forecasts remain available. Cached decisions are reused only after reproducing their complete utility vectors; outcomes are rescored from current labels.

## Continuous ordinary transfers from identical starting states

| Policy | 2025/26 GW20–38 | Current GW1–9 |
|---|---:|---:|
| Ensemble | 1,111 | 481 |
| Frozen statistical reference | 1,060 | 503 |
| Ownership, free transfers only | 1,030 | 561 |
| Official expected points | 875 | 404 |

Each policy starts from the same ownership-selected £100m squad, bank, exact lots and FT. Changes must finance purchases, apply selling-price profit sharing, reconcile club changes, carry bank/FT and pay hits. GW1 initialization grants one FT for GW2. Only each deadline's frozen forecast is revealed; later weekly predictions are not supplied early. Actions freeze before actual scoring. Ownership percentages are ranking signals, so that baseline makes only free transfers rather than comparing percentages against a four-point hit.

The earlier-season ensemble-minus-reference gain interval is [-2.74, 7.32]: the ordinary-policy gate fails. This is a one-deadline policy test excluding chips and future lookahead. It does not certify the primary multiweek/chip/rival pipeline. The ensemble remains a separate current-squad comparison: Cherki → Mbeumo, no hit, Haaland captain, with hold/no-hit alternatives and solve gaps. The real account and primary model/plan are unchanged.

## Remaining acceptance limits

- The integrated benchmark below now measures the deployed reference-family expected-points pipeline, including lookahead and chips. It does not validate every alternative rival/rank objective or rejected research model.
- Neural fusion, decision critics and learned continuation have trained artifacts and comparisons; failed replacement gates keep them out of the primary policy. More parameters do not establish predictive advantage.
- The selected live primary plan has a 12-point first hit and 2.31% additive MIP gap. Its approximately 4.3 modeled-point advantage over the same-budget no-hit incumbent excludes model/solver uncertainty and does not prove the paid hit optimal.
- Optional source coverage is incomplete: no compatible free Premier League odds were returned, and shot-feed failures retain verified cached data. Unsupported social opinions receive no invented numerical uplift.
- The implementation catalog traces selected tested behaviors, not exhaustive acceptance of every supplied-document requirement.
- Frozen prospective full-system forecasts/policies must accrue finalized outcomes before claims of repeatable superiority. Already-inspected benchmark weeks cannot be tuned and then called untouched tests.

Evidence: [reference replay](../artifacts/evaluations/gw1-gw9-replay.json), [matched controls](../artifacts/evaluations/weekly-baseline-comparison.json), [published fusion](../artifacts/evaluations/published-fusion.json), [continuous policy](../artifacts/evaluations/published-fusion-continuous.json). Prepared-data commands: `fpl train-consensus`, `fpl validate-consensus-policy`, `fpl compare-consensus`. These execute real transformations and solvers, not fixed-score fixtures.

## Production forecaster and full-window planner

The integrated experiment uses the actual production `learned_forecast`, full-market multiweek optimizer, hold/no-hit comparisons, exact transition engine and actual autosub/captain scoring. It replans after each deadline. The lookahead ends at the applicable chip window (GW19/GW38), with four terminal weeks; it does not spend chips just to maximize a scored GW9 endpoint.

| GW | Production multiweek | Same-forecast one-week | Ownership, free transfers |
|---|---:|---:|---:|
| 1 | 47 | 45 | 54 |
| 2 | 103 | 96 | 103 |
| 3 | 35 | 30 | 46 |
| 4 | 59 | 56 | 89 |
| 5 | 38 | 68 | 46 |
| 6 | 57 | 41 | 73 |
| 7 | 86 | 39 | 67 |
| 8 | 55 | 70 | 33 |
| 9 | 56 | 71 | 50 |
| **Total** | **536** | **516** | **561** |

The integrated policy pays no hits in these nine weeks and uses BB in GW1, FH in GW3 and TC in GW7. Controls share the initial squad, budget and chip inventory; their declared ordinary/free-transfer policies do not activate chips. This comparison combines planning and chip-policy effects. It is not a pure isolated estimate of the value of lookahead. The weekly 586 ensemble experiment is a different model and account problem.

The complete earlier-season GW20–38 replay scores **1,098** for the integrated policy, **1,154** for the same-forecast one-week policy and **1,022** for ownership. Integrated hits total four points; it uses TC20, FH21, BB30 and WC31. The two-week block-bootstrap interval for its per-deadline gain over ownership is **[-1.84, 10.16]**; reliable superiority is not established. Its lower return than the one-week control also prevents a claim that extra lookahead/chip complexity improved realized performance. Those controls use exactly the same starting state within this experiment; their figures need not equal controls from a different experiment with different fitted/sampled forecasts or seed solves.

All decisions retain their input version, pre-target fit/calibration groups, models, first-week forecasts/worlds, legal actions and next state. Independent checks match reported actions to sealed decisions and verify account continuity and actual scores. The replay uses 4,000 worlds and 60-second MIPs with real gaps. It remains a later statistical reconstruction using corrected observations. Current weeks were already inspected, and they do not select new model parameters here.

Prospective synchronized accounts start separately from equal hypothetical £100m states with known purchase lots. Real future decisions freeze before their deadlines. Missed captures execute a previously frozen standing hold, and finalized scores are revised without double counting. No prospective superiority is asserted before those outcomes exist. These accounts do not change the real account or submit FPL transfers.

Evidence: [integrated production replay](../artifacts/evaluations/integrated-pipeline-replay.json). Commands: `fpl replay-pipeline`, `fpl freeze-accounts`, `fpl score-accounts`.
