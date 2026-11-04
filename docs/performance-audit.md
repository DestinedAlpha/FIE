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

- A synchronized historical full-pipeline transfer/chip/rival benchmark with every deployed component fitted before each deadline remains outstanding. The new continuous test covers ordinary one-deadline decisions only.
- Neural fusion, decision critics and learned continuation have trained artifacts and comparisons; failed replacement gates keep them out of the primary policy. More parameters do not establish predictive advantage.
- The selected live primary plan has a 12-point first hit and 2.31% additive MIP gap. Its approximately 4.3 modeled-point advantage over the same-budget no-hit incumbent excludes model/solver uncertainty and does not prove the paid hit optimal.
- Optional source coverage is incomplete: no compatible free Premier League odds were returned, and shot-feed failures retain verified cached data. Unsupported social opinions receive no invented numerical uplift.
- The implementation catalog traces selected tested behaviors, not exhaustive acceptance of every supplied-document requirement.
- Frozen prospective full-system forecasts/policies must accrue finalized outcomes before claims of repeatable superiority. Already-inspected benchmark weeks cannot be tuned and then called untouched tests.

Evidence: [reference replay](../artifacts/evaluations/gw1-gw9-replay.json), [matched controls](../artifacts/evaluations/weekly-baseline-comparison.json), [published fusion](../artifacts/evaluations/published-fusion.json), [continuous policy](../artifacts/evaluations/published-fusion-continuous.json). Prepared-data commands: `fpl train-consensus`, `fpl validate-consensus-policy`, `fpl compare-consensus`. These execute real transformations and solvers, not fixed-score fixtures.
