# Model card · executed 29 October 2026

## Independent forecast paths

`shared-components-v1` is the decision simulator. It uses official fixture histories, stable player codes across seasonal archives, exposure shrinkage, shared team goal processes and coherent on-pitch goal attribution. Its player/GW means and distributions come from actual simulated worlds. Blanks have zero fixtures; doubles aggregate their constituent fixtures. Currently unassigned fixtures are recorded separately rather than given invented dates.

The minutes model separates availability, lineup selection, starts, substitutions and cameos. Clubs choose one GK and ten outfield starters without imposing an FPL formation on a football team. Up to five paired substitutions preserve the number of footballers. Shared player health uniforms induce dependence across games; uncertain future injury recovery is a disclosed prior. Goal timing controls clean-sheet exposure, including later goals conceded after a red card. Goalkeepers share their team's shots faced; player goals and own goals sum to team goals in every simulated fixture audit.

BPS is a match-conditioned proxy, followed by exact tie ranking, rather than a reconstructed official event formula. Defensive actions use an exposure-conditioned Poisson prior with typed historical coverage. Rare missed penalties and penalty saves can be scored exactly by the deterministic rules engine but are not separately parameterized in this baseline simulation. Tactical role competition, all-competition workload and detailed shot/set-piece inputs are not independently validated production features. Current news/status and explicit, expiring belief scenarios remain distinguishable from trained effects.

The direct challenger is a 100-tree LightGBM regressor (12 leaves, learning rate .04, minimum 45 rows per leaf, L2=5), averaged over three seeds for held-out evaluation. It uses 13 lag-only features: position, home venue, start/cameo probabilities, conditional start minutes, goal/assist/action/save/card rates, prior points, exposure and rest. It never consumes a present-day injury flag, current price, ownership or future fixture outcome in a historical fold.

## Evaluation

The protocol is frozen before fitting. Training uses strictly earlier deadline groups, including both fixtures of any double under one decision cutoff. Statistical archive registrations are season-scoped; current-season position-change histories are not reconstructed. Historical corrections and retrospective archive capture prevent labeling this strict point-in-time replay.

32,866 held-out observations: shrinkage RMSE 2.064208; challenger RMSE 2.011335. Paired deadline-block bootstrap interval for the difference: [-0.070503, -0.040107]. Predicted-start cohort >=60%: 8,695 observations, RMSE 3.120784 versus 3.093972. Reported start Brier score: 0.114113. Per-position and cohort metrics are in the scorecard. Reliability bins are diagnostics, not a silently applied calibration fit on test labels.

The challenger passes the declared statistical RMSE comparison. It does not replace the component model's joint worlds, and this test does not establish improved continuous transfer-policy performance. No ensemble weights are fitted on held-out test data. No competitor's author-reported metric is copied into this scorecard.

OpenFPL's MIT license, sample inference requirements and README were inspected; its sample format requires FPL/Understat features. Dastan's public methodology and participation/regular-starter distinctions were inspected. Neither published weights nor claimed metrics are represented as reproduced in this release. The optimizer was implemented independently; no upstream solver code was copied.

## Decision quality and uncertainty

The full-market multiweek MIP is exact for its stated additive formulation, including bank/lot/FT transitions and joint chip decisions. Solver timeouts return a checked feasible incumbent with an actual bound and gap. The final season-tail value is a disclosed heuristic: 10% of the following four weeks' squad xPts. Future market prices stay constant in this formulation; an observed price change invalidates exact quotes and triggers recomputation.

Common worlds then score exact FPL autosubs and captain fallback. Rival probabilities compare jointly in those same worlds; they are conditional on illustrative hold/greedy policies, published bank and assumed sale quotes. They are not probabilities that private future transfers or injuries are already known. Ties are shown separately because final transfer tie-breaks need final policy state. A 2,000-world binomial standard error describes Monte Carlo noise, not model correctness.

All original strategy families were run through GW19, including FH at GW9/12/15/19 and independent chip timing. The independent candidate's mean was 597.69 simulated net points versus 525.86 for hold; this is an internal projection comparison, not a realized gain or performance benchmark. The best candidate in that run used FH13 and BB17. These are conditional baseline outputs, not permanent tactical instructions.

## Intended use

Personal decision support with reconciled current account inputs. No automated account execution. No claim of top-1% strength, vendor superiority or achieved future season performance. Hosted contextual extraction needs a real eligible unbilled account and key; no authenticated hosted-provider quality benchmark has been claimed. The machine-readable release artifact records this explicitly.
