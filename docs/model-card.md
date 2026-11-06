# Current model card · 0.12 · 6 November 2026

The production system combines LightGBM conditional points, a separately validated role-aware immediate participation classifier, and the accepted future-fixture expert. Exact FPL account transitions and common outcome worlds feed the multiweek HiGHS planner. Locally trained neural components, weights, dependencies and controls have been removed.

## Training and information

Compatible data covers 2022/23 through completed 2026/27 GW9. Upcoming GW10 has no outcome label. The base fit uses 113,870 rows before `2026/27-GW02`, with 5,280 later rows reserved for calibration. The role head uses 113,301 fit rows and 5,254 calibration rows. The future expert uses 747,484 fit rows and 18,930 calibration rows, with fit labels through GW1 and calibration labels through GW9. Using completed labels includes their declared calibration role; they are not all silently added to base fitting.

The role classifier uses 57 inputs, including recent starts, archived season role, official availability, observed workload and source masks. Historical joins use stable player codes and hash-pinned bootstrap archives. Other-competition match enrichment is retrospective with conservative completion cutoffs, rather than proven original publication times. Provider xG definitions remain distinct. Missing sources keep masks.

Conditional scoring, minute-band probabilities, defensive-contribution exposure, empirical residual tails and within-team dependence are separate parts of the statistical forecast. The independent physical simulator remains a labelled sensitivity. These worlds are not a reconstruction of every real match event.

## Role-aware participation acceptance

The mixture is selected on 2024/25 GW1–19 only. Eight earlier deadlines are reserved for calibration in each expanding eight-deadline fold. Later 2024/25 and all 2025/26 provide acceptance and replication; confidence intervals resample complete deadlines.

| Held-out period | Participation Brier, previous → role | Point MSE, previous → role |
|---|---:|---:|
| 2024/25 GW20–38, 14,492 rows | 0.308270 → 0.298704 | 3.340446 → 3.310779 |
| 2025/26, 29,644 rows | 0.290465 → 0.278451 | 3.602296 → 3.544659 |

Regular-starter point error and all four position cohorts also improve. Both point-error and participation deadline-block intervals exclude zero. [Executed protocol and results](../artifacts/evaluations/role_minutes_validation.json). This is forecast acceptance; it does not establish a particular transfer gain or a new continuous-account score.

## Decision evidence and limits

The earlier future-expert continuous replay scores 1,202 versus 1,098 over 19 prior-season weeks and 538 versus 536 over current GW1–9. Those frozen historical protocols precede the new participation head and resource safeguards. They are not retroactively relabelled as validation of the complete 0.12 policy. Independent weekly budget teams solve a different problem from a continuous account. See the [performance audit](performance-audit.md).

The legal-action tree critic is refitted after removing the retired forecast input. Its calibration chooses zero influence; the primary optimizer remains active. The separate continuation replacement also lacks its full policy gate. Rejected alternatives are completed experiments, not hidden sources of numerical confidence.

Paid transfers and optional current chips require evidence beyond unresolved counterfactual search. Chip controls preserve later use and the same hit budget; committed, explicitly fixed and expiring entitlements retain their rules. Solver allowances are sensitivities in the additive surrogate, not rigorous nonlinear autosub bounds or football confidence intervals. Future proposed actions require refreshed data and re-optimization.

Prospective accounts freeze models and actions before the deadline and score only after official finalization. No transfer is submitted by this project. Competitive superiority, consistent 550+ returns and globally optimal future decisions are not established. The [release audit](../artifacts/evaluations/release_0_12.json) records actual implementation checks and current artifact connections.
