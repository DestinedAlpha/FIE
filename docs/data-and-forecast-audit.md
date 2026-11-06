# Data, forecasting and planning audit · 0.13

Evaluated 6 November 2026. This release repairs confirmed data, migration and planning gaps. It does not claim perfect forecasts or globally optimal future teams.

## Repaired data contracts

Older local news packets omitted season starts/minutes even though the original full FPL bootstrap had been saved before the deadline. Upgrading upstream archives did not upgrade these local packets. One historical GW9 view therefore supplied missing season-role fields where live serving supplied seven starts and 630 minutes for Fernandes.

Reconstruction now uses the immutable full bootstrap captured before that same deadline. It checks season/next-event identity, preserves the original source capture/hash and records the later derivation time separately. It cannot borrow later season totals. The wider schema also preserves xG/xA, threat, creativity, ICT, BPS and defensive actions.

Role aggregation uses float32 for both stored and live sequences. Previously four starts in five could become 0.799805 in stored float16 and 0.800000 when served, crossing a tree split. Role version 3 binds the context, feature and validation IDs. Startup migrates older artifacts and refreshes forecasts after a model change. The feature-builder tuple-unpacking crash is fixed.

The [executed parity audit](../artifacts/evaluations/role_contract_audit.json) compares production input construction with historical tensors for **669 GW9 player-fixtures**: all **57 inputs match exactly**, including all 26 core fields. Maximum difference is **0**.

Full training composes accepted participation before comparing source challengers. Decision and manager audits now consume earlier-only composed predictions rather than relabelling raw base predictions as the complete model.

Population and mini-league collectors request only gameweeks an entry actually played. The previous sampler excluded a legal GW9 entrant after requesting its nonexistent GW8 picks; the refreshed sample contains all 112 requested managers with no retrieval errors.

## Actual numerical consumers

| Component | Inputs/use |
|---|---|
| Immediate conditional points | Lagged xG/xA, points, minutes, actions, price, team/opponent strength and form |
| Immediate participation | 57 inputs including archived season role, recent starts, availability and workload |
| Future expert | 138 inputs including advanced FPL statistics, fixtures and lead time |
| Understat | Provider-specific shot/role enrichment in tested source models and physical-world sensitivities |
| Top managers | Released decisions train behavior and rival scenarios; ownership does not automatically add points |
| Scout/community | Linked evidence and explicit sensitivities; opinions are not outcome labels |

The Sources panel displays these actual connections. Official FPL statistics and Understat remain distinct; this is not a claim of access to a proprietary Opta subscription.

## Validation and rejected changes

Selection uses 2024/25 GW1–19, acceptance GW20–38, and replication all 2025/26. Each expanding fold reserves eight earlier deadlines for calibration.

| Held-out period | Base point MSE | Repaired role point MSE |
|---|---:|---:|
| 2024/25 GW20–38 | 3.340446 | 3.310779 |
| 2025/26 | 3.602296 | 3.544459 |

The previous role version's replication MSE was 3.544659, so the incremental error gain is small. The contract repair must not be advertised as a large predictive breakthrough. [Executed role validation](../artifacts/evaluations/role_minutes_validation.json).

Experiments tested multi-scale player/team form, season xG/xA/threat/creativity/BPS/actions, shot enrichment, price-free/price-inclusive LightGBM and XGBoost heads, and online corrections from earlier forecast errors. These complete experiments failed forecast/captaincy acceptance and remain inactive. For example, the advanced XGBoost candidate slightly improved average replication RMSE but reduced the global captain diagnostic from 262 to 226. Their reports explicitly identify the older role-v2 reference and are not relabelled as evaluation of the later repair.

[Statistics experiment](../artifacts/evaluations/immediate_points_validation.json) · [Online-bias experiment](../artifacts/evaluations/online_calibration_validation.json). `uv run fpl audit-scoring` reproduces comparisons without replacing production.

The updated weekly £100m diagnostic scores **575**: **48, 93, 54, 63, 65, 44, 61, 65, 82** for GW1–9. Each week fits/calibrates on earlier data and uses 20,000 worlds, normal captaincy and exact autosubs. This is hypothetical weekly Free Hit selection, not a persistent account or future guarantee. The previous 569 receipt remains in release v0.12.1. [Current squads and proof](../artifacts/evaluations/gw1-gw9-role-replay.json).

## Transfer and captain explanations

The actual pre-GW9 archive records Fernandes's FPL form at **0.7**, confirming poor recent returns. Its season xG/xA is **2.66/1.50**, versus Palmer's **1.50/1.09**. Recent realized points and underlying chances therefore disagree. The core points head already includes recent five-match returns and historical expected involvement; increasing a short-form penalty is a tested model choice rather than a justified manual override from the subsequent result.

Pedro → Palmer cannot be a direct transfer: a forward cannot replace a midfielder. Legal bundles preserve squad composition and finance every purchase. The official GW9 snapshot marked Pedro injured with zero availability. A Palmer route needs a midfielder sale or a complete two-position bundle.

The earlier GW6 forecast projected Tavernier at 3.83 and Anderson at 4.61. Expected defensive-contribution points were approximately 0.20 and 1.02 respectively. An attacking-stat advantage is therefore not the entire FPL comparison; whether that expectation is accurate remains a forecasting question.

The captain panel now ranks XI candidates by expected **extra captain points including the best vice fallback**, and shows that vice alongside ordinary player points. Bench players no longer appear as captain candidates. Tests distinguish a nonappearance from a one-minute cameo.

## Planning repairs

Full-market ordinary, chip-calendar and wildcard action solves supply additional starting trajectories. Each must satisfy every constraint and improve the same complete additive objective before replacing an incumbent. Cash bounds follow fixed-price account wealth. Transfer-free continuous squad relaxations provide a separate valid additive upper bound, including independent chip windows across GW19/GW20. Unfinished joint-search box bounds no longer have to be the only reported ceiling.

Bounds remain for the additive surrogate, not exact nonlinear autosubs. Paid-transfer and optional-chip evidence safeguards remain active. No named player is forced into recommendations. Account data and keys are preserved; the application submits no FPL transfer.

## Current GW10 counterfactuals

The refreshed actual-account comparison uses the same 20,000 frozen worlds and independently continues each legal first action through GW19. GW10 expected scores below include the hit and exact autosubs/captain fallback.

| First action | GW10 expected net points | Hit |
|---|---:|---:|
| Roll | 51.859 | 0 |
| Cherki → Tavernier | 52.438 | 0 |
| Gvardiol → Van Dijk | 53.755 | 0 |
| Cherki → Tavernier, Gvardiol → Gabriel | 50.608 | 4 |

The selected no-hit Van Dijk route remains a model preference, not proof of the best actual football outcome. Each continuation retains its finite-search diagnostics. [Frozen comparison and full ledgers](../artifacts/evaluations/transfer_research.json).
