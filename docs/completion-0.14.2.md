# FIE 0.14.2 completion and verification

The current engineering scope is implemented and exercised. The [205-item disposition ledger](../artifacts/evaluations/release_0_14_2.json) retains all 189 original findings and adds 16 confirmed issues from this implementation. It links present implementation, executed regressions and actual-data evidence. Scientific hypotheses, unavailable external observations and deliberate modelling approximations are distinguished from defects. Original historical reports and supplied research documents remain unchanged.

## Implemented corrections

- Official registry and fixtures are validated and published together in one database transaction. A failed or malformed partner response retains the prior coherent market.
- Missing optional manager/workload sources and unavailable shot refreshes no longer abort independent forecasting/training. Partial workload batches retain the last complete data. Legacy cached workload is re-normalized from retained raw files; daily refresh has failure backoff.
- Forecasts pin live workload and companion models. Upgraded dependencies invalidate stale recommendations. Every numerical forecast is checked against the same worlds used by the optimizer, including coverage, probabilities, minute bounds, zero-minute points, quantiles and component totals. Double-gameweek p60 now means at least one individual match reaches 60 minutes.
- Full model compositions publish atomically after successful fitting and cancellation checks. Retained weights cannot be truncated by a failed writer. Manager-weight paths now include the source and feature contracts; two overwritten legacy companion files were restored, with the current versions retained separately. All 341 recovered original weights match their recorded hashes.
- Windows initialization and planning run as visible background jobs after the interface opens. Reloads discover running jobs, idle views discover scheduled refreshes, and cancellation/shutdown signal ongoing calculations. Live optimization has no forced time limit and targets zero additive-MIP gap. Explicitly budgeted historical experiments remain labelled.
- Rival comparisons work after GW9. Planner defaults remain valid after GW19. Account editing preserves selling-quote verification and uses concurrent-version checks. Public synchronization retained the completed Gabriel/Tavernier moves, bank and free transfers.
- Hosted extraction filters expired capabilities and outdated quality results, uses demonstrated current extraction reliability, resolves Unicode/full-name collisions and excludes related-story text. Missing publication zones remain bounded uncertainty, rather than fabricated UTC timestamps. Unsupported, stale or ambiguous evidence is rejected and queued with reasons. Neither hosted opinion nor manager ownership silently adds expected points.
- The full portable package now also carries the updater checksum manifest. Existing-installation updates preserve credentials, accounts, accepted models and evidence, with transactional rollback and SQLite backups.

## Research and numerical checks

FiveStat's September 2026 methodology and current public predictions, Core Insights' integration review/data, the open FPL solver, existing temporal/football studies and available alternative datasets were reviewed. These methods informed validation and comparison; their constants were not copied into accepted weights without acceptance evidence. Restricted sources were not bypassed.

The current Core Insights refresh retrieved 180/180 files, retained 3,451 accepted player-match records and quarantined 16 conflicting groups. The role train/serve audit matches all 669 checked player-fixtures across 57 inputs. The top-manager history contains 1,000 managers; its transfer/chip models predict rival behavior, rather than treating their decisions as future scoring labels.

A public FiveStat adapter validates player identity, club, position, fixture and capture cutoff. Its 128-player/640-fixture capture supplies 512 aligned comparisons with the current forecast. Mean absolute point disagreement is approximately 0.616; this is disagreement, not measured prediction error or an accepted ensemble gain.

The actual 670-player, GW11–38 forecast has 18,760 player-gameweek rows. All world/summary checks pass. Official xG/xA and per-90 rates are finite and consistent within published rounding. Signed BPS values are legal. Actual tree split gains confirm that team/opponent strength, expected attacking rates, exposure and other declared features are consumed; importance is not causal attribution. The primary conditional-points model does not uniquely identify separate predicted xG, clean-sheet and bonus components. Those outputs remain labelled physical or independent alternatives, rather than invented explanations of its total.

The pre-GW10 diagnostic uses only earlier labels and the dated registry: Gabriel **5.49695 xPts / 87.1359 minutes**, Virgil **5.20385 / 88.4846**. Their prior xG was 1.47 versus 0.36, with BPS 197 versus 141. This reproduces the immediate comparison and does not support the old Van Dijk conclusion on immediate points alone. It does not prove the causal contribution of each statistic or force a player-specific transfer rule.

A new chronological club-minute reconciliation study improved points MSE and participation Brier scores, but reduced the captain diagnostic from 129 to 124 in acceptance and 262 to 260 in replication. It failed its declared gate and remains inactive. Previous rejected advanced-statistics, bias and neural candidates remain inactive. No accepted numerical family was replaced to fit the inspected nine weeks.

## Executed verification

- **317 backend tests**, Ruff, formatting, typing and frontend build; **3 frontend unit tests**.
- **24 desktop/mobile browser journeys**, including account editing, market, planning, exports, research/rival views, background-job discovery/cancellation and the second-half horizon.
- Windows and Linux CI, including Windows dependency/bootstrap, PowerShell parsing, launcher, packaged updater with a folder containing spaces, retained configuration/weights, built interface and loopback HTTP smoke checks.
- Actual official refresh/account synchronization, complete workload refresh, cached Understat fallback, current rival comparison and current companion-model loading.
- Actual free hosted calls: both current providers pass synthetic guard checks; the inspected authentic corpus scores Groq 16/16 and Gemini 12/16. An unsupported Gemini batch and invalid live-article responses were rejected. These are extraction checks, not football-performance improvements.
- Fresh empty-model training, forecast generation with absent optional feeds, and a complete unrestricted one-week solve finishing **Optimal / gap 0**. An actual longer unrestricted search emitted progress, remained HTTP-responsive and cancelled without replacing the selected plan or account.
- Installed package source against a copy of the real approximately 1 GB database: accepted base/role/future/manager weights load, readiness and HTTP exports pass, **425 numerical text weights** and credentials remain unchanged, and actual backup integrity and recovery pass. These full-data installation checks ran on Linux; the separate Windows CI executes the supported launcher and regression workflows.

The complete composed continuous-account rerun independently checks all 27 actions, model hashes, chronological cutoffs and account transitions. The retained 84-action/28-deadline pipeline proof and earlier-season 19-week future-model proof were also reverified.

| GW | Independent weekly squad | Continuous combined | Continuous reference | One-week/no-chip control |
|---|---:|---:|---:|---:|
| 1 | 48 | 41 | 41 | 48 |
| 2 | 93 | 94 | 93 | 84 |
| 3 | 54 | 36 | 36 | 50 |
| 4 | 63 | 61 | 61 | 71 |
| 5 | 65 | 53 | 53 | 56 |
| 6 | 44 | 30 | 29 | 41 |
| 7 | 61 | 49 | 49 | 54 |
| 8 | 65 | 86 | 86 | 46 |
| 9 | 82 | 60 | 62 | 49 |
| Total | **575** | **510** | **510** | **499** |

Independent weekly £100m squads are not a legal repeated-Free-Hit strategy on one account. Continuous results use complete legal transfers, hits, chips, captaincy and autosubs, with matched starting accounts. These inspected historical reconstructions use explicit research budgets; their results do not certify an unlimited multiweek optimum or consistent future 550+ returns. The earlier-season future-model replay retains **1,202 versus 1,098** under its separate protocol.

## Delivery and external boundaries

Use the [small update](https://github.com/DestinedAlpha/FIE/releases/download/v0.14.2/FPL-Intelligence-0.14.2-Update.zip) for an existing installation. Extract separately, close FIE, run **Update existing.cmd**, select the existing application folder, then open **Start.cmd**. The private full portable handoff contains the preserved data/models for installation recovery; credentials are excluded from both archives. A separate delivery receipt records the final archive hashes and actual download verification, avoiding a self-referential package checksum.

No known actionable engineering defect remains from this verification. Understat's five live season requests returned unusable HTML; 39,804 retained validated observations and independent official/Core sources remain available. Missing proprietary tracking, incompatible odds markets or unobserved competitions are not fabricated. Public endpoints cannot disclose unrevealed private transfers. Future football outcomes, causal superiority and guarantees across every possible team are scientific boundaries, not unfinished implementations.
