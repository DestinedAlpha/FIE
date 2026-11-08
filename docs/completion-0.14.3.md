# FIE 0.14.3 — implementation and release evidence

This release completes the current actionable engineering and selected research cycle. The accepted role-v3, conditional-points and future-expert composition remains active. New methods were evaluated against that composition; a reduction in average error alone did not authorize replacing it.

## Implemented

- Population diagnostics cover 62,659 held-out player-fixtures across 2024/25, 2025/26 and completed 2026/27 GW1–9. Every reconstructed accepted point mean matches the retained role-v3 evaluation exactly. Twelve expanding folds reserve eight earlier calibration deadlines and seal probabilities, conditional returns, residual sampling and weight hashes. Reports include points, minutes, starts, appearances, 60-minute probabilities, haul probabilities, calibration bins, CRPS and intervals; 19 cohorts per season include positions, premiums, starters, rotation, promoted clubs, team strength and fixture difficulty. Double fixtures are aggregated before gameweek/captain comparisons.
- The API, CLI and Model & data interface now expose recurring forecast-cohort checks, historical evidence and timestamp/identity/fixture-aligned independent disagreements. Flags distinguish participation differences from conditional-scoring assumptions; they are inspection categories, not causal attribution or proof that another provider is correct. Existing-installation updates display bundled evidence without importing accounts or changing models.
- Forecast validation now verifies displayed quantiles and haul probabilities against the exact optimizer worlds, reconciles fixture counts/points/minutes to gameweek summaries and rejects invalid fixture probabilities. The actual 20,000-world, 18,760-player-gameweek production forecast passes these checks.
- Horizon research caches now bind their dataset, parameters and feature contracts and verify retained weight checksums. Legacy unsealed caches remain preserved. New research prediction files are immutable and content-addressed.
- Understat joins now distinguish tactical roles from FPL positions, handle complete surname prefixes and verified official display aliases, normalize equivalent club names and accept stable provider-ID/FPL-code links only with a unique bijection supported by at least three retained fixtures. Fixture/club/goalkeeper constraints and ambiguity rejection remain enforced. Source-player/fixture duplication blocks publication.
- Offline cache recovery increased the retained corpus from **39,804 to 47,053 observations**, preserving all original keys and immutable original records. All **7,249** additions change their corresponding live enrichment vectors and respect strict completion cutoffs. The remaining **1,194** identities lack a unique supported match and are excluded. A cancellable background API/UI/CLI operation reprocesses this cache without contacting Understat; job results return a bounded summary. Live source availability is labelled separately.
- Installation/update text now describes the actual conditional migration behavior; a compatible update preserves accepted weights and does not force a complete retrain.

## Research resolved

The current [FiveStat methodology](https://www.fivestat.co.uk/methodology), [Core Insights](https://github.com/olbauday/FPL-Core-Insights), [Solio solver](https://github.com/solioanalytics/open-fpl-solver), [OpenFPL](https://github.com/daniegr/OpenFPL), The Open Model, StatsBomb coverage/licensing and available football-data sources were inspected. Capture hashes and dispositions accompany the release. Their reported accuracy is not presented as an FIE result. StatsBomb's open Premier League seasons are 2003/04 and 2015/16; Sideline Quant's advertised current feed returned 404. No restricted endpoint was bypassed and no paid call was used.

A functioning FiveStat-inspired challenger evaluated hierarchical attack/defence strength, home advantage, recent adjustments, independent/additive bivariate Poisson score matrices, team-goal allocation and player season/recent attacking shares. It reports scoreline NLL, RPS, 1X2 Brier, clean-sheet, over/under and BTTS errors. The tested recent-team weights were 0/0.15/0.30/0.50 and shared covariances 0/0.05/0.15; earlier-period selection chose **0 and 0**. Player shares and point blends were selected separately. Opta-style underlying xA and expected FPL assists have distinct targets and units. Appearance, goals, assists, clean sheets, conceded goals, saves, cards, bonus, penalties, own goals and DEFCON reconcile within the stated component approximation.

| Held-out period | Accepted RMSE | Global challenger RMSE | Accepted captain points | Challenger captain points |
|---|---:|---:|---:|---:|
| 2024/25 GW20–38 | 1.81955 | 1.80912 | 129 | 151 |
| Full 2025/26 | 1.88267 | 1.87533 | 262 | 250 |

The global blend improves average error but fails the declared captain gate. All five predeclared global/position families were resolved using configurations selected only on 2024/25 GW1–19 and simultaneous 99% deadline-block intervals. None passes every gate. The defender specialist fails confidence and captain checks; the midfield specialist falls from 262 to 261 replication captain points. These are completed rejection decisions, not pending production integrations. No rejected component affects the production optimizer.

The component audit evaluates expected goals, underlying xA, FPL assists, clean-sheet points, conceded points, saves, bonus and DEFCON against compatible observed targets. Team/player goal allocation, FPL-assist components and total components reconcile to floating-point precision. Missing observations are counted and excluded. The direct learned-points head does not uniquely identify separate xG, clean-sheet or BPS contributions; its explanations do not invent them. Distant expert inputs include observed BPS; the separately evaluated event/BPS models retain their own acceptance status.

The frozen pre-GW10 Gabriel/Virgil case remains **5.49695 versus 5.20385 xPts**, based on earlier labels. Current public projections are never backdated into that comparison. No player-specific coefficient or transfer rule was added.

## Verification and preservation

The machine-readable [release ledger](../artifacts/evaluations/release_0_14_3.json) rechecks all 206 inherited findings, their current source/test paths and evidence, and records the new implementation dispositions. Original documents and prior results remain available.

Executed evidence includes backend regressions, frontend unit/build checks, desktop/mobile browser journeys, the actual offline background API operation, 669 historical/live role-input comparisons, full forecast/world reconciliation, model checksums, SQLite integrity, packaged update/install preservation and backup/recovery. Windows/Linux CI and the final downloaded package are identified in the delivery receipt. All 341 recovered original weight hashes and the production model/account/forecast/selection pointers are preserved.

The independent historical verification rechecks 27 composed-policy actions, 84 pipeline actions and the separate 19-week/9-week future-policy seals, training cutoffs, account continuity and actual scoring. Retained GW1–9 totals remain **575** for independent weekly squads and **510 / 510 / 499** for the composed continuous account, matched future reference and one-week/no-chip control. Independent weekly squads are not legal repeated Free Hits. The earlier-season future-policy comparison remains **1,202 versus 1,098** under its separate protocol. These research searches use their recorded finite budgets; live optimization remains unlimited and cancellable. No new strategy gain is claimed for the rejected challenger.

## Install

For an existing installation, download [FPL-Intelligence-0.14.3-Update.zip](https://github.com/DestinedAlpha/FIE/releases/download/v0.14.3/FPL-Intelligence-0.14.3-Update.zip), verify the accompanying SHA256 file, extract separately, close FIE and run **Update existing.cmd**. Select the existing application folder, then open **Start.cmd**. Credentials and account data stay local. The full prepared installation is delivered through the private handoff branch; older releases remain preserved.

There is no known unaddressed actionable engineering defect from this cycle. Live Understat access, unsupported ambiguous identities, unavailable proprietary observations and future football outcomes are external/scientific boundaries. The release does not claim universal statistical optimality, guaranteed future scores or that an unlimited additive solve knows future results.
