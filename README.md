# FPL Intelligence

A private local FPL decision workspace: learned player-point distributions, exact account ledgers, full-market joint transfer/chip planning, explicit rival scenarios and source-grounded optional hosted research. Numerical training runs on a CPU; paid APIs and local language models are not required.

The six supplied documents remain byte-for-byte unchanged in [`research/`](research/). Design documents are not evidence of competitive performance.

**Current acceptance status:** the role-v3 configuration scores **575** in independent weekly £100m teams. Historical future-expert validation scored **1,202 versus 1,098** in an earlier-season 19-week continuous replay and **538 versus 536** over GW1–9 under its earlier protocol. These different account problems do not certify the complete current policy. **Consistent 550+ continuous-account performance is not established.** See the [data/forecast audit](docs/data-and-forecast-audit.md) and [performance audit](docs/performance-audit.md).

## What changed in 0.14.5

Refresh data / Load team now retrieves official data and synchronizes the revealed account without training, forecasting or starting an unlimited transfer search. **Update forecasts** handles finalized labels, accepted-model learning and forecasts separately; **Optimize plan** remains explicitly unlimited. Calculations run in supervised processes: Cancel stops an unresponsive worker, cancels queued follow-ups and releases the controls only after exit. Progress reports the actual phase. Cross-process account locks and atomic normalized-data replacement protect concurrent edits and interrupted writes. Accepted numerical models are unchanged. [Verification and update instructions](docs/completion-0.14.5.md).

## What changed in 0.14.4

Completed controlled temporal/calibration research and real automatic-learning verification. Accepted production weights remain unchanged: neither shorter/cross-fitted calibration nor the selected decay candidate justifies replacement. Fixed finalized-data refresh eligibility, companion coverage/publication, typed feature checksums, season-specific scoring/cutoffs and isolated report writes. Model & data now shows actual fitting/calibration coverage. The normal refresh learned GW9, forecast all 671 players and produced an independently checked legal plan; unchanged data caused zero additional fits. Final regressions: **361 backend tests, 3 frontend tests and 28 browser journeys**. [Completion and evidence](docs/completion-0.14.4.md) · [Audit dispositions](artifacts/evaluations/release_0_14_4.json).

## What changed in 0.14.3

Added population-wide forecasting/calibration diagnostics, integrated API/UI monitoring and a controlled FiveStat-inspired component/scoreline study. The global and positional challengers failed their complete acceptance gates, so accepted production weights remain unchanged. Corrected cached Understat identity joins recovered 7,249 observations without network requests (47,053 retained total), with an offline recovery control and strict ambiguity/cancellation guards. Forecast display/world reconciliation and research-cache integrity are stronger. [Completion and research evidence](docs/completion-0.14.3.md) · [Download update](https://github.com/DestinedAlpha/FIE/releases/download/v0.14.3/FPL-Intelligence-0.14.3-Update.zip).

## What changed in 0.14.2

Completed transactional refreshes, optional-source fallback, workload migration/cadence, full forecast-to-world checks, companion lineage, atomic model publication, grounded hosted extraction, current rival comparisons and cancellable Windows startup. Added identity/fixture/cutoff-matched FiveStat comparisons. Accepted numerical models remain preserved; the new minutes-balancing candidate failed its captaincy gate and was rejected.

Verification includes 320 backend tests, 24 desktop/mobile browser journeys, Windows/Linux CI, fresh model training, an optimal unrestricted one-week solve, real long-search cancellation and installation/backup recovery against the real database. The original 189 findings and 17 new issues have [evidence-backed dispositions](artifacts/evaluations/release_0_14_2.json). [Full completion report, numerical checks and installation instructions](docs/completion-0.14.2.md). [Download the existing-installation update](https://github.com/DestinedAlpha/FIE/releases/download/v0.14.2/FPL-Intelligence-0.14.2-Update.zip).

## What changed in 0.14.1

Shot ingestion now diagnoses HTML error pages, validates provider schemas before caching, and preserves verified observations across partial refreshes. Hosted evidence has independent availability-polarity and training-status guards; upgrades also revalidate old cached claims. The real-news extraction benchmark supplements the earlier synthetic examples without turning language-model opinions into point adjustments.

`fpl replay-composed` evaluates the combined role, conditional-points and future-fixture models through continuous legal accounts, with matched reference and simpler controls. Sealed decisions are independently re-scored; results appear in Model & data. It is a bounded historical diagnostic and does not certify unlimited live search or future superiority. `fpl benchmark-authentic-extraction` checks the frozen real-news corpus using eligible free providers. See [the research review](docs/research-review-2026-11-06.md).

The completed GW1–9 continuous diagnostic scored **510 / 510 / 499** net points for the combined policy, the future-model reference and the one-week/no-chip control. All 27 actions were independently verified. These results do not establish consistent 550+ performance or a protected acceptance of the unlimited current policy.

## What changed in 0.14

Live Optimize plan and Refresh data searches have no forced solve-time cutoff and target zero MIP gap, with progress and cooperative cancellation. Explicit finite research budgets remain available for reproducible comparisons. Exact search still optimizes an additive forecast objective, not unknowable future football outcomes.

The audit fixes account-editor verification/provenance, version conflicts, late-result publication, scheduler recovery, updater hash checking and rollback/recovery, missing-label and conflicting-duplicate guards, extractor status gates and stale exported schemas. Accepted numerical weights remain unchanged. The 575-point weekly scorecard and older continuous-policy results are retained historical protocols; they do not certify the new unlimited-search policy. See [the audit](docs/current-audit-0.14.md).

## Windows

**[Download the 0.14.5 existing-installation update](https://github.com/DestinedAlpha/FIE/releases/download/v0.14.5/FPL-Intelligence-0.14.5-Update.zip).** This small update requires your existing installation with its data and models.

For an existing installation, extract the new private download into a **separate folder** and run **Update existing.cmd**. Select the existing folder containing Start.cmd. The updater backs up the database/application, preserves your `.env` and account data, updates the application, migrates incompatible companions when necessary and identifies stale forecasts/plans for the explicit update/optimization controls. Compatible accepted models are preserved. It does not apply any FPL account transfer.

For a fresh installation, double-click **Start.cmd** in the extracted package. The prepared ZIP contains the built interface, numerical data and a compressed consistent database. The launcher restores it before provider setup and opens a healthy loopback server. uv installs the CPU numerical dependencies automatically; the prepared package does not require Node.

**Configure keys.cmd** reruns secure provider prompts. Gemini and Groq authenticate independently. An invalid or unavailable optional provider is reported without stopping the numerical planner or disabling a working provider. Keys stay local and are excluded from source, downloads and reports.

## Source installation

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Node 24. Run from the repository root:

```sh
uv sync --frozen --extra dev
npm --prefix apps/web ci --no-audit --no-fund
npm --prefix apps/web run build
uv run fpl ingest
uv run fpl reconcile-team
uv run fpl train
uv run fpl refresh-current
uv run fpl serve
```

Open http://127.0.0.1:8000. `reconcile-team` imports the **dated** research seed; confirm current account facts before acting. The private prepared package uses the user's separately confirmed selling quotes, bank, FT and chips. Unknown purchase prices remain null. The original seed cannot overwrite a confirmed account.

`fpl train` imports the four preceding compatible season archives if absent, builds shared train/serve features, runs expanding chronological folds, calibrates minutes on separate earlier deadlines and fits the production points model. The Refresh data / Load team buttons reconcile revealed deadline squads and retrieve current official data. Update forecasts learns newly finalized labels and regenerates forecasts; Optimize plan runs the separate transfer/chip search. The CLI `fpl refresh-current` retains its complete workflow. Confirm any additional unrevealed transfers in My squad; public endpoints cannot see them.

## What changed in 0.13

Reconstructed incomplete local context from actual pre-deadline full snapshots, fixed historical/live precision and context binding, and repaired migration. All 57 inputs match across 669 audited GW9 player-fixtures. The independent weekly diagnostic scores **575**. Captain explanations show vice fallback; source panels identify actual consumers. Planning has full-market starting trajectories and independent relaxation bounds. More complex scoring/bias candidates failed acceptance and remain inactive. [Evidence and limitations](docs/data-and-forecast-audit.md).

## What changed in 0.12.1

The weekly replay now evaluates the accepted role-aware configuration, refitting strictly before each deadline. That configuration scored **569** over GW1–9; the current diagnostic is 575. The dashboard selects matching evaluated configurations and preserves earlier reference benchmarks separately. [Weekly results and GW10 transfer comparison](docs/gw1-9-and-gw10-review.md).

## What changed in 0.12

Removed local neural training/inference, weights, dependencies and controls. Existing-installation migration preserves the account, keys, accepted tree models and legal training outcomes, and refits companion tree models after removing the retired forecast input.

Added a separately validated role-aware minutes classifier using recent starts, archived season role, availability and observed workload. Selection uses 2024/25 GW1–19; later 2024/25 and all 2025/26 are held out. It improves participation calibration and point error in both periods. Ordinary retraining learns newly finalized results. Its forecast validation is separate from the older full-policy scorecards.

Fixed unused substitutes resetting physical rest; frozen older models retain their original transform. Core Insights imports quarantine conflicting duplicate matches. Paid transfers require evidence beyond unresolved no-hit search. Optional current chip use also requires support over an ordinary-transfer control with later chips available; committed, explicitly fixed and expiring entitlements retain their rules. No named player is forced into or excluded from recommendations.

[Research and implementation decisions](docs/research-review-2026-11-06.md) · [Source fusion](docs/source-fusion.md) · [Decision learning](docs/decision-learning.md) · [Performance audit](docs/performance-audit.md).

## Capabilities

- Official full-market registry, fixtures, all histories, entry/league data, immutable hashed raw snapshots, Parquet features and portable SQLite state.
- Integer selling prices and gain sharing, acquisition lots, formations, club/FT limits, hits/refunds, chip windows, temporary Free Hit squads, restoration, captain/vice, autosubs and scoring. Every plan is independently replayed.
- Full selectable-market multiweek HiGHS optimization, joint chips/transfers, legal warm incumbents, actual solver gaps, common-world rescoring and whole-policy chip counterfactuals. A timeout result is a feasible candidate, not a claim of global optimality.
- Learned points/minutes distributions through GW38; empirical residual tails and measured within-team role dependence. These are statistical outcome worlds, not reconstructed match-event XIs. Rival probabilities remain conditional on an approximate joint model and explicit opponent policies.
- Price, fixture and availability scenarios with a shared decision prefix until observation, adaptive controls, no invented branch probabilities and labelled conditional purchase-lot assumptions.
- Supported official-source claims, freshness/quote/provenance validation, duplicate/contradiction handling, Gemini primary and eligible Groq fallback, free-only quota guards and no paid grounding/tools.
- Captain comparison, bias/decision/uncertainty scorecards, actual-manager comparators, season ledger, JSON/CSV/Markdown/full-market exports, ICS deadlines, pre-deadline frozen shadow decisions and relocated backups.

## Useful commands

```sh
uv run fpl doctor
uv run fpl decision-audit
uv run fpl policy-replay
uv run fpl uncertainty-audit
uv run fpl manager-benchmark
uv run fpl configure-providers --unbilled
uv run fpl research-refresh
uv run fpl rivals --case --seconds 45
uv run fpl chip-comparison
uv run fpl freeze
uv run fpl evaluate --prospective
uv run fpl replay
uv run fpl export
uv run fpl backup
```

Only attest `--unbilled` for genuinely free, unbilled accounts. Capabilities expire after seven days; inference reserves local quotas and never escalates to paid features. Provider keys are automatically loaded from `.env` without overriding inherited variables. The numerical model does not rely on hosted AI.

## Verify

```sh
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run mypy src
uv run pytest -q
npm --prefix apps/web run build
npm --prefix apps/web test
```

Desktop/mobile Playwright journeys run against a seeded local server. CI runs offline backend/frontend checks on Linux and Windows, parses the PowerShell launch/update scripts, and starts the Windows server from a compressed packaged database. Runtime data and credentials are ignored by the public Git repository. See [operations](docs/operations.md), [methodology](docs/methodology.md) and [API contracts](packages/contracts/openapi.json).
