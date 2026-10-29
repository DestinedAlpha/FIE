# FPL Intelligence

A private local FPL decision workspace: learned player-point distributions, exact account ledgers, full-market joint transfer/chip planning, explicit rival scenarios and source-grounded optional hosted research. Numerical training runs on a CPU; paid APIs and local language models are not required.

The six supplied documents remain byte-for-byte unchanged in [`research/`](research/). Design documents are not evidence of competitive performance.

## Windows

For an existing installation, extract the new private download into a **separate folder** and run **Update existing.cmd**. Select the existing folder containing Start.cmd. The updater backs up the database/application, preserves your `.env` and account data, updates the application, retrains against current completed data and recomputes the plan. It does not apply any FPL account transfer.

For a fresh installation, double-click **Start.cmd** in the extracted package. The prepared ZIP contains the built interface, numerical data and a compressed consistent database. The launcher restores it before provider setup and opens a healthy loopback server. uv installs Python dependencies, including CPU-only PyTorch for the neural-training controls, automatically; the prepared package does not require Node.

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
uv run fpl forecast --end 38 --worlds 4000
uv run fpl optimize --end 19 --seconds 90
uv run fpl serve
```

Open http://127.0.0.1:8000. `reconcile-team` imports the **dated** research seed; confirm current account facts before acting. The private prepared package uses the user's separately confirmed selling quotes, bank, FT and chips. Unknown purchase prices remain null. The original seed cannot overwrite a confirmed account.

`fpl train` imports the four preceding compatible season archives if absent, builds shared train/serve features, runs expanding chronological folds, calibrates minutes on separate earlier deadlines and fits the production points model. After new matches, refresh data and retrain/reforecast/replan. Account transfers are never assumed to have been submitted; reconcile actual squad changes separately.

## What changed in 0.4

Added a purpose-built temporal multi-source neural challenger, archived pre-deadline official news, verified prior-qualified manager inputs, current top-1,000 transfer-behavior analysis, all-competition workload, purged four-week values and legal random-account transfer learning. The Model & data and Rivals pages show completed comparisons and their actual source coverage.

On the same 34,828 held-out player-fixtures, point RMSE improves from 1.9785 to 1.9095 for multi-source trees and 1.9195 for the neural network. The new transfer actor does not beat the reference in evolving-account replay, so the established transfer model remains active. A separate critic learns from 912 synthetic legal accounts; its later-test gain interval includes zero, and personalized alternatives stay labelled as research. Model size and a small score lead are not promotion evidence.

The current-ranked cohort contains **1,000 actually retrieved managers and 8,000 released manager/gameweeks**. Their previous squads and transfers inform a conditional behavioral model and explicit rival branches. This is separate from the 100 managers qualified on prior-season results. Scout claims inform source-linked risk analysis; Reddit supplies leads to verifiable originals, without raw Reddit training or a blocked-access bypass.

The optimizer reuses stronger validated feasible incumbents, and no-hit comparisons include already-known alternatives so a weak timeout result cannot exaggerate the benefit of hits. Provider reconfiguration and package updates preserve existing keys and account data. [Full methods, executed results and limits](docs/source-fusion.md).

## What changed in 0.3

The former component simulator could produce repeated defender captaincies. Its points forecasts had not been tested by the separate direct-model benchmark. The production planner now consumes **the same learned-points model and outcome sampler that are evaluated**, with one shared feature transformation. Price, opponent strength, role, recent form and stable-code prior-season histories are inputs. Current/future outcomes and hindsight winner selections are excluded from historical feature cutoffs. Early 2022 fixtures whose starting indicators were all zero are treated as missing start labels.

The former physics-style component simulator remains available only through explicit research-mode Python calls and cannot replace production forecasts or produce an execution-ready plan. The old benchmark artifacts are retained as earlier experiments, not current readiness evidence.

## Reference planner evidence (0.3 architecture)

| Check | Result |
|---|---|
| Held-out player-fixtures | 34,967, 2025/26 plus completed current-season events |
| Point RMSE, rolling prior → learned | 2.0786 → 1.9752 |
| Likely-starter RMSE | 3.1247 → 3.0335 |
| Neural challenger RMSE | 1.9726; paired gain interval includes zero |
| Held-out captain bonuses: learned / price / rolling prior | 323 / 274 / 283 |
| 30-week transfer policy: learned / rolling prior | 1708 / 1527 observed net points |
| Likely starters inside central 80% / 90% intervals | 87.9% / 94.1% |
| Manager comparator | 50 actual squads; 1 qualifies on repeated prior top-10,000 finishes |

The neural network was tested rather than presumed necessary. Its small RMSE difference did not establish a reliable gain over trees. Neither this comparison nor historical returns establish top-1% strength. Historical archives contain retrospective corrections/prices; these results are **statistical replay**, not strict historical news reconstruction or prospective outcomes. Current-rank sampling remains biased and is not relabeled as an elite-manager population.

Hindsight-budget teams are independently optimized as a diagnostic ceiling. Their realized winners are never used as privileged test-week training labels. Full source, protocol, bias cohorts and uncertainty measurements are in the [model card](docs/model-card.md) and [`artifacts/evaluations/`](artifacts/evaluations/).

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
