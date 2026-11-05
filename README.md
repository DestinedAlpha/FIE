# FPL Intelligence

A private local FPL decision workspace: learned player-point distributions, exact account ledgers, full-market joint transfer/chip planning, explicit rival scenarios and source-grounded optional hosted research. Numerical training runs on a CPU; paid APIs and local language models are not required.

The six supplied documents remain byte-for-byte unchanged in [`research/`](research/). Design documents are not evidence of competitive performance.

**Current acceptance status:** the validated future-fixture expert scores **1,202 versus 1,098** in a matched earlier-season 19-week continuous replay, and **538 versus 536** over current GW1–GW9. Future prediction and distribution error also improve. Separately, independent weekly £100m teams score 551 for the refitted reference and 586 for a published-signal ensemble; those are different account problems. **Consistent 550+ continuous-account performance and future top-1% strength are not established.** See the [performance audit](docs/performance-audit.md).

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
uv run fpl refresh-current
uv run fpl serve
```

Open http://127.0.0.1:8000. `reconcile-team` imports the **dated** research seed; confirm current account facts before acting. The private prepared package uses the user's separately confirmed selling quotes, bank, FT and chips. Unknown purchase prices remain null. The original seed cannot overwrite a confirmed account.

`fpl train` imports the four preceding compatible season archives if absent, builds shared train/serve features, runs expanding chronological folds, calibrates minutes on separate earlier deadlines and fits the production points model. `fpl refresh-current` and the Refresh data button reconcile revealed deadline squads, advance to the next deadline, update finalized labels and recompute the recommendation. Confirm any additional unrevealed transfers in My squad; public endpoints cannot see them.

## What changed in 0.11

[Executed correction checks](artifacts/evaluations/release_0_11.json).

Fixed paid-transfer selection: a tiny long-horizon advantage with a narrow Monte Carlo interval can no longer justify a hit while the full no-hit search remains materially unresolved. The selector exposes the additive solver allowance separately from sampling and forecast error, retains the better-supported no-hit route, and shows each week's net payback. Material paid improvements can still pass; there is no hard transfer/hit cap or player-name rule.

The planner displays the forecasts behind withheld transfers: fixtures, expected minutes/starts, official availability, current-season starts and independent official next-week estimates. Those official estimates are sensitivities, not ground truth or silent model replacements. Persisted expected-points paid recommendations require the current evidence policy after an upgrade. Replay protocols identify the changed selector; older frozen scores remain historical evidence for the earlier policy.

Existing installations can use the small application update: extract it separately and run `Update existing.cmd`. It preserves the account, keys and complete numerical models/data, then rebuilds forecasts and recommendations. The update is verified by actually installing the downloaded archive into an existing test installation and checking the database, keys, model files and backup.

## What changed in 0.10

Completed the review of all 36 numbered sections in both original design documents. The [specification review](docs/specification-review.md) indexes their body clauses and connects every section to concrete implementation evidence; release verification rejects unreviewed source changes or missing sections. This supersedes the earlier selected-behaviour-only review without claiming exhaustive testing of every input.

The chip report now evaluates every feasible date for all four available chip types, with separate first/second-half entitlements, complete transfer preparation/cleanup and common-world scores. Other chips remain jointly optimized in each comparison. Date-specific solver diagnostics and downloadable ledgers appear in the planner. Full training considers these candidates before final policy selection.

Fixed Free Hit planning after already-confirmed transfers: future weeks remain eligible, and a current Free Hit uses the original deadline squad/bank, reverses applicable earlier costs and preserves the requested continuation constraints. Regression tests exercise actual solver/account transitions across chip periods and restoration.

## What changed in 0.9

Connected a validated future-fixture expert to the production forecaster, ordinary weekly retraining, full transfer/chip optimizer, scenario overrides, dashboard and existing-installation updates. Three heads learn minutes, conditional points and starts from 138 dated inputs, including advanced statistics, prices, ownership and missingness. Target windows are purged across fitting/calibration/test boundaries. Published next-week estimates are excluded from distant weeks; current-week worlds remain unchanged. Forecast, marginal distribution and full continuous-policy gates all pass. Final fitting uses completed labels through GW9 with reserved calibration data.

Ingestion now preserves actual predeadline registries for subsequent learning. Historical feature chunks are reused when unchanged. Independent verification covers the new sealed teams, training cutoffs, model files, current-world identity, continuous finances and actual returns; the real account stays separate from historical and prospective comparisons. Frontier refinement retains the stronger existing policy. Large replay/recourse scratch files use the data/workspace disk.

The prepared release includes the new models, completed scorecards, latest personalized report and built interface. The updater preserves accounts and keys, imports the new acceptance evidence and attaches an accepted expert during retraining. Full training includes this component; failed experiments remain comparisons. [Forecast and policy results](docs/performance-audit.md) · [Learning contract](docs/data-contract.md).

## What changed in 0.8

Connected the production forecaster, full chip-window transfer planner and exact account engine in a continuous historical replay. Models fit only earlier completed observations; availability gates freeze before each evaluation period. Each decision retains its models, forecasts, input/cutoff proofs and account transition. Independent checks match the sealed actions, verify all model files, and replay finances, chips, autosubs and actual scoring across deadlines. Fixed initial-selection FT allocation: one free transfer for the next week.

Added synchronized prospective accounts with equal initial budgets and known purchase lots. Refreshes freeze decisions before deadlines, then score finalized matches, carry accounts forward, handle a missed capture using an earlier standing hold instruction and apply official score corrections without double counting. These hypothetical accounts are separate from the user's actual team. The Model & data page exposes both evaluations and downloadable frozen teams.

The Windows package now uses checksummed compressed database parts below GitHub's individual-file limit. Restoration is atomic, legacy packages remain supported, and updates preserve existing accounts and keys. Completed replay work can be recovered from portable decisions. [Data and learning contract](docs/data-contract.md) · [Functional coverage](docs/completion-matrix.md).

## What changed in 0.7

Added archived official expected points, form and points per game to multi-source football features, with missingness masks and correct allocation of weekly expected points across announced doubles. Select the reference/crowd blend on 2025/26 GW1–19 and check GW20–38 before reporting the already-inspected current-season diagnostic. Its 87-point earlier-season weekly-team gain passes the weekly gate, but it fails a separate ordinary-transfer gate from identical starting states. Failed evidence remains visible.

Model & data shows complete GW1–GW9 tables, matched controls, weekly versus continuous-account results and legal current-squad ensemble alternatives, including hold and no-hit. Its current one-deadline analysis is Cherki → Mbeumo, no hit, Haaland captain. This does not replace the selected multiweek plan.

Fixed floating-point archive fixture IDs, accepted proven local pre-deadline GW9 captures, zeroed baselines in announced blanks, preserved frozen earlier weeks during replay extension, handled club changes and GW1 FT initialization in continuous replay, reproduced predictions before reusing cached solves, and included these research artifacts in Windows updates. Commands: `fpl train-consensus`, `fpl validate-consensus-policy`, `fpl compare-consensus`. The prepared package contains completed results.

[Executed 0.7 checks](artifacts/evaluations/release_0_7.json). The full multiweek/chip/rival policy still needs synchronized competitive validation; this release does not claim exhaustive blueprint acceptance or guaranteed future scores.

## What changed in 0.6

Fixed deadline rollover and public account synchronization, role-matched transfer displays, incremental hit accounting, stale-plan controls and joint captain/vice fallback. Recommendations compare hold and independently solved no-new-hit policies under the same sampled outcomes. Explicit transfer/chip constraints also apply when selecting alternatives.

Added learned injury recovery and double-gameweek rotation, competition-specific suspension service, separate attacking channels, structural score-process comparisons, optional dated odds calibration and separately tested decay/finishing corrections. Each change has a later-deadline acceptance gate. Existing neural/tree models retain the feature transformations frozen in their training lineage. Unsupported additions remain disabled.

The full workflow connects qualified and current top-1,000 manager behavior, separate neural/tree forecasts, random-account transfer learning, rival branches, population scenarios, source-supported sensitivities and complete policy recourse. The Model & data page exposes data coverage and actual adoption decisions. Historical GW1–8 teams use archived pre-deadline registration, prices and fixtures, refit each week, and score target outcomes only after freezing their choices.

[Implementation coverage](docs/completion-matrix.md) · [Football models and temporal limits](docs/football-models.md) · [Executed 0.6 release audit](artifacts/evaluations/release_0_6.json).

## What changed in 0.5

Completed the connected decision-learning path: full-market random-account trajectories evolve transfers and chips across four weeks, an integrated tree/listwise neural critic compares legal actions, and a separately gated model values squad, bank, FT and remaining chips. The neural forecaster feeds the optimizer, manager-choice models and decision comparison. Failed empirical gates retain the established policy.

Actual top-1,000 histories now train conditional purchases, sales and chip behavior as well as transfer counts. Future rival branches update their own squads, bank, FT and history. A separate rank-stratified population sample includes inactive entries. These remain explicitly conditional scenarios, with sampling and policy limitations.

Added verified pre-deadline price/news learning, early-versus-wait recourse, purchase-lot-aware affordability, report-value research priorities, full-horizon observable scenario re-optimization, structural football-world policy comparison, independently retrained source ablations, activated-chip handling and optional local automatic refresh. Every search permits the full registered market and all 15 ordinary transfers, with actual gaps and timeouts disclosed.

Source-off point RMSE is **1.9534 history-only → 1.9169 official news → 1.9095 full context**, across 34,828 matched held-out rows. Conditional manager purchase/sale log loss improves **4.553 → 3.608 / 3.160 → 1.808**; the chip challenger fails and its baseline remains active. The 48-hour price model improves log loss **0.1283 → 0.0827** across 46 held-out deadlines. Conditional BPS improves RMSE **5.672 → 5.020** and match-wide bonus **0.409 → 0.398**; those tests condition on known events and do not prove future end-to-end accuracy.

[Implementation and evidence matrix](docs/completion-matrix.md) · [Decision-learning methods and limitations](docs/decision-learning.md) · [Executed release audit](artifacts/evaluations/release_0_5.json).

For full CPU retraining use `uv run --extra neural fpl train-complete`. It is a long job; the prepared package already includes trained artifacts. `strategy-analysis`, `timing-analysis`, `refine-frontier` and `population-analysis` rerun individual connected analyses. The interface exposes the same durable jobs. Fresh data and finalized labels trigger recomputation; submitted account changes must be reconciled independently.

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
