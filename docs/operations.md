# Local operation and recovery

## Updating a laptop installation

Extract the new private package into a separate folder. Run `Update existing.cmd` and select the old folder containing `Start.cmd`. The updater stops only FPL servers whose interpreter is inside that folder, validates the application file manifest, creates a consistent account database/application backup, and copies application files while excluding `.env` and `data/`. It then starts the existing installation with `-RebuildModel`: current official data is refreshed, the four preceding archives are imported if absent, the learned-points model is trained/calibrated/evaluated, and the revealed deadline squad is reconciled, and a new forecast/plan is computed. Newer confirmed account edits are preserved. An unresolved selling quote retains a valid hold analysis and opens the interface.

The current account must still reflect actual submitted FPL actions. The updater cannot infer private transfers or acquisition prices. Model changes invalidate old forecasts/plans. A retired component forecast is never presented as execution-ready. Keep the `artifacts/runtime/before-update-*` backup until the new installation passes its checks.

For a fresh installation use `Start.cmd`. The package contains `data/state.sqlite.gz`; the launcher restores it atomically **before** optional provider setup can create a database. The source checkout's readiness check tests actual data/forecast contents rather than the existence of an empty SQLite file. The prepared ZIP includes the built frontend; no Node installation is needed.

`Configure keys.cmd` reopens secure key prompts. Per-provider HTTP/network failures are reported independently. A failed Groq key does not stop a valid Gemini provider or the numerical application. The original unremembered laptop error cannot be diagnosed precisely without its text; the setup path now has explicit nonfatal handling and credential-safe diagnostics.

## State and access

The API binds to loopback only. Trusted host and origin checks reject cross-origin mutation and DNS-rebinding hosts. No credentials are sent to the browser. `FPL_DATA_DIR` selects the local data directory. Provider keys are optional environment variables; `.env.example` contains names only. Never commit account cookies, authenticated exports or a real `.env`.

`fpl doctor` checks SQLite integrity, source freshness, versions and the free-cost guard. The UI shows source age, frozen forecast identity, state status and solver diagnostics. Re-ingestion creates a new market version and invalidates older exact-price plans. Long jobs retain queued/running/completed/failed state in SQLite; on restart, unfinished jobs become explicitly interrupted. Resubmit interrupted work. A stale state/forecast response is archived but cannot overwrite the current selected plan.

## Refresh cadence

Run `fpl ingest` after matches/corrections, `fpl forecast` after material availability/fixture changes, and `fpl optimize` before the deadline after reconciling private account values. Retries are bounded to three attempts, concurrency to four, and 429 Retry-After is respected within a bounded delay. Failed core refreshes retain old snapshots; optional comparator failures do not replace missing forecasts with zero. The UI keeps the last legal plan and its age/version visible.

Local cron or Windows Task Scheduler may invoke these same commands from the repository directory. A sleeping laptop or delayed scheduler is not a deadline guarantee. Refresh/recompute controls and CLI provide the on-demand recovery path. No paid or external scheduler has been created.

## Backup and restoration

`uv run fpl backup --destination artifacts/runtime/backup` creates a transactionally consistent SQLite backup and copies immutable raw inputs, scenarios, models, features and normalized files. The release verification copied this backup to a different temporary directory, loaded it using `Store(new_directory)`, and reproduced all 2,000 selected-plan scenario outcomes exactly. Artifact paths resolve relative to the selected data root.

To restore, stop the server, copy the backup directory into a new local directory and point `FPL_DATA_DIR` at it. Run `uv run fpl doctor`, `uv run fpl replay`, then `uv run fpl serve`. Keep the original data directory until the restore passes. SQLite state alone cannot reproduce forecasts without the corresponding immutable snapshots/scenario files.

## Hosted research

The root `.env` is automatically loaded without overriding inherited variables. Configure an eligible Gemini or Groq key and run `uv run fpl configure-providers --unbilled` only after verifying that account status. This authenticates model catalogs without generation and creates seven-day local capabilities. `uv run fpl research-refresh` uses the selected model and official source discovery. Daily local soft limits are 20 requests and 160,000 reserved tokens per provider. Expired capabilities or exhausted quotas queue work and preserve numerical operation. To configure a custom capability, copy `configs/hosted_free.example.json` to an ignored local path and verify the current model, free text capability, purpose/region eligibility, unbilled account and expiry. The default example is intentionally disabled and expired. No tool/search/batch/caching feature is enabled implicitly.

```sh
uv run fpl research --url https://www.arsenal.com/news --capability data/free-capability.json
```

Without `--capability`, the command only retrieves bounded official-domain text. Structured evidence can be imported with `--claims path.json`; explicit beliefs with `--belief path.json`. Extracted quotes must be verbatim and source-hashed; unknown entities/URLs/schema fields are rejected. Provider output has no executable tools and cannot set bank values or numeric beliefs. Source instructions are untrusted text.

The request boundary rejects unknown/paid/expired capabilities, reserves a conservative token/call quota before dispatch and performs no paid escalation. A provider outage or quota exhaustion leaves numerical forecasts and manual evidence entry operational. An absent key is not reported as a successful hosted extraction. Both user-provided free, unbilled providers have been authenticated and benchmarked. The chosen primary is Gemini 3.5 Flash Lite; Groq GPT OSS 120B is the fallback. Six synthetic cases measure extraction and source-status handling; real production claims retain their original source quotes. No paid grounding, tools or batch endpoint is enabled.

## Exports and reproducibility

`fpl export` writes validated plan JSON, the complete CSV ledger, a readable Markdown report an ICS calendar based on official UTC deadlines, the full-market forecast CSV and matching scenario/chip/rival alternatives. `fpl freeze` stores an immutable pre-deadline shadow decision; `fpl evaluate --prospective` also reports finalized decision results. `fpl snapshot` exports source hashes. `fpl replay` loads the frozen plan market and rules rather than silently substituting today's prices. `fpl evaluate --prospective` scores only forecasts captured before a subsequently completed deadline; an empty eligible set is reported as empty.

Public repository artifacts contain aggregate validation results and the research package the user asked to publish. Runtime exports, source archives, account state, backups and keys are excluded from Git by default.

## Windows handoff

Extract the private prepared ZIP and run `Start.cmd`. The ZIP includes the built UI, a consistent SQLite backup, immutable numerical inputs and the confirmed account state. Provider credentials, executable environments, node_modules and optional 80 MB vendor caches are excluded. Enter keys into the secure launcher prompts; their local file is restricted to the current Windows user. The public source checkout requires Node 24 for its first interface build; the prepared ZIP does not. uv installs the pinned Python environment. The server chooses an available loopback port and opens the browser after its health check succeeds.

The bundle contains private team/account state. Keep it private. Build a fresh portable handoff with `uv run python scripts/prepare_bundle.py`; it verifies ZIP CRC integrity. The Linux/Windows CI matrix verifies backend dependencies and offline tests; Windows CI also parses the launcher. A remote development environment cannot install the application on an inaccessible laptop.

## Connected training and refresh (0.5)

`uv run --extra neural fpl train-complete` runs chronological source-fusion, horizon, manager-count/choice, price/news, structural-component and evolving-account learning, then forecasts, refines the multiweek frontier, checks horizon sensitivity, compares football worlds, evaluates the population and freezes a shadow decision. It is a long CPU workflow. Models pass their own gates; a failed challenger does not stop the usable reference pipeline or silently replace it. Completed per-deadline trajectory checkpoints resume after interruptions. The final dataset receives an independent chip-aware outcome replay before model fitting.

The interface's **Refresh and recompute** performs a dated official refresh and receding-horizon planning. It retrains on newly finalized events, never provisional scores. **Enable automatic refresh** is opt-in and runs only while the local application is open: normally every four hours, at most 30 minutes within six hours of a deadline, and five minutes within the final hour. A serial durable queue prevents overlapping API jobs. Sleep/offline computers cannot provide a hosted scheduler; the next open session checks freshness.

Planning states are stored separately from the actual confirmed account. Full scenario recourse re-optimizes only after each observation; hypothetical prices use explicit purchase-lot bounds. Legal pending transfers and already activated chips are carried into later decisions. No refresh, optimizer, training job or updater submits a transfer to FPL.

`uv run --extra neural python scripts/verify_release.py` verifies current artifact connections, trained-model checksums, structural-world invariants and the independent replay over the selected planning horizon. The release audit records actual acceptance outcomes. The prepared package excludes transient training checkpoints and credentials; updates merge immutable research without replacing the selected account state or local keys.


## Current account and historical replay (0.6)

**Refresh data** reconciles the latest public deadline picks and transfer costs, reconstructs saved free transfers, preserves newer confirmed edits, and recomputes the next available deadline. It does not infer private transfers after that deadline. Save current values in the account editor to confirm any further moves. A passed deadline invalidates old actions, including alternative-plan links.

`uv run --extra neural fpl refresh-current` performs the same reconciliation/refresh workflow. Interrupted source-fusion folds and evolving-account deadlines have durable checkpoints. A rejected refit cannot replace the last accepted reference model. Optional source failures retain verified observations; source attempts record bytes, latency, retry and circuit state.

`uv run --extra neural fpl replay-weekly` builds independent £100m GW1–8 teams using pinned pre-deadline bootstrap and fixture versions. Each week refits from earlier completed data, reserves an earlier calibration slice, freezes the selected squad/XI/bench/captain, then reads that week's actual points. Hindsight teams are separately labelled diagnostics. This is a reconstructed historical experiment, not a prospective record or a continuous account with unlimited free transfers.

`uv run --extra neural python scripts/run_checks.py` records actual static, backend, frontend and desktop/mobile browser results. Run a local server first for browser checks. `scripts/verify_release.py` then verifies connected model checksums, the current financial ledger and weekly cutoff/model proofs. The release audit consumes the recorded checks; it does not assume hardcoded passing test counts.


## Integrated replay and prospective accounts (0.8)

`uv run --extra neural fpl replay-pipeline` executes the reference-family production forecaster and full chip-window planner on 2025/26 GW20–38 and current GW1–9. The normal budget is 4,000 common worlds and 60 seconds per MIP, with independently solved controls and no-hit comparisons. It is a long CPU evaluation. It never changes the real account, selected model, current forecast or selected recommendation. `--current-only` saves a separate current-season experiment; it cannot replace the full two-period scorecard. The planning window still extends to the chip deadline rather than GW9.

Each frozen decision retains training/calibration cutoffs, actual source versions, first-week worlds and model checksums. An independent verifier checks legal transitions, actual scores, sealed actions and uninterrupted account continuity. Completed work has per-deadline checkpoints and portable sealed bundles. A completed result is reverified before reuse; changed data or implementation invalidates reuse. Historical runs are labelled later reconstructions, never future predictions.

`fpl freeze-accounts` freezes three equal-£100m hypothetical continuous accounts: the primary multiweek policy, ownership with free transfers and official expected points with free transfers. Known purchase lots avoid inventing private acquisition costs. `fpl score-accounts` advances finances after released deadlines, scores only finalized matches and recomputes official revisions. Refresh and complete-training workflows also connect these operations. The interface exposes capture status, frozen captaincy, actual net points and downloads. A missed capture executes the previously frozen standing hold; it does not create a new prediction after the deadline.

Automatic refresh remains an explicit local control and operates only while the application is open. A closed/offline laptop cannot retrieve new observations. No scorecard reports future matches as completed evidence, and these hypothetical accounts never submit FPL transfers.

Portable database parts are listed in `data/state.sqlite.manifest.json`; each part and the reconstructed database have checksums. The launcher restores them and checks SQLite integrity before opening the app. It also accepts older `state.sqlite.gz` packages. Restoration refuses to replace an existing account database. The updater merges research artifacts into a backed-up installation while retaining its account/model selections and keys.

`uv run --extra neural python scripts/verify_download.py PATH_TO_DOWNLOADED_ZIP` verifies the actual GitHub download: application/model hashes, credentials exclusion, database restoration, all historical decision proofs, pending prospective accounts, relocated inference and exact live-plan world scores. Large verification scratch files use the project runtime directory, avoiding memory-backed temporary-download storage.

## Accepted future-fixture model updates

The 0.9 update imports future-model research, frozen policy proofs and distribution validation as immutable artifacts. It preserves the existing account and active model selection during file installation. The launcher's chronological retraining verifies passed earlier-season evidence, refits the accepted expert on completed labels and connects it before refreshing the forecast and plan. A failed or partial/current-season-only experiment cannot enable this path.

Regular `fpl train` and scheduled completed-week refits retain this accepted component. `fpl train-complete` includes the research/acceptance stages when absent. `fpl validate-future-policy` reproduces the earlier-season comparison; `--current` runs a diagnostic with preseason future weights, without using current outcomes for selection. `fpl replay-pipeline` follows the active model family. Completed diagnostic decisions carry forward when extending to a new finalized week.

Large database verification, replay and recourse scratch files stay on the workspace/data disk. The portable package includes checksummed model text, feature data, independent decision proofs and selected worlds. The download verifier checks the actual GitHub archive, relocates model/data paths and reproduces the selected plan's sampled outcomes.
