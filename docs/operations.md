# Local operation and recovery

## Updating a laptop installation

Extract the new private package into a separate folder. Run `Update existing.cmd` and select the old folder containing `Start.cmd`. The updater stops only FPL servers whose interpreter is inside that folder, validates the application file manifest, creates a consistent account database/application backup, and copies application files while excluding `.env` and `data/`. It then starts the existing installation with `-RebuildModel`: current official data is refreshed, the four preceding archives are imported if absent, the learned-points model is trained/calibrated/evaluated, and a new forecast/plan is computed for the preserved account.

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
