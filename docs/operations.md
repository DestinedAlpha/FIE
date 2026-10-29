# Local operation and recovery

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

Configure an eligible Gemini or Groq key through the environment. Copy `configs/hosted_free.example.json` to an ignored local path and verify the current model, free text capability, purpose/region eligibility, unbilled account and expiry. The default example is intentionally disabled and expired. No tool/search/batch/caching feature is enabled implicitly.

```sh
uv run fpl research --url https://www.arsenal.com/news --capability data/free-capability.json
```

Without `--capability`, the command only retrieves bounded official-domain text. Structured evidence can be imported with `--claims path.json`; explicit beliefs with `--belief path.json`. Extracted quotes must be verbatim and source-hashed; unknown entities/URLs/schema fields are rejected. Provider output has no executable tools and cannot set bank values or numeric beliefs. Source instructions are untrusted text.

The request boundary rejects unknown/paid/expired capabilities, reserves a conservative token/call quota before dispatch and performs no paid escalation. A provider outage or quota exhaustion leaves numerical forecasts and manual evidence entry operational. An absent key is not reported as a successful hosted extraction. No real key/account was available in the execution environment, so authentication and live extraction quality remain account configuration/evidence requirements.

## Exports and reproducibility

`fpl export` writes validated plan JSON, the complete CSV ledger, a readable Markdown report and an ICS calendar based on official UTC deadlines. `fpl snapshot` exports source hashes. `fpl replay` loads the frozen plan market and rules rather than silently substituting today's prices. `fpl evaluate --prospective` scores only forecasts captured before a subsequently completed deadline; an empty eligible set is reported as empty.

Public repository artifacts contain aggregate validation results and the research package the user asked to publish. Runtime exports, source archives, account state, backups and keys are excluded from Git by default.
