# FPL Intelligence

A private local FPL workspace with independent player forecasts, exact account ledgers, full-market multiweek optimization, correlated match simulations and mini-league policy comparisons. Python runs the numerical engine; React/TypeScript provides the interface. No paid service or local language model is required.

The six supplied documents are preserved unchanged in [`research/`](research/). They are design inputs, not evidence that a model has achieved a particular season rank.

## Run locally

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and Node 24. Commands below run from the repository root, on Windows or Linux.

```sh
uv sync --frozen --extra dev
npm --prefix apps/web ci --no-audit --no-fund
npm --prefix apps/web run build
uv run fpl ingest
uv run fpl import-history --season 2025-26
uv run fpl reconcile-team
uv run fpl build-features
uv run fpl train
uv run fpl forecast --end 38 --worlds 2000
uv run fpl optimize --end 19 --seconds 60
uv run fpl serve
```

Open **http://127.0.0.1:8000**. The UI also has refresh, forecast, evaluation, optimization and rivalry controls. For frontend development, run `npm --prefix apps/web run dev` alongside the API.

`reconcile-team` imports the supplied **dated** SKIBIDI FC snapshot. Confirm your current squad, exact sell quotes, bank, chips and free transfers in **My squad → Edit account state** before executing a recommendation. The snapshot does not establish a current private account. Unknown purchase prices stay null. IDs are resolved against the current season and position; ambiguous identities fail validation.

For another account, pass `--entry` and `--league` to ingestion, then import a JSON `TeamState` conforming to [`schemas/state.schema.json`](schemas/state.schema.json). `fpl select-initial --event 9 --budget 1000` selects an initial/late-entry squad within the frozen forecast horizon. Adjust event and horizon flags for the actual season; GW9–19 is this supplied case, not a product limit.

## What runs

- **Data:** official player registry, fixtures, all player histories, released entry/history/picks/transfers, event-live, paginated league standings, optional partial Solio comparator and historical season archives. Raw responses are compressed, immutable and hashed. SQLite stores versioned artifacts and append-only account events; Parquet and DuckDB support statistical features.
- **Rules:** integer prices and gain sharing, original purchase lots, formations, club limits, FT rollover/caps, hits and refunds, chip windows, Free Hit restoration, captain/vice, autosubs, Bench Boost, Triple Captain, exact scoring thresholds and BPS tie awards. Plans are independently replayed outside the solver.
- **Forecasts:** exposure-shrunk multi-season profiles, zero/start/cameo minutes, dynamic team goal rates, shared match tempo, finite football XIs, timed goal attribution, saves, defensive actions, discipline and match-contested bonus. Every registered player has a player/GW record through GW38, including blanks. Scenario samples are retained for replay.
- **Planning:** full selectable market, ownership-lot accounting, bank balances, FT state network, transfers, captain, XI, temporary FH squads and joint chip timing in HiGHS. A legal incumbent survives time limits. Exact FPL autosubs and captain fallback are rescored in common worlds. Solver bounds/gaps and the terminal-value assumption stay visible.
- **Rivals:** actual mini-league membership and released picks, effective exposure, hold/greedy opponent policies, joint win probabilities in identical worlds, ties and a visible expected-points sacrifice budget. The original five FH architecture families and hold are executable comparisons.
- **Research:** official-domain source retrieval, supported claim schema, contradiction/supersession links, expiring manual beliefs and separate belief forecasts. Optional Gemini/Groq adapters validate JSON and quotes. Unknown/paid capabilities fail closed; daily quota is reserved before dispatch. No account key is needed for the numerical application.
- **Product:** recommendation, XI/bench, full market, transfer package checks, conditional season ledger, rival analysis, evidence, model/data health, JSON/CSV/Markdown exports and an authoritative deadline ICS calendar.

## Measured evidence

The executed release run fetched **669 players, 20 clubs, 38 events, 669 histories and 300 future fixtures**, with no ingestion errors. It generated **20,070 player/GW distributions in 2,000 shared worlds**. All six GW9–19 candidate ledgers passed independent transition validation.

Deadline-grouped walk-forward evaluation used **32,866 held-out player-fixtures** from the 2025/26 archive and completed 2026/27 fixtures:

| Model | RMSE | MAE |
|---|---:|---:|
| Transparent shrinkage | 2.0642 | 1.3023 |
| LightGBM direct challenger | 2.0113 | 1.0884 |

The paired deadline-bootstrap 95% interval for RMSE difference was **−0.0705 to −0.0401**. The predeclared regular-starter cohort also improved (3.1208 → 3.0940 RMSE). This is **historical statistical replay**, with retrospective completed-match data. It is not a strict historical news replay or prospective competitive result. The correlated decision simulator remains a disclosed component baseline; the tabular result is a separate shadow-model comparison.

See [`artifacts/evaluations/`](artifacts/evaluations/) and the [model card](docs/model-card.md) for actual scope and assumptions. Prospective forecasts are frozen; `fpl evaluate --prospective` only scores records issued before subsequently finalized deadlines. No future deadlines or top-1% results have been fabricated.

## Useful commands

```sh
uv run fpl doctor
uv run fpl calibrate
uv run fpl rivals --case --seconds 45
uv run fpl replay
uv run fpl simulate
uv run fpl export
uv run fpl snapshot
uv run fpl backup
uv run fpl evaluate --prospective
```

Exports and backups live in `artifacts/runtime/`; raw data and account state live in `data/`. Both are ignored by Git. The published repository contains source, contracts, the supplied research and measured aggregate validation artifacts. It does not contain provider keys, cookies or authenticated account exports.

## Verify

```sh
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest -q
npm --prefix apps/web run build
npm --prefix apps/web test
```

For the live local user journeys, start the seeded application, run `npm --prefix apps/web exec playwright install chromium`, then `npm --prefix apps/web run test:e2e`. Desktop and mobile tests cover the recommendation, legal XI, all-market search, calendar, exports and account validation. CI runs the offline rules/oracle/API suite and frontend build/tests.

See [operations](docs/operations.md), [methodology](docs/methodology.md) and [contracts](packages/contracts/openapi.json).
