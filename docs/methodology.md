# Data, rules and optimization contracts

## Temporal data

SQLite snapshots record source, exact URL, capture time, optional publication time, content hash and immutable compressed payload path. JSON hashes normalize object keys before canonical ordering; legacy exact-serialized JSON digests remain verifiable. Raw files publish atomically. Market, state, forecast, plan and experiment artifacts have independent content IDs and are never edited in place. Account changes append events.

Every source/fixture/player identity is scoped to its season. The official stable `code` bridges prior-season IDs; name matching is restricted to seed reconciliation within a position, rejects ambiguity, and saves the resolved ID. Accents are normalized only for that import step. Future exact sell quotes use the supplied current quote until a purchase lot is known; a changed price with an unknown purchase lot requires a fresh quote.

Forecast manifests include all history/archive source hashes, the frozen market and fixture hashes, rules hash, cutoff, issue time, model version, scenario seed and sample file. The audited initial forecast's input manifest was completed after numerical execution without changing its scenarios; both versions are retained. Prospective scoring requires issue time before the actual deadline and finalized outcomes after it.

The official events feed resolves the supplied research's deadline discrepancy. Date strings displayed in the product use Asia/Dubai. Operational refresh uses current API events; no static GW19 deadline is used by scheduling logic.

## Data dictionary

| Record | Fields and units | Missingness policy |
|---|---|---|
| Player | season, official ID/code, club, FPL position, price in integer £0.1m, ownership %, status/news | Removed/unselectable preserved in full registry |
| Fixture | ID, clubs, venue, event, kickoff, results, provisional/final state | Null event retained as unassigned |
| History | fixture minutes/starts/points, xG/xA, saves, BPS, actions, discipline | Numeric strings parsed deliberately; missing actions never become observed zero |
| Lot | player, purchase price, exact sell quote, market price at quote, provenance | Unknown purchase price stays null |
| State | bank, FTs, lots, chips, prior-deadline restoration state, already-made transfers/hits, status | Dated/public/confirmed are distinct |
| Forecast | player/fixture/GW, mean/SD/quantiles, start/0/60+ probabilities, components, coverage | Full registered market and explicit blank records |
| Claim | quote, source/hash, entity/type, origin, clocks, status, contradictions/supersession, expiry | Unsupported quote or invalid identity rejected |
| Plan | action, lot ledger, bank, FTs, hit, permanent/temporary squad, XI/bench/captain/vice, assumptions | Independently transitioned; stale input IDs cannot replace current recommendations |

Schemas live in `schemas/`; API contracts live in `packages/contracts/openapi.json`. The source registry includes source semantics, clocks, refresh and fallback behavior.

## Rules boundary

`Rules.from_bootstrap` reads current composition, formations, club limit, FT cap, transfer cap, chip periods, scoring and explicit event overrides. `official-overlay-v1` supplies official narrative behavior: gain rounding, DC10/12 thresholds, captain fallback, ordered autosubs, bonus ties, chip restoration and preservation of FTs during WC/FH. Goalkeeper goal rewards come from the source.

The transition oracle processes whole bundles. It checks identity/season, equal buys/sells, new-player eligibility, club limits, exact affordability, chip usage/periods and lineups. Hits are event-level charges less already-paid penalties, including chip refunds. FH restores the prior underlying squad and bank. A prior transfer followed by FH requires the actual prior-deadline state. Consecutive FH across GW19/20 is tested. The MIP generator deliberately blocks that prior-transfer/FH branch when restoration cannot be represented; the deterministic importer/transition function supports it with complete state.

## Optimization

The time-expanded HiGHS formulation contains permanent ownership, original acquisition lots, sales, buys, bank, scoring/FH squad, XI, captain, chip products and a free-transfer state network. The network represents exact capped rollover and hit costs; original lots prevent a later repurchase from inheriting old gain sharing. FH affordability distinguishes retained lots from new purchases.

The generator includes every selectable registered player and owned unselectable player. A bounded greedy action search and single-week full-market squad solve provide only feasible incumbents; they do not restrict the final market. Strong chip linking cuts tighten the relaxation. The final incumbent is independently replayed and rescored in common match worlds. Tiny knapsack and actual squad/formation enumeration tests serve as independent oracles.

Future actions depend on today's forecasts and constant-price assumptions. This is receding-horizon planning: trigger-based recomputation after observable injuries, lineups, role changes, prices and fixture announcements. It is not an omniscient or globally optimal multistage probability policy. The rival objective selects among generated feasible candidates under a displayed expected-points sacrifice budget.
