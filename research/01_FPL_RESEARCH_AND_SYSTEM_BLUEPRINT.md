# FPL Intelligence: research and engineering blueprint

Research date: 29 October 2026. User timezone: Asia/Dubai.

## 1. The product to build

Build a personal, evidence-driven FPL decision system covering the complete 38-gameweek season, all 20 Premier League clubs, and every registered player. It must combine independently trained forecasts, football context, exact game rules, correlated match simulation, and sequential squad optimization. Hosted language models supply structured research and explanations; numerical models and the optimizer calculate decisions.

The immediate personalization is SKIBIDI FC, entry 9375678, with an eleven-gameweek GW9–GW19 mini-league objective and a continuing season objective through GW38. That is a use case for a reusable product, rather than a reason to hardcode eleven weeks or particular players.

The strongest defensible route is to reproduce useful open research, establish a trustworthy benchmark, improve minutes and role awareness, and optimize actual decisions. A large prompt alone does not confer forecasting skill. This package turns the ambition into a specification an implementation agent can build and test.

Success means improving the quality of repeatable decisions and demonstrating competitive performance on information available before deadlines. A top-1% finish is the target; it is an uncertain season outcome. Claims of matching or beating Solio/FPL Review require synchronized forecasts and prospective evidence, not persuasive explanations or one successful squad.

### Product requirements

1. Recommend hold/roll, individual transfers, transfer packages, hits, captain/vice, legal XI, ordered bench, and all available chips.
2. Evaluate short-term returns, medium-term squad structure, full-season chip opportunities, flexibility, and mini-league comeback probability.
3. Maintain original full-market forecasts even when third-party projections disappear.
4. Explain the causal reason for each consequential update: minutes, role, penalty share, team strength, fixture availability, or price.
5. Show credible alternative plans and the assumptions under which they become preferable.
6. Run without paid data, paid inference, paid solver licenses, or local LLM weights.
7. Preserve every input, decision, forecast version, assumption, and result needed for replay.
8. Continue producing valid decisions when optional research or hosted AI is unavailable.

## 2. Research findings that change the design

### Public forecasts are useful, but coverage matters

FPL Review documents a simpler free model with a three-gameweek horizon, distinct from its premium system. Its premium methodology includes contextual and minutes modeling; contextual intelligence is therefore a competitive requirement, not an unoccupied niche. [Free-model documentation](https://docs.fplreview.com/the-model/projections/free-model/), [premium-model documentation](https://docs.fplreview.com/the-model/projections/massive-data-model/).

Solio publishes a public feed and describes market-informed stochastic projections and optimization. A direct HTTP test succeeded. The captured JSON had a GW9 timestamp and a 30-player topProjected list, plus other selected leaderboards. It did not supply a complete all-player, multiweek forecast matrix. Integrate it as an optional overlap benchmark and research signal; never fill absent players with zero or pretend the feed covers GW38. [Solio product and feed description](https://fpl.solioanalytics.com/), [tested public JSON](https://fpl.solioanalytics.com/api/data/latest.json).

Scout describes human monitoring around its projections; its full RMT tables are offered to members. Fix and Hub publish projection-based material, but product marketing does not establish comparative superiority or free bulk access. Make their adapters optional and activate only after verifying coverage and access. [Scout RMT](https://www.fantasyfootballscout.co.uk/2026/07/27/fpl-rate-my-team-points-projections-live-for-2026-27), [Fix tools](https://www.fantasyfootballfix.com/why_fix/), [Hub predictions](https://www.fantasyfootballhub.co.uk/predictions).

### An open model is a serious starting point

OpenFPL reports prospective 2024/25 results using earlier FPL and Understat seasons, with competitive results against a commercial benchmark. That supports reproducing an open baseline; it does not prove performance under 2026/27 rules or establish a current overall winner. [OpenFPL paper](https://arxiv.org/abs/2508.09992), [author's repository](https://github.com/daniegr/OpenFPL).

Also inspect Dastan as a challenger and evaluation reference. Its own repository distinguishes gains from participation prediction from weaker evidence for improvements among regular starters. Treat its reported results as author-reported until reproduced, and inspect provenance before reusing data or weights. [Dastan](https://github.com/qazybekb/smartplayfpl-dastan).

### A leaderboard is not the whole decision problem

FPL Bench selects weekly £100m squads, uses fixed role multipliers during selection, and excludes transfers and chips. Its methodology also substitutes zero when provider predictions are missing. Replicate its published protocol when comparing with its leaderboard, then separately report common-coverage and full-season policy results. Its published table is evidence of one benchmark, not a certificate that a model is best. [FPL Bench methodology](https://www.fplbench.com/methodology).

### Community research reveals failure modes

The r/fplAnalytics discussions inspected include an author explicitly acknowledging zero-minute inflation of headline accuracy and another describing threshold-sensitive minutes, DefCon, and match-level bonus modeling. Useful lessons are to report participation separately, score thresholds exactly, and evaluate full distributions. These are implementation hypotheses, not validation of those authors' performance claims. [Zero-minute evaluation discussion](https://www.reddit.com/r/fplAnalytics/comments/1rgumu6/first_attempt_building_an_xpts_model/), [component-model discussion](https://www.reddit.com/r/fplAnalytics/comments/1vioe4q/i_made_an_expected_points_model_here_is_how_ive/).

Extend discovery to r/FantasyPL, r/fantasyfootballcoding, public research repositories, club reporters, press conferences, and public analyst work. Prioritize original evidence over repost counts.

### The current official feed is richer than a static stats table

Direct tests returned HTTP 200 for bootstrap, fixtures, entry information, and a player-history sample. The bootstrap identifies 2026_27 in its asset path and contains 669 players, 20 clubs, 38 events, chip periods, scoring configuration, and price-change fields. The match-history sample includes starts, minutes, xG, xA, BPS, tackles, CBI, recoveries, and defensive_contribution. The public entry name and 418 points matched the supplied account context. These are observed research-time results, not guarantees of future schemas. [Official bootstrap](https://fantasy.premierleague.com/api/bootstrap-static/), [fixtures](https://fantasy.premierleague.com/api/fixtures/).

The official 2026/27 update changes BPS treatment of CBI, being tackled, and goalkeeper saves, and moves final score locking to the next morning. Model rules and result revisions explicitly. [Official updates](https://fantasy.premierleague.com/help/new), [BPS explanation](https://www.premierleague.com/en/news/4679946/).

One official explanatory article gives a different GW19 deadline from the live events feed. The captured API says 1 January 2027, 18:30 UTC; the article says 2 January. This is a concrete reason to resolve deadlines from the current event feed and retain a conflict record. Do not embed either date permanently. [Official overview article](https://www.premierleague.com/en/news/4679873), [live events source](https://fantasy.premierleague.com/api/bootstrap-static/).

## 3. Architecture and language choice

Use Python for ingestion, research orchestration, statistical learning, simulation, evaluation, API, and optimization. Use TypeScript/React for the interface. Use SQL for point-in-time features and analysis. Reuse compiled numerical libraries and HiGHS rather than writing a new solver in C++ or introducing more application languages.

This is a quality decision: correctness and iteration speed matter more than language variety. Add Rust/C++ extensions only if profiling shows a remaining bottleneck after vectorization and compilation.

| Layer | Selected approach | Reason |
|---|---|---|
| Runtime | A supported CPython version compatible with the complete locked dependency set; uv | Reproducible setup on Windows/Linux |
| Ingestion | httpx, bounded asyncio concurrency, retries, conditional requests | Controlled fetches and explicit failures |
| Structured data | Pydantic schemas, PyArrow/Parquet, DuckDB, Polars | Typed ingestion, compact archives, point-in-time SQL |
| Personal state | SQLite transactions and append-only state events | Reliable ownership lots, transfers, settings |
| Numerical base | NumPy, SciPy, statsmodels, scikit-learn | Transparent statistical baselines |
| Challenger models | LightGBM; CatBoost only if it wins a documented comparison | Strong CPU tabular modeling |
| Hierarchical models | Regularized likelihood/MAP first; PyMC offline challenger | Shrinkage without mandatory expensive sampling |
| Simulation | NumPy plus Numba for measured hot loops | Reproducible fast CPU simulation |
| Optimization | highspy/HiGHS, with a thin typed model builder | Free MIP solver and precise solve diagnostics |
| API | FastAPI, generated OpenAPI contracts | Typed boundary between engine and UI |
| UI | TypeScript, React, Vite, TanStack Query/Table, accessible chart library | Rich planner without server-rendering complexity |
| Research | Source registry, keyword/SQL retrieval, BM25 where useful | No local embedding model needed |
| Hosted AI | Gemini adapter with verified free capabilities; optional eligible Groq fallback | No local language-model inference |
| Jobs | CLI jobs and SQLite task queue; Windows Task Scheduler or cron | Low maintenance for one user |
| Quality | pytest, Hypothesis, Ruff, mypy, Vitest, Playwright | Rule/state tests and key user journeys |
| Experiments | Versioned JSON/Parquet manifests and scorecards | No mandatory hosted tracking service |
| Packaging | uv lockfile, frontend lockfile, optional Docker | Portable, explicit environment |

HiGHS is available under MIT and has a Python interface. The current open FPL solver repository uses highspy and is Apache-2.0 licensed: inspect and adapt reusable parts, preserving notices and independently testing season rules. [HiGHS](https://highs.dev/), [open FPL solver](https://github.com/solioanalytics/open-fpl-solver).

DuckDB can query Parquet and work with Polars/Arrow. This fits a personal analytics engine without needing a cloud database. [DuckDB Python documentation](https://duckdb.org/docs/stable/clients/python/overview).

### Deployment selected for the free requirement

Default: run ordinary statistical computation and the private web application on the user's machine. This downloads no LLM, runs no local LLM, and requires no GPU. Hosted AI requests go to an eligible free provider.

Alternative: run numerical notebooks interactively on free Colab, export signed/versioned reports, and view them locally. Colab is a research workspace, not a dependable always-on server. [Colab resource policy](https://research.google.com/colaboratory/faq.html).

Optional cloud presentation: publish only non-sensitive static reports through a free static host. Keep personalized planning and API keys private. Cloudflare Pages documents free-tier deployment limits; do not run Python MIP jobs inside a tiny edge-function request. [Pages documentation](https://developers.cloudflare.com/pages/).

Optional scheduled batch: GitHub Actions within its free entitlement, with bounded artifacts and zero paid spending. A scheduled workflow can be delayed or dropped, so deadline-critical refreshes need an on-demand/local path. [Actions workflow documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax), [schedule reliability](https://docs.github.com/en/enterprise-cloud%40latest/actions/how-tos/troubleshoot-workflows).

## 4. Data acquisition: complete core, optional enrichment

The user's FPL scraping grant is accepted as the authorization basis for covered FPL data. Record its scope in configuration. Other providers have their own access conditions. This is an adapter-design issue: missing optional sources trigger alternate evidence paths, not a blocked product.

### Official endpoint registry

| Endpoint | Purpose | Important limitation |
|---|---|---|
| /api/bootstrap-static/ | Player registry, teams, events, settings, season totals, prices, availability | Current snapshot; archive before each deadline |
| /api/fixtures/ | Fixture identity, event assignment, times, results | A future fixture can move or have a null event |
| /api/element-summary/{id}/ | Player match history and upcoming fixtures | One history row is a fixture, not necessarily a whole GW |
| /api/event/{gw}/live/ | Scoring breakdowns and provisional/final results | Retain revisions and finalization status |
| /api/entry/{id}/ | Public entry summary | Does not establish the current private pre-deadline squad |
| /api/entry/{id}/history/ | Historical points, transfers, chips | Reconcile rather than infer current FT blindly |
| /api/entry/{id}/event/{gw}/picks/ | Revealed squad, captaincy, bench for past/current released GW | Cannot reveal hidden future moves |
| /api/entry/{id}/transfers/ | Released transfer history | Publication timing matters |
| /api/leagues-classic/{id}/standings/ | Public league standings and entry IDs | Pagination, caching, access, and sample timestamps |

Treat these as observed/public application interfaces, not a guaranteed stable API contract. Verify each adapter at installation. All core tables can be built from official data; advanced event enrichment improves coverage but is not required to produce a legal plan.

### Historical and enrichment registry

| Source | Intended use | Fallback and validation |
|---|---|---|
| vaastav archive | Multi-season FPL outcomes and schemas | Compare samples with official data; pin commit; audit each field and season |
| Randdalf/fplcache | Pre-deadline bootstrap snapshots | Audit actual coverage and capture timing; do not assume every deadline exists |
| Understat | Additional shot/xG context and historical team/player process | Verify actual season/player coverage; official FPL xG/xA remains the core alternative |
| Football-Data.co.uk | Match results, team statistics, historical odds | Preserve source timestamps; closing odds can be after an FPL deadline |
| Official PL/club sites | Fixtures, registration, injuries, press conferences | Original club text/video; record publication and observation times |
| Official cup/UEFA/federation schedules | Rest, travel, eligibility, international absence | Cache schedules and result updates by competition |
| Public lineup/role/set-piece evidence | Rotation competition and tactical changes | Analyst claim must point to observable evidence |
| Solio public feed | Partial live external benchmark | Own complete forecasts, overlap-only comparisons |
| FPL Review free | Short-horizon external baseline where accessible | Adapter and manual permitted import; own model covers missing horizons |
| Scout/Fix/Hub | Public commentary and legitimate accessible projections | Optional; no core dependency on member exports |
| LiveFPL | Public ownership/elite cohort context | Independently sampled official released picks with a disclosed sampling frame |
| Reddit/X/YouTube/podcasts | Discover claims, methods, lineup hints | Trace the original report; use permitted feeds/transcripts or supplied excerpts |
| StatsBomb open data | Historical event-model experiments where covered | Inspect competitions first; do not assume live full-PL coverage |

Historical references: [vaastav](https://github.com/vaastav/Fantasy-Premier-League), [fplcache](https://github.com/Randdalf/fplcache), [Football-Data](https://football-data.co.uk/englandm.php), [StatsBomb open data](https://github.com/hudl/open-data).

Reddit discovery is an evidence-finding workflow, not permission to train on users' posts. Its API has separate access requirements and content restrictions. Store only permitted evidence and follow source deletion/retention requirements. If direct access is unavailable, use public indexed discovery to identify original club/reporting sources. [Reddit Data API terms](https://redditinc.com/policies/data-api-terms).

### Refresh budget

Use change detection rather than continuous full recrawls.

- Bootstrap and fixtures: every 4–6 hours normally; increase around relevant news/deadlines within source limits.
- Match history: after completed matches and after official corrections; incremental refresh by affected club/player.
- Official club news: daily across 20 clubs, then targeted refresh following press conferences and high-impact events.
- Research queue: prioritize own squad, plausible replacements, captain candidates, and candidates whose uncertain inputs could change the chosen action.
- Elite samples: update membership and released picks at defined weekly cutoffs, not thousands of entry fetches every few minutes.
- External forecasts: retain generation time and fetch time; near-deadline snapshots where permitted.

Record actual calls, bytes, latency, quota, errors, and coverage. Retry transient failures with exponential backoff/jitter and respect Retry-After. Use a circuit breaker for repeated failure and avoid simultaneous refresh jobs.

## 5. Data model and temporal correctness

Use fixture-level facts and deadline-level decisions.

Core entities:

- player_identity: stable internal UUID, source IDs, aliases, season-specific FPL ID, mapping provenance.
- player_registration: club and FPL position with valid-from/valid-to dates.
- fixture: source identity, competition, opponent, venue, kickoff, event assignment, confirmation status.
- player_fixture_actual: raw and normalized stats, source revision, finalization state.
- snapshot: source URL, request metadata, captured_at, source_published_at, content hash, schema version, season.
- evidence_claim: entity, claim type, original source, publication time, known_at, observation time, valid interval, uncertainty, supersession.
- forecast: issued_at, cutoff_at, target fixture/GW, model/rules/features versions, distribution and components.
- ownership_lot: player, acquisition time, purchase price, current sell quote, account-state provenance.
- team_state: bank, 15 players, lot references, free transfers, chips, pending transfers, objective.
- plan: feasible state sequence, scenario tree, forecasts, solver diagnostics, alternatives, validity time.
- experiment: dataset hashes, splits, seeds, code version, metrics, hardware/runtime, conclusions.

Store at least valid time and knowledge time. A suspension decided on Tuesday and first captured Wednesday cannot be used in Monday's replay. A retrospective article about an old injury is not a historical pre-deadline injury feature.

Raw snapshots are immutable and compressed. Normalized tables are regenerated with documented transformations. Save parser versions and reject silent type coercion.

Essential normalization:

1. Money is integer £0.1m units; a missing price is not zero.
2. Numeric strings are parsed deliberately; missing xG is distinct from genuine 0.00.
3. FPL IDs are season-scoped. Names are display labels, not join keys.
4. Match player mappings with source IDs; ambiguous names require corroborating club/position/date evidence.
5. Preserve changes in player clubs, FPL positions, scorer definitions, and supplier metrics.
6. Sum double/triple gameweek fixtures by event only after predicting each fixture.
7. A blank has no fixtures, not an average projected match.
8. Stale fields include age, not only values.
9. Historical labels can use finalized outcomes; historical features must use information known by the cutoff.
10. Advanced sources are separate measurement systems. Do not average incompatible xG definitions without an explicit observation model.

## 6. Complete feature and statistic coverage

Collect available fields across every GW1–GW38; backfill multiple historical seasons where provenance permits. Comprehensive coverage means recording data availability and testing feature value, not forcing every statistic into the model.

| Family | Core fields | Optional enrichment and intended effect |
|---|---|---|
| Identity/rules | IDs, club, position, eligibility, selectable/transactable flags | Transfer history, temporary IDs, reclassifications |
| Minutes | Starts, minutes, appearances, 0/1–59/60+ outcomes | Start/substitution times, bench membership, injury return stage |
| Attacking output | Goals, FPL assists, xG, xA, xGI | Non-penalty xG, shots, box shots, headers, big chances, shot locations |
| Creation | xA and assists, set-piece order | Key passes, big chances created, crosses, cutbacks, expected threat if available |
| Penalties/set pieces | Penalty order and observed takers | On-pitch penalty shares, corners by side, indirect/direct free kicks, aerial targets |
| Defensive process | Team xG conceded, clean sheets, goals conceded | Shot quality allowed, set-piece vulnerability, possession, pressing style |
| Defensive contribution | Tackles, CBI, recoveries, combined defensive actions | Position-specific hit rate, count variance, threshold probabilities |
| Goalkeepers | Saves, penalties saved, conceded, minutes | Shots faced, big-chance saves, post-shot xG, save locations where available |
| Discipline | Yellows, reds, bans | Competition-specific thresholds, rescissions, accumulated-card cutoffs |
| Bonus | BPS, bonus, relevant underlying actions | Pass completion, big chances missed/created, errors, duels where rules require |
| Team strength | Goals/xG for/against, opponent and home/away | Dynamic attack/defence, score-state adjustment, manager and personnel changes |
| Tactical role | FPL position plus observed role | Wing-back/full-back, number 6/8/10, false nine, wide forward, build-up duties |
| Match context | Kickoff, home/away, opponent, rest | Travel, extra time, Europe/cups, recovery days, international duty |
| Availability | Status/news with timestamps | Training evidence, manager wording, rehabilitation, illness, return restrictions |
| Financial | Current buy price, exact account sell quote, purchase lot, bank | Transfer flows, price indicators, scenario affordability |
| Ownership | Selected %, revealed captaincy/chips | Top cohorts, active-manager sample, mini-league overlap, EO uncertainty |
| Fixture uncertainty | Event assignment and official status | Postponement/rearrangement branches, cup-path dependencies |
| Forecast disagreement | Component and total differences | Research priority, uncertainty, model reliability diagnostics |

Candidate rolling windows: recent fixtures, recent starts, 180/450/900/1,800-minute exposures, season-to-date and multi-season priors. Select windows and recency decay on training/validation folds. Keep home/away and strength adjustments regularized.

Avoid unnecessary narratives: motivation, weather, referee, or travel effects enter production only with reliable data, plausible mechanism, and incremental validation. A manager change is a regime-shift hypothesis with wider uncertainty, not a magic fixture multiplier.

## 7. Forecasting methodology

### 7.1 Establish independent baselines

Implement and archive:

1. Position/price/fixture-aware shrinkage baseline.
2. Exposure-weighted historical minutes and event-rate baseline.
3. OpenFPL reproduction on its published setting where the required inputs are available.
4. Direct tabular expected-points challenger.
5. Component-based probabilistic model.
6. Optional third-party comparisons on identical player/deadline/horizon coverage.

Do not declare any architecture the winner before evaluation. Start with reliable simple models and retain challengers only when they improve the specified outcome.

### 7.2 Minutes as a distribution

Forecast availability, start conditional on availability, cameo conditional on not starting, and minutes conditional on each playing state. Include a point mass at zero and model 60-minute threshold probability explicitly.

E[M] = P(available) × {P(start|available) E[M|start] + P(bench|available) P(cameo|bench,available) E[M|cameo]}.

A player with 60 expected minutes might be a secure 60-minute starter or a mixture of 90 minutes and absence. These have different appearance, clean-sheet, captaincy, and bench values.

Use role competition and lineup constraints across the actual football team. Simulated real lineups have 11 distinct starters and obey plausible football roles; FPL's position labels must not prohibit a real false-nine formation. Shared lineup scenarios prevent two competing players from each being independently modeled as near-certain starters.

Use substitution survival/hazard models or calibrated discrete minute buckets. Learn manager patterns with shrinkage. Injury return has a time-dependent state; absence and future minutes are correlated. For doubles, simulate both matches jointly using rest, order, first-match minutes, and unresolved injury/rotation.

### 7.3 Team strength and score process

Fit time-varying team attack and defence, opponent effects, home advantage, recency, and promoted-team priors. Compare independent Poisson with low-score dependence and overdispersion variants using out-of-sample proper scores. Do not add parameter complexity just because it sounds more realistic.

When permitted pre-cutoff odds are available, derive de-vigged market constraints from compatible markets and fit team goal intensities. Normalize 1X2 probabilities only within an exhaustive market; do not normalize anytime-scorer prices across players. An anytime scoring probability requires a count-model conversion and minutes assumptions.

Blend market and statistical estimates using validated horizon-specific reliability. Never apply team fixture strength twice when it is already present in a provider forecast. Long-horizon strength has wider parameter uncertainty and no assumed live market coverage.

### 7.4 Player attack, roles, and assists

Separate open-play, set-piece, and penalty goal production. Estimate player involvement conditional on minutes, role, teammates, and team attacking intensity. Normalize allocations so simulated credited scorers are compatible with actual simulated team goals.

Use hierarchical shrinkage for finishing and creation: player, role, club, competition, age/experience where supported. Historical overperformance is weak evidence unless persistent over sufficient exposure. Promoted players and incoming transfers receive league-strength-adjusted priors with wider uncertainty.

Assist allocation must reflect FPL assists, including its current definitions; statistical xA alone does not capture every FPL assist. Keep a calibrated residual component for eligible rebound/penalty/free-kick/deflection cases. A scorer cannot assist their own goal; an own goal has distinct attribution.

Future role evolution is a scenario, not a permanent extrapolation of the last match. Repeated advanced positioning changes involvement priors; one tactical experiment has less weight.

### 7.5 Clean sheets and goals conceded

Use exposure intervals and goal timing. Clean-sheet eligibility depends on official minutes and scoring rules. A substituted defender may retain a clean sheet after a later team concession. A dismissed player has different conceding/clean-sheet treatment. Simulate these events explicitly.

For conceded-point deductions and save points, evaluate the discrete score function. E[floor(GC/2)] differs from floor(E[GC]/2); likewise for saves/3. Preserve these distinctions rather than rounding expected counts.

### 7.6 Defensive contribution

Fit count distributions or directly calibrated threshold classifiers by role and position. Use exposure, opponent possession/attacking style, expected defensive workload, own-team tactics, and recent role changes. Compare negative-binomial, hierarchical count, and empirical approaches.

For the captured rules, defensive threshold points are two per fixture: defenders need 10 CBIT; midfielders/forwards need 12 including recoveries. Keep thresholds in season configuration. E[DC points] = reward × P(action count reaches threshold), not reward × average actions/threshold. [Official defensive-contribution explanation](https://www.premierleague.com/en/news/4361991/).

Older seasons may lack these actions or use different measurements. Mark absent fields missing, use partial pooling and modern-era training, and never create a historical DefCon label from invented actions.

### 7.7 Goalkeepers, discipline, and bonus

Model saves conditional on shots on target and team defence; calibrate rare penalty saves conservatively. Incorporate current goalkeeper goal rewards from rules, rather than inheriting an old hardcoded six-point assumption.

Use competition-specific suspension rules, ban service dates, yellow-card cutoffs and red-card appeals. One card is not an independent Poisson count of unlimited yellow cards in the scoring layer.

For bonus, model conditional BPS and its competition across all players in each match, or use a calibrated match-conditioned categorical bonus model. Simulate the official tie rules. Old BPS totals cannot simply be carried forward through a changed action formula. When raw actions are insufficient to rescore history, use a documented season adaptation with greater uncertainty.

### 7.8 Full distribution and coherence

For player i and fixture f, calculate:

Points(i,f) = appearance + goal reward + assist reward + clean-sheet reward + saves + penalty saves + bonus + defensive contribution − goals-conceded deductions − missed penalties − cards − own goals.

All rewards and thresholds come from rules_version. GW points are the sum over fixtures assigned to that event.

Simulate common football worlds: scores, timing, lineups, substitutions, scorers, assistants, saves, discipline, actions and bonus. Teammates share clean-sheet outcomes and fixture disruption; opposing attackers and defenders interact; captaincy and rival outcomes reuse the same realization. Do not sample every player's points independently.

Separate outcome uncertainty from uncertain model parameters, lineup reports, injury duration, and fixtures. Bootstrap or posterior parameter draws produce the latter. Export expected points, SD, P(0 minutes), P(start), P(60+), minute quantiles, point quantiles, P(haul), component expectations, coverage and forecast age.

### 7.9 Ensemble design

Train stacking weights on out-of-fold forecasts only. Constrain or regularize weights, account for residual correlation, and allow position/horizon/playing-time groups only where sample sizes support them.

Combine distributions consistently. A mixture distribution is not the same as averaging samples from models with incompatible point totals. Keep provider inputs separate from own forecasts and distinguish an independent model from an ensemble that consumes a competitor's predictions.

## 8. Contextual intelligence that can be measured

### Evidence pipeline

1. Discover new source items.
2. Fetch original text or permitted transcript; keep publisher and exact timestamps.
3. Resolve player/club/competition identity.
4. Extract structured claims and supporting spans.
5. Detect copied reports and cluster by original evidence.
6. Check contradictions, publication age, relevance and supersession.
7. Propose a change to a defined model parameter or scenario.
8. Apply bounded changes through a calibrated evidence adapter.
9. Recompute affected forecasts and plans.
10. Explain the decision delta and retain the prior state.

Claim types: unavailable, available-for-selection, full training, individual training, fitness test, minutes restriction, role change, formation change, set-piece change, penalty order, disciplinary action, lineup evidence, fixture change, transfer/registration.

“Available” does not mean “will start”; “back training” does not mean “fit for 90”. Preserve manager ambiguity. Distinguish the quote, interpretation, and quantitative impact.

Source reliability is specific to claim type, club, and time. A reporter accurate on lineups need not forecast injury durations well. Repeated syndication of one quote counts as one evidence origin. Upvotes, follower count, and an elite rank are not reliability labels.

### Hosted LLM responsibilities

- Extract entities and claims into validated JSON.
- Explain press-conference language and possible tactical mechanisms.
- Identify contradictions and evidence gaps.
- Suggest targeted research questions.
- Narrate the optimizer's actual results.

The LLM does not generate authoritative xPts tables, change bank balances, infer hidden rival transfers, or claim a solver was run. It receives tools with typed read access and submits proposed evidence updates; numerical/state changes pass independent validators.

No local LLM or local embedding model is required. Use lexical retrieval first. Public football context can be sent to a hosted model where allowed; keep API keys, cookies, manager identity details and private account exports outside prompts.

### Model selection and zero-cost operation

Use a capability registry and account-specific probes. Research-time candidates are gemini-3.8-flash for difficult extraction/synthesis, gemini-3.5-flash-lite for simpler high-volume extraction, and gemini-2.5-pro as a reasoning challenger where available. The inspected pricing page lists free text inference for these candidates, with different tool availability. Default to the strongest candidate that passes the football-claim benchmark and fits the account's free quota; route simple work to a lighter model only if extraction quality holds. Free text does not imply free search, caching, batch, or every tool. [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing).

Read current provider terms for the deployment's purpose and region. Gemini's current terms include developer-use and regional restrictions, so personal/deployed-product eligibility must be checked rather than assumed from an API key working. Keep an eligible free-hosted fallback adapter and deterministic evidence entry. [Gemini terms](https://ai.google.dev/gemini-api/terms), [Groq free-tier information](https://console.groq.com/docs/billing-faqs).

The application performs direct source retrieval itself, so it does not require a paid search tool. An available free grounding capability is optional enrichment. Account quotas come from the account and current documentation; static internet RPM/RPD tables are not capacity guarantees. [Gemini rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits).

Set paid_provider_calls_allowed=false. Prefer an unbilled free project with no automatic upgrade. A provider/model/tool must be explicitly allowlisted as free; an unknown price fails closed. Budget notifications alone are not a spending cap. At quota exhaustion, queue research, reuse timestamped claims, accept structured human evidence, and keep the numerical planner working.

### Evaluating contextual value

Create a labeled set of news items with original text, entities, publication cutoff, correct claim type, uncertainty and subsequent outcomes. Measure extraction correctness, source support, temporal validity, contradiction handling, and downstream calibration.

Run paired forecasts with context enabled/disabled on the same cutoffs. Compare minute/start/60+ accuracy, event forecasting, transfer decisions, and chip policies. Log harmful overrides. Context earns production influence through this test, not because its explanation sounds plausible.

## 9. The exact rules and account-state engine

Load a versioned rules artifact from official documentation plus live settings. Include explicit effective periods and change records. Build the scorer, legality validator and transition engine before optimizing.

Mandatory behavior:

- Exactly 15 players with 2 GK, 5 DEF, 5 MID, 3 FWD under captured current rules.
- XI has exactly one goalkeeper and a legal formation; bench has a reserve goalkeeper and ordered outfield players.
- Distinct captain and vice-captain in XI.
- Maximum three per club on new legal transfer states. Preserve the official grandfathered exception when a real transfer temporarily creates excess club ownership; the next applicable transfer must resolve it.
- Integer prices, acquisition lots, correct sell values, and nonnegative bank at every confirmed transfer bundle.
- Transfer caps, points deductions, free-transfer allocation and rollover, chip windows, initial-entry conditions, and event-specific overrides.
- Correct Free Hit restoration to the pre-gameweek underlying squad and account state, including price movements; transfers made earlier in the same week must not accidentally survive the Free Hit.
- One chip per GW, no use after expiry, and the current consecutive-Free-Hit restriction including GW19→GW20.
- Autosubs and captain fallback based on zero minutes across the whole GW; a one-minute cameo blocks replacement and a single appearance in a double prevents fallback.
- Final versus provisional scoring, disciplinary adjustments, bonus ties and official score corrections.

Relevant official details: [FAQs and edge cases](https://fantasy.premierleague.com/help/faqs), [rules](https://fantasy.premierleague.com/help/rules).

Let f_t be available free transfers immediately before an ordinary deadline and n_t the total confirmed chargeable transfers for that event. For normal weeks, implement f_(t+1)=min(cap, max(0,f_t−n_t)+1), unless an explicit season override applies. Under the captured chip rules, Wildcard/Free Hit preserve the corresponding next-deadline FT count rather than generating an extra transfer; represent this as a tested chip transition. Do not confuse previously saved transfers with the current allocation.

Hits are event-level max(0,n_t−f_t) × current hit cost, not a second charge on transfers already paid for earlier that week. If a Wildcard/FH removes applicable earlier penalties, reverse them in the same event ledger.

For purchase price p and current price c in £0.1m units, ordinary sell value is c when c≤p, otherwise p+floor((c−p)/2), subject to actual account quotes and rules. Unknown purchase lots mean future sell values need conditional scenarios; a supplied current sell price alone does not identify the original purchase price.

## 10. Sequential optimization

### State and actions

State includes squad, purchase lots, bank, FT, chips with periods, prior Free Hit use, cumulative points, pending transfers, fixtures, information cutoff, and objective. Actions include roll, transfer bundles, legal lineups, ordered benches, captain/vice and chip activation. Information arriving later may change future actions.

Use a staged design:

1. Deterministic multiweek MIP provides a legal expected-value baseline.
2. Generate diverse high-quality first actions and transfer architectures.
3. Evaluate candidate policies in correlated simulation.
4. Introduce limited scenario branches for high-value uncertainty.
5. Re-optimize after information becomes observable.

MIP variables cover squad membership, buys/sells, XI, captain/vice, chip use and FT state. A network/state formulation can express piecewise FT and ownership-lot behavior more safely than fragile arithmetic. Chip-specific squads and original-squad restoration need explicit states.

Do not claim a linear sum of player xPts solves exact nonlinear autosub and opponent-win probabilities. Use scenario formulations where tractable or candidate generation plus exact policy rescoring. Report which layer is exact, approximate, or simulation-based.

### Planning horizons

Maintain a GW1–GW38 calendar and evaluate chip windows across the remaining season. Use detailed near-term forecasts and branches, then coarser distant scenarios. Begin with a configurable 6–8-GW tactical horizon and extend to the relevant chip boundary and season end where needed.

A receding horizon must include terminal value for squad quality, bank, remaining transfers and chips. Learn or calibrate this from held-out policy simulations. Validate against longer-horizon solutions to avoid selling future value merely because it lies beyond a cutoff.

For the original case, solve the complete GW9–GW19 interval and assess post-GW19 state value. Show a baseline transfer ledger for all eleven weeks plus a small explicit contingency tree. Distant moves are conditional decisions, not claims of future knowledge.

### Objective functions

Default: maximize expected net FPL points, accounting for autosubs, captaincy, hits, and future state value.

Mini-league mode: maximize probability of beating both rivals at the target deadline, subject to a visible expected-points sacrifice budget. Generate a Pareto frontier rather than silently choosing an arbitrary risk preference.

Top-1% mode: evaluate rank outcomes against a modeled population conditioned on current standings, participation and decisions. Report the sampling frame and model dependence. Overall rank is not identical to beating a single mean field team.

Use variance or CVaR only for an explicit user objective. Ordinary expected-points optimization should not penalize outcome variance by default; forecast uncertainty and estimation error are separate issues.

### Transfer timing and information

Compare early transfer against waiting using expected points, injury/rotation news still to arrive, price-change scenarios, and the chance of losing a uniquely valuable affordable move. Use the account's actual sale-price thresholds.

The value of information is the expected improvement in the best decision after a report arrives. Prioritize research when the likely decision improvement exceeds its compute/quota cost. Do not spend scarce hosted calls on players irrelevant to any feasible alternative.

### Diversification and solver correctness

Generate alternative plans with different first actions, chip dates and squad structures; simple exclusion cuts can avoid duplicate solutions. Always include hold/roll and user-proposed architectures.

Return objective, runtime, feasible incumbent, solver status, best bound, MIP gap, input hash, random seed, candidate restrictions and horizon. A proven MIP optimum is optimal only within its specified inputs/formulation. Scenario sampling, candidate pruning and uncertain forecasts remain limitations.

Never drop all obscure cheap players because of low current ownership. Candidate screening must preserve budget enablers, future role changes, captaincy paths and every relevant position/price band. Audit screened solutions against a wider pool on representative weeks.

## 11. Chip strategy and fixture uncertainty

Evaluate chips jointly with transfers, prices, squads, captaincy and opposing chip opportunities.

- Wildcard: permanent squad value across the horizon, bank/FT consequences, future flexibility, information timing and next chip period.
- Free Hit: incremental whole-plan benefit against the best legal no-FH route; include restoration and delayed transfers. Highest raw FH score does not imply highest FH value.
- Bench Boost: score all 15, joint appearance risk, cameo risk, goalkeeper expenditure, preparation/cleanup transfers and lost investment in the XI.
- Triple Captain: incremental extra captain copy, fallback, minutes and rotation correlation across fixtures.

A chip's benefit is the counterfactual difference between complete policies using common simulated worlds. Bench Boost value is not always simply four displayed bench projections: ordinary autosubs already realize some bench points, and preparation changes the whole team.

Confirm actual fixture count and registration for every GW. Double gameweeks increase opportunities and congestion simultaneously. Simulate correlated postponements and rearrangements from official announcements/cup-path scenarios; do not independently move both sides of the same fixture.

Store fixture states: confirmed, scheduled-but-changeable, unassigned, conditional proposal. A predicted double is a branch with probability and provenance. Refresh after cup draws, progression, weather disruptions, and broadcast changes. Include international tournaments and travel only when relevant to that season.

## 12. Elite managers, EO and rivals

Use separate cohorts:

1. Current top 1k/5k/10k by rank, with snapshot timestamp.
2. Multi-season elite cohort selected using only prior seasons.
3. Active-manager cohort with a published activity definition.
4. User mini-league and explicit rivals.

Current leaders can be selected by luck; a preseason elite panel has different bias. Never choose managers using future season ranks when backtesting. LiveFPL provides public elite and tier views that can seed research, subject to access and freshness checks. [Elite view](https://plan.livefpl.net/elite), [rank tiers](https://plan.livefpl.net/rank_tiers).

Record exact retrieved count, sample IDs/hash, inclusion criteria, misses, pagination, time and uncertainty. “Top 5k analysis” requires those actual records or a documented sampled estimate. No inferred precision from a hundred entries.

Effective ownership is expected scoring exposure: selection, bench/BB use, captain/TC multipliers, vice fallback and autosubs in a stated cohort. It can exceed 100%. It is not just ownership plus displayed captain percentage under all circumstances.

Model rival policies, not private future actions: hold, template transfer, model-optimal feasible transfer, and more aggressive catch-up/protect-lead scenarios. Revealed history updates beliefs. Unknown bank, acquisition prices or chips remain scenario state.

In shared simulations, each player has the same realized points for every manager. This is essential for realistic differential distributions. With gap d_r, estimate P(our future net points − rival future net points > d_r), with official tie treatment if relevant. For two rivals, calculate the joint event, not the product of independent win probabilities.

Report Monte Carlo sampling error separately from model uncertainty. A precise simulation under questionable rival assumptions is not a precise real-world probability.

## 13. Evaluation that can support competitive claims

### Data splits

Train chronologically, tune on later validation blocks, and hold out subsequent seasons or contiguous GW blocks. Split by deadline and fixture groups, not random player rows. Purge overlapping target windows for multi-GW forecasts. Fit preprocessing, calibration, imputation and ensemble weights within training folds.

scikit-learn's time-series splitter illustrates temporal evaluation, but use a custom deadline-group splitter for irregular player-fixture rows and multiweek labels. [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

Historical datasets frequently contain final revised fields. Label such experiments “historical statistical replay” unless true pre-deadline feature provenance is available. Maintain a separate strict point-in-time track. Modern news/context validation begins with archived evidence; an LLM's retrospective football knowledge must not leak into a replay.

### Forecast metrics

- MAE/RMSE and signed bias for all eligible players, relevant ownership bands, likely starters, known starters only as a diagnostic, each position, and price tiers.
- Availability/start/60+ Brier score and log loss; minutes MAE and distribution calibration.
- Goal/assist/CS/DC event calibration and proper scores.
- CRPS or equivalent distribution score, quantile loss, interval coverage and width; handle discrete atoms correctly.
- Ranking/NDCG, captain-candidate performance and realistic fixed-budget squad selection.
- Horizon-specific results for one, two, three and longer windows; double/blank and congestion subgroups.
- Actual-return cohorts only as retrospective diagnostics; do not use knowledge of who hauled to select headline test cases.
- Coverage, missing predictions, forecast age, and near-deadline timing equality.

Use paired deadline/fixture-block bootstrap intervals and multi-seed comparisons. Players in the same match are dependent; resampling individual rows understates uncertainty. Multiple experiments need a documented selection protocol and an untouched prospective track.

### Decision and policy metrics

Replay equal starting squads, prices, knowledge cutoffs, free-transfer states, chip rules, and transaction constraints. Compare own forecast/own optimizer, alternative forecast/same optimizer, and same forecast/alternative optimizer to locate the source of improvement.

Report net season points, hits, captaincy points, chip counterfactual value, missed deadlines, invalid squads, recommendation churn, affordability failures and rank relative to a disclosed field.

Distinguish:

- Forecast experiment: prediction quality without full decision rollout.
- One-step action comparison: each week begins from the same reference squad.
- Continuous policy simulation: each policy owns its evolving squad and bank.
- Prospective shadow team: frozen real decisions recorded before deadlines.

Hindsight “best possible transfers” may measure regret but are not achievable competitors. Do not score predicted xPts against itself as evidence of real improvement.

### Promotion gates

Gate A: all rule/state invariants and source contracts pass; zero illegal recommended actions.

Gate B: the new forecast improves the predeclared main metric or a necessary weak component on held-out data, with no material deterioration hidden in key cohorts.

Gate C: context-on paired testing supports beneficial minutes/availability effects and acceptable extraction error.

Gate D: optimizer passes small-instance exhaustive enumeration and realistic replay; approximations and candidate restrictions are disclosed.

Gate E: prospective synchronized comparison with accessible providers over enough independent deadlines to judge uncertainty. Twenty deadlines can be an initial review point, not automatic proof.

Gate F: full-season policy scorecard, rank population definition, and attributable sources of improvement.

Publish “competitive prototype”, “historically validated”, and “prospectively competitive” as different statuses. A single top-1% season is an outcome; a reliable competitive system needs repeated evidence.

## 14. Product experience

Default home screen: one clear recommendation with its validity timestamp, source freshness, exact transfer bundle, hit, bank, FT state, XI, bench, captain/vice and chip. Include “roll” when it wins.

Expandable evidence:

- Why the recommendation wins and its nearest alternative.
- Immediate and horizon net points; uncertainty and probability of outperforming hold.
- Role/minute/availability evidence and changed parameters.
- Affordability today and adverse-price scenarios.
- Trigger that changes the recommendation.
- Solver status and forecast source/version.

Required workspaces:

1. Squad audit and all-market player explorer.
2. Transfer package comparison and price bridge.
3. Full calendar with blank/double uncertainty and all-competition congestion.
4. Chip planner with whole-policy comparisons.
5. Captaincy and bench optimizer.
6. Rival/EO analysis with explicit assumptions.
7. Research inbox with source-supported claims.
8. Editable beliefs, with expiry, reason and visible forecast delta.
9. Backtest/forecast scorecards and prospective archive.
10. Data health, quota, model version, refresh/recompute controls and export.

The user's requested report should include a complete chronological ledger:

GW | FT before | transfers | hit | buy/sell ledger | bank | permanent squad | temporary chip squad | XI | bench order | captain | vice | chip | expected net points | interval | branch assumptions.

Do not use an LLM-generated table as the ledger source. Render it from validated structured plan data. Export JSON and CSV plus a readable report; include an ICS deadline calendar using authoritative event timestamps.

## 15. Reliability, security, and operation

Use atomic writes, transactional state changes, idempotent jobs, schema drift alerts and a last-known-good artifact. Keep raw data and outputs versioned separately. Offline replay uses only the locked local archive.

Freshness gates are field-specific. A price change invalidates an exact transfer quote; an unrelated old article need not invalidate all projections. A critical unresolved identity or account-state conflict blocks only the affected recommendation and offers a verified feasible alternative.

Fetch external pages as untrusted data. Source text cannot change system instructions, authorize tools, expose keys, or invent model parameters. Validate URLs, content length, redirects and requested output schemas. Prevent arbitrary local-file/metadata-service fetches in the research service.

Bind the personal API to localhost by default. Keep secrets in environment/OS storage, redact logs, validate exports, and do not expose keys in frontend bundles. Sign or hash manifests; do not publish personal account state with public reports. Hosted AI receives the minimum required football context.

Monitor source lag, fetch errors, field coverage, model bias, minute calibration, evidence quality, solver gap/time, quota and last successful pre-deadline run. Back up state and test restoration.

Operational cadence:

- After fixtures: ingest provisional outcomes and all-competition minutes.
- After official lock: save finalized labels and evaluate frozen forecasts.
- Midweek: update roles, injuries, fixture scenarios and medium-term plans.
- After press conferences: process high-impact evidence and recompute affected candidates.
- Before deadline: refresh official data and account state, recheck affordability, freeze report and emit checklist.
- After deadline: archive submitted/recommended state separately; later compare outcome without rewriting history.

No automatic account changes are required for the product. Produce an exact action checklist by default. Account execution can be a separately enabled capability with its own current-state and deadline safeguards.

## 16. Delivery plan and acceptance criteria

Phases are dependency gates, not excuses to ship a thin prototype as the finished product.

| Phase | Build | Exit evidence |
|---|---|---|
| 0: research lock | Source/access registry, grant scope, rules snapshot, benchmark protocol | Actual HTTP/schema tests, conflict-resolution record, free-provider capability checks |
| 1: trustworthy state | Identity mappings, immutable snapshots, account imports, scorer, FT/chip/price transitions | Boundary tests, replay of official examples, validated supplied selling-value total |
| 2: useful baseline | Complete player-fixture forecasts, deterministic optimizer, CLI/report | Every candidate has provenance/coverage; legal affordable plan; hold alternative |
| 3: model quality | Minutes model, team rates, component distributions, direct-model challengers | Frozen held-out scorecards by cohort and horizon |
| 4: context | Source discovery, claim extraction, contradiction handling, evidence updates | Labeled extraction benchmark and context-on/off results |
| 5: strategy | Correlated simulation, autosubs, all chips, scenarios, rivals and timing | Exhaustive tiny-case oracle; common-world policy comparison; solve diagnostics |
| 6: finished product | Full interface, health/status, exports, backups, free-cost enforcement | Working end-to-end journeys and restart/restore/quota tests |
| 7: competitive release | Prospective shadow decisions and synchronized benchmark archive | Publicly auditable scope, uncertainty and real performance evidence |

Parallel implementation can occur where dependencies allow, but a model that lacks Gate 1 correctness must not influence real transfers.

Required first demonstration: import this user's dated context, refresh official identity/rules/fixtures, produce a baseline all-market forecast and legal plan, compare the five original FH strategy families plus alternatives, and show actual executed checks. It must be labeled baseline until validated.

### Required repository structure

~~~
apps/web/
packages/contracts/
src/fpl_intelligence/
  ingestion/  identity/  storage/  rules/  state/
  features/  models/  research/  evidence/
  simulation/  optimization/  rivals/  evaluation/
  reports/  api/  cli/
configs/
schemas/
data/raw/  data/normalized/  data/features/
artifacts/forecasts/  artifacts/plans/  artifacts/evaluations/
tests/unit/  tests/property/  tests/integration/  tests/oracle/  tests/e2e/
docs/methodology/  docs/operations/  docs/model_cards/
~~~

### Required commands and interfaces

CLI: doctor, ingest, snapshot, reconcile-team, build-features, train, calibrate, forecast, research, simulate, optimize, explain, replay, evaluate, export, serve.

API: health/data freshness; player/fixture histories; forecasts; evidence and proposed overrides; account-state validation; asynchronous optimize job; plan alternatives; replay/evaluation scorecards; artifact export.

Optimization requests carry a frozen state ID and forecast ID. Results carry those IDs back. A stale response cannot overwrite a plan for a newer bank, squad or deadline.

## 17. High-value failure cases to test

1. Same player name across positions/clubs; accented aliases; season ID reuse.
2. Own-team starting XI unrevealed before deadline; public picks are one week old.
3. Current selling values known but purchase lots unknown.
4. Profit odd increments, price fall below purchase, repurchase creating a new lot.
5. Transfers already made this week; hit reversal after an eligible chip.
6. FT cap reached, zero FT, late entry, unexpected official allocation override.
7. FH after earlier transfers; restoration after underlying price changes.
8. FH19 prevents FH20 under current consecutive-week rules.
9. Real transfer creates four players from one club.
10. Blank, double, triple, postponed or null-event fixture.
11. Captain misses first double match but appears in second.
12. Captain/vice both absent, cameo blocks fallback, TC fallback.
13. Autosub order with missing defenders; reserve GK only replaces GK.
14. Bench Boost compared with normal autosubs and budget diversion.
15. 59/60-minute boundary, stoppage time, red-card conceding, substituted clean sheet.
16. BPS multiway ties; new goalkeeper scoring; DC threshold exactly below/at/above.
17. Official revision after provisional scoring; unchanged finalized labels preserved.
18. Conflicting injury sources, repeated quote, retracted leak, stale training report.
19. Prompt injection in a scraped article, invalid hosted JSON, invented evidence link.
20. Model retires, API quota exhausted, free-search capability absent, unknown price schedule.
21. Schema changes or removed player; missing data not replaced by zero.
22. Hindsight injuries, closing odds after deadline, retroactively revised ownership.
23. Candidate screening drops the only affordable enabler.
24. Solver times out, reports infeasible, or returns an incumbent with a nonzero gap.
25. Scenario solver chooses future actions before the corresponding information exists.
26. Shared rival captaincy simulated inconsistently.
27. Full-market RMSE improves only because more nonplayers were correctly predicted.
28. User manually changes squad while optimization is running.
29. Computer sleeps or scheduled cloud job misses deadline.
30. Forecast is accurate but explanation cites a different version.

## 18. Resource budget and stopping rules

These are initial engineering budgets, not measurements of an implemented system. Benchmark on the actual machine and adjust within the verified free allowances.

| Workload | Initial budget | Stopping/recovery rule |
|---|---|---|
| Source ingestion | Bounded per-source concurrency; incremental affected records | Stop retrying after a configured small limit; retain source age and use fallback |
| Hosted extraction | Soft budget of 20 ordinary-day calls, up to 60 deadline-day calls only when actual quota permits | Use the minimum of this budget and account request/token limits; reserve capacity for high-value uncertainties |
| Research payload | Retrieve and deduplicate first; send only relevant passages | Avoid full-web-page context dumps and repeated identical calls |
| Daily forecasts | CPU inference over all players/fixtures; update only affected models | A complete baseline must remain available if a challenger exceeds budget |
| Initial strategy screen | Approximately 2,000 shared worlds for candidate screening | Eliminate clearly inferior actions with paired uncertainty checks |
| Final strategy comparison | Start around 20,000 worlds; adapt upward only for material near ties | Bound Monte Carlo error for the decision difference; stop when remaining sampling error cannot change a meaningful action |
| Tactical MIP | Start with a 30–120-second interactive limit and a longer explicit batch mode | Return the legal incumbent, gap and best bound; investigate repeated weak solves |
| Training/search | Fixed trial/time budget, several seeds, nested validation | Stop if gains are below experiment noise or repeatedly fail key cohorts |
| Storage | Compressed immutable snapshots; partition by source/date/season | Retain decision/evaluation evidence, remove regenerable caches first |

For a probability near 50%, 20,000 independent simulation draws have Monte Carlo standard error of about 0.35 percentage points. This is numerical sampling error, not total uncertainty; it says nothing about whether the injury, rival, or scoring model is right. Use paired shared-world differences for plan comparisons and account for weighting/dependence if scenarios are not independent.

Capacity accounting must be explicit:

- Monthly provider requests = actual daily extraction + targeted synthesis + retries, not the advertised token context size.
- Monthly cloud execution minutes = job frequency × measured job duration; include retries and artifact retention.
- Raw storage = snapshot count × compressed size; log retention and model artifacts are separate.
- End-to-end refresh latency = fetch + validation + feature update + forecast + solve + report; do not optimize only solver time.

Avoid a large cross-product of every injury, fixture, price and rival branch. Rank uncertainty by its decision impact, branch only on material observable events, and test whether further branching changes the first action. Reuse scenarios, warm starts and cached feature computations.

### Release benchmarks to measure, not invent

Record hardware, peak RAM, wall time and numerical settings for clean installation, full refresh, incremental news update, forecast generation, tactical solve, full case comparison, and historical replay. Set practical latency targets only after those measurements. Always expose the faster validated baseline if the deeper analysis is still running.

The free-cost gate covers software/data/inference/service charges. It does not claim electricity, an internet connection, development labor or unlimited hosted capacity have no resource cost. No paid component is required in the selected default architecture.

## 19. What has been completed in this research package

Completed: attached prompt analysis; primary-source and community research; live HTTP tests and schema inspection; official season/rule/fixture conflict identification; user-context arithmetic; chosen architecture; forecasting/optimization/evaluation methodology; free-hosted-AI design; data fallbacks; delivery gates; copy-ready implementation prompt and structured context.

The predictive model, optimizer, prospective shadow team and interface are the implementation requested by the accompanying master prompt. They have not been represented as already trained, solved, benchmarked or deployed in this research phase.

Use the master prompt with this blueprint, source ledger and context file together. The implementation agent should execute the phases, produce real artifacts, and let measured evidence decide which challengers enter the final system.
