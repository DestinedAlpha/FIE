# Source ledger and research notes

Inspected on 29 October 2026. This is a focused research record, not a claim to have inspected every web page or independently reproduced every provider result. Product descriptions establish what a provider says it offers; they do not establish its superiority.

## Direct access tests

| Source | Observed result | Design consequence |
|---|---|---|
| [Official bootstrap](https://fantasy.premierleague.com/api/bootstrap-static/) | HTTP 200, parsed JSON; 669 players, 20 teams, 38 events; 2026_27 asset path | Complete live core registry and season settings are available |
| [Official fixtures](https://fantasy.premierleague.com/api/fixtures/) | HTTP 200, parsed JSON | Build fixture/event calendar from actual records |
| [Player-history sample](https://fantasy.premierleague.com/api/element-summary/411/) | HTTP 200; eight current-season history rows in sample | Confirmed fixture-level fields include minutes, starts, expected stats and defensive actions |
| [Public entry](https://fantasy.premierleague.com/api/entry/9375678/) | HTTP 200; name Skibidi FC, 418 points, current_event 8 | Confirms selected public account context; does not establish current private picks or purchase lots |
| [Solio public JSON](https://fpl.solioanalytics.com/api/data/latest.json) | HTTP 200; generatedAt 2026-10-29T06:08:40.671Z; gameweek 9; 30 topProjected entries | Optional partial comparator; not a complete player-by-future-GW matrix |
| [Official rules route](https://fantasy.premierleague.com/help/rules) | HTTP 200 application HTML; web text extraction empty | Use official FAQs/update text plus live config and rendered rule inspection during implementation |

The audit JSON contains snapshot hashes and actual inspected schemas. Raw snapshots are preserved in the research workspace as intermediates. The package does not redistribute third-party full datasets.

## Rules and seasonal behavior

1. [Official 2026/27 updates](https://fantasy.premierleague.com/help/new). Primary official description. Supports the need to adapt BPS, price-change information and finalization behavior to this season.
2. [2026/27 BPS explanation](https://www.premierleague.com/en/news/4679946/). Primary official explanation. Relevant changes concern clearances/blocks/interceptions, being tackled, and goalkeeper save events. Historical BPS adaptation needs underlying-action coverage.
3. [Defensive-contribution explanation](https://www.premierleague.com/en/news/4361991/). Primary official rule explanation. Supports position-specific thresholds and per-match reward cap.
4. [Official FAQs](https://fantasy.premierleague.com/help/faqs). Primary rules/edge-case text. Inspected passages cover free transfers, chips, autosubs, selling values, public visibility and real-transfer club-limit exceptions.
5. [Official changes overview](https://www.premierleague.com/en/news/4679873). Primary official article. Useful for season changes, but its GW19 deadline differs from the captured event API.
6. [2024/25 transfer changes](https://www.premierleague.com/en/news/4058895). Historical official source. Demonstrates why old instructions circulate; do not prefer it over current rules.
7. [Official terms](https://fantasy.premierleague.com/help/terms). Primary season terms. Source access configuration should record the user's grant separately.

### Resolved deadline conflict

Captured current API:

- GW9: 31 October 2026, 11:00 UTC / 15:00 Dubai.
- GW19: 1 January 2027, 18:30 UTC / 22:30 Dubai.

The overview article gives GW19 as 2 January, 13:30 GMT. The current event feed is the operational scheduling input; retain the discrepancy and refresh it before acting. These are research-time observations, not permanent deadlines.

### Rule artifacts still needed during implementation

The live config contains scoring rewards and chip periods, but that alone is not the complete game. Narrative rules still govern thresholds, bonus ties, captain fallback, substitutes, chip restoration and special cases. Build one explicit rules artifact by reconciling these sources; test it against official examples.

## Forecasting providers and comparison

8. [FPL Review free-model documentation](https://docs.fplreview.com/the-model/projections/free-model/). First-party product documentation. Distinguishes the accessible baseline from the premium system and describes a three-GW horizon.
9. [FPL Review Massive Data documentation](https://docs.fplreview.com/the-model/projections/massive-data-model/). First-party methodology overview. Describes statistical/ML inputs, context and updates; detailed proprietary implementation is not established by this page.
10. [FPL Review xMins](https://docs.fplreview.com/the-model/projections/xmins/). First-party explanation of minutes uncertainty and nonlinear points effects. Supports evaluating minutes distributions rather than only a mean.
11. [FPL Review original goalscoring comparison](https://docs.fplreview.com/articles/massive-data/). Historical first-party research. Its selection criteria and old scoring environment limit extrapolation to current all-player forecasting.
12. [Solio](https://fpl.solioanalytics.com/). First-party product/feed documentation. Describes market-informed projections, stochastic components and planner access. The feed was separately tested.
13. [FPL Bench methodology](https://www.fplbench.com/methodology). Benchmark operator's primary methodology. Weekly fixed-budget squad selection, approximate role weights, missing-prediction treatment and exclusion of transfer/chip decisions all affect interpretation.
14. [FPL Bench leaderboard](https://www.fplbench.com/). Dynamic benchmark result. Inspected to understand provider coverage and differing point/error rankings. No numeric leaderboard lead is treated as proof of superiority.
15. [Scout RMT launch](https://www.fantasyfootballscout.co.uk/2026/07/27/fpl-rate-my-team-points-projections-live-for-2026-27). First-party article describing current/historical inputs and human monitoring; complete projection tables are a member product.
16. [Scout tool guide](https://www.fantasyfootballscout.co.uk/2026/08/10/how-to-use-our-rate-my-team-tool-to-get-your-fpl-squad-rated). First-party interface description; useful for product comparison, not independent model validation.
17. [Fantasy Football Fix product description](https://www.fantasyfootballfix.com/why_fix/). First-party marketing/method description. Supports considering lineup and projection tools as competitors; its superlative claims are not independently established here.
18. [Fix feature explanation](https://www.fantasyfootballfix.com/blog-index/five-fix-features-help-you-get-flyer/). Historical first-party explanation of predicted-point features. Do not assume old access terms or model details remain current.
19. [Hub predictions](https://www.fantasyfootballhub.co.uk/predictions). First-party product route. Its existence does not establish a free complete export or enough detail to reproduce its model.
20. [Hub current guide](https://www.fantasyfootballhub.co.uk/articles/fantasy-premier-league-ultimate-guide-fpl-tips). Public analysis/discovery source. Individual public projections can be observations when dated, not a complete dataset.

No premium provider's full projection archive was obtained or benchmarked in this research phase. The chosen solution is independent full coverage plus explicit overlap comparisons and permitted imports.

## Open forecasting and solver research

21. [OpenFPL paper](https://arxiv.org/abs/2508.09992). Original author research, with historical and prospective claims. Valuable reproducible starting point. It predates current defensive-contribution/BPS conditions; re-evaluate under new rules.
22. [OpenFPL code](https://github.com/daniegr/OpenFPL). Author's implementation/model resources. Inspect dependencies, data requirements, license and inference behavior before adapting.
23. [Dastan](https://github.com/qazybekb/smartplayfpl-dastan). Author-published challenger, evaluation and limitations. Promising reference for participation-aware evaluation, not independently verified superiority.
24. [Open FPL solver](https://github.com/solioanalytics/open-fpl-solver). Current repository reached from the old Sertalp Bilal FPL-Optimization-Tools address. Provides deterministic optimization infrastructure using highspy. Code reuse does not establish that all current season rules or stochastic features are already implemented.
25. [HiGHS](https://highs.dev/). Primary solver documentation. Free licensed optimization is feasible; solver optimality still pertains to the actual formulation and inputs.
26. [Multi-stream football feedback research](https://arxiv.org/abs/1912.07441). Original research on incorporating human feedback in FPL prediction. Motivates testing information channels; it does not validate a modern LLM override pipeline.
27. [Skill in FPL](https://arxiv.org/abs/2009.01206). Original research on manager performance and decision-making. Supports studying repeated decisions across noisy outcomes; it is not a rank guarantee.
28. [Dixon–Coles extension paper](https://arxiv.org/abs/2307.02139). Original methodological research describing the dependence structure being extended. Candidate score models should be compared on the actual target data.

## Historical data and manager samples

29. [vaastav archive](https://github.com/vaastav/Fantasy-Premier-League) and [data dictionary](https://github.com/vaastav/Fantasy-Premier-League/blob/master/DATA_DICTIONARY.md). Maintainer-published historical data and definitions. Archive availability does not imply pre-deadline provenance for every field.
30. [Randdalf/fplcache](https://github.com/Randdalf/fplcache). Maintainer's bootstrap archive. Useful for point-in-time availability/price reconstruction; inspect actual dates, revisions and complete deadline coverage.
31. [Football-Data England files](https://football-data.co.uk/englandm.php). Publisher's free CSV catalog. Timestamp discipline is essential for odds; not every recorded price was available before an FPL deadline.
32. [StatsBomb open data](https://github.com/hudl/open-data). Data owner's repository. Competitions are selective; it is not a promise of full current PL event coverage.
33. [LiveFPL elite](https://plan.livefpl.net/elite) and [rank tiers](https://plan.livefpl.net/rank_tiers). First-party cohort/EO views. Dynamic public samples need timestamp and population definitions before use.

Understat is an integration candidate referenced in OpenFPL, not a newly authenticated full-current-season dataset in this session. Detailed current event feeds from other providers likewise require implementation-time coverage probes. Core official expected statistics remain the operational fallback.

## Community investigations

34. [r/fplAnalytics: first xPts model](https://www.reddit.com/r/fplAnalytics/comments/1rgumu6/first_attempt_building_an_xpts_model/). Original community report. The author's acknowledgement of nonplayer-driven metric inflation is a useful diagnostic warning; reported RMSE is not imported as a verified benchmark.
35. [r/fplAnalytics: component-model explanation](https://www.reddit.com/r/fplAnalytics/comments/1vioe4q/i_made_an_expected_points_model_here_is_how_ive/). Original community method discussion. Inspired explicit threshold/count/bonus comparisons, with independent testing required.
36. [Monte Carlo validation discussion](https://www.reddit.com/r/fantasyfootballcoding/comments/1wmmfi4/how_would_you_validate_a_monte_carlo_fpl_points/). Original community question. Supports researching distribution validation and frozen snapshots; the proposed exclusion of nonplayers is not adopted as the only evaluation.
37. [Reddit Data API terms](https://redditinc.com/policies/data-api-terms). Primary service conditions. Access and content-use constraints are separate from the user's FPL grant.

The system should keep researching these communities for new methods and original club evidence. Community consensus is not automatically a numerical feature.

## Hosted AI and free operation

38. [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing). Primary, changeable capability/price source. Inspected specific free text rows and different search/tool availability. Account access still requires a capability probe.
39. [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits). Primary documentation. Exact usable quota is account/project dependent; implementation must check it.
40. [Gemini structured output](https://ai.google.dev/gemini-api/docs/structured-output). Primary interface documentation. Supported schema output does not guarantee a semantically correct extracted claim.
41. [Gemini terms](https://ai.google.dev/gemini-api/terms). Primary purpose, regional and data-handling conditions. Working credentials do not alone establish deployment eligibility.
42. [Groq billing/free tier](https://console.groq.com/docs/billing-faqs), [rate limits](https://console.groq.com/docs/rate-limits), [API errors](https://console.groq.com/docs/errors). Primary fallback-provider information. No free account was authenticated or load-tested here.
43. [Colab FAQ](https://research.google.com/colaboratory/faq.html). Primary resource policy. Free interactive compute is useful; availability is variable.
44. [GitHub Actions workflow documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax), [schedule troubleshooting](https://docs.github.com/en/enterprise-cloud%40latest/actions/how-tos/troubleshoot-workflows). Primary execution and reliability constraints.
45. [Cloudflare Pages](https://developers.cloudflare.com/pages/). Primary static hosting documentation. Optional report/UI publication, not the selected numerical runtime.

## Numerical implementation references

46. [DuckDB Python](https://duckdb.org/docs/stable/clients/python/overview). Primary documentation for local SQL/Parquet analytics.
47. [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html). Primary temporal validation reference. Custom deadline grouping/purging is still necessary.
48. [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html). Primary objective/configuration reference. Select objectives through actual forecast validation.
49. [Numba documentation](https://numba.readthedocs.io/). Primary acceleration reference. Add it only after profiling.

## Research conclusions versus proposed engineering

Observed: successful feed requests, available sample fields, partial Solio coverage, official season changes, differing benchmark scope, context arithmetic and an official deadline discrepancy.

Reported by authors/providers: model quality, methodology summaries, projection horizons and product features.

Proposed here: the combined architecture, evidence adapter, calibrated context testing, full joint scenario policies, acceptance gates, operating budgets and product interface.

Not claimed: exhaustive web coverage; current private account access; actual training of the finished forecast system; execution of an FPL transfer optimizer; reproduction of competitor performance; a verified top-1% probability; or future season outcomes.
