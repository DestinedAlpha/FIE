# Multi-source forecasting and training

The original production forecaster did not combine Reddit, Scout analysis and qualified managers. Its small MLP comparison did not settle whether a purpose-built multi-source network would help. The new implementation makes that comparison explicit; collecting a source is distinct from promoting its predictions.

## Sources and their influence

| Input | Retrieved coverage | Influence and timing |
| --- | --- | --- |
| Official FPL fixture outcomes | Four prior seasons plus completed current matches | Targets and lagged numerical/history inputs; target-gameweek outcomes never enter features |
| Archived official status, injury news, set-piece order, ownership and transfers | 160 decision deadlines | Hash-pinned upstream snapshots; both capture and containing Git commit must precede the deadline |
| Qualified managers | 100 verified managers, 800 released manager/gameweeks | At least two top-10,000 finishes in three prior seasons; only previous-gameweek intended picks/captaincy enter the next decision |
| Other competitions | 3,238 accepted player matches; 1,698 rejected records | Stable player codes, explicit minutes including bench zero, completed at least four hours before the decision. Retrospective enrichment, without original publication-time proof |
| Public Fantasy Football Scout articles | RSS plus isolated public article bodies; two validated current claims after identity review | Exact source-linked claims in the evidence inbox. Specialist recommendations have no unvalidated numerical uplift; locked articles and comments are excluded |
| Reddit | Fifteen accessible r/fplAnalytics leads at initial collection; r/FantasyPL rate limited | Leads to registered original sources only. No Reddit text is used to train weights; blocked access is disclosed |

Manager coverage starts in the current season. Only about 3.9% of historical training rows have this expert feature, so this release cannot establish its long-term incremental value. Available expert cohorts also observe zero ownership, distinguished from an unavailable cohort. Membership is a reproducible prior-results sample, not a representative sample of all elite managers. Survivorship and source selection remain limitations.

Official news coverage means registry snapshots are available, not that every player has an injury article. Missing competitions, international/travel workloads, paywalled analysis and unavailable historic specialist claims remain missing. No archive is invented to fill them.

The identity audit rejected a valid-but-wrong player ID for a Palmer quote, an ambiguous Pedro reference, and a quote spanning two different injured players. The explicitly named Palmer quote was corrected deterministically to the registered Cole Palmer identity. Official injury-only spans were expanded to their exact named registry lines. Quarantined originals remain in the immutable audit history and cannot appear as active evidence or support numerical beliefs. The full batch must pass quote, source-hash and unambiguous identity checks before future provider results are committed.

## Purpose-built network

`fusion.py` implements a dedicated temporal fusion network, rather than a generic prompt or the earlier two-layer MLP:

- Twelve earlier player matches, sixteen numerical channels and observation masks, encoded by two temporal attention layers.
- Numerical form, team/opponent strength and fixture context.
- A convolution over stable hashed official-news tokens plus structured status, availability and set-piece features.
- Separate manager/market and workload branches.
- Two attention layers over the source branches, with unavailable sources masked.
- Joint minutes-band, conditional points, starter and seven observed component auxiliary objectives.

The network has 324,431 trainable parameters. This scale follows the available data and CPU constraints. It is not evidence that a larger architecture cannot improve, and parameter count alone is not a forecast-quality claim. Hosted Gemini/Groq remain source extractors, with free-account/quota checks; neither is treated as the numerical training engine.

The tree challenger sees the same numerical sequence, structured context and a hashed text bag. This prevents attributing the advantage of extra data to neural architecture. Neural normalization is fitted only on earlier training rows. Missing values have explicit masks. All-padding histories remain finite. Neural checkpoints load only tensors with `weights_only=True`, and both neural and tree artifacts are hash checked.

## Validation and operation

Expanding eight-deadline test blocks start at 2025/26 GW1. Four earlier deadlines are held out from base fitting for calibration and neural early stopping. Neural weights reset independently for every fold; seed, epoch budget and selected epochs are recorded. Neither actual test minutes nor hindsight-optimal teams enter inference. Component targets are training labels only; predicted minutes bands are marginalized.

Compare both candidates with the existing production model on identical player/fixture rows. Report RMSE, MAE, bias, likely-starter metrics and deadline-bootstrap intervals. A negative difference favors the candidate. The promotion requirement demands an upper interval below zero plus likely-starter improvement; a candidate is not activated merely because training completed. Any eventual selection needs decision/policy and uncertainty checks before release. Report unavailable prospective results as unavailable.

Hindsight-optimal squads and captaincies remain useful ceilings. They are noisy, lucky realized winners; copying them as if they were knowable before the deadline creates hindsight bias. Actual points/minutes/components, expert comparisons and legal evolving-account replay are separate evaluations.

```sh
uv sync --extra dev --extra neural
uv run --extra neural fpl sources-refresh
uv run --extra neural fpl train-fusion --epochs 20
```

Neural dependencies come from the explicit CPU-only PyTorch index. No GPU, hosted training or paid service is requested. The Windows launcher installs the CPU neural dependencies so its training controls work immediately. Advanced retraining is optional; allow roughly 2 GB of additional process memory and use a machine with at least 8 GB RAM for comfortable training alongside other applications. Prepared reports and the established production planner work without importing PyTorch. Retain the existing validated model if a candidate fails its gate.

The Model & data and Research pages display source coverage and completed model comparisons. Source refreshes preserve independent successful adapters when another source fails. Refreshing evidence never silently replaces an account or executes an FPL transfer.

## Executed 0.4 results

The completed temporal network has 324,431 parameters. On 34,828 matched held-out player-fixtures, reference RMSE is 1.97854, multi-source tree RMSE 1.90951, and neural RMSE 1.91950. Likely-starter RMSE is 3.04985 / 2.95230 / 2.96159 respectively. Both candidates improve point error against the reference under the declared deadline-bootstrap test.

Better player forecasts did not establish a better complete transfer policy. In the matched 30-week replay, the tree's one-week greedy transfer actor scores 1,609 versus the reference actor's 1,735; global captain bonus scores improve to 368 versus 323. Keeping the reference transfers while changing lineups scores 1,733. These negative decision results are retained. The reference model therefore remains the production transfer forecaster; both new models produce full-market analysis and disagreement views rather than silently replacing the planner. The shared-sampler tree audit reports likely-starter coverage 87.27% / 93.69% for central 80% / 90% intervals and CRPS 1.4411. Marginal accuracy does not certify joint rival probabilities.

The top-1,000 adapter actually retrieved 1,000 current-ranked managers, all 8,000 released manager/gameweeks, entry histories and public transfers. This population is separate from the 100 prior-qualified managers. Behavior inputs include lagged transfers/hits, FT, bank/value/points, roster injury and premium exposure, club concentration, captain continuity, turnover, purchase/sale costs and rapid resales. No current rank is a numeric feature. A basic/richer mixture is selected on GW5–6 before reporting GW7–8: log loss 1.14350 versus 1.35498 for the preceding-frequency baseline on 2,000 decisions; classification accuracy 43.10% versus 35.40%. The richer branch receives 25% weight. Current-rank selection is retrospective and the two later deadlines do not establish unbiased elite-skill or prospective calibration.

The model weights first-deadline rival action-count scenarios. Named moves remain feasible forecast-guided alternatives; count 3+ is approximated by up to three ordinary transfers, and later actions follow explicit greedy scenarios. Action-type independence and transfer from top-1,000 behavior to these mini-league rivals are assumptions. Hold/greedy sensitivities remain visible and no private future transfers are inferred. Recent purchases/sales and released ownership are also exposed directly in the interface.

A separate four-week value model learns actual subsequent points with purged label windows. Training and calibration outcome windows must finish before the next partition starts. RMSE is 4.74642 versus 5.08151 for repeating the next-week reference; that comparator is not the full multiweek optimizer. Added influence, creativity, threat, ICT, expected involvement/conceding, penalties and defensive statistics give RMSE 4.74176, but their four-deadline block interval relative to context is [-0.01498, 0.00360], so this extra branch is not selected. Existing xG/xA, actual returns, playing time, BPS, defensive actions and team/opponent statistics remain present.

Counterfactual transfer learning generated 912 legal synthetic accounts at genuinely observed prices across 38 decision deadlines, with varied FT balances. Candidate actions are independently transitioned through the rules engine and scored on actual four-week outcomes. Training label windows end before GW20, calibration decisions use GW20–24, and later tests begin GW28 after a three-week purge, also holding out current-season GW2–5. On 288 test accounts, the critic's mean gain is 6.243 points, with a four-deadline block interval [-0.323, 13.441]. Its calibrated residual weight is 0.25. The confidence gate fails, so it provides a personalized research ranking without replacing the selected plan. Best realized actions among the searched candidates are diagnostics, not a global optimal-team claim or privileged future features. Random states use zero accrued price profit, exclude unsupported current blank-week player rows, and hold lineups for four weeks to isolate transfer value; the full planner separately handles chips and evolving accounts.

Personalized analysis uses the actual confirmed squad, current exact sell quotes, bank and FT. Source-linked injury concerns create explicitly conditional lineup stresses, with no invented probabilities. An injury concern is not a confirmed absence. Hosted paraphrases are replaced with literal supported wording, and ambiguous identity or cross-player quotes are quarantined.

## Solver and recovery corrections

The no-hit comparison now considers stronger already-known feasible policies on the same frozen state, forecast and horizon. It cannot use a weak timeout incumbent to inflate the apparent return from taking hits. The optimizer can also reuse prior actions as incumbents: it replays financial transitions at current prices, remakes lineups with current means, checks every linear constraint and bound, and adopts a seed only if its objective improves. Incompatible chip, forced-transfer and hit constraints exclude that seed. Solver bounds/gaps still describe the declared additive formulation, not proof of globally optimal football decisions.

Each new official refresh freezes local next-deadline context before the deadline. Historical inputs and trained candidates have content-derived paths and verified hashes. Completed counterfactual deadlines can resume from local checkpoints; incomplete research runs never become completed model artifacts. The updater merges immutable research data and models while preserving account state and keys. Configuring one provider preserves the other provider and unrelated environment settings.

Useful operations:

```sh
uv run fpl managers-refresh
uv run fpl train-transfer-value --states 24
uv run fpl analyze-transfers
```

All eight completed gameweeks are used, with distinct roles. The production reference fits earlier seasons and calibrates on GW1–GW8; the new one-week multi-source fit trains through GW4 and reserves GW5–GW8 for calibration. Live profiles use all finalized matches. Four-week labels are available only for starting weeks whose full future window has completed, through GW5. Reserving observations from base fitting is deliberate protection against evaluating calibration on its own training predictions.
