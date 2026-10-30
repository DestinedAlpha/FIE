# Integrated decision learning (0.5)

The system has two distinct neural components. The temporal source-fusion network learns player minutes and points from historical statistics, twelve-match sequences, archived news, market/manager features and workload. Its outputs join the reference and tree forecasts in candidate generation, manager-choice features and decision analysis. A separate listwise decision network learns which legal actions lead to better four-week returns. Neither is a language agent or an independently reliable oracle.

## Random-account training

The proposed experiment is useful when the information cutoff is respected. We sample legal squads, budgets, free transfers and already-used chips. For each historical deadline, full registered-market integer programs generate ordinary transfers, roll and every available chip. The released experiment uses 12 accounts per deadline; the complete retraining command defaults to 24. Ordinary transfer search permits all 15 players. Search timeouts, excluded candidates and gaps remain recorded.

Each candidate evolves for four deadlines. Subsequent transfers, lineups, captaincy, bank, purchase lots, FT and chips are recomputed using information available at that later deadline. Actual minutes and points score the resulting decisions, including autosubs, fallback and hits. They never choose those decisions. This differs from holding the initial squad unchanged for four weeks.

An independent hindsight integer program uses realized points to provide a diagnostic ceiling for the immediate action. Exact realized rescoring and its solver gap are retained. It is an additive one-week ceiling, not a proven globally optimal multiweek team. Hindsight winners are noisy labels: training directly to copy the weekly dream team would encourage chasing outcomes that were unpredictable at the deadline.

The critic fits return residuals and a listwise preference over legal actions. Its input includes all three forecasters, their disagreement, squad quality, bank, FT, hits and chips. Train labels finish before GW20; calibration actions start GW20–24; tests start GW28, with overlapping four-week windows purged. We select the architecture and blend weight on calibration, then use the later test only as an acceptance gate. Four-deadline block intervals reflect shared football outcomes across synthetic accounts. This is decision-focused learning, inspired by predict-then-optimize research (for example https://arxiv.org/abs/1710.08005); it does not claim to implement SPO+.

## Continuation, rivals and information

A nonnegative regularized continuation model values future squad strength, bank, FT and unused chips from the same evolving episodes. Ridge selection uses calibration. Promotion requires both improved later-test prediction and a positive held-out policy-gain interval. Bank rewards are capped at training support. Without that evidence, the planner retains its disclosed fixed terminal sensitivity. Longer and shorter horizon solutions are compared under the same football worlds.

For the actual top 1,000 current-ranked managers, count, conditional buy/sell and chip models use earlier public squads, transfers, affordability and source-conditioned player forecasts. A branch feeds its own simulated actions and expected results into the next week's features. Feasible legal integer programs enforce the chosen count/chip with explicit retries when a sampled constraint cannot be filled. These are scenario policies, not knowledge of unreleased transfers. Prior-qualified experts form a separate, lagged cohort.

A rank-stratified sample includes actual entries across the full standings, including inactive teams. Tier/activity priors are calibrated separately; weighted shared-world rank scenarios state their sampling and model dependence. Transfer preferences learned from elite entries are an explicit assumption outside that cohort.

Price models use paired official snapshots captured and committed before deadlines. The training interval is 40–72 hours; different waits are labeled sensitivities. Timing compares early commitment and waiting with legal adjustment in both branches, exact current selling quotes and conditional future sale bounds. Research priority measures how a report could change the best feasible action. A report is valuable only through a decision it might change.

## Sources and structural models

Source-off tests independently refit history-only, official-news and full-context trees. Archived official news, set-piece duties, lagged qualified-manager signals and all-competition workload feed measured features. Supported public Scout claims feed bounded availability/scenario analysis. Reddit is used to discover verifiable originals; blocked access is not bypassed and raw Reddit text is not a training corpus. Rejected or ambiguous claims cannot re-enter scenario construction.

A separate structural simulator draws real lineups, at most five substitution pairs, shared goals, compatible scorers/assists, exposure-dependent scoring, saves, discipline and match-wide bonus ties. Learned event heads and conditional BPS have their own gates. Known-event BPS validation is not complete future-forecast validation. This simulator is connected to policy sensitivity analysis but cannot silently replace the better-supported production forecast.

## What acceptance means

A smaller prediction error does not automatically imply better transfers or captaincy. The accepted reference transfer policy remains active unless a challenger passes its relevant decision gate. Completed current-season GW1–8 are included according to chronological training/calibration roles; no GW9 result exists yet. Live forecasts and frozen shadow decisions are scored only after official finalization. There is no automatic online learning from unfinalized results, fabricated prospective success or guarantee of top-one-percent performance.

## Executed 0.5 trajectory experiment

The completed run contains **456 legal accounts, 4,632 candidate trajectories and 38 starting deadlines** (2025/26 GW2–35 and 2026/27 GW2–5). Every trajectory spans four evolving deadlines and is independently rescored with future chips included. The untouched later test contains 144 accounts. Tree decision gain is **+0.194 points**, four-deadline block 95% interval **[-1.431, 1.903]**. Neural gain is **-0.389**, interval **[-4.382, 3.576]**. Calibration selects the neural architecture at weight 0.5; its failed later test therefore leaves the reference policy active. We do not switch post hoc to the tree because its test mean happened to be higher.

Continuation RMSE improves **26.242 → 25.463**, with a paired error interval excluding zero. Its actual policy gain is **+0.868**, interval **[-0.583, 2.639]**; the policy gate fails and the learned terminal weights remain a challenger. These are completed negative/uncertain experiments, not unfinished integrations. More model complexity or accepting a failed gate would not establish a better system.
