# Multi-source boosted-tree forecasting

The primary system combines calibrated LightGBM participation and conditional-point heads, a separately accepted future-fixture expert, common outcome worlds and a legal multiweek HiGHS optimizer. A multi-source tree challenger, transfer critic, manager-choice models and physical football worlds have separate acceptance tests. Collecting data does not establish that adding it improves prediction.

## Inputs and timing

- Official FPL outcomes and compatible multi-season archives provide lagged statistics and completed labels. Stable player codes link seasons. Defensive contributions are separated for rule compatibility.
- Hash-pinned pre-deadline bootstrap snapshots provide status, news, set pieces, prices, ownership, transfers, starts and minutes. Both upstream capture and containing Git commit precede the deadline. Locally observed snapshots retain actual capture times.
- Twelve-match sequences and observation masks provide recent role. The immediate participation expert adds current-season role, recent starts, archived news and seven workload inputs; it does not extrapolate unreleased future manager teams.
- Understat shot/channel data, advanced FPL statistics and disciplinary observations enter shared enrichment transforms used by the tested multi-source and future trees. Missing values retain masks; provider xG definitions are not silently pooled.
- Prior-qualified manager picks are lagged. The current top-1,000 panel separately supplies behavior labels and conditional rival scenarios. Current rank is not a qualification label for historical training.
- FPL Core Insights supplies other-competition minutes. Zero minutes mean an observed nonappearance. Match/player identity, fixture dates, completion cutoffs and duplicate consistency are checked. Travel and unobserved competitions remain unknown. This is retrospective match enrichment, not proof of original publication time.
- Public Scout claims remain source-linked evidence and named sensitivities. Reddit provides research leads to originals. Neither opinion nor a manager purchase automatically adds expected points.

## Acceptance and serving

The role-aware classifier selects its mixture on 2024/25 GW1–19, reserves eight earlier deadlines for calibration in each expanding fold, accepts on GW20–38 and replicates across all 2025/26. Deadline-block intervals, regular starters and all four positions are reported in `role_minutes_validation.json`. Current-season outcomes do not select its configuration. The classifier modifies immediate participation only; conditional scoring and the accepted distant-fixture head retain their contracts. Explicit availability overrides and already-served suspensions use the baseline scenario path. Probability calibration is not applied twice.

The multi-source tree challenger has independent prediction, captain/policy and world-distribution gates. The legal-action tree critic learns residual four-week returns; acceptance uses purged later accounts. Rejected replacements stay comparisons. The new minutes forecast test does not retroactively certify older continuous-policy scores or guarantee better future transfers.

Newly built workload features count positive physical exposure. Old frozen tree models explicitly use `legacy-v1`; their inputs are not silently changed under fixed weights. New role models bind a validation ID, feature ID, context archive, workload transform, checksummed text weights and completed-label cutoff. Source transforms are shared by historical fitting and live serving.

```sh
uv run fpl sources-refresh
uv run fpl train-fusion
uv run fpl train-role-minutes
uv run fpl maintain-models
```

The launcher removes retired local tensor weights, refits companion trees against their supported feature contracts and synchronizes dependencies. Historical content-addressed experiment receipts remain immutable provenance; unsupported candidates cannot run or be selected.

Optional current chip use is compared with a full ordinary-transfer control that preserves later chips and the same hit budget. A precise simulated mean cannot settle an advantage smaller than unfinished additive search. Committed, explicitly fixed and expiring current entitlements retain their rules.

When a role-aware primary is active, challenger training reconstructs that composed reference in earlier-only folds, including current-season diagnostics. Its final fitted weights never score their own training weeks. References that already include availability are not downweighted a second time. Cached comparison folds bind the reference predictions as well as source inputs.
