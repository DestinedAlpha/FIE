# FIE 0.14.4 — temporal learning and verified automatic updates

Baseline: public v0.14.3, commit `605b9a8`. The accepted conditional-points, role-v3 and 50% future-expert composition is preserved. Seven numerical weight files, every calibrator, residual pool, dependence specification and minutes quantile match the independently executed post-GW9 refit exactly. Account, market, forecast and selected-plan pointers were preserved.

## Training-policy decision

The immediate experiment compares the incumbent against shorter independent calibration, 26/52/104-deadline half-lives, rolling 52/78-deadline windows, current-season upweighting, chronological cross-fitting, adaptive disagreement weighting, and fixed long/recent ensembles. Recent player/team inputs stay fixed, so changes in observation age are not confused with changes in form or availability. All same-deadline fixtures stay together. Current 2026/27 outcomes never select a configuration.

Selection uses 2024/25 GW1–19. The selected 52-deadline decay candidate then receives a separate confirmation at the actual weekly refit cadence: every deadline in 2024/25 GW20–38 and all 38 deadlines of 2025/26. Its small RMSE gains have weighted-MSE confidence intervals crossing zero in both independent periods. It does not earn promotion. Captaincy is assessed with deadline-block confidence intervals; an uncertain one-point aggregate difference is not an automatic rejection criterion.

| Weekly-refit period | Incumbent RMSE | Decay RMSE | Weighted-MSE difference 95% interval |
|---|---:|---:|---:|
| 2024/25 GW20–38 | 1.819847 | 1.819771 | −0.044259 to +0.021392 |
| Full 2025/26 | 1.881515 | 1.880317 | −0.034462 to +0.018397 |

The separate future-expert experiment compares eight and four purged calibration deadlines with chronological cross-fitting that lets the final weights use all eligible earlier labels. It evaluates the accepted 50% production mixture, independently refitting at the scheduled source deadlines, while excluding long-horizon labels unavailable at each teacher's information cutoff. The earlier-period selection objective chooses the eight-deadline incumbent. Neither a later holdout winner nor current-season hindsight replaces that selector.

The resulting operational policy retains expanding historical fits with eight separate calibration deadlines. All 2025/26 labels contribute to fitting. For the present GW1–9 corpus, core and role weights fit through 2026/27 GW1 and calibrate on GW2–9; future fitting targets likewise end at GW1 and calibration targets at GW9. Recent features still change with current information. Calibration is learning; it is not an assertion that every calibration observation also fitted the underlying tree weights.

Executed results and frozen protocols: [screening](../artifacts/evaluations/training_policy_0_14_4.json), [weekly confirmation](../artifacts/evaluations/training_policy_weekly_0_14_4.json), [future calibration](../artifacts/evaluations/future_temporal_0_14_4.json). Reproduce with `uv run fpl validate-training-policy`, `uv run fpl validate-training-policy --weekly`, and `uv run fpl validate-future-calibration --source-path <the sealed study data.parquet>`.

## Engineering corrections

- Refresh eligibility fingerprints finalized observations and consumed dependencies, rather than changing future-fixture payloads. Unchanged data reuses the accepted composition. Ordinary updates refit the frozen accepted family instead of repeating historical model selection.
- Core, role, future and training-state publication share one transaction with cancellation, source-stability and concurrent-publication guards. A required companion with older label coverage rejects the whole update. Failures preserve the preceding composition and expose a warning.
- The real integration test reproduced a minutes-head bug: a feature cache could include more labels than its retained weights had consumed. Matching those cached matrices alone could incorrectly reuse the old head. Reuse now also requires actual completed-label coverage; the UI applies the same distinction when reporting provenance.
- Full player-history ingestion validates response schemas and completeness. A partial finalized batch stops numerical publication and is retried. Current-season archive rows cannot bypass the official `data_checked` guard. Cancellation also stops queued source requests.
- Decision features now distinguish their pre-Parquet semantic record identity from their typed-file checksum. Actual typed historical research no longer rejects valid data by comparing those different hashes. Legacy records rebuild from verified facts; new cached bytes are checked explicitly.
- The official 2026/27 points-lock change is implemented in historical label availability: whole-gameweek finalization at 09:00 Europe/London the following day, including daylight saving. Live ingestion continues to use the authoritative official finalized flag. No retained GW1–GW10 historical input changes under this correction. Pre-2025 scoring excludes DEFCON and retains the earlier six-point goalkeeper-goal reward.
- Model & data now visibly exposes each bound model ID, fitted/calibrated gameweeks, row counts and failed-update status. `/api/models/training` and `fpl training-status` expose the same provenance. Metadata from newer feature-cache rows cannot overstate an older head's training coverage.
- Isolated learning/verification stores write their core, minutes, workload and feature reports locally, preserving production research evidence.

The 2026/27 BPS source review confirms reduced CBI weighting, removal of the being-tackled penalty, and revised goalkeeper save/penalty-save BPS. The primary model learns observed points/bonus labels; it does not apply obsolete deterministic BPS coefficients. Proprietary big-chance/save-location/tackled events are not invented to create retrospective bonus awards. Earlier DEFCON targets remain explicitly unavailable, rather than observed zeros. Full component and population diagnostics from v0.14.3 remain attached to their executed protocols.

## Real execution and release verification

The actual official refresh retrieved all 671 requested player histories with GW9 finalized and GW10 still ineligible as a training label. Starting from retained pre-GW9 weights in an isolated installation, the corrected normal workflow refitted and calibrated all three components, published them together, generated 18,788 player-gameweek forecasts and reconciled their 4,000-world samples exactly. It produced a nine-deadline legal plan, independently replayed through bank, free-transfer, transfer-position, hit, chip and squad constraints. A subsequent unchanged-data update produced zero additional fits. Source, weights and calibration parity also validate the retained production composition without replacing it.

The verification database was sealed into checksummed parts and independently restored. Its real source captures, numerical artifacts, worlds and account continuity remain in the prepared private data payload. [Execution receipt](../artifacts/evaluations/learning_workflow_0_14_4.json); [finalization comparison](../artifacts/evaluations/finalization_cutoffs_0_14_4.json).

The final regression, actual v0.14.3-code upgrade, installed-product API/export/model checks, fresh data recovery, database integrity, Windows/Linux CI and downloaded release checks are identified in the versioned release ledger and delivery receipt. Upgrade verification loads the published v0.14.3 application with the real database before installing this release, rather than using a version-marker-only approximation. Live UI/CLI optimization remains unlimited and cancellable; the historical verification deliberately records its finite search budget and does not claim a proven unlimited multiweek optimum.

All 216 inherited audit dispositions retain their earlier evidence and receive current source/test traceability. New confirmed defects and completed experiments have explicit dispositions in the [release ledger](../artifacts/evaluations/release_0_14_4.json). The settled FiveStat/Core Insights research remains preserved; it was not rerun merely to add complexity. Understat's 47,053 validated observations and exclusion of 1,194 ambiguous identities remain intact, without retrying restricted live access.

No forecast improvement is claimed for rejected alternatives. The improvement delivered here is a verified, consistent automatic learning process, accurate training provenance and safer data/publication/recovery behavior. Unavailable external observations and uncertain future results are distinct from actionable engineering work.
