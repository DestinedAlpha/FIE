# Model card · executed 29 October 2026

## Forecasts and promotion gates

`shared-components-v2` produces player/fixture components and player/gameweek distributions through GW38 in common match worlds. It models finite starting XIs, paired substitutions, correlated health and tempo, timed goals and clean sheets, saves, overdispersed defensive actions, discipline and rare missed/saved penalties. Exact deterministic FPL scoring and BPS tie awards follow simulation. BPS inputs remain a disclosed proxy, rather than an event-level reconstruction of the official BPS formula.

A LightGBM minutes suite uses chronological train/calibration/test partitions, isotonic calibration fitted before the test period, hash-verified model text and coherent minutes bands. Training/calibration/test counts are 20,714 / 7,362 / 6,901. Held-out start Brier improves 0.1382105 → 0.1081523; appearance Brier improves 0.1497423 → 0.1282986. This suite passed its gate and supplies simulation selection probabilities. Availability, selection and news provenance remain distinct.

The regularized recency-weighted Poisson team-strength challenger worsened held-out goal NLL (1.55145 → 1.58194), so the original strength prior remains active. An out-of-fold fitted ridge mixture also failed its improvement gate and remains inactive. Promotion never occurs merely because an advanced model exists.

The separate direct-points challenger uses 13 lag-only features and LightGBM. Deadline-grouped walk-forward evaluation contains 32,866 held-out rows: shrinkage RMSE 2.064208, direct challenger 2.011335; paired deadline-block bootstrap difference interval [-0.070503, -0.040107]. The predeclared regular-starter cohort improves 3.120784 → 3.093972 RMSE. News, current ownership, current prices and future match outcomes are excluded from historical feature folds.

## Reproduction and continuous policy evidence

Dastan numerical models are pinned to commit `758db9247e8b3a9a870c991064f74d8339d61cd0`. Forty downloaded JSON files are SHA-256 checked; the independent inference runtime honors each booster's best iteration. The released packaging holdout contains 6,582 rows and reproduces RMSE 1.794446. This verifies packaging/inference, not independent generalization: upstream calibration used the released holdout. No pickle or upstream executable module is loaded.

OpenFPL released historical vectors were joined with validated player identities and recomputed on 18,173 rows (RMSE 2.017455). This is not a fresh live Understat feature reproduction. Missing live feature feeds are never replaced with invented values. MIT notices and licenses accompany both sources. These different datasets cannot establish a direct ranking against our walk-forward scorecard.

An evolving-account replay from 2025/26 GW9–38 uses the same initial squad, strictly earlier training data, lag-only features, actual archived prices, actual points, bank, FT, hits, captaincy and autosubs. Net points: hold/shrinkage 1,529; hold/direct 1,523; transfer/shrinkage 1,506; transfer/direct 1,613. Transfer policies each made 31 transfers and paid 4 hit points. All transitions passed independent legality/affordability checks. Chips are excluded to isolate transfer/forecast effects. Archive prices and retrospective corrections make this statistical replay, not strict point-in-time news reconstruction.

## Context, opponents and adaptation

Both user-attested unbilled providers authenticated successfully. Six synthetic labelled transcripts include an injection attempt. Gemini 3.5 Flash Lite and Groq GPT OSS 120B both achieved quote/claim F1 1.0; source-status accuracy was 5/6 and 2/6 respectively. Gemini is selected, Groq provides eligible fallback. This small synthetic benchmark cannot prove real-world factual completeness or downstream points gains. Synthetic claims cannot enter production evidence. Official-source retrieval, freshness limits, server-owned provenance, atomic validation, quote matching, syndication clustering and explicit contradiction/supersession links apply before persistence.

Actual manager exposure uses 50 retrieved managers and entry-bootstrap intervals. This sample is current rank based, not a certified prior-elite cohort or overall population estimate. Mini-league outcomes compare explicit hold/greedy rival policies in identical worlds; unknown private future transfers are not claimed known. Joint strict-win probabilities and tie mass are reported separately.

Price, injury, minutes and fixture branches become observable only at their reveal event. Every preceding transfer, XI, captain and chip is identical. Conditional future purchase lots are labelled assumptions; confirmed current selling quotes remain authoritative. After observation a disclosed receding-horizon transfer policy adapts, with an identically adaptive unmodified control for paired effects. Scenario probabilities must be explicitly supplied for every branch and sum to one; otherwise there is no expected mixture. These branches do not replace the production forecast.

Whole-policy chip counterfactuals reoptimize without a chip, including preparation and cleanup, and rescore in identical worlds. Reported intervals reflect Monte Carlo noise; solver gaps and model uncertainty remain separate. The multiweek optimizer is exact for its additive formulation, then exact FPL autosubs/captain fallback are rescored. A legal incumbent and actual bounds/gaps are retained on timeout. Terminal value remains a disclosed 10% of the next four weeks' squad xPts. Future price sensitivity belongs to explicit scenarios rather than an invented price prediction.

## Prospective evaluation and intended use

Immutable forecasts and shadow decisions are frozen before deadlines. Finalized outcomes alone are scored, deduplicating decisions to the last eligible version before each event. A frozen decision does not submit an FPL account action. The current next deadline has not occurred; there is no prospective season-performance evidence yet.

Use this project for personal decision support with reconciled exact account inputs. The confirmed screenshot sale quotes and supplied bank/FT/chips are privately imported; unknown acquisition prices remain null. No top-1% rank, future return, perfect extraction or zero-error guarantee is claimed. Statistical archives, model assumptions and numerical solver tolerances are explicit in the aggregate artifacts.
