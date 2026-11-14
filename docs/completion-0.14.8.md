# FIE 0.14.8

Corrected the free-transfer selection guard: an unavailable solver bound no longer forces a roll. The extra saved transfer is already carried through the rolled continuation; paid hits retain their separate points-cost guard. Old recommendations are invalidated for recalculation.

Free Hit squads now receive an independent full-market bench-coverage search and exact shared-world rescoring. A candidate replaces the existing temporary squad only when its simulated expectation improves. Permanent account continuity is still checked independently. Unlimited Free Hit planning after prior transfers also handles the unlimited setting correctly.

Forecasting reuses each player's unchanged historical profile, preserving all 18,816 tested input rows exactly while reducing input preparation from 33.1 to 9.4 seconds on the same machine. The UI reports forecast stages, gameweek simulation progress and elapsed time. Worker startup timestamps no longer overwrite later progress timestamps. Normal terminal output omits repetitive HTTP polling messages.

The dashboard identifies provisional lineup baselines, shows conditional future actions, separates Free Hit starters from substitutes, and exposes the strongest tested free-transfer alternative. Source rate-limit cooldowns persist across refreshes; unavailable Reddit feeds are reported clearly and are not replaced with invented observations.

Accepted numerical weights remain unchanged. A controlled correction for established players with older bench histories produced tiny two-season improvements but failed its confidence gate, so it was not promoted. Player-specific exclusions and club bans were not added. Prediction disagreement remains distinct from software correctness.

Extract the update separately, close the existing application and command window, and run **Update existing.cmd**, selecting the existing folder. Account data, credentials and accepted weights are preserved. Release verification is recorded in `updates/v0.14.8-delivery.json`.
