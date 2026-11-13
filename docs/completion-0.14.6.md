# FIE 0.14.6 — completed runtime and recommendation fixes

The v0.14.5 Windows check failed and its worker then hung during shutdown. This release replaces the blocking stdin watchdog with OS parent monitoring, collects atomic completed results independently of interpreter cleanup, and stops non-cooperative calculation processes. Cancellation remains available during native model/solver work. A single API owns each installation, queued follow-ups stop on cancellation, and the Windows launcher reuses the running installation.

## Planning and refresh

Normal planning shares one 30-second search budget across the full-market candidate and matched comparisons. Its default is the next five gameweeks; browsing a different gameweek no longer silently changes that horizon. Exact mode retains unlimited search. Auxiliary warm starts keep their small budgets in exact mode. The engine retains independently validated feasible incumbents when a budget expires, exposes real bounds and gaps, and makes a legal current lineup available during search.

The immediate transfer is now compared with rolling this deadline while retaining later transfer and chip choices. Keeping an unchanged squad throughout the entire horizon remains a separate comparator. The dashboard shows matched net gains, expected minutes, observed season xG/xA and defensive contributions. No named player is forced into or excluded from the numerical recommendation.

Live transfer counts, ownership and price-change forecasts can change between consecutive data refreshes. A labelled forecast snapshot remains usable for up to one hour when only these flow fields change. Actual prices, availability, news, fixtures, football statistics, scoring rules and model dependencies still invalidate it immediately. Historical replay always requires the exact captured snapshot. The UI distinguishes a frozen flow snapshot from forecasts that need updating.

Health polling no longer scans the entire database. Integrity checking runs in the supervised startup job, with its actual result and timestamp exposed separately; the Windows browser launcher uses a lightweight runtime probe. Latest-artifact queries use an index instead of repeatedly sorting large forecast bodies. HTTP status polling retains the job and cancellation controls across connection failures.

## Models and actual output

The accepted conditional-points, role-minutes and future-fixture model families are retained. Their ordinary production refit/calibration completed through finalized **2026/27 GW10**, using the compatible historical corpus and the accepted separate calibration windows. All original weight files and preceding state/model/forecast/plan objects remain preserved. A complete prepared download includes the updated weights.

The earlier pre-deadline GW10 forecast was checked against finalized observations. It projected Calafiori at 3.10 points and 59 minutes; he scored 6 in 68 minutes. Hill's 4.57-point projection compared with 4 actual points. This diagnostic does not train on the outcome and then present it as a pre-deadline prediction. [Frozen comparison](../artifacts/evaluations/frozen_gw10_0_14_6.json).

The latest checked GW11 recommendation remains **Calafiori → Hill, no hit, Haaland captain and Gabriel vice**. The model projects approximately 69 minutes for Calafiori and 88 for Hill, with observed defensive contributions of 42 and 115 respectively. Its immediate team gain is about 1.15 expected points; the matched five-week policy comparison is about 2.94 points. Those numbers describe the scored candidates from the normal search, whose remaining optimality gaps are displayed.

All **18,816 player-gameweek projections** passed finite/probability/minute bounds, full-market coverage and exact reconciliation with the optimizer's worlds. Zero-minute scenarios now also have zero pre-sampling expectation and zero scoring components. Historical validation identifies the retained accepted family separately from its updated data snapshot.

The [FiveStat methodology](https://www.fivestat.co.uk/methodology) and [Core Insights dataset](https://github.com/olbauday/FPL-Core-Insights) remain research/source references. Existing controlled component and temporal studies are preserved; no rejected architecture was promoted merely to add complexity. Official advanced statistics and the existing validated source consumers remain connected.

## Executed release evidence

The real production workflow completed synchronization, the accepted-model refit and calibration, a full forecast, legal planning, actual process cancellation and a subsequent refresh. The refit plus forecast took 643.7 seconds on this machine; a subsequent forecast took 143.4 seconds, normal planning 34.4 seconds, refresh 3.8 seconds and cancellation 0.81 seconds. Refresh/Load team does not start model fitting or optimization.

[Production execution and exact model IDs](../artifacts/evaluations/production_workflow_0_14_6.json). Final regression, Linux/Windows launcher/updater, actual installed upgrade/backup/recovery, prepared payload and downloaded-byte receipts are recorded in `updates/v0.14.6-delivery.json`. The original audit and research evidence remains in the preceding release ledgers.

## Install

Close the old FIE app and command window. [Download the update](https://github.com/DestinedAlpha/FIE/releases/download/v0.14.6/FPL-Intelligence-0.14.6-Update.zip), extract it separately and run **Update existing.cmd**, selecting your existing folder. Your account, keys and accepted models are preserved. The complete prepared private download also includes current numerical data and the GW10 weights. No FPL transfers are submitted by installation or analysis.
