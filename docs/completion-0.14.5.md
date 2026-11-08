# FIE 0.14.5 — responsive refresh and process cancellation

The reported hang was real: v0.14.4 could start an unlimited multiweek optimization from Refresh data / Load team. A thread cancellation flag could not reliably interrupt native solver/model code or a blocked source request. Requesting cancellation did not prove the calculation had stopped.

## Delivered behavior

- **Refresh data / Load team** only retrieves official bootstrap, fixtures and revealed account information. It does not train, forecast, request all player histories, fetch auxiliary sources or optimize transfers. Current snapshots retain their market identity when their contents are unchanged. A legitimately changed market/account marks old recommendations stale.
- **Update forecasts** separately ingests required finalized observations, updates the accepted numerical models when needed and regenerates forecasts. It does not start transfer/chip searches, consensus comparisons or shadow planning. The complete CLI workflow remains available.
- **Optimize plan** remains an explicitly requested unlimited search with zero target MIP gap. It can take substantial time; the UI reports its actual phase and available bounds. No forced optimization timeout was reintroduced.
- Each calculation runs in a separate supervised process. Cancel first requests cooperative exit, then terminates an unresponsive process tree after a short grace period. The job reaches `cancelled` only after the worker exits. Queued follow-ups are cancelled too. A manual cancellation also prevents the automatic refresh scheduler from restarting work for 15 minutes.
- Original validated request bodies and omitted-field semantics survive process dispatch, including automatic chip selection. Worker startup does not repeat application recovery or start another scheduler. The worker uses the installed application's source; parent disappearance closes its watchdog pipe.
- Cross-process account locks protect synchronization/publication against concurrent edits. Normalized Parquet materializations are replaced atomically, preventing interrupted writes or hardlinked verification copies from altering an earlier file. Accepted numerical weights, credentials and account records are preserved.

## Executed evidence

The actual default process-isolated API retrieved live official data and the configured revealed account in **4.33 seconds**. Cancelling a real forecasting worker took **0.80 seconds**, and another actual refresh completed in **4.75 seconds** afterward. Accepted model, forecast and plan pointers survived cancellation; all original primary-installation pointers remained unchanged. These are measured local timings, not a network or hardware guarantee. [Execution receipt](../artifacts/evaluations/responsive_jobs_0_14_5.json).

Regression coverage also runs a real process that ignores every cancellation hook, verifies its termination, cancellation of queued work, preservation of primary artifacts and successful dispatch of the next request. Desktop and mobile journeys verify that controls stay disabled during stopping and become usable after terminal cancellation. Identical-data identity and atomic materialization tests cover interrupted refresh safety. Final regression, Linux/Windows CI, actual installed-code upgrade, backup/recovery and downloaded-package receipts are recorded in the release delivery record.

No numerical model or research acceptance was changed to address this runtime defect. The v0.14.4 research and audit evidence remain preserved.

## Install

Close the old FIE application **and its command window**. [Download the v0.14.5 update](https://github.com/DestinedAlpha/FIE/releases/download/v0.14.5/FPL-Intelligence-0.14.5-Update.zip), extract it into a separate folder and run **Update existing.cmd**. Select your current FIE folder and restart. Refreshing the browser alone cannot replace the old running backend. The updater preserves local data, keys and accepted models and makes a recovery backup.
