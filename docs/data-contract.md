# Data and learning contract

The machine-readable acquisition contract is [source_registry.json](../configs/source_registry.json). Strict account, market, action and lineup schemas are in [domain.py](../src/fpl_intelligence/domain.py). Raw snapshots are retained before normalization and addressed by SHA-256. Model artifacts retain their feature specification, fitted parameters, calibration boundaries and checksums.

| Data | Meaning and units | Missingness and identity |
|---|---|---|
| Player | FPL ID within a season; official stable player code across compatible seasons | Ambiguous name/shot joins are excluded; historical club and position stay historical |
| Prices and bank | Integer tenths of £1m; 73 means £7.3m | Purchase price may be unknown; an exact selling quote does not reveal acquisition cost |
| Minutes and starts | Fixture minutes, observed starts, rolling exposure and start/60-minute probabilities | Absent labels remain missing; early archives with unusable start indicators do not create false nonstarts |
| Attack | FPL goals/assists, expected goals/assists/involvement, rolling rates and team/opponent strength | Source definitions remain distinct; generic expected assists are not assumed to be FPL assists |
| Shot/role enrichment | Available open-play, set-piece, penalty and shot-location/role observations | Unresolved identities and inaccessible updates are reported; verified cached observations retain their age |
| Defence | CBI, tackles, recoveries, action counts and position-specific threshold probabilities | Missing component histories use exposed priors and coverage indicators, never invented event counts |
| Goalkeeping and discipline | Saves, penalty events, BPS/bonus, cards, competition-specific bans and appeals | Discrete rewards/deductions follow the frozen rules; uncertainty about an appeal remains explicit |
| News and roles | Original text, official availability, set-piece order, quoted claims, confidence and expiry | Quotes must match the captured source; unsupported opinions cannot become numerical adjustments |
| Manager signals | Ownership, captaincy, chips, purchases/sales, cohort sample and effective scoring exposure | Current-rank analysis is separate from earlier-qualified expertise; future rank cannot define old features |
| Workload | Available non-PL matches/minutes, rest and schedule indicators | Coverage masks distinguish an unavailable schedule from a verified rest period |
| Published forecasts | Archived `ep_next`, form and points per game | Optional comparison inputs with masks; weekly EP is apportioned across announced fixtures, not copied through the season |
| Forecast outputs | Expected points, SD, quantiles, minute quantiles, P(0), P(start), P(60+), component and fixture coverage | Include cutoff, model/rules/input IDs, world count and seed; missing comparator coverage is not zero |
| Actions | Complete transfer bundle, chip, legal XI, ordered bench, captain/vice and exact ledger | Independent transition/scoring checks apply to every exported policy and contingent branch |

## Clocks and labels

`source_published_at` is the publisher's clock when available. `captured_at` records ingestion; `known_at` records account/evidence knowledge. An old publication timestamp cannot make a newly retrieved private claim prospectively available in the past. Imported corrected match archives are labelled historical reconstruction.

Historical production replay installs a checksummed observation view. Only earlier gameweeks with kickoff plus a conservative four-hour completion margin strictly before the decision cutoff are visible. Training and calibration groups precede the target group. Availability replacement gates are frozen before each evaluation period. Today's history snapshots cannot silently fill the historical view.

The real refresh workflow ingests newly finalized labels, reconciles the public deadline squad and refits/calibrates the accepted model. Recent completed matches also update player histories immediately. Calibration observations are deliberately reserved from the fitting slice; “uses completed data” does not mean the same row is used simultaneously as training and untouched validation.

## How decisions learn

The direct models learn actual fixture points and participation. Shared source tensors add dated news, ownership/expert signals, workload, advanced statistics and 12-match sequences with observation masks. Tree and temporal neural challengers use those tensors in training and serving. A collected field can inform a tested challenger without being forced into the primary policy after a failed replacement gate.

Random legal accounts evolve through observed prices and exact transfer/chip rules. Legal actions receive later realized returns; purged chronological splits train and assess action critics and state value. Hindsight best teams and transfers provide regret diagnostics. Winners from a test week cannot be fed back into that week's prediction and then presented as an out-of-sample result.

The production multiweek replay freezes all policy actions before actual scoring, executes only the first action, and carries the resulting account to the next deadline. It keeps the full chip window beyond a short scored interval. Prospective synchronized accounts separately freeze real future decisions, apply the prior standing hold if a capture is missed, and rescore official revisions without duplicating accumulated points.

Failed model gates retain the accepted model. Future observations can improve learning, but neither more data nor a larger network guarantees improved realized returns.
