# Current role model: GW1–9 and GW10 transfer comparison

Evaluated 6 November 2026. These are independent £100m squads each week, equivalent to a hypothetical Free Hit every week. Normal captaincy and exact autosubs apply; no Triple Captain or Bench Boost. The accepted role configuration is fixed, while conditional points and minutes are refitted before each deadline with eight earlier calibration deadlines. Final fitted current-season weights never score their own past training weeks.

| GW | Actual points | Captain |
|---|---:|---|
| 1 | 48 | Haaland |
| 2 | 91 | B.Fernandes |
| 3 | 54 | Haaland |
| 4 | 63 | B.Fernandes |
| 5 | 65 | Haaland |
| 6 | 44 | B.Fernandes |
| 7 | 61 | Haaland |
| 8 | 61 | Haaland |
| 9 | 82 | Haaland |
| **Total** | **569** | |

Each week uses 20,000 outcome worlds. All nine additive XI/captain squad solves report gap zero. This is a reconstructed historical experiment, with retrospectively imported completed-match facts; it does not establish consistent future returns or a continuous-account score. The [complete scorecard](../artifacts/evaluations/gw1-gw9-role-replay.json) includes every squad, lineup, frozen decision, fitting/calibration cutoff and model checksum.

## The proposed Gabriel/Tavernier hit

The comparison uses the same saved GW10–19 forecast, bank £1.6m and one FT, without changing the live account. Gabriel alone costs £0.7m more than available funds. Selling Cherki as well makes the double transfer affordable and leaves £0.5m.

| First move | GW10 gross | Hit | GW10 net | GW10–13 net with future ordinary transfers optimized |
|---|---:|---:|---:|---:|
| Roll | 52.01 | 0 | 52.01 | 226.18 |
| Gvardiol → Van Dijk | 53.91 | 0 | 53.91 | 228.08 |
| Cherki → Tavernier | 52.53 | 0 | 52.53 | 225.89 |
| Cherki → Tavernier + Gvardiol → Gabriel | 54.52 | 4 | 50.52 | 225.96 |

Four-week continuations allow future ordinary transfers and their actual FT/hit/bank transitions; chips are disabled equally to isolate the transfer choice. Conditional additive solver gaps are below 0.1%. These bounds do not prove global optimality of nonlinear autosub outcomes.

If both squads are held without later transfers through GW15, the Gabriel/Tavernier pair projects 324.58 points after the hit versus 324.30 for the Van Dijk route: only +0.28, far smaller than football/model uncertainty. The pair has a plausible longer-term case, rather than a clear modeled advantage.

## Where the numbers come from

Official FPL xG/xA, starts/minutes, points and defensive actions provide the underlying data. The immediate conditional-points head uses lagged/shrunk historical rates, team/opponent strength and form. The role head uses archived role, availability and other-competition workload. This remains a model estimate, not a fact about future performance.

| Player | Season xG | Season xA | Current-season minutes | Model GW10 minutes | Model GW10 points | Official published next-week estimate |
|---|---:|---:|---:|---:|---:|---:|
| Cherki | 1.76 | 1.81 | 472 | 41.0 | 2.89 | 2.8 |
| Tavernier | 3.71 | 1.36 | 750 | 75.5 | 4.07 | 6.0 |
| Gvardiol | 0.80 | 0.99 | 655 | 71.3 | 3.19 | 1.8 |
| Gabriel | 1.47 | 0.25 | 810 | 84.0 | 5.30 | 8.2 |
| Virgil | 0.36 | 0.22 | 810 | 88.6 | 5.21 | 3.5 |

Gabriel has substantially higher season xG than Van Dijk, and the official published estimate favors him much more strongly than our regression does. That disagreement lowers confidence in calling Van Dijk optimal. The four-point cost explains why the pair loses the immediate comparison; the six-week near-tie does not justify certainty either way. Published EP is an independent sensitivity, not ground truth or an automatic model override.

[Full transfer calculations, paired worlds, continuation diagnostics and component values](../artifacts/evaluations/gw10-user-options.json).

The dashboard and weekly API now select the scorecard for the evaluated participation configuration, preserve legacy benchmark results separately, and omit unmatched older controls. To reproduce: `uv run fpl replay-weekly --worlds 20000 --seconds 60 --end-event 9`; add `--no-current-model` for the earlier reference family.
