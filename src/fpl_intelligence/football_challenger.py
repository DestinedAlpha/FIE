"""FiveStat-inspired, deadline-safe scoreline/player-share research challenger.

Independent implementation: parameters are candidates, not borrowed evidence.
This module never changes production weights or the selected account strategy.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import poisson

from .availability import moment
from .forecast_population import point_metrics
from .storage import atomic_json, digest, now

ALPHAS = (0.0, 0.15, 0.30, 0.50)
COVARIANCES = (0.0, 0.05, 0.15)
RECENT_SHARES = (0.0, 0.30, 0.50)
BLENDS = (0.10, 0.25, 0.50, 1.0)


def score_matrix(home, away, covariance=0.0, maximum=24):
    """Shared additive Poisson; preserve specified marginal means exactly.

    Tail loss is checked before normalization, never silently discarded.
    """
    if (
        not np.isfinite([home, away, covariance]).all()
        or min(home, away) < 0
        or not 0 <= covariance <= min(home, away)
    ):
        raise ValueError("Invalid bivariate Poisson rates")
    grid = np.arange(maximum + 1)
    h, a, c = (
        poisson.pmf(grid, home - covariance),
        poisson.pmf(grid, away - covariance),
        poisson.pmf(grid, covariance),
    )
    matrix = np.zeros((maximum + 1, maximum + 1))
    for k in grid:
        matrix[k:, k:] += c[k] * np.outer(h[: maximum + 1 - k], a[: maximum + 1 - k])
    if 1 - matrix.sum() > 1e-8:
        raise ValueError("Score grid truncates material probability")
    return matrix / matrix.sum()


def match_metrics(matrix, home_goals, away_goals):
    n = len(matrix)
    h, a = np.indices(matrix.shape)
    probabilities = np.array([matrix[h < a].sum(), np.trace(matrix), matrix[h > a].sum()])
    actual = int(home_goals > away_goals) * 2 if home_goals != away_goals else 1
    truth = np.eye(3)[actual]
    return {
        "nll": float(-np.log(max(matrix[int(home_goals), int(away_goals)], 1e-15))),
        "rps": float(np.mean((np.cumsum(probabilities)[:2] - np.cumsum(truth)[:2]) ** 2)),
        "brier": float(np.sum((probabilities - truth) ** 2)),
        "home_cs": float(matrix[:, 0].sum()),
        "away_cs": float(matrix[0, :].sum()),
        "over25": float(matrix[h + a >= 3].sum()),
        "btts": float(matrix[(h > 0) & (a > 0)].sum()),
        "home_mean": float(matrix.sum(1) @ np.arange(n)),
        "away_mean": float(matrix.sum(0) @ np.arange(n)),
    }


def allocation(rates, expected_minutes, team_goals):
    weights = np.maximum(0, np.asarray(rates)) * np.asarray(expected_minutes) / 90
    total = weights.sum()
    return weights * team_goals / total if total > 0 else np.zeros_like(weights)


def _past_matches(facts, group, cutoff):
    from datetime import timedelta

    return [
        r for r in facts if r["group"] < group and moment(r["kickoff"]) + timedelta(hours=4) < moment(cutoff)
    ]


def team_predictions(store, rows, directory):
    import hashlib
    import json

    from .score_process import design, fit_strength, fixture_facts
    from .shot_data import club_key

    facts = fixture_facts(store)
    facts_by = {(r["season"], r["fixture_id"]): r for r in facts}
    clubs = sorted({r[k] for r in facts for k in ("home", "away")})
    groups = defaultdict(list)
    for r in rows:
        groups[r["deadline_group"]].append(r)
    contract = {
        "facts": digest(facts),
        "deadlines": digest(sorted({(r["deadline_group"], r["deadline"]) for r in rows})),
        "alphas": ALPHAS,
        "covariances": COVARIANCES,
        "version": 1,
    }
    path = directory / "team-predictions.parquet"
    receipt = directory / "team-predictions.json"
    if path.exists() and receipt.exists():
        old = json.loads(receipt.read_text())
        if (
            old["contract"] == json.loads(__import__("json").dumps(contract))
            and old["hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            return pl.read_parquet(path).to_dicts(), facts_by
    output = []
    for group, part in sorted(groups.items()):
        cutoff = min(r["deadline"] for r in part)
        past = _past_matches(facts, group, cutoff)
        keys = sorted({(r["season"], r["fixture_id"]) for r in part})
        test = [facts_by[k] for k in keys]
        beta, _ = fit_strength(past, clubs, cutoff, 0.98)
        base = np.exp(design(test, clubs) @ beta)
        # Opponent-adjusted recent residuals, shrunk by five prior matches.
        recent = defaultdict(list)
        history = np.exp(design(past, clubs) @ beta)
        for f, expect in zip(past, history):
            for home in (True, False):
                club = f["home"] if home else f["away"]
                own = float(f["home_goals"] if home else f["away_goals"])
                conceded = float(f["away_goals"] if home else f["home_goals"])
                recent[club].append(
                    (own, conceded, float(expect[0 if home else 1]), float(expect[1 if home else 0]))
                )
        factors = {}
        for club in clubs:
            tail = recent[club][-5:]
            factors[club] = [
                (sum(r[k] for r in tail) + 5) / (sum(r[k + 2] for r in tail) + 5) for k in (0, 1)
            ]
        for f, baseline in zip(test, base):
            for alpha in ALPHAS:
                h, a = club_key(f["home"]), club_key(f["away"])
                rates = np.clip(
                    baseline
                    * np.array(
                        [
                            (1 - alpha + alpha * factors[h][0]) * (1 - alpha + alpha * factors[a][1]),
                            (1 - alpha + alpha * factors[a][0]) * (1 - alpha + alpha * factors[h][1]),
                        ]
                    ),
                    0.1,
                    5.0,
                )
                for cov in COVARIANCES:
                    matrix = score_matrix(float(rates[0]), float(rates[1]), min(cov, float(min(rates))))
                    output.append(
                        {
                            "season": f["season"],
                            "event": f["event"],
                            "fixture_id": f["fixture_id"],
                            "group": group,
                            "deadline": cutoff,
                            "past_max": max(r["group"] for r in past),
                            "home": h,
                            "away": a,
                            "alpha": alpha,
                            "covariance": cov,
                            "actual_home": f["home_goals"],
                            "actual_away": f["away_goals"],
                            **match_metrics(matrix, f["home_goals"], f["away_goals"]),
                        }
                    )
        print("Scoreline challenger", group, len(past), flush=True)
    pl.DataFrame(output).write_parquet(path)
    atomic_json(receipt, {"contract": contract, "hash": hashlib.sha256(path.read_bytes()).hexdigest()})
    return output, facts_by


STATS = (
    "expected_goals",
    "expected_assists",
    "bonus",
    "yellow_cards",
    "red_cards",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "own_goals",
)


def histories(originals):
    from .decision import player_key
    from .enrichment import epoch

    lookup = defaultdict(list)
    for r in originals:
        if r.get("kickoff_time"):
            lookup[player_key(r)].append(
                (epoch(r["kickoff_time"]) + 4 * 3600, f"{r['season']}-GW{int(r['round']):02d}", r)
            )
    for v in lookup.values():
        v.sort(key=lambda x: (x[0], x[1]))
    return lookup, {k: [v[0] for v in part] for k, part in lookup.items()}


def rate_vector(history, season, prior, recent_weight):
    """Per90 rates; small samples shrink toward position priors from past only."""
    appeared = [r for r in history if float(r.get("minutes") or 0) > 0]
    seasonal = [r for r in appeared if r["season"] == season]
    if not seasonal:
        seasonal = appeared[-24:]
    recent = appeared[-5:]

    def rates(part):
        result = []
        for j, key in enumerate(STATS):
            known = [r for r in part if r.get(key) not in (None, "")]
            exposure = sum(float(r["minutes"]) / 90 for r in known)
            result.append((sum(float(r[key]) for r in known) + 5 * prior[j]) / (exposure + 5))
        return np.array(result)

    return (1 - recent_weight) * rates(seasonal) + recent_weight * rates(recent)


def component_predictions(store, rows, team_rows, facts_by, alpha, share_weight):
    from .decision import canonical_rows, player_key
    from .enrichment import epoch
    from .shot_data import club_key

    originals, _, _ = canonical_rows(store)
    lookup, dates = histories(originals)
    original_map = {(r["season"], int(r["fixture"]), int(r["element"])): r for r in originals}
    # Position priors are frozen before the first evaluation deadline.
    earliest = min(r["deadline"] for r in rows)
    past = [
        r
        for r in originals
        if r.get("kickoff_time") and epoch(r["kickoff_time"]) + 4 * 3600 < epoch(earliest)
    ]
    priors: dict[int, list[float]] = {}
    for pos in range(1, 5):
        priors[pos] = []
        for key in STATS:
            part = [
                r
                for r in past
                if int(r["position"]) == pos
                and r.get(key) not in (None, "")
                and float(r.get("minutes") or 0) > 0
            ]
            priors[pos].append(
                sum(float(r[key]) for r in part) / max(1, sum(float(r["minutes"]) / 90 for r in part))
            )
    team_map = {
        (r["season"], r["fixture_id"]): r for r in team_rows if r["alpha"] == alpha and r["covariance"] == 0
    }
    groups = defaultdict(list)
    for r in rows:
        groups[r["season"], r["fixture_id"], club_key(r["team"])].append(r)
    output = []
    attack_prior = sum(float(r.get("assists") or 0) for r in past) / max(
        1, sum(float(r.get("goals_scored") or 0) for r in past)
    )
    paired = [
        r
        for r in past
        if r.get("expected_assists") not in (None, "") and r.get("expected_goals") not in (None, "")
    ]
    opta_ratio = sum(float(r["expected_assists"]) for r in paired) / max(
        1, sum(float(r["expected_goals"]) for r in paired)
    )
    for (season, fixture, club), part in groups.items():
        match = team_map[season, fixture]
        home = club == match["home"]
        gf = match["home_mean"] if home else match["away_mean"]
        ga = match["away_mean"] if home else match["home_mean"]
        rate_rows = []
        for r in part:
            key = player_key(original_map[season, fixture, r["player_id"]])
            cutoff = epoch(r["deadline"])
            stop = bisect_left(dates.get(key, []), cutoff)
            past_player = [p[2] for p in lookup.get(key, [])[:stop] if p[1] < r["deadline_group"]]
            rate_rows.append(rate_vector(past_player, season, priors[int(r["position"])], share_weight))
        rates = np.asarray(rate_rows)
        minutes = np.array([r["expected_minutes"] for r in part])
        gx = allocation(rates[:, 0], minutes, gf)
        # FPL assist conversion estimated only on the earlier positional corpus.
        ax = allocation(rates[:, 1], minutes, gf * min(1, attack_prior))
        opta_xa = allocation(rates[:, 1], minutes, gf * min(1, opta_ratio))
        for j, r in enumerate(part):
            mean = 0.0
            pos = int(r["position"])
            components: dict[str, float] = defaultdict(float)
            for band in range(4):
                if band == 0:
                    continue
                band_minutes = r[f"m{band}"]
                prob = r[f"p{band}"]
                exposure = band_minutes / 90
                goal = gx[j] * band_minutes / max(minutes[j], 1e-9)
                assist = ax[j] * band_minutes / max(minutes[j], 1e-9)
                conceded = ga * exposure
                values = {
                    "appearance": 1 if band == 1 else 2,
                    "goals": (10 if pos == 1 else 6 if pos == 2 else 5 if pos == 3 else 4) * goal,
                    "assists": 3 * assist,
                    "clean_sheet": (4 if pos <= 2 else 1 if pos == 3 else 0) * np.exp(-conceded)
                    if band >= 2
                    else 0.0,
                    "goals_conceded": -(conceded / 2 - (1 - np.exp(-2 * conceded)) / 4) if pos <= 2 else 0.0,
                    "bonus": rates[j, 2] * exposure,
                    "cards": -rates[j, 3] * exposure - 3 * rates[j, 4] * exposure,
                    "saves": float(poisson.sf(np.arange(2, 45, 3), rates[j, 5] * exposure).sum())
                    if pos == 1
                    else 0.0,
                    "penalties": (5 * rates[j, 6] - 2 * rates[j, 7]) * exposure,
                    "own_goals": -2 * rates[j, 8] * exposure,
                    "defcon": 2 * r[f"dc{band}"],
                }
                for key, value in values.items():
                    components[key] += prob * value
                mean += prob * sum(values.values())
            actual = original_map[season, fixture, r["player_id"]]
            output.append(
                {
                    **r,
                    "component": float(mean),
                    "xg": float(gx[j]),
                    "xa": float(opta_xa[j]),
                    "expected_fpl_assists": float(ax[j]),
                    "team_xg": gf,
                    "opponent_xg": ga,
                    "cs": float(np.exp(-ga)),
                    "actual_goals": float(actual.get("goals_scored") or 0),
                    "actual_assists": float(actual.get("assists") or 0),
                    "actual_xg": None
                    if actual.get("expected_goals") in (None, "")
                    else float(actual["expected_goals"]),
                    "actual_xa": None
                    if actual.get("expected_assists") in (None, "")
                    else float(actual["expected_assists"]),
                    **{"component_" + k: float(v) for k, v in components.items()},
                }
            )
    return output


def run(store):
    from threadpoolctl import threadpool_limits

    from .forecast_population import predictions
    from .immediate_points import interval, metrics

    with threadpool_limits(limits=1):
        rows, diagnostics = predictions(store)
        directory = store.artifact_path(diagnostics["directory"] + "/protocol.json").parent
        protocol = {
            "version": "football-components-v2",
            "reference": "accepted role-v3 plus conditional-points; future-only expert unchanged",
            "alphas": ALPHAS,
            "covariances": COVARIANCES,
            "recent_shares": RECENT_SHARES,
            "point_blends": BLENDS,
            "specialists": ["all", "defenders", "goalkeepers", "midfielders", "forwards"],
            "selection": "2024/25 GW1-19 only",
            "acceptance": "2024/25 GW20-38 and separate full 2025/26 replication",
            "current": "2026/27 GW1-9 diagnostic only; never selects",
            "gate": "Both periods: upper deadline-block point-MSE delta interval below zero, no regular/premium/position RMSE or captain diagnostic deterioration. Passing forecasts additionally require distribution and legal-account policy gates before promotion.",
            "team_selection": "Earlier selection scoreline NLL; report RPS, 1X2 Brier, clean-sheet Brier, O/U and BTTS",
            "limitation": "Previously studied historical archives, not blinded prospective evidence. No external projection captured today is backdated.",
        }
        atomic_json(directory / "football-protocol.json", protocol)
        team_rows, facts = team_predictions(store, rows, directory)
        selection = [r for r in team_rows if r["season"] == "2024/25" and r["event"] <= 19]
        alpha, cov = min(
            ((a, c) for a in ALPHAS for c in COVARIANCES),
            key=lambda pair: np.mean(
                [r["nll"] for r in selection if r["alpha"] == pair[0] and r["covariance"] == pair[1]]
            ),
        )
        variants = {}
        for w in RECENT_SHARES:
            variants[w] = cached_components(store, rows, team_rows, facts, alpha, w, directory)
            print("Player component challenger", w, len(variants[w]), flush=True)
        candidates: list[dict[str, Any]] = []
        for share, part in variants.items():
            for specialist, pos in (
                ("all", None),
                ("defenders", 2),
                ("goalkeepers", 1),
                ("midfielders", 3),
                ("forwards", 4),
            ):
                for blend in BLENDS:
                    subset = [r for r in part if r["season"] == "2024/25" and r["event"] <= 19]
                    loss = np.mean(
                        [
                            (
                                r["reference"]
                                + blend
                                * (r["component"] - r["reference"])
                                * (pos is None or int(r["position"]) == pos)
                                - r["actual"]
                            )
                            ** 2
                            for r in subset
                        ]
                    )
                    candidates.append(
                        {
                            "share_recent": share,
                            "specialist": specialist,
                            "position": pos,
                            "blend": blend,
                            "selection_mse": float(loss),
                        }
                    )
        selected = min(candidates, key=lambda r: r["selection_mse"])
        output = variants[selected["share_recent"]]
        for r in output:
            r["candidate"] = r["reference"] + selected["blend"] * (r["component"] - r["reference"]) * (
                selected["position"] is None or int(r["position"]) == selected["position"]
            )
            r["delta"] = (r["candidate"] - r["actual"]) ** 2 - (r["reference"] - r["actual"]) ** 2
        periods = {}
        for name, part in (
            ("acceptance_2024", [r for r in output if r["season"] == "2024/25" and r["event"] > 19]),
            ("replication_2025", [r for r in output if r["season"] == "2025/26"]),
            ("current_diagnostic", [r for r in output if r["season"] == "2026/27"]),
        ):
            if not part:
                continue
            ref, can = metrics(part, "reference"), metrics(part, "candidate")
            positions = {
                str(p): {
                    k: point_metrics([r for r in part if int(r["position"]) == p], k)
                    for k in ("reference", "candidate", "component")
                }
                for p in (1, 2, 3, 4)
            }
            ci = interval(part, "delta")
            periods[name] = {
                "rows": len(part),
                "reference": ref,
                "candidate": can,
                "component": metrics(part, "component"),
                "positions": positions,
                "delta_mse_95ci": ci,
                "passed": bool(
                    ci[1] < 0
                    and can["regular_rmse"] <= ref["regular_rmse"]
                    and (can["premium_rmse"] or 0) <= (ref["premium_rmse"] or 0)
                    and can["captain_points"] >= ref["captain_points"]
                    and all(v["candidate"]["rmse"] <= v["reference"]["rmse"] for v in positions.values())
                ),
            }
        team_metrics = {}
        for season in ("2024/25", "2025/26", "2026/27"):
            part = [
                r
                for r in team_rows
                if r["season"] == season
                and r["alpha"] == alpha
                and r["covariance"] == cov
                and (season != "2024/25" or r["event"] > 19)
            ]
            if not part:
                continue
            team_metrics[season] = {
                "fixtures": len(part),
                **{k: float(np.mean([r[k] for r in part])) for k in ("nll", "rps", "brier")},
                "goal_mae": float(
                    np.mean(
                        [
                            abs(r[side + "_mean"] - r["actual_" + side])
                            for r in part
                            for side in ("home", "away")
                        ]
                    )
                ),
                "cs_brier": float(
                    np.mean(
                        [
                            (
                                r[side + "_cs"]
                                - int(r["actual_" + ("away" if side == "home" else "home")] == 0)
                            )
                            ** 2
                            for r in part
                            for side in ("home", "away")
                        ]
                    )
                ),
                "over25_brier": float(
                    np.mean(
                        [(r["over25"] - int(r["actual_home"] + r["actual_away"] >= 3)) ** 2 for r in part]
                    )
                ),
                "btts_brier": float(
                    np.mean(
                        [(r["btts"] - int(r["actual_home"] > 0 and r["actual_away"] > 0)) ** 2 for r in part]
                    )
                ),
            }
        path, checksum = seal_predictions(directory, "football-player", output)

        passed = all(periods[p]["passed"] for p in ("acceptance_2024", "replication_2025"))
        result = {
            "at": now(),
            "protocol": protocol,
            "population_diagnostics_id": diagnostics["id"],
            "selected": selected,
            "selected_alpha": alpha,
            "selected_covariance": cov,
            "selection_candidates": candidates,
            "periods": periods,
            "team_metrics": team_metrics,
            "component_diagnostics": component_diagnostics(store, output),
            "forecast_gate_passed": passed,
            "promoted": False,
            "decision": "Requires distribution and policy gates"
            if passed
            else "Rejected: predeclared forecast/captain/subgroup gates failed; accepted production composition retained",
            "prediction_path": str(path),
            "prediction_hash": checksum,
        }
        result["id"] = store.put("football_challenger", result)
        atomic_json(Path("artifacts/evaluations/football_challenger_0_14_3.json"), result)
        return result


def resolve_specialists(store):
    """Resolve every predeclared positional family, with simultaneous intervals.

    The global blend failed its captain gate. Family configurations still use
    only the original selection period; no holdout metric tunes a weight.
    """
    import hashlib

    from threadpoolctl import threadpool_limits

    from .immediate_points import metrics

    base = store.get("football_challenger")
    diagnostics = store.get("population_diagnostics", base["population_diagnostics_id"])
    directory = store.artifact_path(diagnostics["directory"] + "/protocol.json").parent
    protocol = {
        "at": now(),
        "parent": base["id"],
        "families": base["protocol"]["specialists"],
        "selection": "Original 2024/25 GW1-19 loss selects exactly one configuration per already declared family",
        "interval": "Deadline-block bootstrap, 5000 repeats; two-sided 99% intervals per family (Bonferroni 5), both untouched acceptance periods must pass",
        "gate": base["protocol"]["gate"],
        "reason": "Global point improvement failed captain gate; test the predeclared specialists rather than selecting a holdout-fitted replacement",
        "promotion": False,
    }
    atomic_json(directory / "specialist-protocol.json", protocol)
    rows = []
    for fold in diagnostics["folds"]:
        path = directory / (fold["fold"]["test"][0].replace("/", "-") + ".parquet")
        if hashlib.sha256(path.read_bytes()).hexdigest() != fold["prediction_hash"]:
            raise ValueError("Population checkpoint integrity failed")
        restored = pl.read_parquet(path).to_dicts()
        for row in restored:
            row.update({f"m{k}": float(fold["band_minutes"][k]) for k in range(4)})
        rows.extend(restored)
    with threadpool_limits(limits=1):
        teams, facts = team_predictions(store, rows, directory)
        winners = [
            min(
                (v for v in base["selection_candidates"] if v["specialist"] == name),
                key=lambda r: r["selection_mse"],
            )
            for name in protocol["families"]
        ]
        families = {}
        variants = {}
        paths = {}
        for weight in sorted({r["share_recent"] for r in winners}):
            variants[weight] = cached_components(
                store, rows, teams, facts, base["selected_alpha"], weight, directory
            )
            path, checksum = seal_predictions(directory, f"football-share-{weight:g}", variants[weight])
            paths[str(weight)] = {"path": str(path), "hash": checksum}
            print("Specialist shared components", weight, flush=True)
        for winner in winners:
            part = variants[winner["share_recent"]]
            values = []
            for r in part:
                pred = r["reference"] + winner["blend"] * (r["component"] - r["reference"]) * (
                    winner["position"] is None or int(r["position"]) == winner["position"]
                )
                values.append(
                    {
                        **r,
                        "candidate": pred,
                        "delta": (pred - r["actual"]) ** 2 - (r["reference"] - r["actual"]) ** 2,
                    }
                )
            periods = {}
            for name, subset in (
                ("acceptance_2024", [r for r in values if r["season"] == "2024/25" and r["event"] > 19]),
                ("replication_2025", [r for r in values if r["season"] == "2025/26"]),
            ):
                groups = defaultdict(list)
                for r in subset:
                    groups[r["deadline_group"]].append(r["delta"])
                means = np.array([np.mean(v) for v in groups.values()])
                rng = np.random.default_rng(29431)
                samples = means[rng.integers(0, len(means), (5000, len(means)))].mean(1)
                ci = np.quantile(samples, [0.005, 0.995]).tolist()
                ref, can = metrics(subset, "reference"), metrics(subset, "candidate")
                positions = {
                    str(p): {
                        k: point_metrics([r for r in subset if int(r["position"]) == p], k)
                        for k in ("reference", "candidate")
                    }
                    for p in (1, 2, 3, 4)
                }
                passed = bool(
                    ci[1] < 0
                    and can["regular_rmse"] <= ref["regular_rmse"]
                    and (can["premium_rmse"] or 0) <= (ref["premium_rmse"] or 0)
                    and can["captain_points"] >= ref["captain_points"]
                    and all(v["candidate"]["rmse"] <= v["reference"]["rmse"] for v in positions.values())
                )
                periods[name] = {
                    "rows": len(subset),
                    "reference": ref,
                    "candidate": can,
                    "positions": positions,
                    "delta_mse_99ci": ci,
                    "passed": passed,
                }
            families[winner["specialist"]] = {
                "selected": winner,
                "periods": periods,
                "forecast_gate_passed": all(r["passed"] for r in periods.values()),
            }
            print(
                "Specialist gates",
                winner["specialist"],
                families[winner["specialist"]]["forecast_gate_passed"],
                flush=True,
            )
        result = {
            "at": now(),
            "protocol": protocol,
            "families": families,
            "predictions": paths,
            "promoted": False,
            "passing": [name for name, v in families.items() if v["forecast_gate_passed"]],
        }
        result["id"] = store.put("football_specialists", result)
        atomic_json(Path("artifacts/evaluations/football_specialists_0_14_3.json"), result)
        return result


def cached_components(store, rows, teams, facts, alpha, weight, directory):
    import hashlib
    import inspect
    import json
    import os
    import tempfile

    from .decision import canonical_rows

    originals, _, _ = canonical_rows(store)
    key = digest(
        {
            "version": 2,
            "rows": digest(rows),
            "observations": digest(originals),
            "teams": digest(teams),
            "alpha": alpha,
            "weight": weight,
            "stats": STATS,
            "formula": hashlib.sha256(
                "\n".join(
                    inspect.getsource(f) for f in (component_predictions, rate_vector, allocation, histories)
                ).encode()
            ).hexdigest(),
        }
    )
    path = directory / ("components-v2-" + key[:24] + ".parquet")
    receipt = path.with_suffix(".json")
    if path.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["key"] != key or saved["hash"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Football component checkpoint integrity failed")
        return pl.read_parquet(path).to_dicts()
    values = component_predictions(store, rows, teams, facts, alpha, weight)
    descriptor, name = tempfile.mkstemp(dir=directory, suffix=".parquet")
    os.close(descriptor)
    temporary = Path(name)
    try:
        pl.DataFrame(values).write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(
        receipt, {"key": key, "hash": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(values)}
    )
    return values


def component_diagnostics(store, predictions):
    """Score identifiable components with compatible targets, never direct-head attribution."""
    from .decision import canonical_rows
    from .forecast_population import binary_metrics

    originals, _, _ = canonical_rows(store)
    actual = {(r["season"], int(r["fixture"]), int(r["element"])): r for r in originals}
    periods = {}

    def known(row, key):
        value = actual[row["season"], row["fixture_id"], row["player_id"]].get(key)
        return None if value in (None, "") else float(value)

    for season in ("2024/25", "2025/26", "2026/27"):
        part = [r for r in predictions if r["season"] == season and (season != "2024/25" or r["event"] > 19)]
        if not part:
            continue
        fields = {}
        for label, forecast_field, truth_field, factor, positions in (
            ("goals", "xg", "goals_scored", 1, (1, 2, 3, 4)),
            ("fpl_assists", "expected_fpl_assists", "assists", 1, (1, 2, 3, 4)),
            ("underlying_xg", "xg", "expected_goals", 1, (1, 2, 3, 4)),
            ("underlying_xa", "xa", "expected_assists", 1, (1, 2, 3, 4)),
            ("bonus", "component_bonus", "bonus", 1, (1, 2, 3, 4)),
            ("goalkeeper_save_points", "component_saves", "saves", 1, (1,)),
            ("clean_sheet_points", "component_clean_sheet", "clean_sheets", 1, (1, 2, 3)),
            ("goals_conceded_points", "component_goals_conceded", "goals_conceded", 1, (1, 2)),
        ):
            eligible = [
                r for r in part if int(r["position"]) in positions and known(r, truth_field) is not None
            ]
            truth = np.array([known(r, truth_field) * factor for r in eligible], float)
            pred = np.array([r[forecast_field] for r in eligible], float)
            if label == "goalkeeper_save_points":
                truth = np.floor(truth / 3)
            elif label == "goals_conceded_points":
                truth = -np.floor(truth / 2)
            elif label == "clean_sheet_points":
                truth *= np.array([4 if int(r["position"]) <= 2 else 1 for r in eligible])
            fields[label] = {
                "rows": len(eligible),
                "missing": sum(int(r["position"]) in positions for r in part) - len(eligible),
                "predicted_mean": float(pred.mean()) if len(pred) else None,
                "observed_mean": float(truth.mean()) if len(truth) else None,
                "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))) if len(truth) else None,
                "bias": float(np.mean(pred - truth)) if len(truth) else None,
            }
            if label in ("goals", "fpl_assists") and len(eligible):
                pg = []
                for r in eligible:
                    rate = r[forecast_field] / max(r["expected_minutes"], 1e-9)
                    pg.append(sum(r[f"p{k}"] * (1 - np.exp(-rate * r[f"m{k}"])) for k in range(1, 4)))
                fields[label]["any_return"] = binary_metrics(truth > 0, pg)
        dc = [
            r
            for r in part
            if int(r["position"]) in (2, 3, 4)
            and known(r, "defensive_contribution") is not None
            and season >= "2025/26"
        ]
        fields["defcon"] = binary_metrics(
            [known(r, "defensive_contribution") >= (10 if int(r["position"]) == 2 else 12) for r in dc],
            [sum(r[f"p{k}"] * r[f"dc{k}"] for k in range(4)) for r in dc],
        )
        teams = defaultdict(list)
        for r in part:
            teams[r["fixture_id"], r["team"]].append(r)
        fields["internal_reconciliation"] = {
            "team_goal_allocation_max_absolute": float(
                max(abs(sum(r["xg"] for r in v) - v[0]["team_xg"]) for v in teams.values())
            ),
            "goal_component_marginal_max_absolute": float(
                max(
                    abs(
                        r["component_goals"]
                        / (
                            10
                            if int(r["position"]) == 1
                            else 6
                            if int(r["position"]) == 2
                            else 5
                            if int(r["position"]) == 3
                            else 4
                        )
                        - r["xg"]
                    )
                    for r in part
                )
            ),
            "fpl_assist_component_max_absolute": float(
                max(abs(r["component_assists"] / 3 - r["expected_fpl_assists"]) for r in part)
            ),
            "point_component_sum_max_absolute": float(
                max(
                    abs(sum(v for k, v in r.items() if k.startswith("component_")) - r["component"])
                    for r in part
                )
            ),
        }
        periods[season] = fields
    return {
        "periods": periods,
        "scope": "FiveStat-inspired research components, not decomposition or accepted estimates from the direct learned-points head. xA is an underlying chance-creation proxy; expected_fpl_assists includes the separately estimated FPL attribution ratio. Rate targets, binary event probabilities and FPL point components are scored separately. Missing observations are excluded and counted. The rejected challenger is not sent to the optimizer.",
        "bps": "The accepted immediate direct-points head does not identify expected BPS; lagged bonus is consumed there. Distant expert features include observed BPS. Separately gated BPS/event-model research remains separate; no expected-BPS output is fabricated.",
    }


def seal_predictions(directory, prefix, values):
    """Retain old experimental outputs when data, code or parameters change."""
    import hashlib
    import os
    import tempfile

    descriptor, name = tempfile.mkstemp(dir=directory, suffix=".parquet")
    os.close(descriptor)
    temporary = Path(name)
    try:
        pl.DataFrame(values).write_parquet(temporary)
        checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
        target = directory / (prefix + "-" + checksum[:24] + ".parquet")
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
                raise ValueError("Retained prediction checksum mismatch")
        else:
            os.replace(temporary, target)
        return target, checksum
    finally:
        temporary.unlink(missing_ok=True)
