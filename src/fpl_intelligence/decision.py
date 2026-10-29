"""Deadline-safe learned points forecasts, calibrated worlds and decision audits.

The production target is observed FPL points, not the team selected with hindsight.
Historical oracle teams are diagnostics and are never fed back into test features.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.special import ndtr
from scipy.stats import nbinom
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, histories, number, profile, vector
from .history import archived_rows
from .ingestion import load_market
from .models import curve_apply, curve_fit, reliability
from .storage import Store, atomic_json, digest, now

VERSION = "learned-points-v1"
EXTRA = [
    "price_lag",
    "goals_actual90",
    "assists_actual90",
    "bonus90",
    "last3_minutes",
    "last5_points",
    "long_probability",
    "team_attack",
    "team_defence",
    "opponent_attack",
    "opponent_defence",
    "season_week",
    "history_missing",
]
NAMES = FEATURE_NAMES + EXTRA
PARAMETERS: dict[str, Any] = {
    "n_estimators": 180,
    "num_leaves": 20,
    "learning_rate": 0.035,
    "min_child_samples": 80,
    "reg_lambda": 10,
    "verbosity": -1,
    "n_jobs": 1,
    "random_state": 47,
}
BAND_MINUTES = np.array([0.0, 24.0, 74.0, 90.0])


def minute_band(minutes):
    return np.select([minutes <= 0, minutes < 60, minutes < 90], [0, 1, 2], default=3).astype(int)


def context_vector(position, home, past, kickoff, team, opponent, week):
    """One shared train/serve transformation; the current outcome is never an input."""
    p = profile(position, past)
    exp = sum(number(r, "minutes") / 90 for r in past[-24:])
    extras = {
        "price_lag": number(past[-1], "value", {1: 45, 2: 45, 3: 60, 4: 65}[position])
        if past
        else {1: 45, 2: 45, 3: 60, 4: 65}[position],
        "goals_actual90": (sum(number(r, "goals_scored") for r in past[-24:]) + 5 * p["goals90"]) / (exp + 5),
        "assists_actual90": (sum(number(r, "assists") for r in past[-24:]) + 5 * p["assists90"]) / (exp + 5),
        "bonus90": (sum(number(r, "bonus") for r in past[-24:]) + 2) / (exp + 5),
        "last3_minutes": np.mean([number(r, "minutes") for r in past[-3:]]) if past else 0.0,
        "last5_points": np.mean([number(r, "total_points") for r in past[-5:]]) if past else 2.0,
        "long_probability": p["p_long_start"],
        "team_attack": team[0],
        "team_defence": team[1],
        "opponent_attack": opponent[0],
        "opponent_defence": opponent[1],
        "season_week": week,
        "history_missing": float(not past),
    }
    return vector(position, home, p, kickoff) + [float(extras[k]) for k in EXTRA], p


def strength(rows):
    rows = rows[-16:]
    return (
        (sum(r[0] for r in rows) + 8 * 1.4) / (len(rows) + 8),
        (sum(r[1] for r in rows) + 8 * 1.4) / (len(rows) + 8),
    )


def canonical_rows(store: Store):
    players, market = load_market(store)
    rows = [dict(r) for r in archived_rows(store)]
    team_names = {t["id"]: t["name"] for t in market["teams"]}
    fixtures = {f["id"]: f for f in market["fixtures"]}
    finalized = {e["id"] for e in market["events"] if e.get("data_checked")}
    for identity, history in histories(store, players).items():
        player = players[identity]
        for r in history:
            if r["round"] not in finalized:
                continue
            f = fixtures[r["fixture"]]
            rows.append(
                {
                    **r,
                    "season": market["season"],
                    "element": identity,
                    "code": player.code,
                    "position": player.position,
                    "team": team_names[f["team_h"] if r["was_home"] else f["team_a"]],
                }
            )
    by_fixture = defaultdict(list)
    for r in rows:
        by_fixture[r["season"], int(r["fixture"])].append(r)
    metadata = {}
    unknown_starts = 0
    for key, group in by_fixture.items():
        sides = {}
        for home in (True, False):
            names = Counter(str(r["team"]) for r in group if bool(r["was_home"]) == home)
            if not names:
                raise ValueError(f"Historical fixture {key} lacks a team side")
            sides[home] = names.most_common(1)[0][0]
        exemplar = group[0]
        metadata[key] = {
            "home": sides[True],
            "away": sides[False],
            "home_goals": number(exemplar, "team_h_score"),
            "away_goals": number(exemplar, "team_a_score"),
            "event": int(exemplar["round"]),
        }
        # Early 2022 data has starts filled with zero for entire fixtures. Those
        # are missing labels, not 22 observed substitute appearances.
        missing_starts = sum(number(r, "starts") for r in group) == 0
        for r in group:
            r["team"] = sides[bool(r["was_home"])]
            r["opponent_name"] = sides[not bool(r["was_home"])]
            r["start_observed"] = not missing_starts
            if missing_starts:
                r["starts"] = None
                unknown_starts += 1
    rows.sort(
        key=lambda r: (
            r["season"],
            int(r["round"]),
            r.get("kickoff_time") or "",
            int(r["fixture"]),
            int(r["element"]),
        )
    )
    unique = {(r["season"], int(r["element"]), int(r["fixture"])): r for r in rows}
    return list(unique.values()), metadata, {"unknown_start_labels": unknown_starts}


def player_key(row):
    return ("code", int(row["code"])) if row.get("code") else (row["season"], int(row["element"]))


def build_dataset(store: Store):
    rows, fixture_meta, quality = canonical_rows(store)
    grouped = defaultdict(list)
    for row in rows:
        grouped[f"{row['season']}-GW{int(row['round']):02d}"].append(row)
    player_history: dict[Any, list[dict]] = defaultdict(list)
    club_history: dict[str, list[tuple[float, float]]] = defaultdict(list)
    records = []
    for group, current in sorted(grouped.items()):
        for row in current:
            position = int(row["position"])
            past = player_history[player_key(row)]
            x, p = context_vector(
                position,
                bool(row["was_home"]),
                past,
                row.get("kickoff_time"),
                strength(club_history[row["team"]]),
                strength(club_history[row["opponent_name"]]),
                int(row["round"]),
            )
            dc_known = row.get("defensive_contribution") not in (None, "") and row["season"] >= "2025/26"
            dc = (
                2
                * int(
                    position != 1 and number(row, "defensive_contribution") >= (10 if position == 2 else 12)
                )
                if dc_known
                else 0
            )
            target = number(row, "total_points")
            # Goalkeepers' goals rose from six to ten in the 2025 rules.
            base_target = (
                target
                - dc
                + (4 * number(row, "goals_scored") if position == 1 and row["season"] < "2025/26" else 0)
            )
            records.append(
                dict(zip(NAMES, x))
                | {
                    "season": row["season"],
                    "event": int(row["round"]),
                    "deadline_group": group,
                    "player_id": int(row["element"]),
                    "code": int(row["code"]) if row.get("code") else None,
                    "fixture_id": int(row["fixture"]),
                    "team": row["team"],
                    "opponent": row["opponent_name"],
                    "target": target,
                    "base_target": base_target,
                    "minutes_target": number(row, "minutes"),
                    "start_target": number(row, "starts") if row["start_observed"] else None,
                    "dc_target": dc / 2 if dc_known else None,
                    "dc_probability": dc_probability(position, p, BAND_MINUTES)
                    if row["season"] >= "2025/26"
                    else [0.0] * 4,
                    "baseline": p["points_mean"],
                    "price_observed": bool(past and past[-1].get("value") is not None),
                }
            )
        # Update only after ALL fixtures of this deadline, including doubles.
        for row in current:
            player_history[player_key(row)].append(row)
            player_history[player_key(row)] = player_history[player_key(row)][-24:]
        for key in sorted({(r["season"], int(r["fixture"])) for r in current}):
            f = fixture_meta[key]
            club_history[f["home"]].append((f["home_goals"], f["away_goals"]))
            club_history[f["away"]].append((f["away_goals"], f["home_goals"]))
    path = store.root / "features" / "decision.parquet"
    path.parent.mkdir(exist_ok=True)
    frame = pl.DataFrame(records, infer_schema_length=None)
    frame.write_parquet(path)
    report = {
        "at": now(),
        "version": VERSION,
        "path": str(path),
        "hash": digest(records),
        "rows": len(records),
        "seasons": sorted({r["season"] for r in rows}),
        "features": NAMES,
        "quality": quality,
        "cutoff": "Every feature uses strictly earlier deadlines; same transform in production",
        "limits": [
            "Historical archive corrections are retrospective, not captured before historical deadlines",
            "Old-rule bonus awards cannot be fully reconstructed; defensive-contribution targets are separated",
        ],
    }
    report["id"] = store.put("decision_features", report)
    return report, frame


def dc_probability(position, p, minutes):
    if position == 1:
        return [0.0] * len(minutes)
    mean = np.array(minutes) / 90 * p["actions90"]
    shape = p.get("action_shape", 20)
    result = nbinom.sf((10 if position == 2 else 12) - 1, shape, shape / (shape + np.maximum(mean, 1e-9)))
    return np.where(np.array(minutes) >= 1, result, 0).tolist()


def fit_components(training, calibration, neural=False):
    x = training.select(NAMES).to_numpy()
    xc = calibration.select(NAMES).to_numpy()
    yband = minute_band(training["minutes_target"].to_numpy())
    minutes = lgb.LGBMClassifier(**PARAMETERS).fit(x, yband)
    if minutes.classes_.tolist() != [0, 1, 2, 3]:
        raise ValueError("Missing minutes band in training")
    observed = training.filter(pl.col("start_target").is_not_null())
    starter = lgb.LGBMClassifier(**PARAMETERS).fit(
        observed.select(NAMES).to_numpy(), observed["start_target"].to_numpy()
    )
    appeared = training.filter(pl.col("minutes_target") > 0)
    xp = np.column_stack(
        [appeared.select(NAMES).to_numpy(), minute_band(appeared["minutes_target"].to_numpy())]
    )
    target = appeared["base_target"].to_numpy()
    points = lgb.LGBMRegressor(**PARAMETERS).fit(xp, target)
    raw = np.asarray(minutes.predict_proba(xc))
    truth = minute_band(calibration["minutes_target"].to_numpy())
    curves = {str(k): curve_fit(raw[:, k], (truth == k).astype(int)) for k in range(4)}
    cs = calibration.filter(pl.col("start_target").is_not_null())
    curves["start"] = curve_fit(
        np.asarray(starter.predict_proba(cs.select(NAMES).to_numpy()))[:, 1], cs["start_target"].to_numpy()
    )
    network = None
    if neural:
        network = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                alpha=10,
                learning_rate_init=0.001,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.12,
                random_state=47,
            ),
        )
        network.fit(xp, target)
    return {"minutes": minutes, "starter": starter, "points": points, "curves": curves, "network": network}


def predict_models(bundle, matrix, dc, neural=False):
    raw = (
        np.asarray(bundle["minutes"].predict_proba(matrix))
        if hasattr(bundle["minutes"], "predict_proba")
        else np.asarray(bundle["minutes"].predict(matrix, num_threads=1))
    )
    probabilities = np.column_stack([curve_apply(raw[:, k], bundle["curves"][str(k)]) for k in range(4)])
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-9)
    conditional = np.zeros((len(matrix), 4))
    model = bundle["network"] if neural else bundle["points"]
    for k in range(1, 4):
        conditional[:, k] = model.predict(np.column_stack([matrix, np.full(len(matrix), k)]))
    means = (probabilities * (conditional + 2 * np.asarray(dc))).sum(axis=1)
    return means, probabilities, conditional


def score_metrics(actual, predicted):
    return {
        "n": len(actual),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "bias": float(np.mean(predicted - actual)),
    }


def paired_bootstrap(rows, field, baseline="baseline"):
    groups = sorted({r["deadline_group"] for r in rows})
    summaries = np.array(
        [
            [
                sum((r[field] - r["actual"]) ** 2 for r in rows if r["deadline_group"] == g),
                sum((r[baseline] - r["actual"]) ** 2 for r in rows if r["deadline_group"] == g),
                sum(r["deadline_group"] == g for r in rows),
            ]
            for g in groups
        ]
    )
    rng = np.random.default_rng(47)
    sampled = summaries[rng.integers(0, len(groups), (1000, len(groups)))].sum(axis=1)
    differences = np.sqrt(sampled[:, 0] / sampled[:, 2]) - np.sqrt(sampled[:, 1] / sampled[:, 2])
    return np.quantile(differences, [0.025, 0.975]).tolist()


def ensure_history(store: Store):
    from .history import import_season

    _, market = load_market(store)
    year = int(market["season"][:4])
    with store.db() as db:
        present = {
            row[0]
            for row in db.execute("SELECT DISTINCT source FROM snapshots WHERE source LIKE 'archive_%'")
        }
    for start in range(year - 4, year):
        season = f"{start}-{str(start + 1)[2:]}"
        if f"archive_{season}" not in present:
            import_season(store, season)


def train_decisions(store: Store, neural=True):
    protocol = {
        "version": VERSION,
        "features": NAMES,
        "parameters": PARAMETERS,
        "test": "2025/26 and current completed season, expanding four-deadline test blocks",
        "calibration": "eight earlier deadlines excluded from base fitting in every fold",
        "targets": "Observed points conditional on minutes band; band is marginalized, never taken from test actuals",
        "neural": "Standardized MLP (32,16), alpha=10, 100 iterations, trained only inside earlier folds",
        "promotion": "Upper deadline-bootstrap RMSE delta < 0 vs lag shrinkage, plus regular-starter RMSE improvement",
        "hindsight": "Oracle teams and best actual captain are diagnostic upper bounds, never training inputs",
    }
    protocol_id = store.put("decision_protocol", protocol)
    ensure_history(store)
    feature, frame = build_dataset(store)
    groups = sorted(frame["deadline_group"].unique().to_list())
    testing = [g for g in groups if g >= "2025/26-GW01"]
    if len(groups) < 60 or not testing:
        raise ValueError("Decision training requires multiple complete seasons and held-out deadlines")
    predictions = []
    fold_metrics = []
    for first in range(0, len(testing), 4):
        block = testing[first : first + 4]
        before = [g for g in groups if g < block[0]]
        cut = before[-8]
        train = frame.filter(pl.col("deadline_group") < cut)
        cal = frame.filter((pl.col("deadline_group") >= cut) & (pl.col("deadline_group") < block[0]))
        test = frame.filter(pl.col("deadline_group").is_in(block))
        fitted = fit_components(train, cal, neural)
        x = test.select(NAMES).to_numpy()
        mean, probabilities, _ = predict_models(fitted, x, test["dc_probability"].to_list())
        network = predict_models(fitted, x, test["dc_probability"].to_list(), True)[0] if neural else None
        for i, row in enumerate(test.to_dicts()):
            predictions.append(
                {
                    k: row[k]
                    for k in (
                        "season",
                        "event",
                        "deadline_group",
                        "player_id",
                        "position",
                        "team",
                        "price_lag",
                        "price_observed",
                        "fixture_id",
                    )
                }
                | {
                    "actual": row["target"],
                    "minutes": row["minutes_target"],
                    "baseline": row["baseline"],
                    "predicted": float(mean[i]),
                    "network": float(network[i]) if network is not None else None,
                    "prior_start": row["p_start"],
                    "p_any": float(1 - probabilities[i, 0]),
                    "p60": float(probabilities[i, 2:].sum()),
                    "history_count": row["exposure"],
                }
            )
        fold_metrics.append(
            {"test": block, "training_rows": len(train), "calibration_rows": len(cal), "test_rows": len(test)}
        )
        print(f"Validated {block[0]} through {block[-1]}", flush=True)
    y = np.array([r["actual"] for r in predictions])
    b = np.array([r["baseline"] for r in predictions])
    p = np.array([r["predicted"] for r in predictions])
    regular = np.array([r["prior_start"] >= 0.6 for r in predictions])
    interval = paired_bootstrap(predictions, "predicted")
    promoted = (
        interval[1] < 0
        and score_metrics(y[regular], p[regular])["rmse"] < score_metrics(y[regular], b[regular])["rmse"]
    )
    summary = {
        "at": now(),
        "version": VERSION,
        "protocol_id": protocol_id,
        "feature_id": feature["id"],
        "seasons": feature["seasons"],
        "baseline": score_metrics(y, b),
        "candidate": score_metrics(y, p),
        "paired_rmse_difference_95ci": interval,
        "regular_starters": {
            "baseline": score_metrics(y[regular], b[regular]),
            "candidate": score_metrics(y[regular], p[regular]),
        },
        "appearance": reliability(
            np.array([r["minutes"] > 0 for r in predictions]), np.array([r["p_any"] for r in predictions])
        ),
        "by_position": {
            str(pos): {
                "baseline": score_metrics(y[mask], b[mask]),
                "candidate": score_metrics(y[mask], p[mask]),
            }
            for pos in range(1, 5)
            if (mask := np.array([r["position"] == pos for r in predictions])).any()
        },
        "folds": fold_metrics,
        "promoted": bool(promoted),
        "predictions": predictions,
        "production_target": "The exact same calibrated mixture is consumed by forecast worlds and optimizer",
        "limits": feature["limits"],
    }
    if neural:
        n = np.array([r["network"] for r in predictions])
        summary["neural"] = {
            "metrics": score_metrics(y, n),
            "paired_vs_gbm_95ci": paired_bootstrap(predictions, "network", "predicted"),
            "promotion": "An independently versioned production path is required before any neural replacement",
        }
    # Honest final production fit: base fitting excludes the most recent eight
    # calibration deadlines, so labels never leak into their own calibration.
    cut = groups[-8]
    train = frame.filter(pl.col("deadline_group") < cut)
    cal = frame.filter(pl.col("deadline_group") >= cut)
    fitted = fit_components(train, cal, False)
    directory = store.root / "models" / digest({"protocol": protocol, "feature": feature["hash"]})[:16]
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for key in ("minutes", "starter", "points"):
        path = directory / (key + ".txt")
        fitted[key].booster_.save_model(str(path))
        hashes[key] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    residuals, dependence = residual_contract(fitted, cal)
    observed_minutes = cal["minutes_target"].to_numpy()
    minute_quantiles = {
        str(k): np.quantile(
            observed_minutes[minute_band(observed_minutes) == k], np.linspace(0, 1, 101)
        ).tolist()
        for k in range(4)
    }
    summary.update(
        directory=directory.name,
        model_hashes=hashes,
        curves=fitted["curves"],
        residuals=residuals,
        dependence=dependence,
        minute_quantiles=minute_quantiles,
        production_training_rows=len(train),
        production_calibration_rows=len(cal),
        production_fit_before=cut,
        training_serve_transform="context_vector, stable player code, last 24 earlier fixtures",
    )
    summary["id"] = store.put("decision_model", summary)
    atomic_json(
        Path("artifacts/evaluations/decision_model.json"),
        {k: v for k, v in summary.items() if k not in ("predictions", "residuals", "curves")},
    )
    return summary


def residual_contract(fitted, cal):
    appeared = cal.filter(pl.col("minutes_target") > 0)
    band = minute_band(appeared["minutes_target"].to_numpy())
    prediction = fitted["points"].predict(np.column_stack([appeared.select(NAMES).to_numpy(), band]))
    residual = appeared["base_target"].to_numpy() - prediction
    rows = appeared.to_dicts()
    pools = {}
    for position in range(1, 5):
        for k in range(1, 4):
            selected = np.array([int(r["position"]) == position and band[i] == k for i, r in enumerate(rows)])
            values = residual[selected]
            if len(values) < 30:
                values = residual[band == k]
            # Centering preserves the tested conditional expectation. Empirical
            # residuals retain discrete-outcome skew and heavy upper tails.
            pools[f"{position}:{k}"] = np.quantile(values - values.mean(), np.linspace(0, 1, 201)).tolist()
    groups = defaultdict(list)
    for row, res in zip(rows, residual):
        groups[
            row["season"], row["fixture_id"], row["team"], "defence" if row["position"] <= 2 else "attack"
        ].append(float(res))
    dependence = {}
    for role in ("defence", "attack"):
        numerator = denominator = 0.0
        variance = max(
            float(
                np.var(
                    [res for row, res in zip(rows, residual) if (row["position"] <= 2) == (role == "defence")]
                )
            ),
            0.1,
        )
        for key, vals in groups.items():
            if key[-1] != role or len(vals) < 2:
                continue
            a = np.asarray(vals)
            numerator += a.sum() ** 2 - np.dot(a, a)
            denominator += len(a) * (len(a) - 1) * variance
        dependence[role] = float(np.clip(numerator / max(denominator, 1), 0, 0.6))
    return pools, dependence


def load_bundle(store, spec):
    import hashlib

    models = {}
    for key in ("minutes", "starter", "points"):
        path = store.root / "models" / spec["directory"] / (key + ".txt")
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["model_hashes"][key]:
            raise ValueError("Decision model integrity verification failed")
        models[key] = lgb.Booster(model_file=str(path))
    return models | {"curves": spec["curves"], "network": None}


def future_inputs(store, market, players, start, end, scenario=None, beliefs=()):
    rows, fixtures, _ = canonical_rows(store)
    past = defaultdict(list)
    clubs = defaultdict(list)
    for row in rows:
        past[player_key(row)].append(row)
        past[player_key(row)] = past[player_key(row)][-24:]
    for (season, identity), f in sorted(
        fixtures.items(), key=lambda kv: (kv[0][0], kv[1]["event"], kv[0][1])
    ):
        clubs[f["home"]].append((f["home_goals"], f["away_goals"]))
        clubs[f["away"]].append((f["away_goals"], f["home_goals"]))
    names = {t["id"]: t["name"] for t in market["teams"]}
    scheduled = [dict(f) for f in market["fixtures"]]
    for change in (scenario or {}).get("fixture_events", []):
        match = next((f for f in scheduled if f["id"] == change["fixture_id"]), None)
        if match is None:
            raise ValueError("Unknown fixture in scenario")
        match["event"] = change.get("event")
        match["kickoff_time"] = None
    records = []
    for f in scheduled:
        event = f.get("event")
        if f.get("finished") or event is None or not start <= event <= end:
            continue
        for player in players.values():
            if player.team not in (f["team_h"], f["team_a"]):
                continue
            home = player.team == f["team_h"]
            history = past[("code", player.code)]
            x, p = context_vector(
                player.position,
                home,
                history,
                f.get("kickoff_time"),
                strength(clubs[names[player.team]]),
                strength(clubs[names[f["team_a"] if home else f["team_h"]]]),
                event,
            )
            values = dict(zip(NAMES, x))
            if f.get("kickoff_time"):
                from datetime import datetime

                kickoff = datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))
                previous = [
                    datetime.fromisoformat(other["kickoff_time"].replace("Z", "+00:00"))
                    for other in scheduled
                    if other.get("kickoff_time")
                    and player.team in (other["team_h"], other["team_a"])
                    and other["kickoff_time"] < f["kickoff_time"]
                ]
                if previous:
                    values["rest_days"] = min(
                        30.0, max(0.0, (kickoff - max(previous)).total_seconds() / 86400)
                    )
            cap = 90
            availability = (
                player.chance if player.chance is not None else (1.0 if player.status == "a" else 0.5)
            )
            availability = 1 - (1 - availability) * np.exp(-(event - start) / 5)
            if not player.selectable:
                availability = 0
            for belief in beliefs:
                if belief["player_id"] == player.id:
                    values[belief["parameter"]] = belief["value"]
            for change in (scenario or {}).get("players", []):
                if change["player_id"] == player.id and change["from_event"] <= event <= change.get(
                    "through_event", end
                ):
                    if "availability" in change:
                        availability = change["availability"]
                    for parameter, value in change.get("parameters", {}).items():
                        if parameter == "minutes_cap":
                            cap = value
                        else:
                            values[parameter] = value
            records.append(
                {
                    "player_id": player.id,
                    "fixture_id": f["id"],
                    "event": event,
                    "team": player.team,
                    "opponent": f["team_a"] if home else f["team_h"],
                    "x": [values[k] for k in NAMES],
                    "profile": p,
                    "availability": availability,
                    "minutes_cap": cap,
                    "dc_probability": dc_probability(player.position, p, BAND_MINUTES),
                }
            )
    return records, scheduled


def sample_point_outcomes(
    rng,
    probabilities,
    conditional,
    dc_probabilities,
    residual_pools,
    position,
    band_uniform,
    residual_uniform,
):
    """Shared production/evaluation sampler; integer points and exact zero-minute mass."""
    count = len(band_uniform)
    band = np.minimum(np.searchsorted(np.cumsum(probabilities), band_uniform), 3)
    outcomes = np.zeros(count)
    for k in range(1, 4):
        mask = band == k
        pool = np.asarray(residual_pools[f"{position}:{k}"])
        levels = np.linspace(0, 1, len(pool))
        pool = pool - np.trapezoid(pool, levels)
        outcomes[mask] = conditional[k] + np.interp(residual_uniform[mask], levels, pool)
    base = np.floor(outcomes) + (rng.random(count) < outcomes - np.floor(outcomes))
    base[band == 0] = 0
    dc = (rng.random(count) < np.asarray(dc_probabilities)[band]) & (band > 0)
    return (base + 2 * dc).astype(np.int16), band, dc


def learned_forecast(
    store: Store, start_event=None, end_event=38, worlds=2000, seed=11, use_beliefs=False, scenario=None
):
    import time

    started = time.monotonic()
    spec = store.get("decision_model")
    if not spec["promoted"]:
        raise ValueError("Decision forecasts failed the held-out promotion gate")
    if not 100 <= worlds <= 20000:
        raise ValueError("Invalid scenario budget")
    players, market = load_market(store)
    start = start_event or next((e["id"] for e in market["events"] if e.get("is_next")), 38)
    if not 1 <= start <= end_event <= 38:
        raise ValueError("Invalid forecast horizon")
    from .research import active_beliefs
    from .rules import Rules

    beliefs = active_beliefs(store) if use_beliefs else []
    rows, scheduled = future_inputs(store, market, players, start, end_event, scenario, beliefs)
    if not rows:
        raise ValueError("No scheduled fixtures in forecast horizon")
    bundle = load_bundle(store, spec)
    matrix = np.asarray([r["x"] for r in rows])
    _, probabilities, conditional = predict_models(bundle, matrix, [r["dc_probability"] for r in rows])
    raw_start = bundle["starter"].predict(matrix, num_threads=1)
    starts = curve_apply(raw_start, spec["curves"]["start"])
    ids = sorted(players)
    index = {p: i for i, p in enumerate(ids)}
    arrays: dict[str, Any] = {}
    by_event = defaultdict(list)
    fixture_records = []
    records = []
    audit: list[dict] = []
    shared_player = np.random.default_rng(seed).normal(size=(worlds, len(ids)))
    for i, row in enumerate(rows):
        by_event[row["event"]].append((i, row))
    team_factors = {}
    for event in range(start, end_event + 1):
        points = np.zeros((worlds, len(ids)), dtype=np.int16)
        minutes = np.zeros_like(points)
        component_totals = {
            "appearance": np.zeros(len(ids)),
            "defcon": np.zeros(len(ids)),
            "other_learned_points": np.zeros(len(ids)),
        }
        start_prob = np.zeros(len(ids))
        fixture_count = np.zeros(len(ids), dtype=int)
        for i, row in by_event[event]:
            pid = row["player_id"]
            column = index[pid]
            player = players[pid]
            rng = np.random.default_rng(np.random.SeedSequence([seed, row["fixture_id"], pid]))
            probs = probabilities[i].copy()
            probs[1:] *= row["availability"]
            probs[0] = 1 - probs[1:].sum()
            cap = row["minutes_cap"]
            if cap < 60:
                probs[1] += probs[2:].sum()
                probs[2:] = 0
            elif cap < 90:
                probs[2] += probs[3]
                probs[3] = 0
            u = ndtr(0.3 * shared_player[:, column] + np.sqrt(0.91) * rng.normal(size=worlds))
            band = np.minimum(np.searchsorted(np.cumsum(probs), u), 3)
            sampled_minutes = np.zeros(worlds)
            for k in range(1, 4):
                mask = band == k
                pool = spec["minute_quantiles"][str(k)]
                sampled_minutes[mask] = np.rint(
                    np.interp(rng.random(mask.sum()), np.linspace(0, 1, len(pool)), pool)
                )
            sampled_minutes = np.minimum(sampled_minutes, cap).astype(np.int16)
            role = "defence" if player.position <= 2 else "attack"
            # The same factor is reused for teammates in this fixture. These are
            # calibrated statistical worlds, not claimed event-level football XIs.
            factor_key = (row["fixture_id"], row["team"], role)
            if factor_key not in team_factors:
                team_factors[factor_key] = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, row["fixture_id"], row["team"], 9107 if role == "defence" else 9109]
                    )
                ).normal(size=worlds)
            team_factor = team_factors[factor_key]
            rho = spec["dependence"][role]
            residual_u = ndtr(np.sqrt(rho) * team_factor + np.sqrt(1 - rho) * rng.normal(size=worlds))
            values, sampled_band, dc = sample_point_outcomes(
                rng,
                probs,
                conditional[i],
                row["dc_probability"],
                spec["residuals"],
                player.position,
                u,
                residual_u,
            )
            if not np.array_equal(band, sampled_band):
                raise ValueError("Minutes and points mixture disagree")
            values[sampled_minutes == 0] = 0
            points[:, column] += values
            minutes[:, column] += sampled_minutes
            appearance = (sampled_minutes > 0).astype(int) + (sampled_minutes >= 60)
            parts = {
                "appearance": float(appearance.mean()),
                "defcon": float(2 * dc.mean()),
                "other_learned_points": float(values.mean() - appearance.mean() - 2 * dc.mean()),
            }
            for key, value in parts.items():
                component_totals[key][column] += value
            start_prob[column] = max(start_prob[column], min(1 - probs[0], starts[i] * row["availability"]))
            fixture_count[column] += 1
            fixture_records.append(
                {
                    "player_id": pid,
                    "fixture_id": row["fixture_id"],
                    "event": event,
                    "mean": float(values.mean()),
                    "minutes": float(sampled_minutes.mean()),
                    "p_zero": float((sampled_minutes == 0).mean()),
                    "p_60": float((sampled_minutes >= 60).mean()),
                    "point_quantiles": np.quantile(values, [0.1, 0.5, 0.9]).tolist(),
                    "components": parts,
                    "model_mean_before_mc": float(
                        np.dot(probs, conditional[i] + 2 * np.asarray(row["dc_probability"]))
                    ),
                    "availability_assumption": float(row["availability"]),
                }
            )
        for col, pid in enumerate(ids):
            p, m = points[:, col], minutes[:, col]
            records.append(
                {
                    "player_id": pid,
                    "event": event,
                    "mean": float(p.mean()),
                    "sd": float(p.std()),
                    "point_quantiles": np.quantile(p, [0.1, 0.5, 0.9]).tolist(),
                    "minute_quantiles": np.quantile(m, [0.1, 0.5, 0.9]).tolist(),
                    "minutes_mean": float(m.mean()),
                    "p_zero": float((m == 0).mean()),
                    "p_start": float(start_prob[col]),
                    "p_60": float((m >= 60).mean()),
                    "p_haul": float((p >= 10).mean()),
                    "components": {k: round(float(v[col]), 4) for k, v in component_totals.items()},
                    "history_count": next(
                        (r["profile"]["history_count"] for _, r in by_event[event] if r["player_id"] == pid),
                        0,
                    ),
                    "action_coverage": next(
                        (
                            r["profile"]["action_coverage"]
                            for _, r in by_event[event]
                            if r["player_id"] == pid
                        ),
                        0,
                    ),
                    "fixture_count": int(fixture_count[col]),
                }
            )
        arrays[f"points_{event}"] = points
        arrays[f"minutes_{event}"] = minutes
    rules = Rules.from_bootstrap(store.latest("bootstrap", content_hash=market["source_hash"])[0])
    body = {
        "issued_at": now(),
        "cutoff": now(),
        "season": market["season"],
        "start_event": start,
        "end_event": end_event,
        "worlds": worlds,
        "seed": seed,
        "model_version": VERSION,
        "decision_model_id": spec["id"],
        "rules_hash": rules.hash,
        "market_id": market["id"],
        "input_hashes": [market["source_hash"], market["fixture_hash"], spec["feature_id"]],
        "input_manifest": {
            "decision_model_id": spec["id"],
            "training_feature_id": spec["feature_id"],
            "seasons": spec["seasons"],
        },
        "players": records,
        "fixtures": fixture_records,
        "match_audit": audit,
        "player_ids": ids,
        "belief_scenario": [b["id"] for b in beliefs],
        "scenario": scenario,
        "minutes_model_id": spec["id"],
        "team_strength_model_id": None,
        "unassigned_fixtures": [f["id"] for f in scheduled if f.get("event") is None],
        "assumptions": [
            "Held-out validated conditional-points model, calibrated minutes bands, and empirical residual worlds",
            "Within-team role factors are estimated from earlier calibration residuals; dependence remains approximate",
            "Points distributions are statistical outcome forecasts, not an event-level reconstruction",
            "Exact autosubs, captain fallback, transfers and chips are scored on these SAME worlds",
            "Current news changes availability; future recovery is an explicit horizon assumption",
            "Future prices are fixed outside named price scenarios; no manager ownership is treated as ground truth",
        ],
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    path = store.root / "scenarios" / (digest(body) + ".npz")
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **arrays)
    body["scenario_path"] = str(path)
    body["id"] = store.put("scenario_forecast" if scenario else "forecast", body)
    return body


def validate_uncertainty(store: Store, worlds=512):
    """Run the exact outcome sampler with each fold's strictly earlier residuals."""
    feature = store.get("decision_features")
    frame = pl.read_parquet(store.artifact_path(feature["path"]))
    groups = sorted(frame["deadline_group"].unique().to_list())
    testing = [g for g in groups if g >= "2025/26-GW01"]
    records = []
    for first in range(0, len(testing), 4):
        block = testing[first : first + 4]
        before = [g for g in groups if g < block[0]]
        cut = before[-8]
        train = frame.filter(pl.col("deadline_group") < cut)
        cal = frame.filter((pl.col("deadline_group") >= cut) & (pl.col("deadline_group") < block[0]))
        test = frame.filter(pl.col("deadline_group").is_in(block))
        fitted = fit_components(train, cal, False)
        pools, _ = residual_contract(fitted, cal)
        mean, probability, conditional = predict_models(
            fitted, test.select(NAMES).to_numpy(), test["dc_probability"].to_list()
        )
        rng = np.random.default_rng(100 + first)
        for i, row in enumerate(test.to_dicts()):
            points, _, _ = sample_point_outcomes(
                rng,
                probability[i],
                conditional[i],
                row["dc_probability"],
                pools,
                int(row["position"]),
                rng.random(worlds),
                rng.random(worlds),
            )
            quantile = np.quantile(points, [0.05, 0.1, 0.9, 0.95])
            actual = row["target"]
            ordered = np.sort(points)
            crps = np.abs(points - actual).mean() - np.dot(
                2 * np.arange(1, worlds + 1) - worlds - 1, ordered
            ) / (worlds**2)
            records.append(
                {
                    "regular": row["p_start"] >= 0.6,
                    "covered80": quantile[1] <= actual <= quantile[2],
                    "covered90": quantile[0] <= actual <= quantile[3],
                    "width80": quantile[2] - quantile[1],
                    "crps": float(crps),
                    "mean_error": float(points.mean() - mean[i]),
                }
            )
        print("Distribution audit", block[-1], flush=True)

    def aggregate(rows):
        return {
            "n": len(rows),
            "coverage80": float(np.mean([r["covered80"] for r in rows])),
            "coverage90": float(np.mean([r["covered90"] for r in rows])),
            "width80": float(np.mean([r["width80"] for r in rows])),
            "crps": float(np.mean([r["crps"] for r in rows])),
            "mean_sampling_bias": float(np.mean([r["mean_error"] for r in rows])),
        }

    report = {
        "at": now(),
        "decision_model_id": store.get("decision_model")["id"],
        "worlds_per_row": worlds,
        "all": aggregate(records),
        "regular_starters": aggregate([r for r in records if r["regular"]]),
        "track": "Exact production sampler, earlier-fold fitted models and residuals",
        "limits": [
            "Marginal interval coverage does not establish correctness of the joint opponent-policy probabilities",
            "Discrete points make nominal central coverage conservative",
        ],
    }
    report["id"] = store.put("decision_uncertainty", report)
    atomic_json(Path("artifacts/evaluations/decision_uncertainty.json"), report)
    return report
