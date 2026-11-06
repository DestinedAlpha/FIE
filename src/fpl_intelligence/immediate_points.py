"""Deadline-safe form and advanced-statistics expert for immediate scoring.

Participation, DefCon and the distant-fixture expert remain separate. Families
and blend weights are selected on an earlier season, never the current failures.
"""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .decision import NAMES, PARAMETERS, fit_components, minute_band, predict_models
from .fusion_data import SEQUENCE_FIELDS
from .storage import atomic_json, digest, now

VERSION = "immediate-statistics-v2"
PARAMS = PARAMETERS | {"n_estimators": 260, "num_leaves": 20, "reg_lambda": 25}
XGB_PARAMS = {
    "n_estimators": 320,
    "max_depth": 4,
    "min_child_weight": 80,
    "reg_lambda": 30,
    "learning_rate": 0.035,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "n_jobs": 1,
    "random_state": 137,
}
WINDOWS = (1, 3, 5, 10)
RECENT = ("expected_goals", "expected_assists", "total_points", "bps", "bonus", "goals_conceded")
SEASON_RATES = (
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "bps",
    "bonus",
    "saves",
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
    "defensive_contribution",
)
FAMILIES = ("form", "advanced", "advanced_price", "advanced_xgb")
WEIGHTS = (0.25, 0.5, 1.0)

TEAM_NAMES = tuple(
    f"{side}_last{window}_{stat}"
    for side in ("own", "opponent")
    for window in (3, 5, 10)
    for stat in ("goals_for", "goals_against", "xg_for", "xg_against", "xg_observed", "matches")
)


class TeamForm:
    """Completed team results and underlying xG, with a real-time cutoff."""

    def __init__(self, originals):
        from .enrichment import epoch

        grouped = defaultdict(list)
        for row in originals:
            if row.get("kickoff_time"):
                grouped[row["season"], int(row["fixture"])].append(row)
        self.history = defaultdict(list)
        for (season, fixture), rows in grouped.items():
            sides = {}
            for home in (True, False):
                part = [r for r in rows if bool(r["was_home"]) == home]
                if not part:
                    continue
                known = all(
                    r.get("expected_goals") not in (None, "")
                    for r in part
                    if float(r.get("minutes") or 0) > 0
                )
                sides[home] = (part[0]["team"], sum(float(r.get("expected_goals") or 0) for r in part), known)
            if len(sides) != 2:
                continue
            exemplar = rows[0]
            at = epoch(exemplar["kickoff_time"]) + 4 * 3600
            goals = {
                True: float(exemplar.get("team_h_score") or 0),
                False: float(exemplar.get("team_a_score") or 0),
            }
            group = f"{season}-GW{int(exemplar['round']):02d}"
            for home in (True, False):
                club, xg, known = sides[home]
                self.history[club].append(
                    (
                        at,
                        group,
                        goals[home],
                        goals[not home],
                        xg,
                        sides[not home][1],
                        float(known and sides[not home][2]),
                    )
                )
        for values in self.history.values():
            values.sort()
        self.dates = {club: [r[0] for r in rows] for club, rows in self.history.items()}

    def vector(self, own, opponent, cutoff, group):
        from .enrichment import epoch

        result = []
        for club in (own, opponent):
            stop = bisect_left(self.dates.get(club, []), epoch(cutoff))
            past = [r for r in self.history.get(club, [])[:stop] if r[1] < group]
            for window in (3, 5, 10):
                part = past[-window:]
                known = [r for r in part if r[-1]]
                result += [float(np.mean([r[j] for r in part])) if part else np.nan for j in (2, 3)]
                result += [float(np.mean([r[j] for r in known])) if known else np.nan for j in (4, 5)]
                result += [len(known) / max(1, len(part)), len(part) / window]
        return result


def team_inputs(store, metadata):
    from .decision import canonical_rows

    originals, _, _ = canonical_rows(store)
    lookup = {(r["season"], int(r["fixture"]), int(r["element"])): r for r in originals}
    index = TeamForm(originals)
    return np.asarray(
        [
            index.vector(
                r["team"],
                lookup[r["season"], r["fixture_id"], r["player_id"]]["opponent_name"],
                r["deadline"],
                r["deadline_group"],
            )
            for r in metadata
        ],
        dtype=np.float32,
    )


def names(family):
    core = NAMES if family == "advanced_price" else [n for n in NAMES if n != "price_lag"]
    recent = [
        f"last{w}_{n}"
        for w in WINDOWS
        for n in ("minutes", "starts", *RECENT, *("known_" + n for n in RECENT))
    ]
    advanced = [n for stat in SEASON_RATES for n in ("season_" + stat + "90", "known_" + stat)]
    advanced += [
        "season_minutes",
        "season_starts",
        "published_form",
        "published_ppg",
        "penalty_first",
        "freekick_first",
        "corner_first",
    ]
    from .enrichment import FEATURES

    return (
        core
        + recent
        + list(TEAM_NAMES)
        + ([] if family == "form" else advanced + ["shot_" + n for n in FEATURES])
    )


def matrix(arrays, packets, family):
    if family not in FAMILIES:
        raise ValueError("Unsupported immediate-scoring feature family")
    tab = arrays["tab"]
    core = tab if family == "advanced_price" else np.delete(tab, NAMES.index("price_lag"), axis=1)
    seq, mask = arrays["sequence"].astype(np.float32), arrays["sequence_mask"]
    fields = {name: (i, scale) for i, (name, scale) in enumerate(SEQUENCE_FIELDS)}
    recent = []
    for w in WINDOWS:
        active = mask[:, -w:]
        count = active.sum(1)
        minutes = seq[:, -w:, 0] * 90
        exposure = (minutes * active).sum(1) / 90
        recent += [np.divide(exposure, count, out=np.zeros(len(tab)), where=count > 0)]
        recent += [(seq[:, -w:, 1] * active).sum(1) / np.maximum(count, 1)]
        for name in RECENT:
            j, scale = fields[name]
            known = active & (seq[:, -w:, j + len(SEQUENCE_FIELDS)] > 0)
            values = seq[:, -w:, j] * scale
            # Per-appearance points includes observed zero-minute matches; rate
            # statistics use only minutes with an observed provider value.
            denominator = known.sum(1) if name == "total_points" else (minutes * known).sum(1) / 90
            recent.append((values * known).sum(1) / np.maximum(denominator, 1))
        for name in RECENT:
            j, _ = fields[name]
            recent.append(
                (active & (seq[:, -w:, j + len(SEQUENCE_FIELDS)] > 0)).sum(1) / np.maximum(count, 1)
            )
    columns = [core, np.column_stack(recent), arrays["team_form"]]
    if family != "form":
        season = []
        for raw in packets:
            raw = raw or {}
            minutes = float(raw.get("minutes") or 0)
            exposure = max(1, minutes / 90)
            values = []
            for key in SEASON_RATES:
                known = raw.get(key) not in (None, "")
                values += [float(raw[key]) / exposure if known else np.nan, float(known)]
            values += [
                minutes / 90,
                float(raw.get("starts") or 0),
                float(raw.get("form") or 0),
                float(raw.get("points_per_game") or 0),
            ]
            values += [
                float(raw.get(key) == 1)
                for key in (
                    "penalties_order",
                    "direct_freekicks_order",
                    "corners_and_indirect_freekicks_order",
                )
            ]
            season.append(values)
        if "enrichment" not in arrays:
            raise ValueError("Advanced scoring requires masked shot enrichment, not invented zero statistics")
        columns += [np.asarray(season), arrays["enrichment"]]
    result = np.column_stack(columns).astype(np.float32)
    if result.shape[1] != len(names(family)):
        raise ValueError("Immediate-scoring feature width mismatch")
    return result


def source_packets(store, metadata):
    from .consensus import replay_packets
    from .context_archive import load_context

    context = load_context(store)
    # The reduced historical news archive did not preserve advanced fields.
    # Full deadline-pinned registries already have both capture and Git proof.
    for (season, event), record in replay_packets(store).items():
        if digest(record["bootstrap"]) != record["bootstrap_hash"]:
            raise ValueError("Archived scoring bootstrap checksum mismatch")
        for player in record["bootstrap"]["elements"]:
            if player.get("code"):
                context[season, event, int(player["code"])] = player
    return [context.get((r["season"], r["event"], int(r["code"]))) for r in metadata]


def conditional(head, x, offsets=None):
    result = np.zeros((len(x), 4))
    for band in (1, 2, 3):
        values = np.column_stack([x, np.full(len(x), band)])
        predicted = (
            head.predict(values, num_threads=1) if isinstance(head, lgb.Booster) else head.predict(values)
        )
        result[:, band] = predicted + (offsets or {}).get(str(band), 0)
    return result


def fit(x, arrays, train, cal, family="advanced"):
    bands = minute_band(arrays["minutes"])
    appeared = train[bands[train] > 0]
    if family == "advanced_xgb":
        from xgboost import XGBRegressor

        head = XGBRegressor(**XGB_PARAMS)
    else:
        head = lgb.LGBMRegressor(**PARAMS)
    head.fit(np.column_stack([x[appeared], bands[appeared]]), arrays["base_target"][appeared])
    predicted = conditional(head, x[cal])
    offsets = {}
    for band in (1, 2, 3):
        mask = bands[cal] == band
        residual = arrays["base_target"][cal][mask] - predicted[mask, band]
        offsets[str(band)] = float(residual.sum() / (len(residual) + 100))
    return head, offsets


def interval(rows, field, repeats=2000):
    groups = defaultdict(list)
    for row in rows:
        groups[row["deadline_group"]].append(row[field])
    values = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(683631)
    samples = values[rng.integers(0, len(values), (repeats, len(values)))].mean(1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def metrics(rows, field):
    errors = np.array([r[field] - r["actual"] for r in rows])
    relevant = np.array([r["prior_start"] >= 0.6 for r in rows])
    premium = np.array([r["price_lag"] >= 90 for r in rows]) & relevant
    groups = defaultdict(list)
    for row in rows:
        groups[row["deadline_group"], row["player_id"]].append(row)
    events = defaultdict(list)
    for (group, player), part in groups.items():
        events[group].append((sum(r[field] for r in part), -player, sum(r["actual"] for r in part)))
    captains = sum(max(part)[2] for part in events.values())
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "regular_rmse": float(np.sqrt(np.mean(errors[relevant] ** 2))),
        "premium_rmse": float(np.sqrt(np.mean(errors[premium] ** 2))) if premium.any() else None,
        "bias": float(errors.mean()),
        "captain_points": float(captains),
    }


def validate(store):
    from .fusion import load_data
    from .role_minutes import VERSION as ROLE_VERSION
    from .role_minutes import fit_head, historical_inputs, predict

    arrays, meta, features = load_data(store)
    arrays["team_form"] = team_inputs(store, meta)
    packets = source_packets(store, meta)
    context_id = store.get("context_archive")["id"]
    role = historical_inputs(store, arrays, meta, features, context_id)
    matrices = {family: matrix(arrays, packets, family) for family in FAMILIES}
    groups = np.array([r["deadline_group"] for r in meta])
    unique = sorted(set(groups))
    frame = pl.DataFrame(arrays["tab"], schema=NAMES).with_columns(
        pl.Series("minutes_target", arrays["minutes"]),
        pl.Series(
            "start_target",
            [float(v) if known else None for v, known in zip(arrays["start"], arrays["start_mask"])],
        ),
        pl.Series("base_target", arrays["base_target"]),
    )
    key = digest(
        {
            "version": VERSION,
            "features": features["id"],
            "context": context_id,
            "parameters": PARAMS,
            "xgb_parameters": XGB_PARAMS,
            "reference_role_version": ROLE_VERSION,
            "families": FAMILIES,
            "columns": {f: names(f) for f in FAMILIES},
            "packet_hash": digest(packets),
        }
    )
    directory = store.root / "models" / ("immediate-validation-" + key[:24])
    directory.mkdir(parents=True, exist_ok=True)
    rows, folds = [], []
    for season in ("2024/25", "2025/26"):
        testing = sorted(g for g in unique if g.startswith(season))
        for first in range(0, len(testing), 8):
            block = testing[first : first + 8]
            checkpoint = directory / (block[0].replace("/", "-") + ".parquet")
            cut = [g for g in unique if g < block[0]][-8]
            train = np.flatnonzero(groups < cut)
            cal = np.flatnonzero((groups >= cut) & (groups < block[0]))
            test = np.flatnonzero(np.isin(groups, block))
            folds.append(
                {
                    "test": block,
                    "fit_max": str(max(groups[train])),
                    "calibration_max": str(max(groups[cal])),
                    "training_rows": len(train),
                    "calibration_rows": len(cal),
                }
            )
            if checkpoint.exists():
                rows.extend(pl.read_parquet(checkpoint).to_dicts())
                print("Reused immediate-scoring fold", block[0], flush=True)
                continue
            reference = fit_components(frame[train], frame[cal])
            _, _, rc = predict_models(reference, arrays["tab"][test], arrays["dc_prior"][test])
            head, curves = fit_head(role, arrays["minutes"], train, cal)
            probability = predict(head, curves, role[test], arrays["news"][test])
            reference_mean = (probability * (rc + 2 * arrays["dc_prior"][test])).sum(1)
            candidates = {}
            for family, x in matrices.items():
                head, offsets = fit(x, arrays, train, cal, family)
                cc = conditional(head, x[test], offsets)
                for weight in WEIGHTS:
                    candidates[f"{family}_{weight}"] = (
                        probability * ((1 - weight) * rc + weight * cc + 2 * arrays["dc_prior"][test])
                    ).sum(1)
            part = []
            for j, i in enumerate(test):
                item = {
                    k: meta[i][k]
                    for k in (
                        "season",
                        "event",
                        "deadline_group",
                        "player_id",
                        "fixture_id",
                        "position",
                        "prior_start",
                        "price_lag",
                    )
                }
                item.update(actual=float(arrays["target"][i]), reference=float(reference_mean[j]))
                item.update({name: float(values[j]) for name, values in candidates.items()})
                part.append(item)
            pl.DataFrame(part).write_parquet(checkpoint)
            rows.extend(part)
            print("Validated immediate-scoring fold", block[0], "through", block[-1], flush=True)
    selection = [r for r in rows if r["season"] == "2024/25" and r["event"] <= 19]
    fields = [f"{f}_{w}" for f in FAMILIES for w in WEIGHTS]

    def loss(field):
        return float(
            np.mean(
                [(1 + 4 * float(r["prior_start"] >= 0.6)) * (r[field] - r["actual"]) ** 2 for r in selection]
            )
        )

    selected = min(fields, key=loss)
    periods = {}
    for label, period in (
        ("acceptance_2024", [r for r in rows if r["season"] == "2024/25" and r["event"] > 19]),
        ("replication_2025", [r for r in rows if r["season"] == "2025/26"]),
    ):
        for r in period:
            r["squared_delta"] = (r[selected] - r["actual"]) ** 2 - (r["reference"] - r["actual"]) ** 2
        periods[label] = {
            "rows": len(period),
            "reference": metrics(period, "reference"),
            "candidate": metrics(period, selected),
            "mse_delta_95ci": interval(period, "squared_delta"),
        }
    forecast_passed = all(
        p["mse_delta_95ci"][1] < 0
        and p["candidate"]["regular_rmse"] <= p["reference"]["regular_rmse"]
        and p["candidate"]["premium_rmse"] <= p["reference"]["premium_rmse"]
        for p in periods.values()
    )
    captain_passed = all(
        p["candidate"]["captain_points"] >= p["reference"]["captain_points"] for p in periods.values()
    )
    path = directory / "predictions.parquet"
    pl.DataFrame(rows).write_parquet(path)
    report = {
        "at": now(),
        "version": VERSION,
        "reference_role_version": ROLE_VERSION,
        "reference_decision_model_id": store.get("decision_model")["id"],
        "feature_id": features["id"],
        "context_id": context_id,
        "key": key,
        "selected": selected,
        "selection_losses": {f: loss(f) for f in fields},
        "metrics": periods,
        "forecast_passed": bool(forecast_passed),
        "captain_passed": bool(captain_passed),
        "passed": bool(forecast_passed and captain_passed),
        "folds": folds,
        "prediction_path": str(path),
        "prediction_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "protocol": "Eight-deadline expanding folds with eight earlier calibration deadlines. Family/blend selected on 2024/25 GW1-19; acceptance GW20-38 and replication all 2025/26. No current-season outcomes select the configuration. Participation and DefCon held identical; final weights never score their own history.",
    }
    report["id"] = store.put("immediate_points_validation", report)
    atomic_json(Path("artifacts/evaluations/immediate_points_validation.json"), report)
    return report
