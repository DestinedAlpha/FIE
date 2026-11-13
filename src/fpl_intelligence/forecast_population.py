"""Reproducible population diagnostics for the accepted immediate composition.

Research checkpoints never publish model pointers. Calibration labels and test
labels are separated by whole deadlines, including both fixtures of a double.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from .decision import (
    NAMES,
    fit_components,
    minute_band,
    predict_models,
    residual_contract,
    sample_point_outcomes,
)
from .models import curve_apply
from .storage import atomic_json, digest, now

VERSION = "population-v1"
EVIDENCE_ROOT = Path(__file__).resolve().parents[2] / "artifacts/evaluations"


def saved_evidence(store, kind, filename):
    if store.list(kind, 1):
        return store.get(kind)
    # An existing-installation update preserves its DB. Bundled reports make
    # the new research visible without importing foreign accounts or weights.
    import json

    path = EVIDENCE_ROOT / filename
    return json.loads(path.read_text()) if path.is_file() else None


def binary_metrics(actual, probability):
    y, p = np.asarray(actual, float), np.asarray(probability, float)
    if not len(y):
        return None
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid diagnostic probability")
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    bins = []
    for low in np.arange(0, 1, 0.1):
        mask = (p >= low) & (p < low + 0.1 if low < 0.9 else p <= 1)
        if mask.any():
            bins.append(
                {"n": int(mask.sum()), "predicted": float(p[mask].mean()), "observed": float(y[mask].mean())}
            )
    return {
        "n": len(y),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))),
        "bias": float(np.mean(p - y)),
        "ece": sum(b["n"] * abs(b["predicted"] - b["observed"]) for b in bins) / len(y),
        "bins": bins,
    }


def point_metrics(rows, field="reference"):
    if not rows:
        return None
    error = np.array([r[field] - r["actual"] for r in rows])
    return {
        "n": len(rows),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
    }


def gameweek_rows(rows):
    """Aggregate means and outcomes before captain/DGW comparisons."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["deadline_group"], r["player_id"]].append(r)
    return [
        {
            "deadline_group": group,
            "player_id": pid,
            "fixture_count": len(part),
            "actual": sum(r["actual"] for r in part),
            "reference": sum(r["reference"] for r in part),
        }
        for (group, pid), part in groups.items()
    ]


def report(rows):
    masks = {"all": rows}
    masks.update({f"position_{p}": [r for r in rows if int(r["position"]) == p] for p in range(1, 5)})
    masks.update(
        {
            "regular": [r for r in rows if r["prior_start"] >= 0.6],
            "rotation": [r for r in rows if 0.1 <= r["prior_start"] < 0.6],
            "premium": [r for r in rows if r["price_lag"] >= 90],
            "home": [r for r in rows if r["home"]],
            "away": [r for r in rows if not r["home"]],
            "low_history": [r for r in rows if r["exposure"] < 3],
        }
    )
    for key, values in (
        ("team_group", ("strong", "middle", "weak")),
        ("fixture_difficulty", ("easy", "middle", "hard")),
    ):
        for value in values:
            masks[f"{key}_{value}"] = [r for r in rows if r.get(key) == value]
    masks["promoted_clubs"] = [r for r in rows if r.get("promoted_club")]
    out = {}
    for label, part in masks.items():
        if not part:
            continue
        start = [r for r in part if r["start_known"]]
        out[label] = {
            "points": point_metrics(part),
            "any": binary_metrics([r["minutes"] > 0 for r in part], [1 - r["p0"] for r in part]),
            "sixty": binary_metrics([r["minutes"] >= 60 for r in part], [r["p2"] + r["p3"] for r in part]),
            "start": binary_metrics([r["started"] for r in start], [r["p_start"] for r in start]),
            "minute_mae": float(np.mean([abs(r["expected_minutes"] - r["minutes"]) for r in part])),
            "mean_crps": float(np.mean([r["crps"] for r in part])),
            "interval_80_coverage": float(np.mean([r["q10"] <= r["actual"] <= r["q90"] for r in part])),
            "haul": binary_metrics([r["actual"] >= 10 for r in part], [r["p_haul"] for r in part]),
        }
    weekly = gameweek_rows(rows)
    out["gameweeks"] = {
        "all": point_metrics(weekly),
        "double": point_metrics([r for r in weekly if r["fixture_count"] > 1]),
    }
    return out


def decorate_cohorts(rows, metadata, tab):
    """Cohort definitions use only features already known at each deadline."""
    lookup = {(r["season"], r["event"], r["fixture_id"], r["player_id"]): i for i, r in enumerate(metadata)}
    clubs = defaultdict(set)
    for r in metadata:
        clubs[r["season"]].add(r["team"])
    decorated = []
    groups = defaultdict(list)
    for row in rows:
        r = row.copy()
        i = lookup[r["season"], r["event"], r["fixture_id"], r["player_id"]]
        r.update(
            {
                k: float(tab[i, NAMES.index(k)])
                for k in ("team_attack", "team_defence", "opponent_attack", "opponent_defence")
            }
        )
        prior = f"{int(r['season'][:4]) - 1}/{int(r['season'][:4]) % 100:02d}"
        r["promoted_club"] = bool(clubs.get(prior) and r["team"] not in clubs[prior])
        groups[r["deadline_group"]].append(r)
        decorated.append(r)
    for part in groups.values():
        strengths = {r["team"]: r["team_attack"] / max(0.01, r["team_defence"]) for r in part}
        low, high = np.quantile(list(strengths.values()), [0.25, 0.75])
        difficulties = [r["opponent_attack"] / max(0.01, r["opponent_defence"]) for r in part]
        # Each opponent is represented once, preventing squad-size weighting.
        easy, hard = np.quantile(sorted(set(difficulties)), [1 / 3, 2 / 3])
        for r, difficulty in zip(part, difficulties):
            r["team_group"] = (
                "strong"
                if strengths[r["team"]] > high
                else "weak"
                if strengths[r["team"]] < low
                else "middle"
            )
            r["fixture_difficulty"] = (
                "hard" if difficulty > hard else "easy" if difficulty < easy else "middle"
            )
    return decorated


def predictions(store):
    from .fusion import load_data
    from .models import save_booster_atomic
    from .role_minutes import VERSION as ROLE_VERSION
    from .role_minutes import fit_head, historical_inputs, predict

    arrays, meta, features = load_data(store)
    validation = store.get("role_minutes_validation")
    if (
        validation["feature_id"] != features["id"]
        or validation["version"] != ROLE_VERSION
        or validation["weight"] != 1
    ):
        raise ValueError("Diagnostics require the accepted role-v3 composition and pinned features")
    context = validation["context_archive_id"]
    contract = {
        "version": VERSION,
        "feature_id": features["id"],
        "role_validation": validation["id"],
        "context": context,
        "draws": 512,
        "seed": 7381,
    }
    directory = store.root / "models" / ("population-" + digest(contract)[:24])
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "protocol.json", contract)
    groups = np.array([r["deadline_group"] for r in meta])
    unique = sorted(set(groups))
    role = historical_inputs(store, arrays, meta, features, context)
    frame = (
        pl.DataFrame(meta)
        .select("season", "fixture_id", "team")
        .hstack(pl.DataFrame(arrays["tab"], schema=NAMES))
        .with_columns(
            pl.Series("minutes_target", arrays["minutes"]),
            pl.Series(
                "start_target",
                [float(v) if known else None for v, known in zip(arrays["start"], arrays["start_mask"])],
            ),
            pl.Series("base_target", arrays["base_target"]),
        )
    )
    all_rows, folds = [], []
    for season in ("2024/25", "2025/26", "2026/27"):
        testing = [g for g in unique if g.startswith(season)]
        for first in range(0, len(testing), 8):
            block = testing[first : first + 8]
            cut = [g for g in unique if g < block[0]][-8]
            train = np.flatnonzero(groups < cut)
            cal = np.flatnonzero((groups >= cut) & (groups < block[0]))
            test = np.flatnonzero(np.isin(groups, block))
            fold = {
                "test": block,
                "fit_max": str(max(groups[train])),
                "calibration_min": str(min(groups[cal])),
                "calibration_max": str(max(groups[cal])),
                "training_rows": len(train),
                "calibration_rows": len(cal),
                "test_rows": len(test),
            }
            name = block[0].replace("/", "-")
            checkpoint, receipt = directory / (name + ".parquet"), directory / (name + ".json")
            if checkpoint.exists() and receipt.exists():
                import json

                saved = json.loads(receipt.read_text())
                if (
                    saved["fold"] != fold
                    or saved["prediction_hash"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                ):
                    raise ValueError("Population checkpoint integrity failed")
                for key, value in saved["models"].items():
                    if hashlib.sha256((directory / value["file"]).read_bytes()).hexdigest() != value["hash"]:
                        raise ValueError("Population weight integrity failed: " + key)
                restored = pl.read_parquet(checkpoint).to_dicts()
                for row in restored:
                    row.update({f"m{k}": float(saved["band_minutes"][k]) for k in range(4)})
                all_rows.extend(restored)
                folds.append(saved)
                continue
            ref = fit_components(frame[train], frame[cal])
            _, _, conditional = predict_models(ref, arrays["tab"][test], arrays["dc_prior"][test])
            head, curves = fit_head(role, arrays["minutes"], train, cal)
            p = predict(head, curves, role[test], arrays["news"][test])
            starts = curve_apply(
                ref["starter"].predict_proba(arrays["tab"][test])[:, 1], ref["curves"]["start"]
            )
            starts = np.minimum(starts, 1 - p[:, 0])
            pools, dependence = residual_contract(ref, frame[cal])
            bands = minute_band(arrays["minutes"][cal])
            band_minutes = [0] + [float(np.mean(arrays["minutes"][cal][bands == k])) for k in range(1, 4)]
            rng = np.random.default_rng(contract["seed"] + first + int(season[:4]))
            rows = []
            for j, i in enumerate(test):
                r = meta[i].copy()
                draws, _, _ = sample_point_outcomes(
                    rng,
                    p[j],
                    conditional[j],
                    arrays["dc_prior"][i],
                    pools,
                    r["position"],
                    rng.random(512),
                    rng.random(512),
                )
                sd = np.sort(draws.astype(float))
                actual = float(arrays["target"][i])
                crps = np.abs(sd - actual).mean() - np.dot(2 * np.arange(1, 513) - 513, sd) / 512**2
                r.update(
                    actual=actual,
                    minutes=float(arrays["minutes"][i]),
                    started=float(arrays["start"][i]),
                    start_known=bool(arrays["start_mask"][i]),
                    p_start=float(starts[j]),
                    expected_minutes=float(p[j] @ band_minutes),
                    reference=float(p[j] @ (conditional[j] + 2 * arrays["dc_prior"][i])),
                    home=bool(arrays["tab"][i, NAMES.index("home")]),
                    exposure=float(arrays["tab"][i, NAMES.index("exposure")]),
                    crps=float(crps),
                    q10=float(np.quantile(draws, 0.1)),
                    q90=float(np.quantile(draws, 0.9)),
                    p_haul=float((draws >= 10).mean()),
                )
                r.update({f"p{k}": float(p[j, k]) for k in range(4)})
                r.update({f"m{k}": float(band_minutes[k]) for k in range(4)})
                r.update({f"c{k}": float(conditional[j, k]) for k in range(4)})
                r.update({f"dc{k}": float(arrays["dc_prior"][i, k]) for k in range(4)})
                rows.append(r)
            pl.DataFrame(rows).write_parquet(checkpoint)
            models = {}
            for key, model in {**{k: ref[k] for k in ("minutes", "starter", "points")}, "role": head}.items():
                path = directory / (name + "-" + key + ".txt")
                save_booster_atomic(model.booster_, path)
                models[key] = {"file": path.name, "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
            saved = {
                "fold": fold,
                "models": models,
                "curves": ref["curves"],
                "role_curves": curves,
                "band_minutes": band_minutes,
                "dependence": dependence,
                "prediction_hash": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            atomic_json(receipt, saved)
            all_rows.extend(rows)
            folds.append(saved)
            print("Population probabilities", block[0], block[-1], len(rows), flush=True)
    reference_path = store.artifact_path(validation["prediction_path"])
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != validation["prediction_hash"]:
        raise ValueError("Accepted OOF predictions failed integrity verification")
    reference = pl.read_parquet(reference_path).select(
        "season", "event", "player_id", "fixture_id", "candidate"
    )
    new = pl.DataFrame(all_rows).join(
        reference, on=["season", "event", "player_id", "fixture_id"], how="inner"
    )
    if (
        len(new) != len(reference)
        or new.select("season", "event", "player_id", "fixture_id").is_duplicated().any()
    ):
        raise ValueError("Accepted OOF comparison population differs")
    parity = float(np.max(np.abs(new["candidate"].to_numpy() - new["reference"].to_numpy())))
    if parity > 1e-6:
        raise ValueError(f"Accepted OOF composition parity failed: {parity}")
    decorated = decorate_cohorts(all_rows, meta, arrays["tab"])
    result = {
        "at": now(),
        "version": VERSION,
        "contract": contract,
        "directory": str(directory),
        "folds": folds,
        "accepted_mean_parity_max_absolute": parity,
        "periods": {
            season: report([r for r in decorated if r["season"] == season])
            for season in ("2024/25", "2025/26", "2026/27")
        },
        "scope": "Immediate role-v3 / accepted conditional-points composition; later leads retain their separately gated expert. Component predictions are assessed separately, never invented as decompositions of this direct model.",
    }
    result["id"] = store.put("population_diagnostics", result)
    atomic_json(Path("artifacts/evaluations/population_0_14_3.json"), result)
    return all_rows, result


def monitor(store, forecast, independent=None):
    """Cheap recurring diagnostics for every served player/week and source cohort."""
    from .independent_sources import compare_fivestat
    from .ingestion import load_market

    players, _ = load_market(store, forecast["market_id"])
    groups = defaultdict(list)
    for row in forecast["players"]:
        p = players.get(row["player_id"])
        if p:
            groups[int(p.position), row["event"] - forecast["start_event"]].append(row)
    cohorts = [
        {
            "position": p,
            "lead": lead,
            "players": len(part),
            "mean_points": float(np.mean([r["mean"] for r in part])),
            "mean_minutes": float(np.mean([r["minutes_mean"] for r in part])),
            "mean_haul_probability": float(np.mean([r["p_haul"] for r in part])),
            "minutes_above_match_exposure": sum(
                r["minutes_mean"] > 90 * r["fixture_count"] + 1e-7 for r in part
            ),
        }
        for (p, lead), part in sorted(groups.items())
    ]
    comparison = independent if independent is not None else compare_fivestat(store, forecast)
    disagreements = []
    aligned = defaultdict(list)
    for r in comparison.get("records", []):
        p = players.get(r["player_id"])
        if p is None:
            continue
        reason = (
            "participation_assumptions"
            if abs(r["minutes_difference"]) >= 10
            else "conditional_scoring_assumptions"
        )
        item = {**r, "position": int(p.position), "classification": reason, "observed_accuracy_claim": False}
        aligned[int(p.position), r["event"] - forecast["start_event"]].append(item)
        if abs(r["points_difference"]) >= 1.5:
            disagreements.append(item)
    summaries = [
        {
            "position": p,
            "lead": lead,
            "fixtures": len(part),
            "point_bias": float(np.mean([r["points_difference"] for r in part])),
            "point_mae_disagreement": float(np.mean([abs(r["points_difference"]) for r in part])),
            "minute_bias": float(np.mean([r["minutes_difference"] for r in part])),
            "material_disagreements": sum(abs(r["points_difference"]) >= 1.5 for r in part),
        }
        for (p, lead), part in sorted(aligned.items())
    ]
    historical = saved_evidence(store, "population_diagnostics", "population_0_14_3.json")
    model = store.get("decision_model", forecast["decision_model_id"])
    role = (
        store.get("role_minutes_model", model["role_minutes_model_id"])
        if model.get("role_minutes_model_id")
        else None
    )
    compatible = bool(
        historical
        and role
        and historical["contract"]["role_validation"] == role["validation_id"]
        and historical["contract"]["feature_id"] == role["feature_id"]
    )
    challenger = saved_evidence(store, "football_challenger", "football_challenger_0_14_3.json")
    return {
        "version": VERSION,
        "forecast_id": forecast["id"],
        "decision_model_id": forecast["decision_model_id"],
        "cohorts": cohorts,
        "independent_status": comparison.get("status"),
        "independent_source_id": comparison.get("source_id"),
        "aligned_cohorts": summaries,
        "material_disagreements": sorted(
            disagreements, key=lambda r: abs(r["points_difference"]), reverse=True
        ),
        "historical": {
            "id": historical["id"],
            "compatible": compatible,
            "family_compatible": bool(
                role
                and historical["contract"]["role_validation"] == role["validation_id"]
                and model.get("version") == "learned-points-v1"
            ),
            "historical_feature_id": historical["contract"]["feature_id"],
            "active_feature_id": role["feature_id"] if role else None,
            "periods": historical["periods"],
            "scope": historical["scope"],
        }
        if historical
        else None,
        "challenger": {
            "id": challenger["id"],
            "decision": challenger["decision"],
            "selected": challenger["selected"],
            "periods": challenger["periods"],
            "promoted": challenger["promoted"],
            "component_diagnostics": challenger.get("component_diagnostics"),
        }
        if challenger
        else None,
        "interpretation": "Disagreement flags identify participation versus conditional-scoring assumptions, not evidence that a provider is correct. Direct conditional points do not identify separate xG, CS or bonus contributions. Historical scores apply to pinned OOF compositions, not a future accuracy guarantee.",
    }
