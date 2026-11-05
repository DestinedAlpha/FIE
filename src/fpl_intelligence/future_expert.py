"""A gated future-fixture expert; the immediate distribution stays unchanged."""

from __future__ import annotations

import hashlib

import lightgbm as lgb
import numpy as np
import polars as pl

from .consensus import mix_distributions
from .fixture_horizon import FEATURES, feature_vector, predict
from .models import curve_apply


def load_expert(store, spec):
    if list(FEATURES) != spec["features"]:
        raise ValueError("Future expert feature contract changed; refit before serving")
    models = {}
    for name, item in spec["models"].items():
        path = store.artifact_path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["hash"]:
            raise ValueError("Future expert model checksum mismatch")
        models[name] = lgb.Booster(model_file=str(path))
    return {**spec, **models}


def combine(store, decision_model, rows, scheduled, market, players, p, c, starts, start_event, cutoff):
    identity = decision_model.get("future_horizon_model_id")
    included = np.zeros(len(rows), dtype=bool)
    if not identity:
        return p, c, starts, included
    spec = store.get("future_horizon_model", identity)
    if not spec["forecast_gate_passed"] or not 0 < spec["weight"] <= 1:
        raise ValueError("Future expert lacks a valid forecast gate")
    if not spec.get("policy_gate_passed") and not store.list("replay_context", 1):
        raise ValueError("Future expert remains experimental until the full policy gate passes")
    if spec.get("policy_gate_passed"):
        distribution = store.get("future_distribution_validation", spec.get("distribution_validation_id"))
        if not spec.get("distribution_validation_id") or not distribution["passed"]:
            raise ValueError("Future expert requires accepted distribution validation")
    indices = [i for i, r in enumerate(rows) if 0 < r["event"] - start_event <= spec["max_lead"]]
    if not indices:
        return p, c, starts, included
    bundle = load_expert(store, spec)
    teams = {t["id"]: t for t in market["teams"]}
    fixtures = {f["id"]: f for f in scheduled}
    matrix = []
    factors = []
    for i in indices:
        row = rows[i]
        player = players[row["player_id"]]
        changed = not np.isclose(row["availability"], row.get("base_availability", row["availability"]))
        intervention = changed or row.get("suspension_served", False)
        if intervention:
            raw = {**player.raw, "status": "a", "chance_of_playing_next_round": 100, "ep_next": None}
            player = player.model_copy(update={"raw": raw, "status": "a", "chance": 1.0})
        matrix.append(
            feature_vector(player, row, fixtures[row["fixture_id"]], scheduled, teams, start_event, cutoff)
        )
        factors.append(row["availability"] if intervention else (0.0 if row["availability"] == 0 else 1.0))
    frame = pl.DataFrame(np.asarray(matrix, dtype=np.float32), schema=list(FEATURES)).with_columns(
        pl.Series("dc_prior", [rows[i]["dc_probability"] for i in indices])
    )
    _, hp, hc = predict(bundle, frame)
    hp[:, 1:] *= np.asarray(factors)[:, None]
    hp[:, 0] = 1 - hp[:, 1:].sum(1)
    hs = curve_apply(
        bundle["starter"].predict(frame.select(FEATURES).to_numpy(), num_threads=1), spec["curves"]["start"]
    )
    hs = np.minimum(hs * np.asarray(factors), 1 - hp[:, 0])
    selected = np.asarray(indices)
    rp = p[selected].copy()
    available = np.asarray([rows[i]["availability"] for i in indices])
    rp[:, 1:] *= available[:, None]
    rp[:, 0] = 1 - rp[:, 1:].sum(1)
    mixed_p, mixed_c, mixed_s = mix_distributions(
        (rp, c[selected], starts[selected] * available), (hp, hc, hs), spec["weight"]
    )
    p = p.copy()
    c = c.copy()
    starts = starts.copy()
    p[selected], c[selected], starts[selected] = mixed_p, mixed_c, mixed_s
    included[selected] = True
    return p, c, starts, included


def frozen_spec(store, refinement, before):
    """A frozen earlier-period expert for a research replay, never live promotion."""
    import json

    if not refinement["forecast_gate_passed"]:
        raise ValueError("Future forecast acceptance failed")
    family, weight = refinement["selected"].rsplit("_", 1)
    directory = store.artifact_path(refinement["directory"] + "/protocol.json").parent
    folder = directory / (before.replace("/", "-") + "-" + family)
    body = json.loads((folder / "model.json").read_text())
    models = {
        name: {"path": str(folder / (name + ".txt")), "hash": checksum}
        for name, checksum in body["hashes"].items()
    }
    return {
        "version": "future-horizon-v1",
        "features": list(FEATURES),
        "models": models,
        "curves": body["curves"],
        "weight": float(weight),
        "max_lead": 18,
        "fit_before": before,
        "training_evidence": {
            k: body[k] for k in ("fit_max_label", "cal_max_label", "cut", "training_rows", "calibration_rows")
        },
        "refinement_id": refinement["id"],
        "forecast_gate_passed": True,
        "policy_gate_passed": False,
        "scope": "Future fixtures only; current-gameweek reference probabilities and worlds are preserved",
    }
