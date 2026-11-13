"""Refit an accepted future-fixture expert on newly completed deadlines."""

from __future__ import annotations

import hashlib
import inspect
import json

import polars as pl
from threadpoolctl import threadpool_limits

from .availability import moment
from .fixture_horizon import DATA_CONTRACT, FEATURES, PARAMS, build_data, fit_models, partitions
from .future_expert import load_expert
from .storage import atomic_json, digest, now


def refit_expert(store, acceptance_id=None):
    """Only a passed earlier-season full-policy comparison can authorize refits."""
    acceptance = store.get("future_policy_replay", acceptance_id)
    market = store.get("market")
    if not acceptance["policy_gate_passed"] or acceptance["season"] >= market["season"]:
        raise ValueError("Future expert requires accepted earlier-season policy evidence")
    if (
        acceptance["verification"]["sealed_deadlines"] != len(acceptance["weeks"])
        or len(acceptance["weeks"]) < 19
    ):
        raise ValueError("Future expert requires the complete independently checked policy window")
    refinement = store.get("horizon_refinement", acceptance["protocol"]["refinement_id"])
    if not refinement["forecast_gate_passed"]:
        raise ValueError("Future expert forecast gate failed")
    distribution = store.get("future_distribution_validation")
    if (
        not distribution["passed"]
        or distribution["protocol"]["refinement_id"] != refinement["id"]
        or distribution["protocol"]["baseline_id"] != acceptance["protocol"]["baseline_id"]
    ):
        raise ValueError("Future expert requires matching marginal distribution acceptance")
    family, weight = refinement["selected"].rsplit("_", 1)
    event = next((e for e in market["events"] if e.get("is_next")), None)
    if event is None:
        raise ValueError("No upcoming deadline for a future expert refit")
    from .observations import information_time

    cutoff = information_time(store)
    replay = bool(store.list("replay_context", 1))
    if moment(cutoff) > moment(event["deadline_time"]) or (
        not replay and moment(cutoff) == moment(event["deadline_time"])
    ):
        raise ValueError("The upcoming deadline has passed; refresh before refitting")
    before = f"{market['season']}-GW{event['id']:02d}"
    from .search_control import report_phase

    report_phase("Preparing future-fixture training data")
    path = build_data(store)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pl.read_parquet(path)
    known = frame.filter(
        (pl.col("source_group") < before)
        & (pl.col("label_end") < before)
        & (pl.col("label_known") < moment(cutoff).timestamp())
    )
    completed = (
        known.select("source_group", "label_end", "label_known")
        .unique()
        .sort("source_group", "label_end", "label_known")
        .to_dicts()
    )
    signature = digest(
        {
            "data": checksum,
            "before": before,
            "completed": completed,
            "family": family,
            "parameters": PARAMS,
            "acceptance": acceptance["id"],
            "distribution_validation_id": distribution["id"],
            "features": FEATURES,
            "training_method": [inspect.getsource(fit_models), inspect.getsource(partitions)],
        }
    )
    directory = store.root / "models" / ("future-refit-" + signature[:20])
    manifest = directory / "model.json"
    if manifest.exists():
        spec = json.loads(manifest.read_text())
        load_expert(store, spec)
    else:
        report_phase("Fitting the accepted future-fixture model")
        with threadpool_limits(limits=1):
            bundle = fit_models(frame, before, family, moment(cutoff).timestamp())
        directory.mkdir(parents=True, exist_ok=True)
        models = {}
        for name in ("minutes", "points", "starter"):
            target = directory / (name + ".txt")
            from .models import save_booster_atomic

            save_booster_atomic(bundle[name].booster_, target)
            models[name] = {"path": str(target), "hash": hashlib.sha256(target.read_bytes()).hexdigest()}
        spec = {
            "version": "future-horizon-v1",
            "data_contract": DATA_CONTRACT,
            "family": family,
            "parameters": PARAMS,
            "features": list(FEATURES),
            "models": models,
            "curves": bundle["curves"],
            "weight": float(weight),
            "max_lead": 18,
            "fit_before": before,
            "fit_as_of": cutoff,
            "data_path": str(path),
            "data_hash": checksum,
            "training_evidence": {
                k: bundle[k]
                for k in ("fit_max_label", "cal_max_label", "cut", "training_rows", "calibration_rows")
            },
            "refinement_id": refinement["id"],
            "acceptance_id": acceptance["id"],
            "distribution_validation_id": distribution["id"],
            "forecast_gate_passed": True,
            "policy_gate_passed": True,
            "scope": "Future fixtures through an 18-week lead; unchanged immediate reference and shared residual worlds",
            "learning": "Refit minutes, conditional points and starts on completed earlier labels, with eight purged calibration deadlines; fixed accepted family and blend weight",
        }
        atomic_json(manifest, spec)
    return spec | {"id": store.put("future_horizon_model", spec)}


def activate_expert(store, acceptance_id=None):
    """Fit and connect an accepted expert; callers then regenerate forecast/plan."""
    from .future_policy_verification import verify_future_policy

    accepted = store.get("future_policy_replay", acceptance_id)
    if not accepted["policy_gate_passed"]:
        raise ValueError("Future expert policy gate failed; reference retained")
    verify_future_policy(store, accepted)
    original = store.get("decision_model")
    if original.get("fusion_model_id"):
        raise ValueError("This acceptance does not certify a different immediate fusion family")
    expert = refit_expert(store, accepted["id"])
    if store.get("decision_model")["id"] != original["id"]:
        raise ValueError("The primary model changed during the future expert fit")
    model = {k: v for k, v in original.items() if k != "id"}
    model["future_horizon_model_id"] = expert["id"]
    model["id"] = store.put("decision_model", model)
    bind_immediate_audits(store, model)
    return model


def repair_data_contract(store):
    """Refit the existing accepted family after a confirmed target/feature defect.

    Preserve the entire immediate head and every original artifact. Publication
    of the repaired future weights and composed model is one guarded transaction.
    """
    if not store.list("decision_model", 1):
        return None
    original = store.get("decision_model")
    identity = original.get("future_horizon_model_id")
    if not identity or not original.get("promoted"):
        return None
    previous = store.get("future_horizon_model", identity)
    if previous.get("data_contract") == DATA_CONTRACT:
        return None
    from .model_refresh import observation_signature
    from .search_control import check_cancelled, report_phase
    from .training import DeferredModelStore

    report_phase("Repairing future scoring targets and missing team-strength inputs")
    before = observation_signature(store)
    staged = DeferredModelStore(store)
    expert = refit_expert(staged, previous["acceptance_id"])
    if expert.get("data_contract") != DATA_CONTRACT:
        raise ValueError("Future refit did not repair its training-data contract")
    model = {k: v for k, v in original.items() if k != "id"}
    model["future_horizon_model_id"] = expert["id"]
    model["id"] = staged.put("decision_model", model)
    if observation_signature(store) != before:
        raise ValueError("Football observations changed during scoring repair; old model retained")
    staged.put(
        "model_training_state",
        {
            "at": now(),
            "decision_model_id": model["id"],
            "observations": before,
            "refitted": True,
            "policy": "existing accepted family; repaired future scoring inputs",
        },
    )
    check_cancelled()
    staged.publish(original["id"])
    bind_immediate_audits(store, model)
    result = {
        "at": now(),
        "data_contract": DATA_CONTRACT,
        "previous_decision_model_id": original["id"],
        "decision_model_id": model["id"],
        "previous_future_model_id": identity,
        "future_model_id": expert["id"],
        "immediate_weights_changed": False,
        "correction": "Canonical base_target used once; unknown provider strengths masked",
        "evaluation_scope": "Existing accepted family with corrected data; preceding historical scores are retained as original evidence, not relabelled as evaluations of repaired weights",
    }
    result["id"] = store.put("future_data_repair", result)
    return result


def bind_immediate_audits(store, model):
    """Reuse executed immediate-head audits only when that entire head is identical."""

    def strip(body):
        return {k: v for k, v in body.items() if k not in ("id", "future_horizon_model_id")}

    rebound = {}
    for kind in ("decision_audit", "decision_policy", "decision_uncertainty", "manager_audit"):
        if not store.list(kind, 1):
            continue
        report = store.get(kind)
        original_id = report.get("decision_model_id")
        if not original_id or original_id == model["id"]:
            continue
        original = store.get("decision_model", original_id)
        if strip(original) != strip(model):
            continue
        body = {k: v for k, v in report.items() if k != "id"}
        body.update(
            decision_model_id=model["id"],
            reused_from=report["id"],
            compatibility_verified_at=now(),
            compatibility_scope="Executed audit of the identical immediate reference head; future expert has separate forecast, distribution and policy validation",
        )
        rebound[kind] = store.put(kind, body)
    return rebound


def train_future_component(store):
    """Run all acceptance stages when absent, otherwise refit the frozen winner."""
    from .future_policy_replay import run_future_policy
    from .horizon_refinement import run_refinement
    from .horizon_trial import run_trial
    from .pipeline_jobs import run_pipeline_job

    if not store.list("horizon_trial", 1):
        run_trial(store)
    refinement = (
        store.get("horizon_refinement") if store.list("horizon_refinement", 1) else run_refinement(store)
    )
    if not refinement["forecast_gate_passed"]:
        return refinement
    if not store.list("pipeline_replay", 1):
        run_pipeline_job(store)
    accepted = store.get("future_policy_replay") if store.list("future_policy_replay", 1) else None
    if accepted is None or accepted["protocol"]["refinement_id"] != refinement["id"]:
        accepted = run_future_policy(store)
    if not accepted["policy_gate_passed"]:
        return accepted
    from .future_distribution import run_distribution_gate

    distribution = run_distribution_gate(store, accepted["protocol"]["baseline_id"])
    if not distribution["passed"]:
        return distribution
    if store.get("decision_model").get("fusion_model_id"):
        return accepted | {
            "activation": "The separately promoted immediate fusion family requires its own matched acceptance"
        }
    activate_expert(store, accepted["id"])
    return store.get("future_horizon_model", store.get("decision_model")["future_horizon_model_id"])
