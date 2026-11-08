"""Execute the actual refresh/retraining workflow in a disposable real-data copy.

Retained pre-GW9 weights initialize the copy. Official endpoints supply today's
finalized observations. No training, forecast or solver success is mocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

from fpl_intelligence.composed_replay import attach_role
from fpl_intelligence.consensus import replay_packets
from fpl_intelligence.decision import load_bundle
from fpl_intelligence.future_expert import load_expert
from fpl_intelligence.model_refresh import observation_signature, training_needed
from fpl_intelligence.pipeline_replay import install_feature_view, link_file
from fpl_intelligence.search_control import search_session
from fpl_intelligence.settings import load_environment
from fpl_intelligence.storage import Store, atomic_json, now
from fpl_intelligence.training import train_current
from fpl_intelligence.training_status import status
from fpl_intelligence.workflow import refresh_cycle
from fpl_intelligence.worlds import load_worlds


def clone(parent, target):
    stage = Store(target)
    # Retain all research manifests and dated inputs; omit superseded huge
    # forecast payloads and shot-cache duplicates from this disposable copy.
    with parent.db() as source, stage.db() as db:
        for table in ("objects", "snapshots"):
            query = "SELECT * FROM " + table
            if table == "objects":
                query += " WHERE kind NOT IN ('forecast','scenario_forecast','component_forecast','top_manager_history','shot_archive') OR rowid IN (SELECT max(rowid) FROM objects WHERE kind IN ('forecast','top_manager_history','shot_archive') GROUP BY kind)"
            for row in source.execute(query):
                db.execute(
                    "INSERT OR IGNORE INTO " + table + " VALUES(" + ",".join("?" for _ in row) + ")",
                    tuple(row),
                )
    for category in ("raw", "features", "models", "scenarios", "normalized"):
        for source in (parent.root / category).rglob("*"):
            if source.is_file():
                link_file(source, stage.root / source.relative_to(parent.root))
    return stage


def run(resume=False):
    load_environment()
    parent = Store("data")
    pointers = {
        k: parent.get(k)["id"] for k in ("state", "market", "decision_model", "forecast", "selected_plan")
    }
    directory = Path("artifacts/runtime/learning-workflow-0.14.4")
    if directory.exists() and not resume:
        raise ValueError("An existing verification directory must be reviewed before reuse")
    stage = Store(directory / "data") if resume else clone(parent, directory / "data")
    packets = replay_packets(parent)
    baseline = parent.get("pipeline_replay")
    row = next(
        r
        for r in baseline["weeks"]
        if r["season"] == "2026/27" and r["event"] == 9 and r["policy"] == "primary_multiweek"
    )
    frozen = json.loads(parent.artifact_path(row["decision_file"]).read_text())
    model = {k: v for k, v in parent.get("decision_model").items() if k != "id"}
    model.update({k: v for k, v in frozen["model"].items() if k != "id"})
    model.update(at=now(), completed_labels_through="2026/27-GW08")
    source = parent.get("decision_features", parent.get("decision_model")["feature_id"])
    earlier = pl.read_parquet(parent.artifact_path(source["path"])).filter(
        pl.col("deadline_group") < "2026/27-GW09"
    )
    model["feature_id"] = install_feature_view(stage, earlier, source)["id"]
    for key, item in frozen["model_artifacts"].items():
        link_file(
            parent.artifact_path(item["path"]), stage.root / "models" / "primary-reference" / (key + ".txt")
        )
    # All model paths are remapped into the copy by Store.artifact_path.
    model["id"] = stage.put("decision_model", model)
    weekly = next(
        r for r in parent.get("role_weekly_replay")["weeks"] if r["season"] == "2026/27" and r["event"] == 9
    )
    model, role = attach_role(parent, stage, weekly, model)
    composed = parent.get("composed_policy_replay")["weeks"][-1]
    body = json.loads(parent.artifact_path(composed["decision_file"]).read_text())
    future = body["future_model"] | {
        "acceptance_id": parent.get("future_policy_replay")["id"],
        "distribution_validation_id": parent.get("future_distribution_validation")["id"],
        "policy_gate_passed": True,
    }
    future["id"] = stage.put("future_horizon_model", future)
    model = {k: v for k, v in model.items() if k != "id"} | {"future_horizon_model_id": future["id"]}
    model["id"] = stage.put("decision_model", model)
    packet = packets["2026/27", 9]
    market = stage.get("market")
    if not resume:
        stage.put(
            "market",
            {k: v for k, v in market.items() if k != "id"} | {"events": packet["bootstrap"]["events"]},
        )
    state = parent.get("state")
    stage.put("state", {k: v for k, v in state.items() if k != "id"} | {"entry_id": None})
    # Validate actual retained weights before the live refresh.
    load_bundle(stage, model)
    load_expert(stage, future)
    old = {
        "core": model["id"],
        "role": model["role_minutes_model_id"],
        "future": future["id"],
        "role_calibrated_through": role["completed_labels_through"],
    }
    print("Real official refresh starting", old, flush=True)
    with search_session(unlimited=False):
        result = refresh_cycle(stage)
    print("Real refresh finished", result, flush=True)
    updated = stage.get("decision_model")
    assert updated["id"] != old["core"]
    assert updated["role_minutes_model_id"] != old["role"]
    assert updated["future_horizon_model_id"] != old["future"]
    assert not result.get("training_warning"), result
    assert result.get("forecast_id") and result.get("plan_id"), result
    forecast = stage.get("forecast", result["forecast_id"])
    assert forecast["decision_model_id"] == updated["id"]
    from fpl_intelligence.forecast_validation import validate_forecast

    with load_worlds(stage, forecast) as worlds:
        shape = worlds[f"points_{forecast['start_event']}"].shape
        assert shape[1] == len(forecast["player_ids"])
        validation = validate_forecast(forecast, worlds)
    observations = observation_signature(stage)
    assert not training_needed(stage, updated, observations)
    counts = {
        k: len(stage.list(k, 10000)) for k in ("decision_model", "role_minutes_model", "future_horizon_model")
    }
    reused = train_current(stage)
    assert reused["id"] == updated["id"]
    assert counts == {k: len(stage.list(k, 10000)) for k in counts}
    assert pointers == {k: parent.get(k)["id"] for k in pointers}
    report = {
        "at": now(),
        "status": "passed",
        "source": "Actual official live endpoints and retained pre-GW9 model artifacts; no mocked fitting, forecasts or optimization",
        "prior": old,
        "refresh": result,
        "training": status(stage),
        "unchanged_data_additional_fits": 0,
        "production_pointers_unchanged": True,
        "forecast_validation": validation,
        "world_shape": shape,
        "directory": str(directory),
        "budget": "Actual normal refresh forecasting and chip-window planning; explicitly bounded historical-verification search, live UI/CLI remain unlimited and cancellable",
    }
    atomic_json(Path("artifacts/evaluations/learning_workflow_0_14_4.json"), report)
    return report


if __name__ == "__main__":
    run(resume="--resume" in sys.argv)
