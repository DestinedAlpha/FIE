"""Read-only provenance of the model composition actually in use."""

from functools import lru_cache

import polars as pl


@lru_cache(maxsize=32)
def group_bounds(path, checksum, through=None):
    # Content-addressed metadata only; never load another feature matrix.
    frame = pl.read_parquet(path, columns=["season", "deadline_group"])
    if through:
        frame = frame.filter(pl.col("deadline_group") <= through)
    groups = sorted(frame["deadline_group"].unique().to_list())
    if len(groups) < 9:
        raise ValueError("Insufficient completed groups to establish training bounds")
    cut = groups[-8]
    fit = frame.filter(pl.col("deadline_group") < cut)
    cal = frame.filter(pl.col("deadline_group") >= cut)
    return {
        "fit_min": fit["deadline_group"].min(),
        "fit_max": fit["deadline_group"].max(),
        "calibration_min": cal["deadline_group"].min(),
        "calibration_max": cal["deadline_group"].max(),
        "training_rows": len(fit),
        "calibration_rows": len(cal),
        "seasons": frame.group_by("season").len().sort("season").to_dicts(),
    }


def component_bounds(path, checksum, spec):
    evidence = spec.get("training_evidence") or spec.get("fit_evidence") or {}
    through = spec.get("completed_labels_through") or evidence.get("calibration_max_group")
    bounds = dict(group_bounds(path, checksum, through))
    bounds.update({k: evidence[k] for k in bounds if k in evidence})
    for old, new in (("fit_max_group", "fit_max"), ("calibration_max_group", "calibration_max")):
        if old in evidence:
            bounds[new] = evidence[old]
    return bounds


def status(store):
    if not store.list("decision_model", 1):
        return {"components": [], "status": "Training required", "warning": None}
    model = store.get("decision_model")
    if not model.get("feature_id"):
        return {
            "decision_model_id": model["id"],
            "components": [],
            "status": "Training required",
            "warning": None,
        }
    feature = store.get("decision_features", model["feature_id"])
    components = [
        {
            "name": "Immediate points",
            "id": model["id"],
            "feature_id": feature["id"],
            **component_bounds(
                str(store.artifact_path(feature["path"])), feature.get("file_hash", feature["hash"]), model
            ),
        }
    ]
    if model.get("role_minutes_model_id"):
        role = store.get("role_minutes_model", model["role_minutes_model_id"])
        features = store.get("fusion_features", role["feature_id"])
        components.append(
            {
                "name": "Role and minutes",
                "id": role["id"],
                "feature_id": features["id"],
                **component_bounds(
                    str(store.artifact_path(features["metadata_path"])),
                    features.get("metadata_hash", features["id"]),
                    role,
                ),
            }
        )
    if model.get("future_horizon_model_id"):
        future = store.get("future_horizon_model", model["future_horizon_model_id"])
        evidence = future["training_evidence"]
        components.append(
            {
                "name": "Future fixtures",
                "id": future["id"],
                "fit_max": evidence["fit_max_label"],
                "calibration_min": evidence["cut"],
                "calibration_max": evidence["cal_max_label"],
                "training_rows": evidence["training_rows"],
                "calibration_rows": evidence["calibration_rows"],
                "fit_before": future["fit_before"],
                "fit_as_of": future["fit_as_of"],
                "data_hash": future["data_hash"],
            }
        )
    market = store.get("market") if store.list("market", 1) else None
    finalized = [e["id"] for e in market["events"] if e.get("data_checked")] if market else []
    through = f"{market['season']}-GW{max(finalized):02d}" if market and finalized else None
    scheduled = {f.get("event") for f in market.get("fixtures", [])} if market else set()
    label_events = set(finalized) & scheduled
    labels_through = f"{market['season']}-GW{max(label_events):02d}" if market and label_events else None
    warnings = store.list_matching(
        "model_update_attempt", {"decision_model_id": model["id"], "status": "failed"}, 1
    )
    warning = store.get("model_update_attempt", warnings[0]["id"])["reason"] if warnings else None
    return {
        "decision_model_id": model["id"],
        "components": components,
        "finalized_through": through,
        "labels_expected_through": labels_through,
        "status": "Current completed labels represented"
        if labels_through and all(c["calibration_max"] >= labels_through for c in components)
        else "Refresh required"
        if labels_through
        else "No finalized current-season results",
        "warning": warning,
        "policy": model.get("training_policy", "expanding-reserved8-v1"),
        "explanation": "The latest eight completed deadlines calibrate probabilities independently; earlier labels fit weights. Current form, roles, availability and fixtures still update forecast inputs. After new finalized labels, the accepted family refits and all three components publish together. Unchanged effective training data reuses the accepted composition.",
    }
