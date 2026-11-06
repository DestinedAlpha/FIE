"""Idempotent retirement of removed model weights and incompatible companions."""

from __future__ import annotations

from .storage import now


def maintain_models(store):
    from .decision_training import ACTION_FEATURES, train_decision_models
    from .manager_policy import PLAYER_FEATURES, train_manager_policy

    removed = 0
    model_root = (store.root / "models").resolve()
    for path in model_root.rglob("*"):
        if path.suffix in (".pt", ".pth") and path.is_file() and path.resolve().is_relative_to(model_root):
            path.unlink()
            removed += 1
    # Content-addressed historical reports stay immutable. The active source
    # report contains only supported candidates and the original tree checksums.
    if store.list("fusion_model", 1):
        old = store.get("fusion_model")
        if "network_spec" in old or "network" in old.get("candidates", {}):
            body = {k: v for k, v in old.items() if k not in ("id", "network_spec")}
            for key in ("candidates", "metrics"):
                if isinstance(body.get(key), dict):
                    body[key] = {k: v for k, v in body[key].items() if k == "tree"}
            if body.get("selected") not in (None, "tree"):
                body["selected"] = None
            body.update(at=now(), retired_companion_from=old["id"])
            store.put("fusion_model", body)
    if store.list("decision_model", 1):
        active = store.get("decision_model")
        original = active
        while active.get("fusion_model_id") and active.get("fusion_kind") != "tree":
            active = store.get("decision_model", active["previous_decision_model_id"])
        if active["id"] != original["id"]:
            body = {k: v for k, v in active.items() if k != "id"}
            body.update(at=now(), retired_companion_from=original["id"])
            store.put("decision_model", body)
    refits = []
    input_audit = None
    if all(
        store.list(kind, 1)
        for kind in ("decision_model", "market", "decision_features", "fusion_features", "context_archive")
    ):
        from .role_minutes import VERSION, ensure

        active = store.get("decision_model")
        if active.get("promoted") and not active.get("fusion_model_id"):
            identity = active.get("role_minutes_model_id")
            role = store.get("role_minutes_model", identity) if identity else {}
            if role.get("version") != VERSION or not role.get("context_archive_id"):
                updated = ensure(store, active)
                if updated["id"] != active["id"]:
                    refits.append("role_minutes_contract")
                    if store.list("replay_inputs", 1):
                        from .role_minutes import audit_input_contract

                        try:
                            proof = audit_input_contract(store)
                            input_audit = {
                                "id": proof["id"],
                                "passed": proof["passed"],
                                "rows": proof["rows"],
                            }
                        except ValueError as error:
                            input_audit = {
                                "status": "Historical parity check unavailable",
                                "reason": str(error),
                            }
    if store.list("manager_policy", 1) and list(
        store.get("manager_policy").get("player_features", [])
    ) != list(PLAYER_FEATURES):
        train_manager_policy(store)
        refits.append("manager_policy")
    if store.list("decision_critic", 1) and list(store.get("decision_critic").get("features", [])) != list(
        ACTION_FEATURES
    ):
        train_decision_models(store)
        refits.append("decision_critic")
    result = {
        "at": now(),
        "removed_weights": removed,
        "refitted_companions": refits,
        "account_state_changed": False,
        "role_input_audit": input_audit,
    }
    if removed or refits:
        store.put("model_maintenance", result)
    return result
