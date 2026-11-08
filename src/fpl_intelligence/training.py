"""Publish a complete numerical model composition only after successful fitting."""

import asyncio

from threadpoolctl import threadpool_limits

from .search_control import check_cancelled
from .storage import Store, digest


class DeferredModelStore(Store):
    def __init__(self, store):
        super().__init__(store.root)
        self.pending: dict[tuple[str, str], dict] = {}
        self.role_refit_required = False

    def put(self, kind, body, *, connection=None):
        if kind not in (
            "decision_model",
            "role_minutes_model",
            "future_horizon_model",
            "model_training_state",
        ):
            return super().put(kind, body, connection=connection)
        identity = digest(body)
        self.pending[kind, identity] = dict(body)
        return identity

    def get(self, kind, identity=None):
        matching = [
            key for key in self.pending if key[0] == kind and (identity is None or key[1] == identity)
        ]
        if matching:
            key = matching[-1]
            return self.pending[key] | {"id": key[1]}
        return super().get(kind, identity)

    def list(self, kind, limit=100):
        from .storage import now

        pending = [{"id": key[1], "created_at": now()} for key in reversed(self.pending) if key[0] == kind]
        ids = {r["id"] for r in pending}
        return (pending + [r for r in super().list(kind, limit) if r["id"] not in ids])[:limit]

    def publish(self, expected_model_id=None, *, require_empty=False):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            check_cancelled()
            if expected_model_id is not None or require_empty:
                row = db.execute(
                    "SELECT id FROM objects WHERE kind='decision_model' ORDER BY created_at DESC,rowid DESC LIMIT 1"
                ).fetchone()
                if (row[0] if row else None) != expected_model_id:
                    raise ValueError("Accepted model changed during training; late publication rejected")
            for (kind, _), body in self.pending.items():
                super().put(kind, body, connection=db)


def train_current(store, *, refit=True):
    from .decision import train_decisions
    from .role_minutes import ensure

    staged = DeferredModelStore(store)
    active = store.get("decision_model") if store.list("decision_model", 1) else {}
    from .model_refresh import observation_signature, refit_reference

    operational = (
        active.get("version") == "learned-points-v1"
        and active.get("promoted")
        and not active.get("fusion_model_id")
    )
    before = observation_signature(store) if operational else None
    receipts = (
        store.list_matching("model_training_state", {"decision_model_id": active.get("id")}, 1)
        if operational
        else []
    )
    prior = store.get("model_training_state", receipts[0]["id"]) if receipts else None
    unchanged = bool(prior and prior["observations"] == before)
    with threadpool_limits(limits=1):
        check_cancelled()
        result = active
        if refit and not unchanged:
            if active.get("fusion_model_id"):
                from .context_archive import import_context
                from .fusion import refit_active_fusion

                asyncio.run(import_context(staged))
                result = refit_active_fusion(staged)
            elif operational:
                result = refit_reference(staged, active)
                staged.role_refit_required = True
            else:
                result = train_decisions(staged)
        result = ensure(staged, result)
        if not result.get("promoted"):
            raise ValueError("Training did not produce an accepted model; active model retained")
        if active.get("role_minutes_model_id") and not result.get("role_minutes_model_id"):
            raise ValueError("Training would discard accepted role minutes; active model retained")
        if operational and result.get("completed_labels_through"):
            through = result["completed_labels_through"]
            if (
                result.get("role_minutes_model_id")
                and staged.get("role_minutes_model", result["role_minutes_model_id"]).get(
                    "completed_labels_through", ""
                )
                < through
            ):
                raise ValueError(
                    "Role weights do not represent the newly completed labels; previous composition retained"
                )
            if (
                result.get("future_horizon_model_id")
                and staged.get("future_horizon_model", result["future_horizon_model_id"])[
                    "training_evidence"
                ]["cal_max_label"]
                < through
            ):
                raise ValueError(
                    "Future weights do not represent the newly completed labels; previous composition retained"
                )
        if operational:
            after = observation_signature(store)
            if before != after:
                raise ValueError(
                    "Finalized training observations changed during fitting; previous composition retained"
                )
            from .storage import now

            if refit and not unchanged:
                staged.put(
                    "model_training_state",
                    {
                        "at": now(),
                        "decision_model_id": result["id"],
                        "observations": after,
                        "refitted": True,
                        "policy": result.get("training_policy", "expanding-reserved8-v1"),
                    },
                )
        current = store.get("decision_model") if store.list("decision_model", 1) else {}
        if current.get("id") != active.get("id"):
            raise ValueError("Accepted model changed during training; late publication rejected")
        staged.publish(active.get("id"), require_empty=not bool(active))
    return result
