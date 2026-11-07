"""Publish a complete numerical model composition only after successful fitting."""

import asyncio

from threadpoolctl import threadpool_limits

from .search_control import check_cancelled
from .storage import Store, digest


class DeferredModelStore(Store):
    def __init__(self, store):
        super().__init__(store.root)
        self.pending: dict[str, dict] = {}

    def put(self, kind, body, *, connection=None):
        if kind != "decision_model":
            return super().put(kind, body, connection=connection)
        identity = digest(body)
        self.pending[identity] = dict(body)
        return identity

    def get(self, kind, identity=None):
        if kind == "decision_model" and self.pending:
            key = identity or next(reversed(self.pending))
            if key in self.pending:
                return self.pending[key] | {"id": key}
        return super().get(kind, identity)

    def publish(self):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            check_cancelled()
            for body in self.pending.values():
                super().put("decision_model", body, connection=db)


def train_current(store, *, refit=True):
    from .decision import train_decisions
    from .role_minutes import ensure

    staged = DeferredModelStore(store)
    active = store.get("decision_model") if store.list("decision_model", 1) else {}
    with threadpool_limits(limits=1):
        check_cancelled()
        result = active
        if refit:
            if active.get("fusion_model_id"):
                from .context_archive import import_context
                from .fusion import refit_active_fusion

                asyncio.run(import_context(staged))
                result = refit_active_fusion(staged)
            else:
                result = train_decisions(staged)
        result = ensure(staged, result)
        if not result.get("promoted"):
            raise ValueError("Training did not produce an accepted model; active model retained")
        if active.get("role_minutes_model_id") and not result.get("role_minutes_model_id"):
            raise ValueError("Training would discard accepted role minutes; active model retained")
        staged.publish()
    return result
