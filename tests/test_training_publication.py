from concurrent.futures import CancelledError
from threading import Event

import pytest

from fpl_intelligence.models import save_booster_atomic
from fpl_intelligence.search_control import search_session
from fpl_intelligence.storage import Store
from fpl_intelligence.training import train_current


@pytest.mark.parametrize("failure", ["fit", "cancel", "downgrade"])
def test_failed_composition_retains_active_model_and_forecasts(tmp_path, monkeypatch, failure):
    store = Store(tmp_path)
    original = store.put("decision_model", {"promoted": True, "role_minutes_model_id": "accepted"})
    cancel = Event()

    def base(staged):
        body = {"promoted": True, "version": "new-base"}
        return body | {"id": staged.put("decision_model", body)}

    def role(staged, model):
        assert store.get("decision_model")["id"] == original
        if failure == "fit":
            raise ValueError("Role fit failed")
        if failure == "cancel":
            cancel.set()
        return model

    monkeypatch.setattr("fpl_intelligence.decision.train_decisions", base)
    monkeypatch.setattr("fpl_intelligence.role_minutes.ensure", role)
    with search_session(cancel=cancel), pytest.raises((ValueError, CancelledError)):
        train_current(store)
    assert store.get("decision_model")["id"] == original
    assert len(store.list("decision_model")) == 1


def test_successful_composition_publishes_complete_resolvable_chain(tmp_path, monkeypatch):
    store = Store(tmp_path)
    old = store.put("decision_model", {"promoted": True})

    def base(staged):
        body = {"promoted": True, "version": "new-base"}
        return body | {"id": staged.put("decision_model", body)}

    def role(staged, model):
        assert store.get("decision_model")["id"] == old
        assert staged.get("decision_model")["id"] == model["id"]
        body = {
            "promoted": True,
            "role_minutes_model_id": "accepted",
            "previous_decision_model_id": model["id"],
        }
        return body | {"id": staged.put("decision_model", body)}

    monkeypatch.setattr("fpl_intelligence.decision.train_decisions", base)
    monkeypatch.setattr("fpl_intelligence.role_minutes.ensure", role)
    result = train_current(store)
    assert store.get("decision_model")["id"] == result["id"]
    assert store.get("decision_model", result["previous_decision_model_id"])["version"] == "new-base"


def test_weight_failure_or_retraining_cannot_corrupt_retained_bytes(tmp_path):
    path = tmp_path / "minutes.txt"
    path.write_text("accepted")

    class Booster:
        def __init__(self, content, fail=False):
            self.content, self.fail = content, fail

        def save_model(self, target):
            from pathlib import Path

            Path(target).write_text(self.content)
            if self.fail:
                raise RuntimeError("Interrupted writer")

    with pytest.raises(RuntimeError):
        save_booster_atomic(Booster("partial", True), path)
    with pytest.raises(ValueError, match="retained"):
        save_booster_atomic(Booster("changed"), path)
    save_booster_atomic(Booster("accepted"), path)
    assert path.read_text() == "accepted"
    assert list(tmp_path.iterdir()) == [path]
    fresh = tmp_path / "points.txt"
    save_booster_atomic(Booster("new"), fresh)
    assert fresh.read_text() == "new"


def test_companion_paths_separate_feature_and_source_contracts(tmp_path):
    from fpl_intelligence.manager_policy import model_directory

    store = Store(tmp_path)
    paths = {
        model_directory(store, "cohort", source, features)
        for source, features in (
            ("source-a", ("price",)),
            ("source-a", ("price", "minutes")),
            ("source-b", ("price",)),
        )
    }
    assert len(paths) == 3
