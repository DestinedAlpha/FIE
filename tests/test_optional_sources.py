import numpy as np
import pytest

from fpl_intelligence.elite import elite_features
from fpl_intelligence.role_minutes import historical_inputs
from fpl_intelligence.storage import Store
from fpl_intelligence.workload import records


def test_uncollected_optional_inputs_have_masks_not_fictitious_observations(tmp_path, monkeypatch):
    store = Store(tmp_path)
    assert elite_features(store)[0] == {} and elite_features(store)[1]["id"] is None
    assert records(store) == []
    monkeypatch.setattr("fpl_intelligence.context_archive.load_context", lambda *_: {})
    arrays = {
        "tab": np.zeros((1, 26)),
        "sequence": np.zeros((1, 12, 32)),
        "sequence_mask": np.zeros((1, 12), dtype=bool),
        "news": np.zeros((1, 13)),
        "quality": np.zeros((1, 4)),
    }
    metadata = [{"season": "2026/27", "event": 1, "code": 77, "deadline": "2026-08-14T17:00:00Z"}]
    result = historical_inputs(store, arrays, metadata, {"lineage": {"workload": None}})
    assert result.shape == (1, 57)
    assert np.isfinite(result).all() and not result[0, -9:].any()
    store.put("workload", {"records": []})
    with pytest.raises(ValueError):
        historical_inputs(store, arrays, metadata, {"lineage": {"workload": "missing-pinned-source"}})


def test_optional_shot_outage_keeps_training_independent_stages_executable(tmp_path, monkeypatch):
    from fpl_intelligence.workflow import football_training

    monkeypatch.chdir(tmp_path)
    store = Store(tmp_path / "data")
    seen = []

    async def unavailable(*_):
        raise ValueError("Understat maintenance")

    monkeypatch.setattr("fpl_intelligence.shot_data.collect_shots", unavailable)
    for module, name in [
        ("discipline", "build_discipline"),
        ("availability", "train_availability"),
        ("score_process", "train_score_process"),
        ("odds", "calibrate_odds"),
    ]:

        def fit(*_, stage=name):
            seen.append(stage)
            return {"id": stage}

        monkeypatch.setattr(f"fpl_intelligence.{module}.{name}", fit)
    result = football_training(store)
    assert len(seen) == 4
    assert next(s for s in result["stages"] if s["stage"] == "shot_dependent_fits")["records"] == 0
    assert not store.list("shot_archive", 1)
