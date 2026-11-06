from types import SimpleNamespace

import numpy as np

from fpl_intelligence.role_minutes import historical_inputs, predict, role_vector


def test_challenger_reference_uses_composed_out_of_fold_points(tmp_path, monkeypatch):
    from fpl_intelligence.fusion import reference_forecasts
    from fpl_intelligence.storage import Store

    store = Store(tmp_path / "data")
    store.put(
        "decision_model",
        {
            "role_minutes_model_id": "accepted-role",
            "predictions": [{"season": "2026/27", "player_id": 7, "fixture_id": 91, "predicted": 100}],
        },
    )
    causal = {
        "season": "2026/27",
        "player_id": 7,
        "fixture_id": 91,
        "predicted": 3.7,
        "availability_included": True,
    }
    seen = []

    def composed(saved, model, feature_id):
        seen.append((model["role_minutes_model_id"], feature_id))
        return [causal]

    monkeypatch.setattr("fpl_intelligence.role_minutes.reference_predictions", composed)
    _, _, rows = reference_forecasts(store, [], "frozen-challenger-inputs")
    assert seen == [("accepted-role", "frozen-challenger-inputs")]
    assert rows["2026/27", 7, 91] == causal


def test_role_history_uses_stable_code_and_frozen_context(monkeypatch):
    packet = {"starts": 7, "minutes": 630}
    seen = []

    def context(store, archive_id):
        seen.append(archive_id)
        return {("2026/27", 9, 777777): packet}

    monkeypatch.setattr("fpl_intelligence.context_archive.load_context", context)
    store = SimpleNamespace(get=lambda *args: {"records": []})
    sequence = np.zeros((1, 12, 32))
    mask = np.ones((1, 12), dtype=bool)
    arrays = {
        "tab": np.zeros((1, 26)),
        "sequence": sequence,
        "sequence_mask": mask,
        "news": np.zeros((1, 13)),
        "quality": np.zeros((1, 5)),
    }
    metadata = [
        {"season": "2026/27", "event": 9, "player_id": 7, "code": 777777, "deadline": "2026-11-06T18:30Z"}
    ]
    x = historical_inputs(
        store, arrays, metadata, {"lineage": {"workload": "frozen-workload"}}, "frozen-context"
    )
    assert seen == ["frozen-context"] and x.shape == (1, 57)
    assert x[0, 26] == 7 / 8 and x[0, 27] == 630 / (90 * 8)
    assert x[0, 33:35].tolist() == [1, 1]


def test_role_vector_keeps_missing_season_data_distinct_from_observed_zero():
    seq = np.zeros((12, 32))
    mask = np.zeros(12, dtype=bool)
    unknown = role_vector({}, seq, mask, 1)
    zero = role_vector({"starts": 0, "minutes": 0}, seq, mask, 1)
    assert unknown[:7] == zero[:7] and unknown[-2:] == [0, 0] and zero[-2:] == [1, 1]


def test_participation_head_normalizes_and_enforces_published_absence():
    class Head:
        def predict_proba(self, x):
            return np.tile([0.1, 0.2, 0.3, 0.4], (len(x), 1))

    from fpl_intelligence.models import curve_fit

    curves = {str(k): curve_fit(np.array([0.0, 1.0]), np.array([0, 1])) for k in range(4)}
    news = np.zeros((2, 13))
    news[1, 5] = 1
    probabilities = predict(Head(), curves, np.zeros((2, 57)), news)
    assert np.allclose(probabilities.sum(1), 1)
    assert probabilities[1].tolist() == [1, 0, 0, 0]
    assert probabilities[0, 3] > probabilities[0, 0]


def test_live_role_head_preserves_future_and_explicit_availability_scenarios(tmp_path, monkeypatch):
    import hashlib

    from fpl_intelligence.role_minutes import VERSION, apply
    from fpl_intelligence.storage import Store

    store = Store(tmp_path / "data")
    path = store.root / "models/minutes.txt"
    path.parent.mkdir()
    path.write_bytes(b"checked-model")
    mid = store.put(
        "role_minutes_model",
        {
            "promoted": True,
            "version": VERSION,
            "feature_id": "frozen-inputs",
            "width": 57,
            "path": str(path),
            "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
            "curves": {},
            "weight": 1,
        },
    )
    store.put("workload", {"records": []})
    monkeypatch.setattr("fpl_intelligence.observations.information_time", lambda _: "2026-11-06T09:00Z")
    monkeypatch.setattr(
        "fpl_intelligence.fusion_data.live_inputs",
        lambda *args: {
            "tab": np.zeros((1, 26)),
            "sequence": np.zeros((1, 12, 32)),
            "sequence_mask": np.zeros((1, 12), dtype=bool),
            "news": np.zeros((1, 13)),
            "quality": np.zeros((1, 5)),
        },
    )
    monkeypatch.setattr("fpl_intelligence.role_minutes.lgb.Booster", lambda **_: object())
    monkeypatch.setattr(
        "fpl_intelligence.role_minutes.predict", lambda *args: np.array([[0.1, 0.2, 0.3, 0.4]])
    )
    row = {"event": 10, "player_id": 7, "availability": 0.5, "base_availability": 0.5}
    rows = [row, row | {"availability": 0}, row | {"event": 11}, row | {"suspension_served": True}]
    before = np.tile([0.4, 0.1, 0.2, 0.3], (4, 1))
    player = SimpleNamespace(code=777777, raw={"starts": 7, "minutes": 630})
    market = {"events": [{"id": e, "deadline_time": "2026-11-06T18:30Z"} for e in range(1, 39)]}
    after, included = apply(store, mid, rows, market, {7: player}, before, 10)
    assert included.tolist() == [True, False, False, False]
    assert after[0].tolist() == [0.1, 0.2, 0.3, 0.4]
    assert np.array_equal(after[1:], before[1:]) and np.array_equal(before[0], [0.4, 0.1, 0.2, 0.3])
