"""Controlled regressions; real historical workflow proof is a separate receipt."""

import asyncio
import hashlib

import numpy as np
import polars as pl
import pytest

from fpl_intelligence.model_refresh import fit_bounds, observation_signature
from fpl_intelligence.storage import Store
from fpl_intelligence.training import DeferredModelStore
from fpl_intelligence.training_policy import mix, training_indices


def test_deadline_splits_keep_double_fixture_labels_together_and_decay_by_observations():
    groups = np.array([f"2025/26-GW{i:02d}" for i in range(1, 20) for _ in range(2)])
    fit, cal, weights, cut = training_indices(groups, "2025/26-GW19", half_life=4)
    assert cut == "2025/26-GW11"
    assert max(groups[fit]) < min(groups[cal]) < "2025/26-GW19"
    assert len(cal) == 16 and weights[0] == weights[1]
    assert weights[8] / weights[0] == pytest.approx(2)
    rolling, _, _, _ = training_indices(groups, "2025/26-GW19", window=3)
    assert len(rolling) == 6 and min(groups[rolling]) == "2025/26-GW08"
    frame = pl.DataFrame({"deadline_group": groups, "season": ["2025/26"] * len(groups)})
    train, calibration, bounds = fit_bounds(frame)
    assert len(train) == 22 and len(calibration) == 16
    assert bounds["fit_max"] < bounds["calibration_min"]


def test_mixture_conserves_probability_and_conditional_mean():
    p = np.array([[0.1, 0.2, 0.3, 0.4]])
    q = p[:, ::-1]
    c, d = np.array([[0, 2, 4, 8]]), np.array([[0, 3, 5, 10]])
    r, v, start = mix((p, c, np.array([0.7])), (q, d, np.array([0.4])), 0.25)
    assert r.sum() == pytest.approx(1)
    assert (r * v).sum() == pytest.approx(0.75 * (p * c).sum() + 0.25 * (q * d).sum())
    assert start[0] == pytest.approx(0.625)


def test_signature_ignores_future_fixture_payload_and_context_download_metadata(tmp_path, monkeypatch):
    store = Store(tmp_path)
    rows = [{"season": "2026/27", "round": 9, "minutes": 90, "points": 5}]
    monkeypatch.setattr("fpl_intelligence.model_refresh.canonical_rows", lambda _: (rows, {}, {}))
    market = {
        "season": "2026/27",
        "events": [{"id": 9, "deadline_time": "2026-10-31T11:00Z"}],
        "fixtures": [],
    }
    store.put("market", market)
    entry = {"season": "2026/27", "event": 9, "source": "context9", "hash": "same", "downloaded": "then"}
    store.put("context_archive", {"entries": [entry]})
    original = observation_signature(store)
    store.put("market", market | {"fixtures": [{"id": 100, "event": 11}]})
    store.put("context_archive", {"entries": [entry | {"downloaded": "now"}, entry | {"event": 10}]})
    assert observation_signature(store) == original
    rows[0]["points"] = 6
    assert observation_signature(store) != original


@pytest.mark.parametrize("failure", ["rollback", "concurrent"])
def test_entire_model_publication_is_one_transaction(tmp_path, monkeypatch, failure):
    store = Store(tmp_path)
    old = {
        kind: store.put(kind, {"accepted": True})
        for kind in ("decision_model", "role_minutes_model", "future_horizon_model", "model_training_state")
    }
    staged = DeferredModelStore(store)
    for kind in reversed(old):
        staged.put(kind, {"new": True})
    if failure == "rollback":
        original = Store.put

        def fail(self, kind, body, **kwargs):
            if kind == "decision_model":
                raise ValueError("Failed final write")
            return original(self, kind, body, **kwargs)

        monkeypatch.setattr(Store, "put", fail)
    else:
        old["decision_model"] = store.put("decision_model", {"concurrent": True})
    with pytest.raises(ValueError):
        staged.publish(expected_model_id=old["decision_model"] if failure == "rollback" else "superseded")
    assert {kind: store.get(kind)["id"] for kind in old} == old


def test_invalid_official_history_does_not_replace_preceding_snapshot(tmp_path, monkeypatch):
    from fpl_intelligence.ingestion import Official

    store = Store(tmp_path)
    body = {"history": [{"round": 1, "minutes": 90}]}
    store.snapshot("history_1", "https://fantasy.premierleague.com/api/element-summary/1/", body)
    adapter = Official(store)

    async def get(*args):
        return {"history": "broken"}

    monkeypatch.setattr(adapter.transport, "get", get)
    assert asyncio.run(adapter.fetch(None, "element-summary/1/", "history_1")) is None
    assert store.latest("history_1")[0] == body


def test_legacy_semantic_hash_is_rebuilt_not_compared_to_parquet_bytes(tmp_path, monkeypatch):
    from fpl_intelligence import decision

    store = Store(tmp_path)
    path = tmp_path / "features" / "legacy.parquet"
    path.parent.mkdir()
    pl.DataFrame({"value": [1.0]}).write_parquet(path)
    store.put("decision_features", {"path": str(path), "hash": "pre-parquet-record-digest"})
    rebuilt = {"file_hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(decision, "build_dataset", lambda _: (rebuilt, pl.read_parquet(path)))
    spec, frame = decision.load_decision_data(store)
    assert spec == rebuilt and frame["value"][0] == 1


@pytest.mark.parametrize(
    "kickoff,before,after",
    [
        ("2026-08-30T15:00Z", "2026-08-31T07:59Z", "2026-08-31T08:01Z"),
        ("2026-11-01T15:00Z", "2026-11-02T08:59Z", "2026-11-02T09:01Z"),
    ],
)
def test_2026_finalization_uses_next_day_uk_clock_including_dst(kickoff, before, after):
    from fpl_intelligence.observations import completed_before

    rows = [{"season": "2026/27", "round": 3, "kickoff_time": kickoff}]
    assert completed_before(rows, before, "2026/27-GW04") == []
    assert completed_before(rows, after, "2026/27-GW04") == rows


def test_pre_defcon_scoring_does_not_award_modern_points():
    from fpl_intelligence.rules import Rules, score_fixture

    events = {"minutes": 90, "goals_scored": 1, "defensive_actions": 20, "goals_conceded": 1}
    assert score_fixture(events, 1, Rules(season="2024/25")) == 8
    assert score_fixture(events, 1, Rules(season="2025/26")) == 12
    assert score_fixture(events, 2, Rules(season="2024/25")) == 8
    assert score_fixture(events, 2, Rules(season="2025/26")) == 10


def test_partial_history_refresh_stops_before_training_or_forecasting(tmp_path, monkeypatch):
    from fpl_intelligence.ingestion import Official
    from fpl_intelligence.workflow import refresh_cycle

    store = Store(tmp_path)
    market = {"season": "2026/27", "events": [{"id": 9, "data_checked": False}]}
    store.put("market", market)
    store.put("state", {})
    calls = []

    async def ingest(self, **kwargs):
        calls.append(kwargs)
        if not kwargs.get("histories", True):
            store.put("market", market | {"events": [{"id": 9, "data_checked": True}]})
            return {}
        return {
            "histories_complete": False,
            "history_successes": 600,
            "history_requests": 671,
            "errors": [{"source": "history_7", "error": "unavailable"}],
        }

    def forbidden(*args, **kwargs):
        raise AssertionError("A partial finalized batch must not enter numerical publication")

    monkeypatch.setattr(Official, "ingest", ingest)
    monkeypatch.setattr("fpl_intelligence.training.train_current", forbidden)
    monkeypatch.setattr("fpl_intelligence.simulation.forecast", forbidden)
    result = refresh_cycle(store)
    assert result["training_published"] is False and result["history_successes"] == 600
    assert len(calls) == 2


def test_cancelled_refresh_does_not_send_additional_source_requests(tmp_path):
    from concurrent.futures import CancelledError
    from threading import Event

    from fpl_intelligence.search_control import search_session
    from fpl_intelligence.transport import PublicTransport

    cancel = Event()
    with search_session(cancel=cancel), pytest.raises(CancelledError):
        cancel.set()
        asyncio.run(
            PublicTransport(Store(tmp_path)).get(None, "https://fantasy.premierleague.com/api/", "test")
        )


def test_role_reuse_requires_actual_fitted_label_coverage_even_if_cache_has_newer_rows(tmp_path, monkeypatch):
    import fpl_intelligence.role_minutes as module

    store = Store(tmp_path)
    old = store.put(
        "role_minutes_model",
        {
            "version": module.VERSION,
            "context_archive_id": "pinned",
            "completed_labels_through": "2026/27-GW08",
        },
    )
    store.put("role_minutes_validation", {"version": module.VERSION, "passed": True})

    async def context(_):
        return {}

    monkeypatch.setattr("fpl_intelligence.context_archive.import_context", context)
    monkeypatch.setattr(
        "fpl_intelligence.fusion_data.build_fusion_data", lambda _: ({}, [], {"id": "new-features"})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("An old head must not reuse unconsumed newer rows from its feature cache")

    monkeypatch.setattr("fpl_intelligence.fusion.load_data", forbidden)
    calls = []
    monkeypatch.setattr(module, "train", lambda _, feature: calls.append(feature) or {"id": "refitted"})
    monkeypatch.setattr(
        module, "attach", lambda _, model, role_id: model | {"role_minutes_model_id": role_id}
    )
    model = {"promoted": True, "completed_labels_through": "2026/27-GW09", "role_minutes_model_id": old}
    result = module.ensure(store, model)
    assert calls == ["new-features"] and result["role_minutes_model_id"] == "refitted"


def test_publication_rejects_a_companion_that_has_not_consumed_new_labels(tmp_path, monkeypatch):
    from fpl_intelligence.training import train_current

    store = Store(tmp_path)
    role = store.put("role_minutes_model", {"completed_labels_through": "2026/27-GW08"})
    active = store.put(
        "decision_model", {"version": "learned-points-v1", "promoted": True, "role_minutes_model_id": role}
    )
    monkeypatch.setattr(
        "fpl_intelligence.model_refresh.observation_signature", lambda _: {"through": "2026/27-GW09"}
    )

    def refit(staged, old):
        model = {
            "version": "learned-points-v1",
            "promoted": True,
            "completed_labels_through": "2026/27-GW09",
            "role_minutes_model_id": role,
        }
        return model | {"id": staged.put("decision_model", model)}

    monkeypatch.setattr("fpl_intelligence.model_refresh.refit_reference", refit)
    monkeypatch.setattr("fpl_intelligence.role_minutes.ensure", lambda _, model: model)
    with pytest.raises(ValueError, match="Role weights do not represent"):
        train_current(store)
    assert store.get("decision_model")["id"] == active
    assert len(store.list("decision_model")) == 1


def test_training_status_never_claims_newer_cached_rows_were_used_by_old_weights(tmp_path):
    from fpl_intelligence.training_status import component_bounds

    path = tmp_path / "metadata.parquet"
    pl.DataFrame(
        {"season": ["2025/26"] * 20, "deadline_group": [f"2025/26-GW{i:02d}" for i in range(1, 21)]}
    ).write_parquet(path)
    spec = {
        "completed_labels_through": "2025/26-GW18",
        "training_evidence": {"fit_max_group": "2025/26-GW10", "calibration_max_group": "2025/26-GW18"},
    }
    bounds = component_bounds(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), spec)
    assert bounds["fit_max"] == "2025/26-GW10" and bounds["calibration_max"] == "2025/26-GW18"
    assert bounds["training_rows"] == 10 and bounds["calibration_rows"] == 8
