import gzip
import hashlib
import json

from fpl_intelligence.decision_training import ACTION_FEATURES, load_episodes
from fpl_intelligence.maintenance import maintain_models
from fpl_intelligence.storage import Store
from fpl_intelligence.workload import canonical_player_matches, workload_features


def test_reused_actions_cannot_come_from_future_forecasts(tmp_path):
    from fpl_intelligence.optimization import incumbent_precedes_forecast

    store = Store(tmp_path / "data")
    early = store.put("forecast", {"cutoff": "2026-11-05T09:00Z"})
    later = store.put("forecast", {"cutoff": "2026-11-06T09:00Z"})
    assert incumbent_precedes_forecast(store, {"forecast_id": early}, store.get("forecast", later))
    assert not incumbent_precedes_forecast(store, {"forecast_id": later}, store.get("forecast", early))


def test_unused_midweek_substitute_does_not_reset_physical_rest():
    played = {
        "player_code": 7,
        "kickoff": "2026-11-01T15:00Z",
        "known_after": "2026-11-01T19:00Z",
        "minutes": 90,
        "extra_time": 0,
    }
    bench = played | {"kickoff": "2026-11-04T20:00Z", "known_after": "2026-11-05T00:00Z", "minutes": 0}
    cutoff = "2026-11-06T18:30Z"
    assert workload_features([played, bench], 7, cutoff) == workload_features([played], 7, cutoff)
    assert workload_features([played, bench], 7, cutoff, "legacy-v1") != workload_features(
        [played], 7, cutoff, "legacy-v1"
    )


def test_conflicting_duplicate_match_excludes_every_copy():
    row = {
        "season": "2026/27",
        "player_code": 7,
        "match_id": "cup-a",
        "minutes": 90,
        "fixture_signature": "one",
    }
    accepted, rejected = canonical_player_matches([row, row.copy()])
    assert accepted == [row] and not rejected
    accepted, rejected = canonical_player_matches([row, row | {"fixture_signature": "conflicting-score"}])
    assert not accepted and len(rejected) == 1
    accepted, rejected = canonical_player_matches(
        [row, row | {"player_code": 8, "fixture_signature": "conflicting-score"}]
    )
    assert not accepted and len(rejected) == 2


def test_weight_cleanup_preserves_account_tree_and_external_files(tmp_path):
    store = Store(tmp_path / "data")
    identity = store.put("state", {"bank": 16, "free_transfers": 1, "lots": [{"player_id": 7}]})
    models = store.root / "models"
    models.mkdir()
    retired = models / "retired.pt"
    retired.write_bytes(b"retired")
    tree = models / "minutes.txt"
    tree.write_bytes(b"validated-tree")
    outside = tmp_path / "other.pt"
    outside.write_bytes(b"unrelated-user-file")
    result = maintain_models(store)
    assert result["removed_weights"] == 1 and not retired.exists()
    assert tree.read_bytes() == b"validated-tree" and outside.exists()
    assert store.get("state")["id"] == identity
    assert maintain_models(store)["removed_weights"] == 0


def test_episode_migration_keeps_returns_and_drops_retired_input(tmp_path):
    store = Store(tmp_path / "data")
    old = list(range(len(ACTION_FEATURES) + 1))
    data = {"actions": [{"x": old, "return": 101, "action": {"buy": [7]}}], "states": []}
    (store.root / "features").mkdir()
    path = store.root / "features/episodes.json.gz"
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    store.put(
        "evolving_dataset",
        {
            "path": str(path),
            "hash": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scoring_version": "all-deadline-chips-v2",
        },
    )
    migrated, spec = load_episodes(store)
    row = migrated["actions"][0]
    assert row["return"] == 101 and row["action"] == {"buy": [7]}
    assert row["x"][:3] == [0, 1, 3] and row["x"][-1] == 1
    assert len(row["x"]) == len(ACTION_FEATURES) and spec["path"] != str(path)
    assert json.loads(gzip.decompress(path.read_bytes())) == data
    assert load_episodes(store)[0] == migrated
