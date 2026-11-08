import hashlib
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from fpl_intelligence import horizon_trial
from fpl_intelligence.storage import Store


class Booster:
    def save_model(self, path):
        Path(path).write_text("retained test model")


def bundle():
    return {k: SimpleNamespace(booster_=Booster()) for k in ("minutes", "points", "starter")} | {
        "curves": {},
        "training_rows": 3,
    }


def test_horizon_challenger_cache_is_bound_to_data_and_checks_weights(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        horizon_trial, "fit_models", lambda frame, *args: calls.append(len(frame)) or bundle()
    )
    monkeypatch.setattr(horizon_trial.lgb, "Booster", lambda **kw: kw["model_file"])
    frame = pl.DataFrame({"x": [1.0, 2.0]})
    horizon_trial.fitted(None, frame, "2024/25-GW01", "uniform", tmp_path)
    horizon_trial.fitted(None, frame, "2024/25-GW01", "uniform", tmp_path)
    assert calls == [2]
    horizon_trial.fitted(None, frame.with_columns(pl.col("x") + 1), "2024/25-GW01", "uniform", tmp_path)
    assert calls == [2, 2]
    path = next(
        p
        for p in tmp_path.glob("*/minutes.txt")
        if __import__("json").loads((p.parent / "model.json").read_text())["contract"]["frame_hash"]
        == hashlib.sha256(frame.hash_rows(seed=137).to_numpy().tobytes()).hexdigest()
    )
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        horizon_trial.fitted(None, frame, "2024/25-GW01", "uniform", tmp_path)


def test_reference_cache_never_reuses_unsealed_legacy_files(tmp_path, monkeypatch):
    store = Store(tmp_path / "data")
    source = store.root / "features" / "input.parquet"
    source.parent.mkdir()
    pl.DataFrame(
        {"deadline_group": [f"2023/24-GW{i:02d}" for i in range(1, 13)], "value": list(range(12))}
    ).write_parquet(source)
    feature = {
        "path": str(source),
        "hash": "semantic-record-identity",
        "file_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    store.put("decision_features", feature)
    legacy = tmp_path / "reference-2024-25-GW01"
    legacy.mkdir()
    (legacy / "model.json").write_text('{"curves":{},"cut":"legacy"}')
    calls = []
    monkeypatch.setattr(horizon_trial, "fit_components", lambda *a: calls.append(True) or bundle())
    monkeypatch.setattr(horizon_trial.lgb, "Booster", lambda **kw: kw["model_file"])
    horizon_trial.reference(store, "2024/25-GW01", tmp_path)
    horizon_trial.reference(store, "2024/25-GW01", tmp_path)
    assert calls == [True] and not (legacy / "minutes.txt").exists()
    path = next(tmp_path.glob("reference-*/minutes.txt"))
    path.write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        horizon_trial.reference(store, "2024/25-GW01", tmp_path)
    source.write_bytes(b"corrupt features")
    with pytest.raises(ValueError, match="feature file checksum"):
        horizon_trial.reference(store, "2024/25-GW01", tmp_path)
