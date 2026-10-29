"""Pinned, hash-verified open numerical models; never loads Python pickles.

Dastan composition/minutes formulas adapted under MIT, Copyright SmartPlayFPL.
OpenFPL historical prediction vectors retain their original supplied identity keys.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import xgboost as xgb

from .evaluation import metrics
from .storage import Store, atomic_json, now

REPOSITORY = "qazybekb/smartplayfpl-dastan"
COMMIT = "758db9247e8b3a9a870c991064f74d8339d61cd0"


def provision(store: Store) -> Path:
    root = store.root / "vendor" / "dastan"
    if (root / "models" / "artifact_manifest.json").exists() and (
        root / "data" / "features.parquet"
    ).exists():
        verify_models(root)
        return root
    with httpx.Client(timeout=60, follow_redirects=False) as client:
        r = client.get(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1")
        r.raise_for_status()
        tree = r.json()["tree"]
    data_files = {
        "data/features.parquet",
        "data/pre_deadline_ep_next.parquet",
        "data/pre_deadline_signals.parquet",
        "data/openfpl_predictions.csv",
        "data/openfpl_row_keys.csv",
        "data/season_registry.json",
        "data/release_manifest.json",
        "data/source_pins.json",
        "LICENSE",
    }
    files = [
        r
        for r in tree
        if r["type"] == "blob"
        and (r["path"] in data_files or (r["path"].startswith("models/") and r["path"].endswith(".json")))
    ]
    if sum(r.get("size", 0) for r in files) > 150_000_000:
        raise ValueError("Pinned open-model download exceeds the numerical artifact budget")

    def download(row):
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(
            f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{row['path']}", timeout=60
        )
        response.raise_for_status()
        expected = hashlib.sha1(
            b"blob " + str(len(response.content)).encode() + b"\0" + response.content
        ).hexdigest()
        if expected != row["sha"]:
            raise ValueError("Pinned Git blob hash mismatch")
        path.write_bytes(response.content)
        return {"path": row["path"], "sha256": hashlib.sha256(response.content).hexdigest()}

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest = list(pool.map(download, files))
    atomic_json(root / "download-manifest.json", {"commit": COMMIT, "files": manifest})
    verify_models(root)
    return root


def verify_models(root: Path) -> dict:
    manifest = json.loads((root / "models" / "artifact_manifest.json").read_text())
    for name, details in manifest["files"].items():
        path = root / "models" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != details["sha256"]:
            raise ValueError(f"Open model checksum failed: {name}")
    return manifest


class DastanRuntime:
    """Exact public 286-feature numerical JSON contract, independent safe loader."""

    def __init__(self, root: Path):
        verify_models(root)
        self.root = root
        self.models = root / "models"
        self.features = json.loads((self.models / "feature_cols.json").read_text())
        self.blend = json.loads((self.models / "blend.json").read_text())["per_position_direct_weight"]
        self.calibration = json.loads((self.models / "bucket_calibration.json").read_text())
        self.minutes = json.loads((self.models / "minutes_calibration.json").read_text())["curve"]
        self.cache: dict[str, Any] = {}

    def head(self, name, matrix):
        if name not in self.cache:
            model = xgb.Booster(params={"nthread": 2})
            model.load_model(self.models / f"{name}.json")
            self.cache[name] = model
        attributes = self.cache[name].attributes()
        stop = int(attributes["best_iteration"]) + 1 if "best_iteration" in attributes else 0
        return self.cache[name].predict(matrix, iteration_range=(0, stop))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Open-model input lacks required features: {sorted(missing)[:5]}")
        result = []
        for pos in ("GKP", "DEF", "MID", "FWD"):
            rows = frame.loc[frame.position.eq(pos)].copy()
            if rows.empty:
                continue
            matrix = xgb.DMatrix(rows[self.features].fillna(0).to_numpy(dtype=np.float32))
            p60 = self.head(f"p60_{pos}", matrix)
            non60 = np.maximum(0, self.head(f"non60_{pos}", matrix))
            bucket = self.head(f"bucket_{pos}", matrix)
            cal = self.calibration.get(pos)
            if cal:
                weights = np.array([cal[f"w{k}"] for k in range(4)])
                bucket = np.clip(bucket, 1e-12, 1) ** cal["gamma"] * weights
                bucket /= np.maximum(1e-12, bucket.sum(axis=1, keepdims=True))
            within = np.column_stack(
                [np.maximum(0, self.head(f"bucketreg_{pos}_{k}", matrix)) for k in range(4)]
            )
            component = p60 * (bucket * within).sum(axis=1) + (1 - p60) * non60
            direct = np.maximum(0, self.head(f"direct_{pos}", matrix))
            curve = self.minutes[pos]
            emins = np.clip(np.maximum(np.interp(p60, curve["p60"], curve["e_minutes"]), 60 * p60), 0, 90)
            pany = np.clip(
                np.maximum.reduce([np.interp(p60, curve["p60"], curve["p_any"]), p60, emins / 90]), 0, 1
            )
            rows["dastan"] = (1 - self.blend[pos]) * component + self.blend[pos] * direct
            rows["p60"], rows["p_any"], rows["expected_minutes"] = p60, pany, emins
            result.append(rows)
        return pd.concat(result).sort_index()


def historical_frame(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / "data" / "features.parquet")
    keys = ["season", "gameweek", "fpl_code"]
    for name, columns in [
        ("pre_deadline_signals.parquet", ["sig_status_risk", "sig_chance_playing", "sig_has_news"]),
        ("pre_deadline_ep_next.parquet", ["ep_next"]),
    ]:
        signal = pd.read_parquet(root / "data" / name)[keys + columns].drop_duplicates(keys)
        frame = frame.merge(signal, on=keys, how="left", validate="many_to_one")
        frame[columns] = frame[columns].fillna(-1)
    frame["ar_ep_next"] = frame["ep_next"]
    # Independently enforce one historical information set per FPL deadline.
    cols = [
        c for c in frame if c.startswith(("player_", "team_", "ar_", "cx_", "p_any_", "p60_", "e_minutes_"))
    ]
    dup = frame.loc[frame.duplicated(keys, keep=False)]
    if not dup.empty and dup.groupby(keys)[cols].nunique().gt(1).any().any():
        raise ValueError("Historical open-model frame leaks within a double-gameweek deadline")
    return frame


def reproduce(store: Store) -> dict:
    root = provision(store)
    runtime = DastanRuntime(root)
    frame = historical_frame(root)
    test = frame.loc[frame.season.eq("2025-26") & frame.gameweek.between(31, 38)]
    output = runtime.predict(test)
    if not np.isfinite(output[["dastan", "p60", "p_any", "expected_minutes"]]).all().all():
        raise ValueError("Nonfinite public-model output")
    bridge = pd.read_csv(root / "data" / "openfpl_row_keys.csv")
    supplied = pd.read_csv(root / "data" / "openfpl_predictions.csv").rename(columns={"gw": "gameweek"})
    joined = bridge.merge(
        supplied[["element", "gameweek", "openfpl_xpts", "actual_points", "minutes"]],
        on=["element", "gameweek"],
        validate="one_to_one",
    )
    report = {
        "at": now(),
        "source_repository": REPOSITORY,
        "source_commit": COMMIT,
        "model_contract": "40 hash-verified JSON files",
        "dastan": {
            "rows": len(output),
            "features": len(runtime.features),
            "metrics": metrics(output.target_points.to_numpy(), output.dastan.to_numpy()),
            "scope": "Released packaging holdout reproduction, not independent generalization",
            "coherent_minutes": bool(
                (output.p_any >= output.p60).all()
                and (output.expected_minutes >= 60 * output.p60 - 1e-6).all()
            ),
        },
        "openfpl": {
            "rows": len(joined),
            "metrics": metrics(joined.actual_points.to_numpy(), joined.openfpl_xpts.to_numpy()),
            "scope": "Recalculated from released historical prediction vectors with validated identity join",
        },
        "limitations": [
            "OpenFPL vectors are a historical benchmark; absent live Understat inputs are never filled with invented statistics",
            "Released Dastan holdout was used by upstream calibration; metrics verify reproduction only",
        ],
    }
    report["id"] = store.put("open_model_reproduction", report)
    atomic_json(Path("artifacts/evaluations/open_models.json"), report)
    return report
