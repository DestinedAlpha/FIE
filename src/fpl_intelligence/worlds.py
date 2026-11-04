"""Bounded per-gameweek world archives, preserving portable legacy NPZ reads."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np

from .storage import atomic_json


def save_worlds(store, identity, arrays):
    total = sum(a.nbytes for a in arrays.values())
    if total < 100_000_000:
        path = store.root / "scenarios" / (identity + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return str(path)
    groups: dict[str, dict[str, np.ndarray]] = {}
    for key, value in arrays.items():
        groups.setdefault(key.rsplit("_", 1)[-1], {})[key] = value
    entries = {}
    for event, values in groups.items():
        path = store.root / "scenarios" / (identity + "-gw" + event + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **cast(Any, values))
        if path.stat().st_size >= 95 * 1024**2:
            raise ValueError("A world chunk exceeds the portable archive budget")
        for key in values:
            entries[key] = {"path": str(path), "hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = store.root / "scenarios" / (identity + "-worlds.json")
    atomic_json(manifest, {"version": "gameweek-chunks-v1", "arrays": entries})
    return str(manifest)


class WorldArchive:
    def __init__(self, store, reference):
        self.store = store
        self.path = store.artifact_path(reference)
        self.legacy: Any = None
        self.cached: Any = None
        self.cached_path = None
        self.manifest = json.loads(self.path.read_text()) if self.path.suffix == ".json" else None
        self.files = list(self.manifest["arrays"]) if self.manifest else []

    def __enter__(self):
        if self.manifest is None:
            self.legacy = np.load(self.path, allow_pickle=False)
            self.files = self.legacy.files
        return self

    def __getitem__(self, key):
        if self.legacy is not None:
            return self.legacy[key]
        assert self.manifest is not None
        item = self.manifest["arrays"][key]
        path = self.store.artifact_path(item["path"])
        if self.cached_path != path:
            if self.cached is not None:
                self.cached.close()
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["hash"]:
                raise ValueError("World archive checksum mismatch")
            self.cached = np.load(path, allow_pickle=False)
            self.cached_path = path
        return self.cached[key]

    def __exit__(self, *args):
        if self.legacy is not None:
            self.legacy.close()
        if self.cached is not None:
            self.cached.close()


def load_worlds(store, forecast):
    return WorldArchive(store, forecast["scenario_path"])


def copy_worlds(store, forecast, destination):
    """Copy a frozen scenario and its referenced chunks for isolated recourse."""
    path = store.artifact_path(forecast["scenario_path"])
    target = Path(destination) / "scenarios"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target / path.name)
    if path.suffix == ".json":
        manifest = json.loads(path.read_text())
        for item in {v["path"]: v for v in manifest["arrays"].values()}.values():
            chunk = store.artifact_path(item["path"])
            shutil.copy2(chunk, target / chunk.name)
