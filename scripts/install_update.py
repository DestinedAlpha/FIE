"""Apply packaged application files while preserving local account data and keys."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

RESEARCH_KINDS = {
    "decision_critic",
    "state_value_model",
    "market_model",
    "market_windows",
    "coherent_model",
    "manager_policy",
    "context_ablation",
    "evolving_dataset",
    "population_sample",
    "population_activity",
    "top_manager_history",
    "manager_behavior",
    "horizon_model",
    "transfer_policy_model",
    "policy_features",
    "context_archive",
    "elite_cohort",
    "workload",
    "fusion_features",
    "fusion_model",
    "fusion_decisions",
    "fusion_comparison",
    "fusion_review",
    "fusion_uncertainty",
    "routed_policy",
    "archive_import",
    "intelligence_refresh",
    "claim_identity_audit",
    "claim_rejection",
    "claim",
    "source_text",
    "community_lead",
    "fusion_policy_tree",
    "fusion_policy_network",
}


def merge_research(source: Path, target: Path):
    """Merge immutable research artifacts without replacing any account/model selection."""
    packed = source / "data/state.sqlite.gz"
    destination = target / "data/state.sqlite"
    if not packed.is_file() or not destination.is_file():
        return 0
    with tempfile.TemporaryDirectory() as temporary:
        restored = Path(temporary) / "research.sqlite"
        with gzip.open(packed, "rb") as original, restored.open("wb") as output:
            shutil.copyfileobj(original, output)
        with (
            closing(sqlite3.connect(restored)) as original,
            closing(sqlite3.connect(destination)) as existing,
        ):
            with existing:
                if not existing.execute("SELECT 1 FROM sqlite_master WHERE name='objects'").fetchone():
                    return 0
                if original.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("Packaged research database failed integrity verification")
                for folder in ("raw", "features", "models"):
                    for path in (source / "data" / folder).rglob("*"):
                        if path.is_file():
                            relative = path.relative_to(source / "data")
                            destination_file = target / "data" / relative
                            if not destination_file.exists():
                                destination_file.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(path, destination_file)
                count = 0
                for row in original.execute("SELECT kind,id,created_at,body FROM objects ORDER BY rowid"):
                    if row[0] in RESEARCH_KINDS:
                        existing.execute("INSERT OR IGNORE INTO objects VALUES(?,?,?,?)", row)
                        count += 1
                for row in original.execute(
                    "SELECT source,url,captured_at,source_published_at,hash,path,status FROM snapshots ORDER BY id"
                ):
                    if row[0].startswith(
                        (
                            "market_window_",
                            "season_rules_",
                            "population_",
                            "context_",
                            "elite_",
                            "workload_",
                            "intelligence_feed_",
                            "research_",
                            "topk_",
                            "archive_",
                            "history_",
                        )
                    ):
                        existing.execute(
                            "INSERT OR IGNORE INTO snapshots(source,url,captured_at,source_published_at,hash,path,status) VALUES(?,?,?,?,?,?,?)",
                            row,
                        )
    return count


def install(source: Path, target: Path):
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError("Extract the update into a separate folder, then select your existing installation")
    if not (target / "pyproject.toml").is_file() or 'name = "fpl-intelligence"' not in (
        target / "pyproject.toml"
    ).read_text(encoding="utf-8"):
        raise ValueError("Select the folder containing your existing FPL Intelligence Start.cmd")
    manifest = json.loads((source / "app-files.json").read_text(encoding="utf-8"))
    files = []
    for name in manifest:
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0] in {"data", ".git", ".venv"}
            or relative.name.startswith(".env")
        ):
            raise ValueError("Update manifest contains a private or unsafe path")
        if not (source / relative).is_file():
            raise ValueError(f"Missing packaged application file: {name}")
        files.append(relative)
    backup = (
        target
        / "artifacts"
        / "runtime"
        / ("before-update-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    )
    backup.mkdir(parents=True, exist_ok=True)
    database = target / "data" / "state.sqlite"
    if database.exists():
        with (
            closing(sqlite3.connect(database)) as original,
            closing(sqlite3.connect(backup / "state.sqlite")) as saved,
        ):
            original.backup(saved)
    for relative in files:
        old = target / relative
        if old.is_file():
            (backup / "app" / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, backup / "app" / relative)
        old.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, old)
    imported = merge_research(source, target)
    return {
        "updated_files": len(files),
        "backup": str(backup),
        "account_database_replaced": False,
        "provider_keys_replaced": False,
        "research_artifacts_merged": imported,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(install(Path(__file__).resolve().parents[1], options.target)))
