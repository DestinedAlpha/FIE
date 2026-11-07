"""Apply packaged application files while preserving local account data and keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path, PureWindowsPath

from fpl_intelligence.database_archive import unpack_database
from fpl_intelligence.storage import atomic_json

RESEARCH_KINDS = {
    "role_minutes_validation",
    "role_minutes_model",
    "role_minutes_reference",
    "immediate_points_validation",
    "online_calibration_validation",
    "role_contract_audit",
    "role_weekly_replay",
    "role_weekly_replay_segment",
    "user_transfer_comparison",
    "model_maintenance",
    "transfer_research",
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
    "decision_features",
    "shot_archive",
    "shot_refresh",
    "availability_model",
    "model_rejection",
    "temporal_model",
    "luck_report",
    "attack_model",
    "score_process_model",
    "discipline_refresh",
    "discipline_report",
    "ban_window",
    "ban_decision",
    "extra_fixture",
    "official_cautions",
    "odds_observation",
    "odds_refresh",
    "odds_model",
    "football_training",
    "replay_inputs",
    "weekly_replay_decision",
    "weekly_replay",
    "weekly_replay_segment",
    "weekly_baseline_comparison",
    "consensus_model",
    "consensus_weight_selection",
    "consensus_decision",
    "consensus_policy_validation",
    "continuous_consensus_decision",
    "pipeline_replay",
    "horizon_trial",
    "horizon_refinement",
    "future_distribution_protocol",
    "future_distribution_validation",
    "future_policy_replay",
    "future_policy_diagnostic",
    "future_policy_suite",
    "future_horizon_model",
}


def merge_research(source: Path, target: Path):
    """Merge immutable research artifacts without replacing any account/model selection."""
    packed = source / "data/state.sqlite.manifest.json"
    legacy = source / "data/state.sqlite.gz"
    destination = target / "data/state.sqlite"
    if not (packed.is_file() or legacy.is_file()) or not destination.is_file():
        return 0
    runtime = target / "artifacts/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime) as temporary:
        restored = Path(temporary) / "research.sqlite"
        unpack_database(source / "data", restored)
        with (
            closing(sqlite3.connect(restored)) as original,
            closing(sqlite3.connect(destination)) as existing,
        ):
            with existing:
                if not existing.execute("SELECT 1 FROM sqlite_master WHERE name='objects'").fetchone():
                    return 0
                if original.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("Packaged research database failed integrity verification")
                for folder in ("raw", "features", "models", "normalized"):
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
                            "shot_",
                            "discipline_",
                            "odds_",
                            "replay_",
                        )
                    ):
                        existing.execute(
                            "INSERT OR IGNORE INTO snapshots(source,url,captured_at,source_published_at,hash,path,status) VALUES(?,?,?,?,?,?,?)",
                            row,
                        )
    return count


def _runtime(target):
    folder = target / "artifacts/runtime"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def restore_transaction(target, backup, journal):
    for name in journal["files"]:
        destination = target / name
        if name in journal["existed"]:
            shutil.copy2(backup / "app" / name, destination)
        else:
            destination.unlink(missing_ok=True)
        destination.with_name(destination.name + ".fie-update").unlink(missing_ok=True)
    if (backup / "state.sqlite").exists():
        with (
            closing(sqlite3.connect(backup / "state.sqlite")) as original,
            closing(sqlite3.connect(target / "data/state.sqlite")) as saved,
        ):
            original.backup(saved)
    journal["status"] = "rolled_back"
    atomic_json(backup / "transaction.json", journal)
    (target / ".fie-update-in-progress").unlink(missing_ok=True)


def install(source: Path, target: Path):
    from fpl_intelligence.pipeline_jobs import replay_lock

    with replay_lock(_runtime(target.resolve()) / "update.lock"):
        return _install(source, target)


def _install(source: Path, target: Path):
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
            not name
            or not relative.parts
            or PureWindowsPath(name).drive
            or "\\" in name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[0].casefold() in {"data", ".git", ".venv"}
            or relative.name.casefold().startswith(".env")
        ):
            raise ValueError("Update manifest contains a private or unsafe path")
        if not (source / relative).is_file():
            raise ValueError(f"Missing packaged application file: {name}")
        files.append(relative)
    metadata = json.loads((source / "update-checksums.json").read_text(encoding="utf-8"))
    hashes = metadata["files"]
    if set(manifest) != set(hashes) or len(manifest) != len(set(manifest)):
        raise ValueError("Update manifest/checksum entries disagree")
    for relative in files:
        if not (source / relative).resolve().is_relative_to(source) or not (
            target / relative
        ).resolve().is_relative_to(target):
            raise ValueError("Update path escapes its application folder")
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != hashes[relative.as_posix()]:
            raise ValueError(f"Update checksum mismatch: {relative}")
    for previous in sorted(_runtime(target).glob("before-update-*/transaction.json")):
        journal = json.loads(previous.read_text())
        if journal.get("status") == "applying":
            restore_transaction(target, previous.parent, journal)
    backup = Path(tempfile.mkdtemp(prefix="before-update-", dir=_runtime(target)))
    database = target / "data" / "state.sqlite"
    if database.exists():
        with (
            closing(sqlite3.connect(database)) as original,
            closing(sqlite3.connect(backup / "state.sqlite")) as saved,
        ):
            original.backup(saved)
    existed = []
    for relative in files:
        old = target / relative
        if old.is_file():
            existed.append(relative.as_posix())
            (backup / "app" / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, backup / "app" / relative)
    journal = {"files": [p.as_posix() for p in files], "existed": existed, "status": "applying"}
    atomic_json(backup / "transaction.json", journal)
    marker = target / ".fie-update-in-progress"
    marker.write_text(str(backup))
    try:
        # Install a small guard before changing any other application file.
        # A process/power interruption must not launch a mixed application tree.
        launcher = target / "Start.cmd"
        if "Start.cmd" in journal["existed"]:
            guard = b'@if exist "%~dp0.fie-update-in-progress" (echo Interrupted update: re-run Update existing.cmd & exit /b 1)\r\n'
            staged_launcher = launcher.with_name("Start.cmd.fie-update")
            staged_launcher.write_bytes(guard + launcher.read_bytes())
            os.replace(staged_launcher, launcher)
        for relative in files:
            old = target / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            staged = old.with_name(old.name + ".fie-update")
            shutil.copy2(source / relative, staged)
            if hashlib.sha256(staged.read_bytes()).hexdigest() != hashes[relative.as_posix()]:
                raise ValueError(f"Copied update checksum mismatch: {relative}")
            os.replace(staged, old)
        imported = merge_research(source, target)
    except Exception:
        restore_transaction(target, backup, journal)
        raise
    journal["status"] = "completed"
    atomic_json(backup / "transaction.json", journal)
    marker.unlink(missing_ok=True)

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
