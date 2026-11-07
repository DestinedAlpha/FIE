"""Verify the actual downloaded portable ZIP against its prepared installation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from dotenv import dotenv_values

from fpl_intelligence.consensus import load_verified_models
from fpl_intelligence.database_archive import unpack_database
from fpl_intelligence.pipeline_replay import verify_pipeline_replay
from fpl_intelligence.shadow_accounts import advance_shadow_accounts
from fpl_intelligence.simulation import score_plan
from fpl_intelligence.storage import Store, atomic_json, now


def verify(archive: Path, *, reuse_verified_data=False):
    source = Path(__file__).resolve().parents[1]
    original = Store(source / "data")
    plan = original.get("plan", original.get("selected_plan")["plan_id"])
    forecast = original.get("forecast")
    kinds = (
        "state",
        "market",
        "decision_model",
        "forecast",
        "selected_plan",
        "pipeline_replay",
        "shadow_cohort",
    )
    kinds += tuple(
        k
        for k in (
            "horizon_refinement",
            "future_policy_replay",
            "future_policy_diagnostic",
            "future_policy_suite",
            "future_distribution_validation",
        )
        if original.list(k, 1)
    )
    expected_ids = {k: original.get(k)["id"] for k in kinds}
    secrets = [v.encode() for k, v in dotenv_values(source / ".env").items() if k.endswith("API_KEY") and v]
    runtime = source / "artifacts/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with (
        ZipFile(archive) as zipped,
        tempfile.TemporaryDirectory(prefix="verify-download-", dir=runtime) as scratch,
    ):
        if zipped.testzip() is not None:
            raise ValueError("Downloaded archive CRC check failed")
        manifests = [n for n in zipped.namelist() if n.endswith("/app-files.json")]
        if len(manifests) != 1:
            raise ValueError("Download does not contain exactly one application")
        prefix = manifests[0][: -len("app-files.json")]
        files = json.loads(zipped.read(manifests[0]))
        if any(Path(n).name.startswith(".env") and Path(n).name != ".env.example" for n in zipped.namelist()):
            raise ValueError("Private environment file found in downloaded archive")
        for name in files:
            if (
                hashlib.sha256(zipped.read(prefix + name)).digest()
                != hashlib.sha256((source / name).read_bytes()).digest()
            ):
                raise ValueError("Downloaded application file differs: " + name)
        for entry in zipped.infolist():
            body = zipped.read(entry)
            if entry.filename.endswith(".gz"):
                body = gzip.decompress(body)
            if any(secret in body for secret in secrets):
                raise ValueError("Credential found in downloaded archive")
        root = Path(scratch)
        model_count = 0
        for path in (source / "data/models").rglob("*"):
            if path.is_file():
                relative = path.relative_to(source)
                body = zipped.read(prefix + relative.as_posix())
                if hashlib.sha256(body).digest() != hashlib.sha256(path.read_bytes()).digest():
                    raise ValueError("Downloaded model differs: " + relative.as_posix())
                model_count += 1
        # Recover all source/feature/model dependencies, without copying unrelated
        # alternative world archives. ZIP names never control arbitrary paths.
        for entry in zipped.infolist():
            if not entry.filename.startswith(prefix + "data/"):
                continue
            relative = Path(entry.filename[len(prefix) :])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe downloaded path")
            if (
                len(relative.parts) > 2
                and relative.parts[1] in {"raw", "models", "features", "normalized"}
                or (relative.name.startswith("state.sqlite.") and len(relative.parts) == 2)
            ):
                if not entry.is_dir():
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    body = zipped.read(entry)
                    retained = source / relative
                    if reuse_verified_data and retained.is_file():
                        if hashlib.sha256(body).digest() != hashlib.sha256(retained.read_bytes()).digest():
                            raise ValueError("Downloaded immutable data differs: " + str(relative))
                        target.hardlink_to(retained)
                    else:
                        target.write_bytes(body)
        unpack_database(root / "data", root / "data/state.sqlite")
        if secrets:
            overlap, tail = max(map(len, secrets)) - 1, b""
            with (root / "data/state.sqlite").open("rb") as database:
                while block := database.read(1024 * 1024):
                    combined = tail + block
                    if any(secret in combined for secret in secrets):
                        raise ValueError("Credential found in restored download")
                    tail = combined[-overlap:] if overlap else b""
        restored = Store(root / "data")
        if any(restored.get(k)["id"] != value for k, value in expected_ids.items()):
            raise ValueError("Downloaded database does not match the prepared account and evidence")
        with restored.db() as database:
            if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Downloaded database failed full integrity verification")
        report = restored.get("pipeline_replay")
        pipeline = verify_pipeline_replay(restored, report)
        future = {}
        if restored.list("future_policy_replay", 1):
            from fpl_intelligence.future_policy_verification import verify_future_policy

            future["earlier_policy"] = verify_future_policy(restored, restored.get("future_policy_replay"))
            if restored.list("future_policy_diagnostic", 1):
                future["current_policy"] = verify_future_policy(
                    restored, restored.get("future_policy_diagnostic")
                )
        identity = restored.get("decision_model").get("future_horizon_model_id")
        if identity:
            from fpl_intelligence.future_expert import load_expert

            spec = restored.get("future_horizon_model", identity)
            before, after = load_expert(original, spec), load_expert(restored, spec)
            for name in spec["models"]:
                values = np.random.default_rng(71).normal(size=(3, before[name].num_feature()))
                if not np.array_equal(
                    before[name].predict(values, num_threads=1), after[name].predict(values, num_threads=1)
                ):
                    raise ValueError("Relocated future expert inference differs")
            if (
                hashlib.sha256(restored.artifact_path(spec["data_path"]).read_bytes()).hexdigest()
                != spec["data_hash"]
            ):
                raise ValueError("Relocated future expert training data differs")
            future["relocated_expert_inference"] = "identical"
        shadow = advance_shadow_accounts(restored)
        finalized = {e["id"] for e in restored.get("market")["events"] if e.get("data_checked")}
        if any(r["event"] not in finalized for r in shadow["outcomes"]):
            raise ValueError("A prospective outcome has not finalized")
        scenario = Path(forecast["scenario_path"]).name
        target = root / "data/scenarios" / scenario
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zipped.read(prefix + "data/scenarios/" + scenario))
        if target.suffix == ".json":
            world_manifest = json.loads(target.read_text())
            used_events = {str(row["event"]) for row in plan["ledger"]}
            needed = {
                v["path"]: v
                for key, v in world_manifest["arrays"].items()
                if key.rsplit("_", 1)[-1] in used_events
            }
            for item in needed.values():
                name = Path(item["path"]).name
                (target.parent / name).write_bytes(zipped.read(prefix + "data/scenarios/" + name))
        expected = score_plan(original, plan, forecast)
        actual = score_plan(restored, restored.get("plan", plan["id"]), restored.get("forecast"))
        if not np.array_equal(expected, actual):
            raise ValueError("Restored plan outcomes differ")
        consensus = original.get("consensus_model")
        for spec in (consensus["tree_spec"], consensus["reference_spec"]):
            before, after = load_verified_models(original, spec), load_verified_models(restored, spec)
            for name, model in before.items():
                values = np.random.default_rng(71).normal(size=(3, model.num_feature()))
                if not np.array_equal(
                    model.predict(values, num_threads=1), after[name].predict(values, num_threads=1)
                ):
                    raise ValueError("Relocated model inference differs")
        if json.loads(zipped.read(prefix + "reports/plan.json"))["id"] != plan["id"]:
            raise ValueError("The downloaded personalized report is stale")
        result = {
            "at": now(),
            "archive": str(archive),
            "bytes": archive.stat().st_size,
            "crc": "passed",
            "immutable_data_recovery": "byte-verified hardlinks" if reuse_verified_data else "ZIP extraction",
            "application_files": len(files),
            "model_files": model_count,
            "database": "ok",
            "credentials": "excluded",
            "pipeline": pipeline,
            "future_expert": future,
            "prospective_pending": shadow["pending_deadlines"],
            "restored_worlds_identical": len(actual),
            "account_ids": expected_ids,
            "plan_id": plan["id"],
        }
    release = version("fpl-intelligence")
    atomic_json(runtime / f"handoff-{release}-verification.json", result)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--reuse-verified-data",
        action="store_true",
        help="Use byte-verified immutable hardlinks when disk space is limited; SQLite and active worlds are still restored from the ZIP",
    )
    arguments = parser.parse_args()
    verify(arguments.archive, reuse_verified_data=arguments.reuse_verified_data)
