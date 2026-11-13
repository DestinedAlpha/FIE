"""Exercise a packaged update against a disposable copy of the real installation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import tomllib
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

from fpl_intelligence.storage import Store, atomic_json, now


def verify(archive):
    source = Path(__file__).resolve().parents[1]
    original = Store(source / "data")
    kinds = ("state", "market", "decision_model", "forecast", "selected_plan")
    expected = {kind: original.get(kind)["id"] for kind in kinds}
    weights = {
        str(path.relative_to(original.root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (original.root / "models").rglob("*.txt")
    }
    runtime = source / "artifacts/runtime"
    baseline = source / "updates/FPL-Intelligence-0.14.5-Update.zip"
    if (
        not baseline.is_file()
        or hashlib.sha256(baseline.read_bytes()).hexdigest()
        != "f01b7c127628dfa785d5cf01df8132d814c668adb967aff55c12a352b62bab60"
    ):
        raise ValueError("Real upgrade verification requires the checksum-verified published v0.14.5 archive")
    with tempfile.TemporaryDirectory(prefix="real-update-", dir=runtime) as temporary:
        scratch = Path(temporary)
        target, stage = scratch / "installed", scratch / "package"
        target.mkdir()
        with ZipFile(baseline) as zipped:
            prefix = "FPL-Intelligence-Update/"
            old_manifest = json.loads(zipped.read(prefix + "app-files.json"))
            for name in [*old_manifest, "app-files.json", "update-checksums.json"]:
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe baseline application path")
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(zipped.read(prefix + name))
        baseline_version = tomllib.loads((target / "pyproject.toml").read_text())["project"]["version"]
        assert baseline_version == "0.14.5"
        secret = source / ".env"
        if secret.exists():
            shutil.copy2(secret, target / ".env")
            (target / ".env").chmod(0o600)
        secret_hash = hashlib.sha256(secret.read_bytes()).hexdigest() if secret.exists() else None
        data = target / "data"
        data.mkdir()
        original.backup(data / "state.sqlite")
        for category in ("raw", "features", "normalized", "scenarios", "models"):
            origin = original.root / category
            if not origin.exists():
                continue
            # Immutable files are hardlinked, avoiding another multi-GB copy.
            shutil.copytree(origin, data / category, copy_function=os.link)
        baseline_check = r"""
import sys,json
from pathlib import Path
p=Path(sys.argv[1]);sys.path.insert(0,str(p/'src'))
from fpl_intelligence.api import create_app
import fpl_intelligence.api
from fastapi.testclient import TestClient
assert Path(fpl_intelligence.api.__file__).is_relative_to(p/'src')
c=TestClient(create_app(p/'data'),base_url='http://127.0.0.1')
assert c.get('/api/health').status_code==200
d=c.get('/api/dashboard');assert d.status_code==200 and d.json()['decision_model']['id']==json.loads(sys.argv[2])['decision_model']
assert c.get('/').status_code==200
print(json.dumps({'baseline_installed_source':True,'dashboard_and_built_ui':True}))
"""
        environment = dict(os.environ, GEMINI_API_KEY="", GROQ_API_KEY="")
        baseline_result = subprocess.run(
            [sys.executable, "-c", baseline_check, str(target.resolve()), json.dumps(expected)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        with ZipFile(archive) as zipped:
            manifest_name = next(name for name in zipped.namelist() if name.endswith("/app-files.json"))
            prefix = manifest_name[: -len("app-files.json")]
            manifest = json.loads(zipped.read(manifest_name))
            for name in [*manifest, "app-files.json", "update-checksums.json"]:
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe packaged path")
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(zipped.read(prefix + name))
        outcome = subprocess.run(
            [sys.executable, str(stage / "scripts/install_update.py"), "--target", str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        installed_check = r"""
import sys, json, hashlib, time
from pathlib import Path
p=Path(sys.argv[1]); sys.path.insert(0,str(p/'src'))
from fpl_intelligence.storage import Store
from fpl_intelligence.maintenance import maintain_models
from fpl_intelligence.decision import load_bundle
from fpl_intelligence.future_expert import load_expert
from fpl_intelligence.manager_policy import policy_models
from fpl_intelligence.readiness import operational_readiness
from fpl_intelligence.api import create_app
from fastapi.testclient import TestClient
import fpl_intelligence.api
assert Path(fpl_intelligence.api.__file__).is_relative_to(p/'src')
s=Store(p/'data'); expected=json.loads(sys.argv[2]); maintain_models(s)
assert {k:s.get(k)['id'] for k in expected}==expected
model=s.get('decision_model'); load_bundle(s,model)
role=s.get('role_minutes_model',model['role_minutes_model_id'])
assert hashlib.sha256(s.artifact_path(role['path']).read_bytes()).hexdigest()==role['hash']
load_expert(s,s.get('future_horizon_model',model['future_horizon_model_id'])); policy_models(s)
assert operational_readiness(s)['ready']
with TestClient(create_app(p/'data'),base_url='http://127.0.0.1') as c:
    expires=time.monotonic()+45
    while c.get('/api/dashboard').json()['active_jobs']:
        assert time.monotonic()<expires,'Unexpected long startup in current installation'
        time.sleep(.1)
    # Exercise the installed supervisor and child imports with real model data.
    job=c.post('/api/forecast',json={'worlds':20000}).json()['job_id']
    expires=time.monotonic()+15
    while not c.get('/api/jobs/'+job).json().get('progress',{}).get('pid'):
        assert time.monotonic()<expires,'Installed calculation worker did not start'
        time.sleep(.05)
    started=time.monotonic()
    c.post('/api/jobs/'+job+'/cancel')
    while c.get('/api/jobs/'+job).json()['status'] not in ('cancelled','failed','completed'):
        assert time.monotonic()-started<8,'Installed worker did not stop'
        time.sleep(.05)
    assert c.get('/api/jobs/'+job).json()['status']=='cancelled'
    cancelled_seconds=time.monotonic()-started
    assert not c.get('/api/dashboard').json()['active_jobs']
    health=c.get('/api/health'); assert health.status_code==200 and health.json()['readiness']['ready']
    dashboard=c.get('/api/dashboard'); assert dashboard.status_code==200 and not dashboard.json()['plan']['stale']
    assert c.get('/api/forecast/diagnostics').status_code==200
    training=c.get('/api/models/training');assert training.status_code==200 and len(training.json()['components'])==3
    plan=s.get('selected_plan')['plan_id']
    for fmt in ('json','csv','md','ics'):
        response=c.get('/api/export/'+plan+'/'+fmt); assert response.status_code==200 and response.content
assert {k:s.get(k)['id'] for k in expected}==expected
print(json.dumps({'installed_modules':True,'accepted_weights_loaded':True,'readiness':True,'http_exports':4,'installed_process_cancellation_seconds':cancelled_seconds}))
"""
        checked = subprocess.run(
            [sys.executable, "-c", installed_check, str(target.resolve()), json.dumps(expected)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        installed = Store(data)
        assert all(
            hashlib.sha256((data / name).read_bytes()).hexdigest() == value for name, value in weights.items()
        )
        if secret_hash:
            assert hashlib.sha256((target / ".env").read_bytes()).hexdigest() == secret_hash
        backups = list((target / "artifacts/runtime").glob("before-update-*/state.sqlite"))
        assert len(backups) == 1
        # Exercise recovery after a real account change and database loss in
        # the disposable copy. Restoring over a full WAL database would require
        # another database-sized journal without testing a different account.
        installed.put("state", {"recovery_probe": True})
        with closing(sqlite3.connect(backups[0])) as saved:
            assert saved.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        for suffix in ("", "-wal", "-shm"):
            (data / ("state.sqlite" + suffix)).unlink(missing_ok=True)
        with (
            closing(sqlite3.connect(backups[0])) as saved,
            closing(sqlite3.connect(data / "state.sqlite")) as destination,
        ):
            saved.backup(destination)
        assert {kind: installed.get(kind)["id"] for kind in kinds} == expected
        with installed.db() as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        result = {
            "at": now(),
            "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
            "baseline_version": baseline_version,
            "baseline_archive_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            "baseline_installed_checks": json.loads(baseline_result.stdout.splitlines()[-1]),
            "database_bytes": (data / "state.sqlite").stat().st_size,
            "installed_source_api_and_models": json.loads(checked.stdout.splitlines()[-1]),
            "accepted_weight_files_preserved": len(weights),
            "account_and_active_objects": "preserved",
            "credentials": "unchanged; not included in report",
            "real_database_backup_integrity_and_recovery": "passed",
            "installer_completed": bool(outcome.returncode == 0),
        }
        atomic_json(runtime / "installed-update-verification.json", result)
        print(json.dumps(result))
    return result


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
