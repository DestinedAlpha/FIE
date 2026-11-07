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
    with tempfile.TemporaryDirectory(prefix="real-update-", dir=runtime) as temporary:
        scratch = Path(temporary)
        target, stage = scratch / "installed", scratch / "package"
        target.mkdir()
        (target / "pyproject.toml").write_text('name = "fpl-intelligence"\nversion = "0.14.1"\n')
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
    health=c.get('/api/health'); assert health.status_code==200 and health.json()['readiness']['ready']
    dashboard=c.get('/api/dashboard'); assert dashboard.status_code==200 and not dashboard.json()['plan']['stale']
    assert c.get('/api/forecast/diagnostics').status_code==200
    plan=s.get('selected_plan')['plan_id']
    for fmt in ('json','csv','md','ics'):
        response=c.get('/api/export/'+plan+'/'+fmt); assert response.status_code==200 and response.content
assert {k:s.get(k)['id'] for k in expected}==expected
print(json.dumps({'installed_modules':True,'accepted_weights_loaded':True,'readiness':True,'http_exports':4}))
"""
        environment = dict(os.environ, GEMINI_API_KEY="", GROQ_API_KEY="")
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
        # Exercise recovery after a real account change in the disposable copy.
        installed.put("state", {"recovery_probe": True})
        with closing(sqlite3.connect(backups[0])) as saved, installed.db() as destination:
            assert saved.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            saved.backup(destination)
        assert {kind: installed.get(kind)["id"] for kind in kinds} == expected
        with installed.db() as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        result = {
            "at": now(),
            "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
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
