"""Verify the downloaded application update and a real installation preserving existing data."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from fpl_intelligence.storage import atomic_json, now


def verify(archive: Path):
    source = Path(__file__).resolve().parents[1]
    with ZipFile(archive) as zipped, tempfile.TemporaryDirectory(dir=source / "artifacts/runtime") as scratch:
        assert zipped.testzip() is None
        entries = [n for n in zipped.namelist() if n.endswith("/update-checksums.json")]
        assert len(entries) == 1
        prefix = entries[0][: -len("update-checksums.json")]
        metadata = json.loads(zipped.read(entries[0]))
        manifest = json.loads(zipped.read(prefix + "app-files.json"))
        assert metadata["requires_existing_installation"] and set(manifest) == set(metadata["files"])
        stage = Path(scratch) / "update"
        stage.mkdir()
        for name in manifest:
            relative = Path(name)
            assert (
                not relative.is_absolute()
                and ".." not in relative.parts
                and relative.parts[0] not in ("data", ".git", ".venv")
            )
            assert not relative.name.startswith(".env")
            data = zipped.read(prefix + name)
            assert hashlib.sha256(data).hexdigest() == metadata["files"][name]
            assert data == (source / relative).read_bytes()
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (stage / "app-files.json").write_text(json.dumps(manifest))
        (stage / "update-checksums.json").write_text(json.dumps(metadata))
        target = Path(scratch) / "existing"
        target.mkdir()
        (target / "pyproject.toml").write_text('name = "fpl-intelligence"\n')
        (target / ".env").write_text("EXISTING_PRIVATE_KEY=preserved\n")
        models = target / "data/models"
        models.mkdir(parents=True)
        model = models / "existing-model.bin"
        model.write_bytes(b"existing-complete-numerical-model")
        retired = models / "retired.pt"
        retired.write_bytes(b"retired-tensor-model")
        with sqlite3.connect(target / "data/state.sqlite") as db:
            db.execute("CREATE TABLE account (bank INTEGER, points INTEGER)")
            db.execute("INSERT INTO account VALUES (16,470)")
        subprocess.run(
            [sys.executable, str(stage / "scripts/install_update.py"), "--target", str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; p=Path(sys.argv[1]); sys.path.insert(0,str(p/'src')); from fpl_intelligence.storage import Store; from fpl_intelligence.maintenance import maintain_models; maintain_models(Store(p/'data'))",
                str(target),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert not retired.exists()
        assert (target / ".env").read_text() == "EXISTING_PRIVATE_KEY=preserved\n"
        assert model.read_bytes() == b"existing-complete-numerical-model"
        with sqlite3.connect(target / "data/state.sqlite") as db:
            assert db.execute("SELECT bank,points FROM account").fetchall() == [(16, 470)]
        assert len(list((target / "artifacts/runtime").glob("before-update-*/state.sqlite"))) == 1
        for name in manifest:
            assert hashlib.sha256((target / name).read_bytes()).hexdigest() == metadata["files"][name]
    result = {
        "at": now(),
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "version": metadata["version"],
        "application_files": len(manifest),
        "crc": "passed",
        "application_checksums": "passed",
        "actual_installation": "passed",
        "account_database": "preserved",
        "keys": "preserved",
        "existing_models": "preserved",
        "retired_weights": "removed by installed migration",
        "backup": "verified",
    }
    atomic_json(source / "artifacts/runtime/update-verification.json", result)
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
