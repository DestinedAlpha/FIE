"""Apply packaged application files while preserving local account data and keys."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


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
        with sqlite3.connect(database) as original, sqlite3.connect(backup / "state.sqlite") as saved:
            original.backup(saved)
    for relative in files:
        old = target / relative
        if old.is_file():
            (backup / "app" / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, backup / "app" / relative)
        old.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, old)
    return {
        "updated_files": len(files),
        "backup": str(backup),
        "account_database_replaced": False,
        "provider_keys_replaced": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(install(Path(__file__).resolve().parents[1], options.target)))
