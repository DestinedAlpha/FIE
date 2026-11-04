"""Verified, size-bounded portable database archives, including legacy ZIPs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


def pack_database(database: Path, directory: Path, part_bytes=64 * 1024**2):
    if not 1024 <= part_bytes <= 90 * 1024**2:
        raise ValueError("Database part size must be between 1KiB and 90MiB")
    directory.mkdir(parents=True, exist_ok=True)
    parts: list[dict] = []
    total, raw_hash = 0, hashlib.sha256()
    with database.open("rb") as original:
        while body := original.read(part_bytes):
            path = directory / f"state.sqlite.part-{len(parts):04d}.gz"
            with path.open("wb") as output:
                with gzip.GzipFile(
                    filename="", fileobj=output, mode="wb", compresslevel=6, mtime=0
                ) as packed:
                    packed.write(body)
            raw_hash.update(body)
            total += len(body)
            with path.open("rb") as packed:
                checksum = hashlib.file_digest(packed, "sha256").hexdigest()
            parts.append({"name": path.name, "bytes": path.stat().st_size, "sha256": checksum})
    manifest = directory / "state.sqlite.manifest.json"
    manifest.write_text(
        json.dumps(
            {"version": 1, "raw_bytes": total, "raw_sha256": raw_hash.hexdigest(), "parts": parts}, indent=2
        )
    )
    return [manifest, *(directory / p["name"] for p in parts)]


def unpack_database(directory: Path, destination: Path):
    """Restore atomically; never replace a user's existing database."""
    if destination.exists():
        raise ValueError("Refusing to replace an existing account database")
    manifest = directory / "state.sqlite.manifest.json"
    spec = json.loads(manifest.read_text()) if manifest.exists() else None
    if spec is not None:
        if spec.get("version") != 1 or not spec.get("parts"):
            raise ValueError("Unsupported or empty database archive manifest")
        paths = []
        for index, item in enumerate(spec["parts"]):
            name = item["name"]
            if (
                not re.fullmatch(r"state\.sqlite\.part-\d{4}\.gz", name)
                or name != f"state.sqlite.part-{index:04d}.gz"
            ):
                raise ValueError("Unsafe or unordered database archive part")
            path = directory / name
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise ValueError("Missing or truncated database archive part: " + name)
            with path.open("rb") as packed:
                if hashlib.file_digest(packed, "sha256").hexdigest() != item["sha256"]:
                    raise ValueError("Database archive checksum mismatch: " + name)
            paths.append(path)
    else:
        paths = [directory / "state.sqlite.gz"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".database-restore-", dir=destination.parent)
    restored = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            for path in paths:
                with gzip.open(path, "rb") as packed:
                    shutil.copyfileobj(packed, output)
            output.flush()
            os.fsync(output.fileno())
        if spec is not None:
            with restored.open("rb") as original:
                checksum = hashlib.file_digest(original, "sha256").hexdigest()
            if restored.stat().st_size != spec["raw_bytes"] or checksum != spec["raw_sha256"]:
                raise ValueError("Restored database does not match its manifest")
        with closing(sqlite3.connect(restored)) as database:
            if database.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Restored account database failed integrity verification")
        if destination.exists():
            raise ValueError("An account database appeared during restoration; retained it")
        restored.replace(destination)
    finally:
        restored.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    options = parser.parse_args()
    unpack_database(options.directory, options.directory / "state.sqlite")
