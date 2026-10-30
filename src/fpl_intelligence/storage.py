"""Immutable content-addressed inputs and transactional, append-only local state."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    # Normalize mapping keys through JSON first: integer keys become strings on
    # disk, and lexical versus numeric ordering must not change the content ID.
    canonical = json.loads(json.dumps(value, allow_nan=False))
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Store:
    def __init__(self, root: str | Path | None = None):
        configured_root: str | Path = root if root is not None else (os.getenv("FPL_DATA_DIR") or "data")
        self.root = Path(configured_root)
        self.root.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY, source TEXT, url TEXT, captured_at TEXT,
                source_published_at TEXT, hash TEXT, path TEXT, status INTEGER,
                UNIQUE(source, captured_at, hash));
            CREATE TABLE IF NOT EXISTS objects (
                kind TEXT, id TEXT, created_at TEXT, body TEXT, PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, object_id TEXT,
                recorded_at TEXT, body TEXT);
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, status TEXT, created_at TEXT, updated_at TEXT,
                request TEXT, result TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS quota (
                day TEXT, provider TEXT, calls INTEGER, tokens INTEGER,
                PRIMARY KEY(day,provider));
            """)

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.root / "state.sqlite", timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def snapshot(self, source: str, url: str, body: Any, published: str | None = None) -> dict:
        sha = digest(body)
        path = self.root / "raw" / f"{sha}.json.gz"
        if not path.exists():
            # Exclusive creation prevents a concurrent refresh from replacing immutable content.
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(gzip.compress(json.dumps(body, sort_keys=True).encode(), mtime=0))
                    f.flush()
                    os.fsync(f.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass
            finally:
                os.unlink(temporary)
        record = dict(
            source=source,
            url=url,
            captured_at=now(),
            source_published_at=published,
            hash=sha,
            path=str(path.relative_to(self.root)),
            status=200,
        )
        with self.db() as db:
            db.execute(
                "INSERT INTO snapshots(source,url,captured_at,source_published_at,hash,path,status) "
                "VALUES(:source,:url,:captured_at,:source_published_at,:hash,:path,:status)",
                record,
            )
        return record

    def latest(
        self, source: str, cutoff: str | None = None, content_hash: str | None = None
    ) -> tuple[Any, dict]:
        with self.db() as db:
            query = "SELECT * FROM snapshots WHERE source=? AND captured_at<=? "
            args = [source, cutoff or now()]
            if content_hash:
                query += "AND hash=? "
                args.append(content_hash)
            row = db.execute(query + "ORDER BY captured_at DESC,id DESC LIMIT 1", args).fetchone()
        if row is None:
            raise ValueError(f"No stored snapshot for {source}; run fpl ingest")
        meta = dict(row)
        raw = gzip.decompress((self.root / meta["path"]).read_bytes())
        body = json.loads(raw)
        if digest(body) != meta["hash"] and hashlib.sha256(raw).hexdigest() != meta["hash"]:
            raise ValueError(f"Snapshot integrity failure: {source}")
        return body, meta

    def put(self, kind: str, body: dict) -> str:
        identity = digest(body)
        with self.db() as db:
            db.execute(
                "INSERT OR IGNORE INTO objects VALUES(?,?,?,?)",
                (kind, identity, now(), json.dumps(body, allow_nan=False)),
            )
        return identity

    def get(self, kind: str, identity: str | None = None) -> dict:
        with self.db() as db:
            row = db.execute(
                "SELECT body,id FROM objects WHERE kind=? "
                + ("AND id=?" if identity else "ORDER BY created_at DESC,rowid DESC LIMIT 1"),
                (kind, identity) if identity else (kind,),
            ).fetchone()
        if row is None and kind == "state" and identity is not None:
            with self.db() as db:
                row = db.execute(
                    "SELECT body,id FROM objects WHERE kind='planning_state' AND id=?", (identity,)
                ).fetchone()
        if row is None:
            raise ValueError(f"No {kind} artifact available")
        return json.loads(row["body"]) | {"id": row["id"]}

    def list(self, kind: str, limit: int = 100) -> list[dict]:
        with self.db() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,created_at FROM objects WHERE kind=? ORDER BY created_at DESC,rowid DESC LIMIT ?",
                    (kind, limit),
                )
            ]

    def event(self, kind: str, object_id: str, body: dict) -> None:
        with self.db() as db:
            db.execute(
                "INSERT INTO events(kind,object_id,recorded_at,body) VALUES(?,?,?,?)",
                (kind, object_id, now(), json.dumps(body)),
            )

    def backup(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as source:
            with closing(sqlite3.connect(destination)) as target:
                source.backup(target)

    def artifact_path(self, reference: str) -> Path:
        value = Path(reference)
        if ".." in value.parts:
            raise ValueError("Unrecognized local artifact reference")
        if len(value.parts) >= 3 and value.parts[-3] == "models":
            return self.root / "models" / value.parts[-2] / value.name
        if len(value.parts) < 2 or value.parts[-2] not in {"scenarios", "models", "features", "normalized"}:
            raise ValueError("Unrecognized local artifact reference")
        return self.root / value.parts[-2] / value.name
