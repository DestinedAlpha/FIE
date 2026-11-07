"""Create the user's private portable handoff; credentials and caches are excluded."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from dotenv import dotenv_values

from fpl_intelligence.database_archive import pack_database
from fpl_intelligence.storage import Store


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "runtime" / "FPL-Intelligence-Windows.zip"
    store = Store(root / "data")
    output.parent.mkdir(parents=True, exist_ok=True)
    secrets = [
        value.encode()
        for name, value in dotenv_values(root / ".env").items()
        if name.endswith("API_KEY") and value
    ]
    files = set(
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, text=True
        ).splitlines()
    )
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        backup = Path(temporary) / "state.sqlite"
        store.backup(backup)
        if secrets:
            overlap = max(map(len, secrets)) - 1
            tail = b""
            with backup.open("rb") as database:
                while block := database.read(1024 * 1024):
                    combined = tail + block
                    if any(secret in combined for secret in secrets):
                        raise ValueError("Credential found in account database; publication blocked")
                    tail = combined[-overlap:] if overlap else b""
        application_files = sorted(
            name
            for name in files
            if (root / name).is_file()
            and not name.startswith(("data/", "artifacts/runtime/", ".github/", "updates/"))
            and not Path(name).name.startswith(".env")
        )
        application_files += [
            path.relative_to(root).as_posix()
            for path in (root / "apps/web/dist").rglob("*")
            if path.is_file()
        ]
        with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=3) as bundle:
            bundle.writestr("FPL-Intelligence/app-files.json", json.dumps(application_files))
            bundle.writestr(
                "FPL-Intelligence/update-checksums.json",
                json.dumps(
                    {
                        "version": version("fpl-intelligence"),
                        "requires_existing_installation": False,
                        "existing_models_preserved": True,
                        "files": {
                            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                            for name in application_files
                        },
                    },
                    indent=2,
                ),
            )
            for name in sorted(files):
                path = root / name
                if (
                    path.is_file()
                    and not name.startswith(("artifacts/runtime/", "data/", "updates/"))
                    and not path.name.startswith(".env")
                ):
                    bundle.write(path, "FPL-Intelligence/" + name)
            for path in (root / "apps" / "web" / "dist").rglob("*"):
                if path.is_file():
                    bundle.write(path, "FPL-Intelligence/" + path.relative_to(root).as_posix())
            for part in pack_database(backup, Path(temporary) / "database"):
                bundle.write(part, "FPL-Intelligence/data/" + part.name)
            for folder in ("raw", "scenarios", "models", "features", "normalized"):
                for path in (store.root / folder).rglob("*"):
                    if path.is_file():
                        bundle.write(path, "FPL-Intelligence/" + path.relative_to(root).as_posix())
            account = store.root / "account-confirmed.json"
            if account.exists():
                bundle.write(account, "FPL-Intelligence/data/account-confirmed.json")
            # The personalized report belongs only in the private handoff.
            report = root / "artifacts/runtime/report"
            for path in report.rglob("*"):
                if path.is_file():
                    bundle.write(path, "FPL-Intelligence/reports/" + path.relative_to(report).as_posix())
        with ZipFile(output) as verify:
            assert "FPL-Intelligence/.env" not in verify.namelist()
            assert verify.testzip() is None
            for entry in verify.infolist():
                body = verify.read(entry)
                if entry.filename.endswith(".gz"):
                    body = gzip.decompress(body)
                if any(secret in body for secret in secrets):
                    raise ValueError("Credential found in private handoff; publication blocked")
    print(json.dumps({"bundle": str(output), "bytes": output.stat().st_size, "contains_credentials": False}))


if __name__ == "__main__":
    main()
