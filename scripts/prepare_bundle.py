"""Create the user's private portable handoff; credentials and caches are excluded."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fpl_intelligence.storage import Store


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "runtime" / "FPL-Intelligence-Windows.zip"
    store = Store(root / "data")
    files = set(
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, text=True
        ).splitlines()
    )
    with tempfile.TemporaryDirectory() as temporary:
        backup = Path(temporary) / "state.sqlite"
        store.backup(backup)
        with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=3) as bundle:
            for name in sorted(files):
                path = root / name
                if (
                    path.is_file()
                    and not name.startswith(("artifacts/runtime/", "data/"))
                    and path.name != ".env"
                ):
                    bundle.write(path, "FPL-Intelligence/" + name)
            for path in (root / "apps" / "web" / "dist").rglob("*"):
                if path.is_file():
                    bundle.write(path, "FPL-Intelligence/" + path.relative_to(root).as_posix())
            bundle.write(backup, "FPL-Intelligence/data/state.sqlite")
            for folder in ("raw", "scenarios", "models", "features", "normalized"):
                for path in (store.root / folder).rglob("*"):
                    if path.is_file():
                        bundle.write(path, "FPL-Intelligence/" + path.relative_to(root).as_posix())
            account = store.root / "account-confirmed.json"
            if account.exists():
                bundle.write(account, "FPL-Intelligence/data/account-confirmed.json")
        with ZipFile(output) as verify:
            assert "FPL-Intelligence/.env" not in verify.namelist()
            assert verify.testzip() is None
    print(json.dumps({"bundle": str(output), "bytes": output.stat().st_size, "contains_credentials": False}))


if __name__ == "__main__":
    main()
