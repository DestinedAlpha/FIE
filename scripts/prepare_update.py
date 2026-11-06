"""Package the complete application and built UI as a small existing-installation update."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from dotenv import dotenv_values


def main():
    root = Path(__file__).resolve().parents[1]
    release = version("fpl-intelligence")
    output = root / "artifacts/runtime" / f"FPL-Intelligence-{release}-Update.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, text=True
    ).splitlines()
    files = {
        name
        for name in names
        if (root / name).is_file()
        and not name.startswith(("data/", "artifacts/runtime/", ".github/", "updates/"))
        and not Path(name).name.startswith(".env")
    }
    files.update(p.relative_to(root).as_posix() for p in (root / "apps/web/dist").rglob("*") if p.is_file())
    if not (root / "apps/web/dist/index.html").is_file():
        raise ValueError("Build the full interface before packaging an update")
    keys = [v.encode() for k, v in dotenv_values(root / ".env").items() if k.endswith("API_KEY") and v]
    hashes = {}
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as zipped:
        prefix = "FPL-Intelligence-Update/"
        for name in sorted(files):
            body = (root / name).read_bytes()
            if any(key in body for key in keys):
                raise ValueError("Credential found in application update")
            hashes[name] = hashlib.sha256(body).hexdigest()
            zipped.writestr(prefix + name, body)
        zipped.writestr(prefix + "app-files.json", json.dumps(sorted(files)))
        zipped.writestr(
            prefix + "update-checksums.json",
            json.dumps(
                {
                    "version": release,
                    "requires_existing_installation": True,
                    "existing_models_preserved": True,
                    "retired_tensor_weights_removed_on_start": True,
                    "files": hashes,
                },
                indent=2,
            ),
        )
        zipped.writestr(
            prefix + "UPDATE-README.txt",
            "Requires an existing FPL Intelligence installation. Extract separately, close the old app, and run Update existing.cmd. Select your existing application folder. Your account, keys, accepted tree models and historical data are preserved. The launcher removes retired tensor weights/dependencies, migrates and refits companion trees, validates the role-aware minutes model, then rebuilds forecasts and recommendations. Initial migration/training takes several minutes on a CPU. No account transfers are submitted.\n",
        )
    with ZipFile(output) as zipped:
        assert zipped.testzip() is None
    print(
        json.dumps(
            {
                "path": str(output),
                "bytes": output.stat().st_size,
                "files": len(files),
                "credentials": "excluded",
                "models": "existing full models preserved",
            }
        )
    )


if __name__ == "__main__":
    main()
