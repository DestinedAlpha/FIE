"""Run release checks and retain actual machine-readable results, never assumed passes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from fpl_intelligence.storage import atomic_json, now


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/runtime/checks"
    output.mkdir(parents=True, exist_ok=True)
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not npm:
        raise ValueError("Node/npm is required for release browser checks")
    commands = [
        ("ruff", [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"], root),
        ("format", [sys.executable, "-m", "ruff", "format", "--check", "src", "scripts", "tests"], root),
        ("mypy", [sys.executable, "-m", "mypy", "src"], root),
        ("backend", [sys.executable, "-m", "pytest", "-q", f"--junitxml={output / 'backend.xml'}"], root),
        ("frontend_build", [npm, "run", "build"], root / "apps/web"),
        (
            "frontend_unit",
            [npm, "test", "--", "--reporter=json", f"--outputFile={output / 'frontend.json'}"],
            root / "apps/web",
        ),
        ("browser", [npm, "run", "test:e2e", "--", "--reporter=json"], root / "apps/web"),
    ]
    results = []
    for name, command, cwd in commands:
        print("Checking", name, flush=True)
        completed = subprocess.run(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        log = output / (name + ".log")
        log.write_text(completed.stdout)
        row = {
            "name": name,
            "returncode": completed.returncode,
            "passed": completed.returncode == 0,
            "log": str(log),
            "finished_at": now(),
        }
        results.append(row)
        if completed.returncode:
            print(completed.stdout[-10000:])
            atomic_json(output / "checks.json", {"at": now(), "results": results, "passed": False})
            raise SystemExit(completed.returncode)
    tree = ET.parse(output / "backend.xml")
    backend = sum(int(e.attrib.get("tests", 0)) for e in tree.iter("testsuite"))
    skipped = sum(int(e.attrib.get("skipped", 0)) for e in tree.iter("testsuite"))
    frontend = json.loads((output / "frontend.json").read_text())
    # npm prints a short banner before Playwright's JSON report.
    browser_log = (output / "browser.log").read_text()
    browser = json.loads(browser_log[browser_log.index("{") :])
    stats = browser["stats"]
    report = {
        "at": now(),
        "passed": all(r["passed"] for r in results),
        "results": results,
        "backend_tests": backend - skipped,
        "backend_skipped": skipped,
        "frontend_unit_tests": frontend["numPassedTests"],
        "browser_journeys": stats["expected"],
        "browser_failed": stats["unexpected"],
        "browser_skipped": stats["skipped"],
    }
    atomic_json(output / "checks.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
