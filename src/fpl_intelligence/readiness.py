"""Trace implemented requirements to executed checks and concrete runtime artifacts."""

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from .storage import atomic_json, now


def build_readiness(store):
    root = Path(__file__).resolve().parents[2]
    catalog = json.loads((root / "configs/implementation_requirements.json").read_text())
    checks = json.loads((root / "artifacts/runtime/checks/checks.json").read_text())
    if not checks["passed"]:
        raise ValueError("Release checks have not all passed")
    store.put("release_checks", checks)
    junit = ET.parse(root / "artifacts/runtime/checks/backend.xml")
    passed_tests = {
        (case.attrib["classname"].replace(".", "/") + ".py", case.attrib["name"].split("[")[0])
        for case in junit.iter("testcase")
        if not any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
    }
    rows = []
    for item in catalog["items"]:
        for field in ("implementation", "test"):
            path, symbol = item[field].split(":")
            tree = ast.parse((root / path).read_text())
            symbols = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            if symbol not in symbols:
                raise ValueError(f"Missing {field} mapping: {item['id']} / {symbol}")
            if field == "test" and (path, symbol) not in passed_tests:
                raise ValueError(f"Mapped regression did not pass in this run: {item['id']}")
        kind = item["runtime_evidence_kind"]
        if kind == "plan":
            artifact = store.get("plan", store.get("selected_plan")["plan_id"])
        else:
            artifact = store.get(kind)
        rows.append(
            {
                **item,
                "implementation_symbol_found": True,
                "mapped_regression_passed": True,
                "runtime_artifact_id": artifact["id"],
            }
        )
    report = {
        "at": now(),
        "requirements_traced": len(rows),
        "verification_scope": "Implementation symbols, executed mapped regressions and existing runtime evidence; not a proof that all possible behaviours are error-free",
        "scope": catalog["scope"],
        "checks_at": checks["at"],
        "items": rows,
        "scientific_limits": [
            "A rejected model is a completed experiment, not a deployed improvement.",
            "Unavailable optional inputs are recorded as unavailable; no observations are invented.",
            "Prospective competitive advantage and future ranks require future finalized matches.",
        ],
    }
    report["id"] = store.put("implementation_audit", report)
    atomic_json(root / "artifacts/evaluations/implementation_audit.json", report)
    return report
