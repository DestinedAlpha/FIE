"""Trace implemented requirements to executed checks and concrete runtime artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .storage import atomic_json, now


def review_specification(root: Path, evidence: list[dict]) -> dict:
    """Fail on unreviewed document changes, omitted sections or broken evidence.

    Each source body line remains visible for human inspection. Passing this
    structural check establishes traceability, not semantic proof of a feature.
    """
    manifest = json.loads((root / "configs/specification_sections.json").read_text(encoding="utf-8"))
    mapped = {row["id"]: row for row in evidence}
    clauses, sections = [], []
    for document in manifest["reviewed_documents"]:
        path = root / document["path"]
        # Git's Windows checkout may change LF to CRLF without changing text.
        # Normalize documentation newlines only; model/input hashes stay binary.
        source = path.read_text(encoding="utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != document["sha256"]:
            raise ValueError(f"Specification changed without review: {document['path']}")
        declared = [s for s in manifest["sections"] if s["document"] == document["path"]]
        if len({s["number"] for s in declared}) != len(declared):
            raise ValueError("Duplicate specification section")
        lookup = {s["number"]: s for s in declared}
        observed = set()
        section = None
        for line, text in enumerate(source.splitlines(), 1):
            heading = re.match(r"^## (\d+)\. (.+)$", text)
            if heading:
                number = int(heading[1])
                if number not in lookup or lookup[number]["title"] != heading[2]:
                    raise ValueError(f"Unreviewed specification section: {document['path']}:{line}")
                section = lookup[number]
                if section["line"] != line or not section["evidence_ids"]:
                    raise ValueError("Specification section needs a current source location and evidence")
                if any(identity not in mapped for identity in section["evidence_ids"]):
                    raise ValueError("Specification references unknown implementation evidence")
                observed.add(number)
                sections.append(section)
            elif text.strip() and not text.startswith("#") and text.strip() not in ("---", "~~~"):
                # The supplied introductory prose is context, indexed separately.
                clauses.append(
                    {
                        "document": document["path"],
                        "line": line,
                        "text": text,
                        "section": section["number"] if section else "preamble",
                        "evidence_ids": section["evidence_ids"] if section else [],
                    }
                )
        if observed != lookup.keys():
            raise ValueError("Specification manifest contains missing or unobserved sections")
    return {
        "review_scope": manifest["review_scope"],
        "documents": manifest["reviewed_documents"],
        "sections_reviewed": len(sections),
        "body_lines_indexed": len(clauses),
        "sections": sections,
        "source_clauses": clauses,
        "verification_scope": "Complete source indexing plus reviewed section-level evidence links. Individual behavioral regressions are reported separately; this is not exhaustive behavioral testing or prospective performance acceptance.",
    }


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
        "specification_review": review_specification(root, rows),
        "scientific_limits": [
            "A rejected model is a completed experiment, not a deployed improvement.",
            "Unavailable optional inputs are recorded as unavailable; no observations are invented.",
            "Prospective competitive advantage and future ranks require future finalized matches.",
        ],
    }
    report["id"] = store.put("implementation_audit", report)
    atomic_json(root / "artifacts/evaluations/implementation_audit.json", report)
    atomic_json(root / "artifacts/evaluations/specification_review.json", report["specification_review"])
    return report
