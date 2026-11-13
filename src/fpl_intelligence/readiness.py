"""Trace implemented requirements to executed checks and concrete runtime artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .storage import atomic_json, now


def workload_is_current(store, forecast, model=None):
    model = model if model is not None else store.get("decision_model")
    if not (model.get("role_minutes_model_id") or model.get("fusion_model_id")):
        return True
    manifest = forecast.get("input_manifest", {})
    current = store.get("workload")["id"] if store.list("workload", 1) else None
    return "live_workload_id" in manifest and manifest["live_workload_id"] == current


def forecast_dependencies_current(store, forecast, model=None):
    if not workload_is_current(store, forecast, model):
        return False
    for key, kind in (
        ("manager_policy_id", "manager_policy"),
        ("manager_behavior_id", "manager_behavior"),
        ("source_challenger_model_id", "fusion_model"),
    ):
        current = store.get(kind)["id"] if store.list(kind, 1) else None
        if forecast.get(key) != current:
            return False
    return True


def selection_is_current(store, plan):
    """Check all serving pointers and the deadline immediately before publication."""
    try:
        state = store.get("state")
        market = store.get("market")
        forecast = store.get("forecast")
        model = store.get("decision_model")
    except ValueError:
        return False
    if (
        any(
            plan.get(key) != value
            for key, value in (
                ("state_id", state["id"]),
                ("market_id", market["id"]),
                ("forecast_id", forecast["id"]),
            )
        )
        or forecast.get("market_id") != market["id"]
        or forecast.get("decision_model_id") != model["id"]
        or not forecast_dependencies_current(store, forecast, model)
    ):
        return False
    from .account_sync import next_event

    return not market.get("events") or state.get("event") == next_event(market)


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


def operational_readiness(store):
    """Serving readiness is distinct from database liveness; never mutate the account."""
    reasons = []
    objects = {}
    for kind in ("state", "market", "forecast", "decision_model", "selected_plan"):
        try:
            objects[kind] = store.get(kind)
        except ValueError:
            reasons.append(f"Missing {kind}")
    if reasons:
        return {"ready": False, "reasons": reasons}
    state, market, forecast, model = (objects[k] for k in ("state", "market", "forecast", "decision_model"))
    weight_paths: list[Path] = []
    if model.get("directory"):
        weight_paths.extend(
            store.root / "models" / model["directory"] / (name + ".txt")
            for name in ("points", "minutes", "starter")
        )
    else:
        reasons.append("Core model weight manifest unavailable")
    for key, kind in (
        ("role_minutes_model_id", "role_minutes_model"),
        ("future_horizon_model_id", "future_horizon_model"),
    ):
        if model.get(key):
            try:
                companion = store.get(kind, model[key])
                references = (
                    [companion["path"]]
                    if "path" in companion
                    else [item["path"] for item in companion.get("models", {}).values()]
                )
                weight_paths.extend(store.artifact_path(reference) for reference in references)
            except (KeyError, ValueError):
                reasons.append(f"Missing {kind} weight manifest")
    if any(not path.is_file() for path in weight_paths):
        reasons.append("One or more fitted weight files are missing")
    if forecast.get("market_id") != market["id"]:
        reasons.append("Forecast uses an older market")
    if forecast.get("decision_model_id") != model["id"]:
        reasons.append("Forecast uses an older model")
    if not workload_is_current(store, forecast, model):
        reasons.append("Forecast needs current workload inputs")
    elif not forecast_dependencies_current(store, forecast, model):
        reasons.append("Forecast needs current companion models")
    from .account_sync import next_event

    if market.get("events") and state.get("event") != next_event(market):
        reasons.append("Account needs deadline reconciliation")
    try:
        plan = store.get("plan", objects["selected_plan"].get("plan_id"))
        if any(
            plan.get(key) != value
            for key, value in (
                ("state_id", state["id"]),
                ("market_id", market["id"]),
                ("forecast_id", forecast["id"]),
            )
        ):
            reasons.append("Selected plan uses superseded inputs")
    except ValueError:
        reasons.append("No selected plan available")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "scope": "Input lineage, weight-file existence and deadline readiness. Weight integrity is verified by model loaders; this is not a predictive-quality certificate",
    }
