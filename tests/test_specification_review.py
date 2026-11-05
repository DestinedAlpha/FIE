import hashlib
import json
from pathlib import Path

import pytest

from fpl_intelligence.readiness import review_specification


def test_specification_review_rejects_unreviewed_changes_and_missing_sections(tmp_path):
    (tmp_path / "configs").mkdir()
    source = tmp_path / "spec.md"
    source.write_text("# Specification\n\n## 1. Data\nKeep dated facts.\n\n## 2. State\nUse integer money.\n")
    manifest = {
        "review_scope": "Test source review",
        "reviewed_documents": [
            {"path": "spec.md", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        ],
        "sections": [
            {"document": "spec.md", "number": 1, "title": "Data", "line": 3, "evidence_ids": ["facts"]},
            {"document": "spec.md", "number": 2, "title": "State", "line": 6, "evidence_ids": ["money"]},
        ],
    }
    path = tmp_path / "configs/specification_sections.json"
    path.write_text(json.dumps(manifest))
    evidence = [{"id": "facts"}, {"id": "money"}]
    result = review_specification(tmp_path, evidence)
    assert result["sections_reviewed"] == 2 and result["body_lines_indexed"] == 2
    assert result["source_clauses"][1]["evidence_ids"] == ["money"]
    source.write_text(source.read_text() + "Handle profit sharing.\n")
    with pytest.raises(ValueError, match="changed without review"):
        review_specification(tmp_path, evidence)
    manifest["reviewed_documents"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["sections"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unreviewed specification section"):
        review_specification(tmp_path, evidence)


def test_shipped_specification_manifest_covers_both_original_documents():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads((root / "configs/implementation_requirements.json").read_text())["items"]
    result = review_specification(root, evidence)
    assert result["sections_reviewed"] == 36
    assert len(result["documents"]) == 2
    assert result["body_lines_indexed"] > 500
