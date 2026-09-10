from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_registry_opens_verified_artifacts_and_displays_all_release_gates() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")

    assert 'id="artifactOpenLink"' in index
    assert "Open verified artifact" in index
    assert "Streaming parity" in index
    assert "Security" in index
    assert "artifactOpenLink" in javascript
    assert "/api/workstation/artifacts/${encodeURIComponent(artifact.id)}/content" in javascript


def test_expression_qualification_is_evidence_driven_not_manually_selected() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")

    assert 'id="expressionQualificationEvidence"' in index
    assert 'id="expressionPromoteButton"' in index
    assert '<output id="expressionQualification"' in index
    assert '<select id="expressionQualification"' not in index


def test_production_qualification_ui_requires_four_distinct_evidence_sources() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")

    for control in (
        "qualificationSubjectEvidence",
        "qualificationAutomatedEvidence",
        "qualificationLongFormEvidence",
        "qualificationExpressionEvidence",
        "qualificationSecurityEvidence",
        "qualificationContentEvidence",
        "qualificationComposeButton",
    ):
        assert f'id="{control}"' in index
    assert 'api("/api/workstation/qualifications/compose"' in javascript
    assert 'new Set(selected).size === selected.length' in javascript
    assert 'requiredCount === 4 && distinct' in javascript
    assert 'content_evidence_artifact_id' in javascript
    assert "backend verifies whether expression evidence is applicable" in index.lower()
    assert "qualification_status: \"qualified\"" not in javascript
