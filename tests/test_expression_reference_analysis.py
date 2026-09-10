from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_webui import _mock_upstream

from aniflive_tts.expression_reference_analysis import (
    ANALYSIS_SCHEMA,
    ExpressionReferenceAnalysisError,
    analyze_pcm,
    analyze_reference_file,
)
from aniflive_tts.webui import create_webui_app
from aniflive_tts.workstation import WorkstationError, WorkstationStore


def _write_reference(
    path: Path,
    *,
    frequency: float = 220.0,
    leading_seconds: float = 0.15,
    tone_seconds: float = 0.9,
) -> bytes:
    sample_rate = 16_000
    leading = np.zeros(round(leading_seconds * sample_rate), dtype=np.float32)
    timeline = np.arange(round(tone_seconds * sample_rate), dtype=np.float32) / sample_rate
    tone = 0.42 * np.sin(2.0 * np.pi * frequency * timeline)
    samples = np.concatenate((leading, tone)).astype(np.float32)
    payload = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(payload)
    return path.read_bytes()


def test_non_neural_reference_analysis_measures_signal_without_affect_labels(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.wav"
    payload = _write_reference(reference)
    digest = hashlib.sha256(payload).hexdigest()

    result = analyze_reference_file(
        reference,
        expected_sha256=digest,
        language="yue",
        transcript="今日 天氣 真係 幾好",
    )

    assert result["schema"] == ANALYSIS_SCHEMA
    assert result["reference_sha256"] == digest
    assert result["provenance"]["neural_inference"] is False
    assert result["provenance"]["decoder"] == "python-wave-pcm-v1"
    assert result["measurements"]["duration_seconds"] == pytest.approx(1.05, abs=0.002)
    assert result["measurements"]["onset_seconds"] == pytest.approx(0.15, abs=0.011)
    assert result["measurements"]["rms_dbfs"] < 0
    pitch = result["measurements"]["pitch"]
    assert pitch["reliable"] is True
    assert pitch["median_hz"] == pytest.approx(220.0, abs=3.0)
    rate = result["measurements"]["speaking_rate"]
    assert rate["unit"] == "characters-per-second"
    assert rate["count"] == 8
    assert '"transcript":' not in json.dumps(result)
    assert len(result["waveform"]["bins"]) == 320
    assert "emotion" not in result
    assert "vad" not in result


def test_pitch_and_speaking_rate_are_omitted_without_reliable_evidence() -> None:
    samples = np.zeros(16_000, dtype=np.float32)
    result = analyze_pcm(
        samples,
        16_000,
        reference_sha256="a" * 64,
        decoder="test-pcm",
        language="en",
    )
    assert result["measurements"]["pitch"]["reliable"] is False
    assert "median_hz" not in result["measurements"]["pitch"]
    assert "speaking_rate" not in result["measurements"]
    assert result["measurements"]["onset_seconds"] is None


def test_reference_analysis_rejects_a_changed_registered_sha(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    _write_reference(reference)
    with pytest.raises(ExpressionReferenceAnalysisError, match="registered SHA256"):
        analyze_reference_file(
            reference,
            expected_sha256="0" * 64,
            language="ja",
        )


def test_expression_store_persists_read_only_analysis_and_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_root = tmp_path / "references"
    import_root.mkdir()
    reference = import_root / "generic-v2proplus.wav"
    _write_reference(reference)
    monkeypatch.setenv("ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", str(import_root))
    store = WorkstationStore(tmp_path / "workstation")
    draft = store.create_expression_draft(
        name="Generic reference",
        profile_id="generic-reference",
        model_id="any-v2proplus-package",
        reference_path=reference,
        language="en",
        emotion="operator-label",
        intensity=0.5,
        vad={"valence": 0.1},
        prosody={"reference_transcript": "This is a measured reference"},
    )

    measured = store.analyze_expression_reference(draft["id"])
    analysis = measured["prosody"]["reference_analysis"]
    assert analysis["reference_sha256"] == draft["reference"]["sha256"]
    assert analysis["measurements"]["speaking_rate"]["unit"] == "words-per-minute"
    assert measured["emotion"] == "operator-label"
    assert measured["vad"] == {"valence": 0.1}

    preserved = store.update_expression_draft(
        draft["id"],
        prosody={
            "reference_transcript": "This is a measured reference",
            "operator_note": "unchanged source",
        },
    )
    assert preserved["prosody"]["reference_analysis"] == analysis
    invalidated = store.update_expression_draft(
        draft["id"],
        prosody={"reference_transcript": "Transcript changed"},
    )
    assert "reference_analysis" not in invalidated["prosody"]
    with pytest.raises(WorkstationError, match="read-only"):
        store.update_expression_draft(
            draft["id"], prosody={"reference_analysis": analysis}
        )


def test_secured_expression_analysis_api_persists_measured_result(
    tmp_path: Path,
) -> None:
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html>", encoding="utf-8")
    store = WorkstationStore(tmp_path / "workstation")
    reference = store.artifact_root / "references" / "voice.wav"
    reference.parent.mkdir(parents=True)
    _write_reference(reference)
    draft = store.create_expression_draft(
        name="Voice",
        profile_id="voice",
        reference_path=reference,
        language="ja",
        emotion="manual-only",
        intensity=0.4,
    )
    app = create_webui_app(
        static_dir=static,
        client=_mock_upstream({}),
        workstation=store,
    )
    endpoint = (
        f"/api/workstation/expression-drafts/{draft['id']}/analyze-reference"
    )

    with TestClient(app, base_url="http://localhost") as client:
        wrong_type = client.post(endpoint, content=b"{}")
        cross_origin = client.post(
            endpoint,
            json={},
            headers={"Origin": "https://attacker.invalid"},
        )
        unexpected = client.post(endpoint, json={"emotion": "happy"})
        response = client.post(endpoint, json={})

    assert wrong_type.status_code == 415
    assert cross_origin.status_code == 403
    assert unexpected.status_code == 400
    assert response.status_code == 200
    assert response.json()["expression"]["emotion"] == "manual-only"
    assert response.json()["analysis"]["reference_sha256"] == draft["reference"]["sha256"]
    assert store.get_expression_draft(draft["id"])["prosody"]["reference_analysis"]


def test_expression_reference_analysis_webui_contract() -> None:
    root = Path(__file__).parents[1] / "webui"
    html = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "studio.js").read_text(encoding="utf-8")
    style = (root / "studio.css").read_text(encoding="utf-8")

    assert 'id="expressionReferenceTranscript"' in html
    assert 'id="expressionReferenceAnalysis"' in html
    assert 'id="expressionWaveform"' in html
    assert 'id="expressionAnalyzeButton"' in html
    assert "/analyze-reference`" in script
    assert "drawExpressionWaveform" in script
    assert "editableExpressionProsody" in script
    assert "delete editable.reference_analysis" in script
    assert ".expression-reference-analysis canvas" in style
    assert ".expression-analysis-metrics" in style
    assert "innerHTML" not in script
