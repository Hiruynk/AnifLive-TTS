from __future__ import annotations

import hashlib
import json

import pytest

from aniflive_tts.workstation_conversion_worker import _apply_content_review


def _fixture(tmp_path):
    audio = tmp_path / "trt.wav"
    audio.write_bytes(b"exact reviewed audio")
    entry = {
        "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        "text": "Expected words",
        "decision": "content-complete",
        "user_response": "Both complete",
        "origin": "explicit user response",
    }
    case = {
        "case": 2, "text": "Expected words", "content_regression": True,
        "tensorrt_hypothesis": "extra expected words",
        "tensorrt_content_error": 0.147,
        "speaker_cosine": 0.997, "new_artifacts": True,
        "paths": {"tensorrt": "trt.wav"},
    }
    return entry, case


def _write(tmp_path, entries):
    path = tmp_path / "review.json"
    path.write_text(json.dumps({
        "schema": "aniflive-conversion-content-review-v1", "reviews": entries,
    }))
    return path


def test_exact_human_review_only_adjudicates_content(tmp_path):
    entry, case = _fixture(tmp_path)
    path = _write(tmp_path, [entry])
    result = _apply_content_review([case], path, tmp_path)
    assert result["applied_cases"] == [2]
    assert case["content_regression"] is False
    assert case["automated_content_regression"] is True
    assert case["tensorrt_hypothesis"] == "extra expected words"
    assert case["tensorrt_content_error"] == 0.147
    assert case["speaker_cosine"] == 0.997
    assert case["new_artifacts"] is True
    assert case["human_content_review"]["review_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("field", "value"), [
    ("audio_sha256", "0" * 64), ("text", "Different words"),
    ("decision", "incomplete"), ("user_response", ""), ("origin", ""),
])
def test_stale_or_unconfirmed_review_cannot_override(tmp_path, field, value):
    entry, case = _fixture(tmp_path)
    entry[field] = value
    result = _apply_content_review([case], _write(tmp_path, [entry]), tmp_path)
    assert result["applied_cases"] == []
    assert case["content_regression"] is True


def test_changed_audio_cannot_reuse_review(tmp_path):
    entry, case = _fixture(tmp_path)
    (tmp_path / "trt.wav").write_bytes(b"new generation")
    result = _apply_content_review([case], _write(tmp_path, [entry]), tmp_path)
    assert result["applied_cases"] == []
    assert case["content_regression"] is True


def test_ambiguous_duplicate_review_fails_closed(tmp_path):
    entry, case = _fixture(tmp_path)
    result = _apply_content_review([case], _write(tmp_path, [entry, entry]), tmp_path)
    assert result["applied_cases"] == []
    assert case["content_regression"] is True


def test_human_content_confirmation_cannot_pass_failed_acoustics(tmp_path):
    from aniflive_tts.workstation_production_gates import conversion_parity_report

    entry, case = _fixture(tmp_path)
    case.update({
        "pytorch_onnx_logits_cosine": 1.0,
        "onnx_trt_logits_cosine": 1.0,
        "greedy_sequence_agreement": 1.0,
        "log_mel_cosine": 0.98,
        "duration_difference_ratio": 0.0,
    })
    _apply_content_review([case], _write(tmp_path, [entry]), tmp_path)
    report = conversion_parity_report(
        [case], checkpoint_manifest_sha256="a" * 64, engine_build_sha256="b" * 64,
    )
    assert report["status"] == "failed"
    assert report["cases"][0]["failures"] == ["log-mel", "artifacts"]


def test_content_review_adapter_declares_optional_contained_path():
    from aniflive_tts.workstation_adapters import _PATH_KEYS, _SPECS

    spec = next(spec for spec in _SPECS if spec.job_type == "conversion.parity")
    assert "content_review" in spec.optional_paths
    assert "content_review" in _PATH_KEYS
