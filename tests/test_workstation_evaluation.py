from __future__ import annotations

import math
from types import SimpleNamespace
import sys

import pytest

from aniflive_tts import workstation_evaluation as evaluation


def test_evaluation_plan_is_bounded_and_rejects_unknown_controls() -> None:
    plan = evaluation.resolve_evaluation_plan(
        {
            "benchmark_sessions": 2,
            "benchmark_warmups": 1,
            "benchmark_runs": 4,
            "benchmark_language": "yue",
            "asr_compute_type": "int8_float16",
        }
    )
    assert plan.benchmark_sessions == 2
    assert plan.benchmark_language == "yue"
    with pytest.raises(evaluation.EvaluationWorkerError, match="unsupported"):
        evaluation.resolve_evaluation_plan({"command": "fake"})
    with pytest.raises(evaluation.EvaluationWorkerError, match="between 1 and 100"):
        evaluation.resolve_evaluation_plan({"benchmark_runs": 0})


def test_real_cer_and_wer_are_edit_distance_metrics() -> None:
    wer = evaluation.content_error(
        "The weather is pleasant today.", "the weather was pleasant today", "en"
    )
    assert wer["metric"] == "wer"
    assert wer["edits"] == 1
    assert math.isclose(wer["error_rate"], 0.2)
    assert math.isclose(
        evaluation.content_error_rate("one two three", "one two", "en"),
        1 / 3,
    )
    cer = evaluation.content_error("今日天氣幾好。", "今日天氣好。", "yue")
    assert cer["metric"] == "cer"
    assert cer["edits"] == 1
    assert cer["reference_units"] == 6


def test_japanese_spoken_content_uses_pinned_phoneme_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readings = {
        "時": "t o k i",
        "とき": "t o k i",
        "スタイル": "s U t a i r u",
    }
    fake = SimpleNamespace(
        __version__="test",
        g2p=lambda text, kana=False: readings[text],
    )
    monkeypatch.setitem(sys.modules, "pyopenjtalk", fake)

    equivalent = evaluation.spoken_content_error("時", "とき", "ja")
    different = evaluation.spoken_content_error("時", "スタイル", "ja")

    assert equivalent["metric"] == "per"
    assert equivalent["error_rate"] == 0.0
    assert different["error_rate"] > 0.0
    assert equivalent["normalizer"] == "pyopenjtalk-phonemes"


def test_runtime_headers_fail_closed_on_fallback() -> None:
    evaluation._validate_headers(
        {
            "x-tensorrt-backend": "TensorRT-11",
            "x-pytorch-fallback": "false",
            "x-tts-model": "voice",
        },
        "ja",
    )
    with pytest.raises(evaluation.EvaluationWorkerError, match="fallback"):
        evaluation._validate_headers(
            {
                "x-tensorrt-backend": "TensorRT-11",
                "x-pytorch-fallback": "true",
                "x-tts-model": "voice",
            },
            "ja",
        )


def _language_evidence(*, content_delta: float = 0.0, speaker_delta: float = 0.0):
    rows = {}
    for language, text in evaluation._LANGUAGE_CASES.items():
        rows[language] = {
            "text": text,
            "asr": {
                "metric": "wer" if language == "en" else "cer",
                "error_rate": 0.02 + content_delta,
                "hypothesis": text + ("錯" if content_delta > 0.005 else ""),
            },
            "quality": {
                "speaker_cosine_stream_vs_identity": 0.99 + speaker_delta,
            },
        }
    return rows


def _benchmark(*, multiplier: float = 1.0):
    return {
        "workload": {"text": "fixed", "language": "ja", "seed": 1234},
        "methodology": {
            "sessions": 10,
            "warmup_requests_per_session": 10,
            "full_wav_requests_per_session": 100,
            "stream_requests_per_session": 100,
            "keepalive_stream_requests_per_session": 100,
        },
        "table": [
            {
                "key": "stream_keepalive_audible_ttfa_p50_ms",
                "median": 75.0 * multiplier,
            },
            {
                "key": "stream_keepalive_audible_ttfa_p95_ms",
                "median": 85.0 * multiplier,
            },
            {"key": "wall_rtf_p50", "median": 0.09 * multiplier},
        ],
    }


def test_matched_baseline_gate_checks_content_speaker_and_performance() -> None:
    identity = {"files": {"model.bin": {"sha256": "a" * 64, "size_bytes": 1}}}
    baseline = {
        "schema": "aniflive-tts-workstation-evaluation-v1",
        "asr": {"model_fingerprint": identity},
        "languages": _language_evidence(),
        "benchmark": _benchmark(),
    }
    passed = evaluation.compare_evaluation_baseline(
        languages=_language_evidence(content_delta=0.004, speaker_delta=-0.004),
        benchmark=_benchmark(multiplier=1.02),
        asr_identity=identity,
        baseline=baseline,
    )
    assert passed["passed"] is True
    failed = evaluation.compare_evaluation_baseline(
        languages=_language_evidence(content_delta=0.006, speaker_delta=-0.006),
        benchmark=_benchmark(multiplier=1.11),
        asr_identity=identity,
        baseline=baseline,
    )
    assert failed["passed"] is False
    assert failed["languages"]["yue"]["content_passed"] is False
    assert failed["languages"]["ja"]["speaker_passed"] is False
    assert (
        failed["performance"]["stream_keepalive_audible_ttfa_p50_ms"]["passed"]
        is False
    )


def test_baseline_gate_rejects_incompatible_asr_or_methodology() -> None:
    identity = {"files": {"model.bin": {"sha256": "a" * 64, "size_bytes": 1}}}
    baseline = {
        "schema": "aniflive-tts-workstation-evaluation-v1",
        "asr": {"model_fingerprint": identity},
        "languages": _language_evidence(),
        "benchmark": _benchmark(),
    }
    with pytest.raises(evaluation.EvaluationWorkerError, match="different offline ASR"):
        evaluation.compare_evaluation_baseline(
            languages=_language_evidence(),
            benchmark=_benchmark(),
            asr_identity={"files": {}},
            baseline=baseline,
        )
    unmatched = _benchmark()
    unmatched["methodology"]["sessions"] = 9
    with pytest.raises(evaluation.EvaluationWorkerError, match="methodology"):
        evaluation.compare_evaluation_baseline(
            languages=_language_evidence(),
            benchmark=unmatched,
            asr_identity=identity,
            baseline=baseline,
        )

@pytest.mark.parametrize("text", ["", "   ", 1, "x" * 4097, "\x00invalid"])
def test_benchmark_text_is_bounded(text):
    with pytest.raises(evaluation.EvaluationWorkerError, match="benchmark_text"):
        evaluation.resolve_evaluation_plan({"benchmark_text": text})


def test_benchmark_preserves_explicit_comparison_workload(tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "benchmark_readme.py").write_text("# fixture")
    monkeypatch.setattr(evaluation, "_EVALUATION_TOOLS", tools)
    text = "今日はいい天気ですね。"
    plan = evaluation.resolve_evaluation_plan({"benchmark_text": text})
    seen = []

    def run(command, **kwargs):
        from pathlib import Path
        assert kwargs["sessions"] == plan.benchmark_sessions
        seen.append(command)
        Path(command[command.index("--report") + 1]).write_text("{}")
        Path(command[command.index("--markdown") + 1]).write_text("fixture")
        return 0, "done"

    monkeypatch.setattr(evaluation, "_run_logged_benchmark", run)
    evaluation._run_benchmark(plan, "voice", tmp_path, runtime_release="1.3.0")
    assert seen[0][seen[0].index("--text") + 1] == text
    assert seen[0][seen[0].index("--release") + 1] == "1.3.0"


def test_chinese_script_normalization_preserves_real_content_errors():
    same = evaluation.orthographic_content_error("測試語音品質", "测试语音品质", "zh")
    changed = evaluation.orthographic_content_error("測試語音品質", "测试鱼音品质", "zh")
    assert same["error_rate"] == 0
    assert changed["edits"] == 1
    assert same["normalizer"] == "opencc-t2s"
    assert same["normalizer_version"]
    dialect = evaluation.orthographic_content_error("我哋一齊測試", "我们一起测试", "yue")
    assert dialect["error_rate"] > 0


def test_baseline_rescores_both_scripts_and_retains_original_metrics():
    identity = {"files": {}}
    current = _language_evidence()
    previous = _language_evidence()
    current["zh"]["asr"].update(hypothesis="今天天气很好，我们来测试语音品质。", error_rate=0.4)
    previous["zh"]["asr"].update(hypothesis=previous["zh"]["text"], error_rate=0.0)
    baseline = {
        "schema": "aniflive-tts-workstation-evaluation-v1",
        "asr": {"model_fingerprint": identity},
        "languages": previous, "benchmark": _benchmark(),
    }
    result = evaluation.compare_evaluation_baseline(
        languages=current, benchmark=_benchmark(), asr_identity=identity, baseline=baseline,
    )
    row = result["languages"]["zh"]
    assert row["content_passed"]
    assert row["content_error_rate"] == row["baseline_content_error_rate"] == 0
    assert row["raw_content_error_rate"] == 0.4
    assert current["zh"]["asr"]["error_rate"] == 0.4
    del previous["zh"]["asr"]["hypothesis"]
    with pytest.raises(evaluation.EvaluationWorkerError, match="original ASR hypotheses"):
        evaluation.compare_evaluation_baseline(
            languages=current, benchmark=_benchmark(), asr_identity=identity, baseline=baseline,
        )


def test_default_japanese_benchmark_is_v13_sentence_not_quality_probe():
    plan = evaluation.resolve_evaluation_plan({})
    assert evaluation.evaluation_workload(plan)["text"] == "今日はいい天気ですね。"
    assert evaluation._LANGUAGE_CASES["ja"] != plan.benchmark_text


def test_preflight_rejects_workload_and_methodology_before_measurement():
    plan = evaluation.resolve_evaluation_plan({
        "benchmark_sessions": 10, "benchmark_warmups": 10, "benchmark_runs": 100,
    })
    baseline = {
        "schema": "aniflive-tts-workstation-evaluation-v1",
        "benchmark": _benchmark(),
    }
    baseline["benchmark"]["workload"] = evaluation.evaluation_workload(plan)
    assert evaluation.preflight_evaluation_baseline(plan, baseline)["status"] == "passed"
    baseline["benchmark"]["workload"]["text"] = evaluation._LANGUAGE_CASES["ja"]
    with pytest.raises(evaluation.EvaluationWorkerError, match="workload.*text"):
        evaluation.preflight_evaluation_baseline(plan, baseline)
    baseline["benchmark"]["workload"] = evaluation.evaluation_workload(plan)
    baseline["benchmark"]["methodology"]["sessions"] = 9
    with pytest.raises(evaluation.EvaluationWorkerError, match="methodology"):
        evaluation.preflight_evaluation_baseline(plan, baseline)


def test_new_performance_policy_accepts_ten_percent_not_more():
    identity = {"files": {}}
    baseline = {"schema": "aniflive-tts-workstation-evaluation-v1",
                "asr": {"model_fingerprint": identity},
                "languages": _language_evidence(), "benchmark": _benchmark()}
    for multiplier, expected in [(1.09, True), (1.10, True), (1.1001, False)]:
        result = evaluation.compare_evaluation_baseline(
            languages=_language_evidence(), benchmark=_benchmark(multiplier=multiplier),
            asr_identity=identity, baseline=baseline,
        )
        assert result["passed"] is expected
        assert result["performance_policy"] == "v14-near-v13-10pct-v1"


def test_completed_measurements_survive_late_comparison_failure(tmp_path, monkeypatch):
    import json
    import numpy as np
    from aniflive_tts import validate as validation

    package = tmp_path / "package"
    voice = package / "voices" / "default"
    voice.mkdir(parents=True)
    (package / "manifest.json").write_text(json.dumps({
        "model_id": "qa", "default_voice_profile": "default",
    }))
    (voice / "profile.json").write_text(json.dumps({"reference_audio": "reference.wav"}))
    (voice / "reference.wav").write_bytes(b"fixture")
    shared, asr_root, source = [tmp_path / name for name in ("shared", "asr", "source")]
    for directory in (shared, asr_root, source):
        directory.mkdir()
    (source / "run_trt_inference.py").write_text("# fixture")
    plan = evaluation.resolve_evaluation_plan({
        "benchmark_sessions": 10, "benchmark_warmups": 10, "benchmark_runs": 100,
    })
    benchmark = _benchmark()
    benchmark["workload"] = evaluation.evaluation_workload(plan)
    identity = {"files": {}}
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "schema": "aniflive-tts-workstation-evaluation-v1",
        "benchmark": benchmark, "languages": _language_evidence(),
        "asr": {"model_fingerprint": identity},
    }))
    monkeypatch.setattr(evaluation, "_SOURCE_DIR", source)
    monkeypatch.setattr(evaluation, "_asr_fingerprint", lambda path: identity)
    monkeypatch.setattr(validation, "validate_model_package",
                        lambda *a, **kw: {"status": "passed", "engine_count": 9})
    monkeypatch.setattr(evaluation.subprocess, "Popen", lambda *a, **kw: object())
    monkeypatch.setattr(evaluation, "_wait_for_service",
                        lambda *a: {"version": "test", "model": "qa"})
    monkeypatch.setattr(evaluation, "_terminate", lambda server: None)
    monkeypatch.setattr(evaluation, "_request",
                        lambda *a, **kw: (200, {"x-tts-sample-rate": "32000"}, b"\x00\x01" * 100))
    monkeypatch.setattr(evaluation, "_validate_headers", lambda *a: None)
    monkeypatch.setattr(evaluation, "_decode_wav", lambda data: (32000, np.ones(100)))
    monkeypatch.setattr(evaluation, "_TensorRTSpeakerEmbedder",
                        lambda package: lambda path: np.ones(4))
    monkeypatch.setattr(evaluation, "_spectral_quality", lambda *a: {
        "log_mel_cosine": 1.0, "duration_difference_ratio": 0.0,
    })
    monkeypatch.setitem(sys.modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=lambda *a, **kw: object()))
    monkeypatch.setattr(evaluation, "_transcribe",
                        lambda model, path, language: evaluation._LANGUAGE_CASES[language])

    def measured(plan, model_id, output, **kwargs):
        # The preflight matched, but the measurement tool returned a different workload.
        current = {**benchmark, "workload": {**benchmark["workload"], "text": "different"}}
        current["session_records"] = [{"session": 1}]
        path, markdown = output / "benchmark.json", output / "benchmark.md"
        path.write_text(json.dumps(current))
        markdown.write_text("fixture")
        (output / "benchmark.log").write_text("fixture")
        return path, markdown
    monkeypatch.setattr(evaluation, "_run_benchmark", measured)
    output = tmp_path / "output"
    report, artifacts = evaluation.run_evaluation({
        "container_input_paths": {
            "model_package": str(package), "shared_dir": str(shared),
            "asr_model": str(asr_root), "baseline_report": str(baseline),
        },
        "settings": {"benchmark_sessions": 10, "benchmark_warmups": 10, "benchmark_runs": 100},
    }, output)
    saved = json.loads((output / "evaluation-report.json").read_text())
    assert saved == report
    assert report["status"] == "failed"
    assert report["gates"]["passed"] is False
    assert report["baseline"]["comparable"] is False
    assert "workload" in report["baseline"]["reason"]
    assert len(report["languages"]) == 5
    assert len(list((output / "audio").glob("*.wav"))) == 10
    assert output / "evaluation-report.json" in artifacts
