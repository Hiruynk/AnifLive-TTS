from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v12_release_defaults_preserve_the_accepted_runtime() -> None:
    semantic = (
        ROOT / "src" / "aniflive_tts" / "backend" / "semantic_runtime.py"
    ).read_text(encoding="utf-8")
    streaming = (ROOT / "src" / "aniflive_tts" / "streaming.py").read_text(
        encoding="utf-8"
    )

    assert '"ANIFLIVE_TTS_REPETITION_PENALTY", "1.0"' in semantic
    assert '"ANIFLIVE_TTS_PERSISTENT_GPT_CONTEXTS", "1"' in semantic
    assert '"ANIFLIVE_TTS_FIRST_SEGMENT_PREVIEW_ONLY", "1"' in streaming
    assert "ANIFLIVE_TTS_FULL_CONTEXT_REFILL_DECODER" not in streaming


def test_failed_semantic_experiments_are_not_production_backends() -> None:
    streaming = (ROOT / "src" / "aniflive_tts" / "streaming.py").read_text(
        encoding="utf-8"
    )

    for backend in (
        '"delta-kv"',
        '"direct-delta"',
        '"fitted-h1"',
        '"mtp2"',
        '"mtp2-direct"',
        '"mtp4"',
        '"mtp4-rejection"',
    ):
        assert backend not in streaming


def test_benchmark_report_keeps_each_model_session_record() -> None:
    benchmark = (ROOT / "scripts" / "benchmark_readme.py").read_text(
        encoding="utf-8"
    )

    assert '"session_records": session_records' in benchmark
    assert '"model": model' in benchmark
    assert '"session": index + 1' in benchmark
    assert '"metrics": metrics' in benchmark
    assert "for index in range(args.sessions):\n        for model in models:" in benchmark


def test_converter_builds_the_serial_gpt_stage_without_aux_streams() -> None:
    converter = (ROOT / "src" / "aniflive_tts" / "converter.py").read_text(
        encoding="utf-8"
    )
    builder = (
        ROOT / "src" / "aniflive_tts" / "backend" / "legacy_converter.py"
    ).read_text(encoding="utf-8")

    assert 'STAGE_MAX_AUX_STREAMS: dict[str, int] = {"gpt_step": 0}' in builder
    assert "max_aux_streams=STAGE_MAX_AUX_STREAMS.get(stage)" in builder
    assert '"max_aux_streams_by_stage": STAGE_MAX_AUX_STREAMS' in converter


def test_warm_retention_uses_one_bounded_tensorrt_pulse() -> None:
    streaming = (ROOT / "src" / "aniflive_tts" / "streaming.py").read_text(
        encoding="utf-8"
    )
    service = (ROOT / "src" / "aniflive_tts" / "service.py").read_text(
        encoding="utf-8"
    )
    pulse = streaming.split("def keepwarm_pulse", 1)[1].split(
        "def _load_mute_matrix", 1
    )[0]

    assert pulse.count("self.engine.model_bert(inputs, sync=False)") == 1
    assert '"ANIFLIVE_TTS_WARM_MAX_PULSE_SECONDS", "0.040"' in service
