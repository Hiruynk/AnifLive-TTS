from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aniflive_tts import container_worker
from aniflive_tts.model_package import write_checksums


def _manifest(tmp_path: Path, *, job_type: str, inputs: dict[str, str]) -> Path:
    document = {
        "schema": "aniflive-tts-worker-preparation-v1",
        "job_id": "job_11111111-1111-4111-8111-111111111111",
        "job_type": job_type,
        "project_id": "tse_22222222-2222-4222-8222-222222222222",
        "project_kind": "tse",
        "resource_class": "gpu-exclusive",
        "created_at": "2026-08-31T00:00:00.000Z",
        "input_paths": inputs,
        "container_input_paths": inputs,
        "settings": {
            "target_threshold": 0.7,
            "review_margin": 0.1,
            "minimum_speech_ms": 80,
        },
        "missing_inputs": [],
        "backend": {"available": True},
        "readiness": "ready",
        "disposition": "execute-linux-docker",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _worker_environment(monkeypatch: pytest.MonkeyPatch, manifest: Path) -> None:
    monkeypatch.setenv("ANIFLIVE_TTS_WORKER_RESULT_SCHEMA", container_worker._RESULT_SCHEMA)
    monkeypatch.setenv("ANIFLIVE_TTS_WORKER_RUN_ID", "a" * 32)
    monkeypatch.setenv("ANIFLIVE_TTS_WORKER_IMAGE_DIGEST", "sha256:" + "b" * 64)
    monkeypatch.setenv(
        "ANIFLIVE_TTS_WORKER_MANIFEST_SHA256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    monkeypatch.setenv("ANIFLIVE_TTS_WORKER_PLATFORM", "linux/amd64")


def _deployment_checkpoint(root: Path, *, corrupt_gpt: bool = False) -> Path:
    gpt = root / "checkpoints" / "gpt" / "voice-e4.ckpt"
    sovits = root / "checkpoints" / "sovits" / "voice_e4_s40.pth"
    gpt.parent.mkdir(parents=True)
    sovits.parent.mkdir(parents=True)
    gpt.write_bytes(b"trained-gpt")
    sovits.write_bytes(b"trained-sovits")
    (root / "deployment-checkpoints.json").write_text(
        json.dumps(
            {
                "schema": "aniflive-tts-v2proplus-deployment-checkpoints-v2",
                "model_family": "gsv-v2proplus",
                "selection": {
                    "method": "validation-checkpoint-selection-v1",
                    "report_sha256": "1" * 64,
                    "validation_manifest_sha256": "2" * 64,
                    "winner_reason": "passed hard gates and ranked first",
                    "test_split_accessed": False,
                },
                "gpt": {
                    "relative_path": gpt.relative_to(root).as_posix(),
                    "sha256": "0" * 64
                    if corrupt_gpt
                    else hashlib.sha256(gpt.read_bytes()).hexdigest(),
                    "size_bytes": gpt.stat().st_size,
                },
                "sovits": {
                    "relative_path": sovits.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(sovits.read_bytes()).hexdigest(),
                    "size_bytes": sovits.stat().st_size,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_tse_worker_writes_real_audio_report_and_broker_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rate = 16_000
    source = tmp_path / "source.wav"
    reference = tmp_path / "reference.wav"
    package = tmp_path / "package"
    package.mkdir()
    separation_model = tmp_path / "separation-model"
    separation_model.mkdir()
    timeline = np.arange(rate, dtype=np.float32) / rate
    tone = 0.3 * np.sin(2 * np.pi * 220 * timeline)
    sf.write(source, np.concatenate((np.zeros(1600), tone, np.zeros(1600))), rate)
    sf.write(reference, tone, rate)

    class FakeEmbedder:
        def __init__(self, _package: Path) -> None:
            pass

        def __call__(self, _audio: np.ndarray, _rate: int) -> np.ndarray:
            return np.asarray([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(container_worker, "TensorRTSpeakerEmbedder", FakeEmbedder)

    class FakeSeparationResult:
        def __init__(self, audio: np.ndarray) -> None:
            self.audio = np.asarray(audio, dtype=np.float32).copy()
            self.accepted = True
            self.minimum_selected_similarity = 0.99
            self.chunks = ()

    class FakeSeparator:
        def __init__(self, path: Path) -> None:
            assert path == separation_model

        def separate_target(self, audio, _rate, **_kwargs):
            return FakeSeparationResult(audio)

    monkeypatch.setattr(
        container_worker,
        "_possible_overlap_detector",
        lambda *_args, **_kwargs: (lambda _audio, _rate: False),
    )
    monkeypatch.setattr(
        "aniflive_tts.tse_separation.MossFormer2TargetSeparator", FakeSeparator
    )
    manifest = _manifest(
        tmp_path,
        job_type="tse.prepare",
        inputs={
            "source": str(source),
            "reference": str(reference),
            "model_package": str(package),
            "separation_model": str(separation_model),
        },
    )
    manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_document["settings"]["separation_segments"] = [
        {"start_seconds": 0.0, "end_seconds": 1.2}
    ]
    manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
    _worker_environment(monkeypatch, manifest)
    result = tmp_path / "output" / "result.json"

    assert container_worker.run("tse", manifest, result) == 0
    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["outcome"] == "completed"
    assert document["payload"]["speaker_backend"].startswith("TensorRT-11")
    assert document["payload"]["target_segments"] == 1
    assert document["payload"]["schema"] == "aniflive-tts-tse-report-v2"
    assert document["payload"]["separation_backend"]["name"] == "MossFormer2_SS_16K"
    assert (
        document["payload"]["separation_backend"]["target_selection"]
        == "TensorRT-11 sv_embedding.engine"
    )
    assert document["payload"]["separation_attempted_segments"] == 1
    assert document["payload"]["separation_accepted_segments"] == 1
    assert document["payload"]["separation_detector_segments"] == 0
    assert document["payload"]["separation_operator_segments"] == 1
    assert document["payload"]["segments"][0]["separation_triggers"] == ["operator"]
    assert document["payload"]["separation_requested_ranges"] == [
        {
            "start_sample": 0,
            "end_sample": 19200,
            "start_seconds": 0.0,
            "end_seconds": 1.2,
        }
    ]
    paths = {item["relative_path"] for item in document["artifacts"]}
    assert paths == {"target-speaker.wav", "tse-report.json"}
    for artifact in document["artifacts"]:
        path = result.parent / artifact["relative_path"]
        assert artifact["size_bytes"] == path.stat().st_size
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({}, "JSON array"),
        ([{"start_seconds": 0.0}], "contain only"),
        ([{"start_seconds": True, "end_seconds": 1.0}], "JSON numbers"),
        ([{"start_seconds": "0.0", "end_seconds": 1.0}], "JSON numbers"),
        ([{"start_seconds": 0.5, "end_seconds": 0.5}], "time bounds"),
        ([{"start_seconds": 0.0, "end_seconds": 1.01}], "time bounds"),
        (
            [
                {"start_seconds": 0.5, "end_seconds": 0.8},
                {"start_seconds": 0.7, "end_seconds": 0.9},
            ],
            "sorted and must not overlap",
        ),
    ],
)
def test_manual_separation_ranges_fail_closed(value, message: str) -> None:
    with pytest.raises(container_worker.ContainerWorkerError, match=message):
        container_worker._manual_separation_ranges(
            value, sample_rate=16_000, sample_count=16_000
        )


def test_manual_separation_ranges_are_sample_exact_and_may_be_adjacent() -> None:
    assert container_worker._manual_separation_ranges(
        [
            {"start_seconds": 0.0, "end_seconds": 0.25},
            {"start_seconds": 0.25, "end_seconds": 1.0},
        ],
        sample_rate=16_000,
        sample_count=16_000,
    ) == ((0, 4000), (4000, 16000))


def test_worker_dispatches_dataset_pipeline_and_records_canonical_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixed-media")
    manifest = _manifest(
        tmp_path,
        job_type="dataset.process",
        inputs={"source": str(source)},
    )
    manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_document["project_id"] = "dataset_22222222-2222-4222-8222-222222222222"
    manifest_document["project_kind"] = "dataset"
    manifest_document["resource_class"] = "gpu-exclusive"
    manifest_document["settings"] = {
        "acquisition_mode": "standard",
        "sources": [str(source)],
        "reference_audio": None,
        "speaker_threshold": 0.72,
        "ambiguity_margin": 0.03,
        "review_margin": 0.08,
        "speaker_component": "eres2netv2-speaker-verifier",
        "vad_component": "fsmn-vad",
        "diarization_component": "sortformer-overlap-diarization",
        "separation_component": "mossformer2-ss-16k",
        "asr_component": "sensevoice-small",
        "require_expressions": False,
        "enable_afftdn": True,
    }
    manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
    _worker_environment(monkeypatch, manifest)

    def fake_pipeline(source_path, destination, **kwargs):
        assert source_path == source
        assert kwargs["config"].enable_afftdn is True
        destination.mkdir(parents=True)
        (destination / "canonical.wav").write_bytes(b"RIFFcanonical")
        (destination / "segments").mkdir()
        (destination / "segments" / "segment-00000.wav").write_bytes(b"RIFFsegment")
        report = {
            "schema": "aniflive-dataset-pipeline-v1",
            "pipeline_identity_sha256": "c" * 64,
            "runtime": {
                "execution_environment": "linux-docker-only",
                "neural_fallback": False,
            },
            "input": {"sha256": "d" * 64},
            "output": {
                "canonical": {
                    "sample_rate": 32_000,
                    "frame_count": 64_000,
                    "denoise": {"enabled": True, "backend": "ffmpeg-afftdn"},
                    "dereverb": {"enabled": False, "backend": "none"},
                },
                "segments": [{"path": "segments/segment-00000.wav"}],
            },
            "transcript": None,
        }
        (destination / "dataset-report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    monkeypatch.setattr(
        "aniflive_tts.dataset_pipeline.run_dataset_pipeline", fake_pipeline
    )
    result = tmp_path / "output" / "result.json"
    assert container_worker.run("dataset", manifest, result) == 0
    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["payload"] == {
        "denoise": {"backend": "ffmpeg-afftdn", "enabled": True},
        "dereverb": {"backend": "none", "enabled": False},
        "duration_seconds": 2.0,
        "execution_environment": "linux-docker-only",
        "neural_fallback": False,
        "pipeline_identity_sha256": "c" * 64,
        "sample_rate": 32_000,
        "schema": "aniflive-dataset-pipeline-v1",
        "segment_count": 1,
        "source_sha256": "d" * 64,
        "transcript": None,
        "asr": None,
    }
    assert {item["relative_path"] for item in document["artifacts"]} == {
        "dataset-pipeline/canonical.wav",
        "dataset-pipeline/dataset-report.json",
        "dataset-pipeline/segments/segment-00000.wav",
    }
    assert {item["kind"] for item in document["artifacts"]} == {"dataset"}


def test_dataset_worker_passes_optional_local_asr_model_and_exports_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"fixed-media")
    asr_model = tmp_path / "asr-model"
    asr_model.mkdir()
    manifest = _manifest(
        tmp_path,
        job_type="dataset.process",
        inputs={"source": str(source), "asr_model": str(asr_model)},
    )
    manifest_document = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_document.update(
        {
            "project_id": "dataset_22222222-2222-4222-8222-222222222222",
            "project_kind": "dataset",
            "resource_class": "gpu-exclusive",
            "settings": {
                "declared_language": "yue",
                "asr_backend": "sensevoice-small",
            },
        }
    )
    manifest.write_text(json.dumps(manifest_document), encoding="utf-8")
    _worker_environment(monkeypatch, manifest)

    def fake_pipeline(source_path, destination, **kwargs):
        assert source_path == source
        assert kwargs["asr_model_path"] == asr_model
        assert kwargs["asr_backend"] == "sensevoice-small"
        assert kwargs["declared_language"] == "yue"
        destination.mkdir(parents=True)
        (destination / "canonical.wav").write_bytes(b"RIFFcanonical")
        (destination / "segments").mkdir()
        (destination / "segments" / "segment-00000.wav").write_bytes(b"RIFFsegment")
        (destination / "asr-transcripts.json").write_text("{}\n", encoding="utf-8")
        report = {
            "schema": "aniflive-dataset-pipeline-v1",
            "pipeline_identity_sha256": "c" * 64,
            "runtime": {
                "execution_environment": "linux-docker-only",
                "neural_fallback": False,
            },
            "input": {"sha256": "d" * 64},
            "output": {
                "canonical": {
                    "sample_rate": 32_000,
                    "frame_count": 64_000,
                    "denoise": {"enabled": False, "backend": "none"},
                    "dereverb": {"enabled": False, "backend": "none"},
                },
                "segments": [{"path": "segments/segment-00000.wav"}],
                "asr_transcripts": {"path": "asr-transcripts.json"},
            },
            "transcript": {"language": "yue", "normalized_text": "我今日好開心。"},
            "asr": {
                "schema": "aniflive-dataset-asr-v1",
                "backend": "sensevoice-small",
                "language": "yue",
                "text": "我今日好開心。",
                "segments": [
                    {
                        "position": 0,
                        "text": "我今日好開心。",
                        "segment_sha256": "e" * 64,
                    }
                ],
                "review_required": True,
                "review_reasons": ["offline-asr-requires-human-review"],
            },
        }
        (destination / "dataset-report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    monkeypatch.setattr(
        "aniflive_tts.dataset_pipeline.run_dataset_pipeline", fake_pipeline
    )
    result = tmp_path / "output" / "result.json"
    assert container_worker.run("dataset", manifest, result) == 0
    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["payload"]["asr"] == {
        "schema": "aniflive-dataset-asr-v1",
        "backend": "sensevoice-small",
        "language": "yue",
        "review_required": True,
        "review_reasons": ["offline-asr-requires-human-review"],
        "segment_count": 1,
        "transcript_character_count": 7,
    }
    assert document["payload"]["transcript"] == {
        "language": "yue",
        "transcript_character_count": 7,
    }
    assert {item["relative_path"] for item in document["artifacts"]} == {
        "dataset-pipeline/asr-transcripts.json",
        "dataset-pipeline/canonical.wav",
        "dataset-pipeline/dataset-report.json",
        "dataset-pipeline/segments/segment-00000.wav",
    }


def test_worker_dispatches_training_executor_and_records_real_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = _manifest(
        tmp_path,
        job_type="training.prepare",
        inputs={"dataset": str(dataset)},
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["project_id"] = "training_22222222-2222-4222-8222-222222222222"
    document["project_kind"] = "training"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    _worker_environment(monkeypatch, manifest)
    report = tmp_path / "output" / "training-report.json"

    def fake_training(_manifest, output):
        assert output == report.parent
        output.mkdir(parents=True, exist_ok=True)
        report.write_text('{"status":"passed"}\n', encoding="utf-8")
        return {"status": "passed", "backend": "fixed-test"}, [report]

    monkeypatch.setattr("aniflive_tts.workstation_training.run_training", fake_training)
    result = tmp_path / "output" / "result.json"

    assert container_worker.run("training", manifest, result) == 0
    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["payload"]["backend"] == "fixed-test"
    assert document["artifacts"][0]["kind"] == "checkpoint"
    assert document["artifacts"][0]["relative_path"] == "training-report.json"


def test_worker_dispatches_evaluation_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    package.mkdir()
    manifest = _manifest(
        tmp_path,
        job_type="evaluation.prepare",
        inputs={"model_package": str(package)},
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["project_id"] = "evaluation_22222222-2222-4222-8222-222222222222"
    document["project_kind"] = "evaluation"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    _worker_environment(monkeypatch, manifest)

    def fake_evaluation(_manifest, output):
        output.mkdir(parents=True, exist_ok=True)
        report = output / "evaluation-report.json"
        report.write_text('{"status":"passed"}\n', encoding="utf-8")
        return {"status": "passed", "backend": "TensorRT-11"}, [report]

    monkeypatch.setattr("aniflive_tts.workstation_evaluation.run_evaluation", fake_evaluation)
    result = tmp_path / "output" / "result.json"
    assert container_worker.run("evaluation", manifest, result) == 0
    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["payload"]["backend"] == "TensorRT-11"
    assert document["artifacts"][0]["kind"] == "evaluation"


def test_engine_worker_converts_a_verified_checkpoint_pair_to_v2proplus_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _deployment_checkpoint(tmp_path / "checkpoint")
    shared = tmp_path / "shared"
    shared.mkdir()
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"RIFF-reference")
    output = tmp_path / "output"
    observed = {}

    def fake_convert_model(**kwargs):
        observed.update(kwargs)
        destination = kwargs["output"]
        destination.mkdir(parents=True)
        (destination / "manifest.json").write_text(
            '{"format":"aniflive-tts-model-package"}\n', encoding="utf-8"
        )
        return destination

    monkeypatch.setattr(container_worker, "_convert_checkpoint_model", fake_convert_model)
    monkeypatch.setattr(
        container_worker,
        "_validate_package",
        lambda package: {
            "status": "passed",
            "engine_count": 9,
            "model_id": "trained-voice",
        },
    )
    payload, artifacts = container_worker._run_engine_build(
        {
            "container_input_paths": {
                "checkpoint": str(checkpoint),
                "shared_dir": str(shared),
                "reference": str(reference),
            },
            "settings": {
                "reference_text": "今日はいい天気ですね。",
                "reference_language": "ja",
                "model_id": "trained-voice",
                "voice_profile": "default",
            },
        },
        output,
    )

    assert observed["gpt"].name == "voice-e4.ckpt"
    assert observed["sovits"].name == "voice_e4_s40.pth"
    assert observed["model_id"] == "trained-voice"
    assert observed["shared_dir"] == shared.resolve()
    assert payload["status"] == "passed"
    assert payload["mode"] == "checkpoint-conversion"
    assert payload["validation"]["engine_count"] == 9
    assert output / "engine-build.json" in artifacts
    assert not (output / ".private-conversion").exists()


def test_engine_worker_rejects_a_checkpoint_pair_with_mismatched_hash(
    tmp_path: Path,
) -> None:
    checkpoint = _deployment_checkpoint(tmp_path / "checkpoint", corrupt_gpt=True)
    with pytest.raises(container_worker.ContainerWorkerError, match="immutable manifest"):
        container_worker._deployment_checkpoint_pair(checkpoint)


def test_conversion_parity_command_publishes_fail_closed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = {}
    for name in (
        "model_package",
        "selected_checkpoints",
        "shared_dir",
        "asr_model",
    ):
        path = tmp_path / name
        path.mkdir()
        roots[name] = str(path)
    reference = tmp_path / "deployment-reference.json"
    reference.write_text('{"status":"human-locked"}\n', encoding="utf-8")
    roots["deployment_reference"] = str(reference)
    manifest = _manifest(tmp_path, job_type="conversion.parity", inputs=roots)
    _worker_environment(monkeypatch, manifest)

    def fake_parity(_manifest, output):
        report = output / "conversion-parity-report.json"
        report.write_text('{"status":"failed"}\n', encoding="utf-8")
        return {"status": "failed"}, [report]

    monkeypatch.setattr(
        "aniflive_tts.workstation_conversion_worker.run_conversion_parity",
        fake_parity,
    )
    result = tmp_path / "output" / "result.json"

    assert container_worker.run("conversion-parity", manifest, result) == 0
    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["payload"]["status"] == "failed"
    assert document["artifacts"][0]["kind"] == "evaluation"


def test_template_reference_is_generic_and_checksum_verified(tmp_path: Path) -> None:
    package = tmp_path / "package"
    profile = package / "voices" / "actor-default"
    profile.mkdir(parents=True)
    reference = profile / "reference.wav"
    reference.write_bytes(b"RIFF-reference")
    (profile / "profile.json").write_text(
        json.dumps(
            {
                "id": "actor-default",
                "reference_audio": "reference.wav",
                "reference_text": "我今日好開心。",
                "reference_language": "yue",
            }
        ),
        encoding="utf-8",
    )
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "format": "aniflive-tts-model-package",
                "model_id": "actor-v2proplus",
                "model_family": "gsv-v2proplus",
                "precision": "FP16",
                "voice_profiles": ["actor-default"],
                "default_voice_profile": "actor-default",
            }
        ),
        encoding="utf-8",
    )
    write_checksums(package)

    values = container_worker._template_reference(package)

    assert values[:5] == (
        reference.resolve(),
        "我今日好開心。",
        "yue",
        "actor-v2proplus",
        "actor-default",
    )


def test_model_package_worker_requires_a_complete_package_directory(tmp_path: Path) -> None:
    engine = tmp_path / "engine"
    engine.mkdir()
    with pytest.raises(container_worker.ContainerWorkerError, match="requires a model package"):
        container_worker._run_model_package(
            {"container_input_paths": {"engine_dir": str(engine)}},
            tmp_path / "output",
        )


def test_worker_rejects_a_manifest_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path, job_type="tse.prepare", inputs={})
    _worker_environment(monkeypatch, manifest)
    monkeypatch.setenv("ANIFLIVE_TTS_WORKER_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(container_worker.ContainerWorkerError, match="checksum"):
        container_worker.run("tse", manifest, tmp_path / "output" / "result.json")
