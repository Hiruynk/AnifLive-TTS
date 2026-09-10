from __future__ import annotations

import hashlib
from pathlib import Path
import wave

import numpy as np

from aniflive_tts.dataset_factory import DatasetFactory
from aniflive_tts.voice_acquisition_report import (
    VOICE_ACQUISITION_REPORT_SCHEMA,
    build_voice_acquisition_report,
)
from aniflive_tts.workstation import WorkstationStore


def _wav(path: Path, seconds: float = 0.4) -> None:
    sample_rate = 32_000
    timeline = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    pcm = np.rint(0.2 * np.sin(2 * np.pi * 220 * timeline) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm.tobytes())


def test_report_is_evidence_only_and_lists_missing_production_gates(tmp_path: Path) -> None:
    media = tmp_path / "media"
    clip = media / "selected.wav"
    reference = media / "reference.wav"
    _wav(clip)
    _wav(reference)
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(
        kind="dataset",
        name="Voice acquisition",
        config={
            "acquisition_mode": "target-speaker",
            "sources": [str(media)],
            "reference_audio": str(reference),
        },
    )
    factory = DatasetFactory(
        store.root / "dataset-factory",
        allowed_source_roots=(media, store.artifact_root),
    )
    factory.ensure_project(project["id"], project["config"])
    imported = factory.import_target_speaker_clips(
        project["id"],
        [
            {
                "path": str(clip),
                "sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
                "position": 0,
                "source_sha256": "c" * 64,
                "source_start_sample": 0,
                "source_end_sample": 12_800,
                "route": "clean",
                "speaker": {"target_similarity": 0.98},
                "acquisition": {"route": "clean"},
                "annotations": {
                    "transcript": "今日はいい天気ですね。",
                    "language": "ja",
                    "asr_backend": "sensevoice-small-cuda-v1",
                    "authoritative": False,
                    "review_required": True,
                },
            }
        ],
    )
    item = imported["items"][0]
    factory.update_annotations(
        item["id"],
        {
            "transcript": "今日はいい天気ですね。",
            "language": "ja",
            "speaker": "voice",
            "expression": "neutral",
        },
        source="manual",
    )
    factory.review(item["id"], decision="accepted")
    factory.verify_annotation(item["id"], kind="transcript")
    factory.verify_annotation(item["id"], kind="speaker")
    factory.verify_annotation(item["id"], kind="expression")
    factory.assign_splits(project["id"], train=1, validation=0, test=0)
    factory.freeze_dataset(project["id"])

    report = build_voice_acquisition_report(store, factory, project["id"])

    assert report["schema"] == VOICE_ACQUISITION_REPORT_SCHEMA
    assert report["dataset"]["accepted"] == 1
    assert report["dataset"]["frozen"] is True
    assert report["transcription"] == {
        "backend": "sensevoice-small-cuda-v1",
        "human_reviewed": 1,
        "pending_review": 0,
    }
    assert report["expressions"] == {"classified": 1, "reference_profiles": 1}
    assert report["training"]["passed"] is False
    assert report["engines"] == {"count": 0, "backend": None}
    assert report["qualification"]["passed"] is False
    assert report["ready_for_release"] is False
    assert report["missing_gates"] == [
        "target-speaker-finalize-evidence",
        "thirty-minute-source-qualification",
        "offline-acquisition-backends",
        "training",
        "nine-tensorrt-engines",
        "production-qualification",
    ]


def test_report_requires_real_long_source_and_offline_backend_provenance(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    reference = media / "reference.wav"
    _wav(reference)
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(
        kind="dataset",
        name="Acquisition evidence",
        config={
            "acquisition_mode": "target-speaker",
            "sources": [str(media)],
            "reference_audio": str(reference),
        },
    )
    factory = DatasetFactory(
        store.root / "dataset-factory",
        allowed_source_roots=(media, store.artifact_root),
    )
    factory.ensure_project(project["id"], project["config"])
    job = store.create_job(job_type="dataset.finalize", project_id=project["id"])
    claim = store.claim_job(job["id"])
    store.update_job(
        job["id"],
        status="succeeded",
        claim_token=claim.token,
        result={
            "backend": {
                "payload": {
                    "schema": "aniflive-target-speaker-finalize-v1",
                    "source_seconds": 1_860.0,
                    "reference_seconds": 6.0,
                    "speaker_backend": "workstation-eres2netv2-tensorrt11",
                    "vad": {"backend": "fsmn-vad-funasr-1.4.11-cuda-v1"},
                    "separator": "MossFormer2_SS_16K",
                    "transcription_backend": "sensevoice-small-cuda-v1",
                }
            }
        },
    )

    report = build_voice_acquisition_report(store, factory, project["id"])

    assert report["source_audio_seconds"] == 1_860.0
    assert report["maximum_single_source_seconds"] == 1_860.0
    assert report["acquisition_evidence"] == {
        "finalized_sources": 1,
        "speaker_backend": ["workstation-eres2netv2-tensorrt11"],
        "vad_backend": ["fsmn-vad-funasr-1.4.11-cuda-v1"],
        "separator": ["MossFormer2_SS_16K"],
    }
    assert "target-speaker-finalize-evidence" not in report["missing_gates"]
    assert "thirty-minute-source-qualification" not in report["missing_gates"]
    assert "offline-acquisition-backends" not in report["missing_gates"]


def test_standard_report_uses_process_evidence_and_optional_expression_policy(
    tmp_path: Path,
) -> None:
    media = tmp_path / "media"
    clip = media / "segment.wav"
    _wav(clip)
    store = WorkstationStore(tmp_path / "workstation")
    project = store.create_project(
        kind="dataset",
        name="Standard acquisition evidence",
        config={
            "acquisition_mode": "standard",
            "sources": [str(clip)],
            "require_expressions": False,
        },
    )
    factory = DatasetFactory(
        store.root / "dataset-factory",
        allowed_source_roots=(media, store.artifact_root),
    )
    factory.ensure_project(project["id"], project["config"])
    imported = factory.import_standard_clips(
        project["id"],
        [
            {
                "path": str(clip),
                "sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
                "position": 0,
                "source_sha256": "d" * 64,
                "source_start_sample": 0,
                "source_end_sample": 12_800,
                "annotations": {
                    "transcript": "今日はいい天気ですね。",
                    "language": "ja",
                    "asr_backend": "sensevoice-small-cuda-v1",
                    "authoritative": False,
                    "review_required": True,
                },
            }
        ],
    )
    item = imported["items"][0]
    factory.update_annotations(
        item["id"],
        {
            "transcript": "今日はいい天気ですね。",
            "language": "ja",
            "speaker": "voice",
        },
        source="manual",
    )
    factory.review(item["id"], decision="accepted")
    factory.verify_annotation(item["id"], kind="transcript")
    factory.verify_annotation(item["id"], kind="speaker")
    job = store.create_job(job_type="dataset.process", project_id=project["id"])
    claim = store.claim_job(job["id"])
    store.update_job(
        job["id"],
        status="succeeded",
        claim_token=claim.token,
        result={
            "backend": {
                "payload": {
                    "schema": "aniflive-dataset-pipeline-v1",
                    "duration_seconds": 826.645344,
                }
            }
        },
    )

    report = build_voice_acquisition_report(store, factory, project["id"])

    assert report["source_audio_seconds"] == 826.645344
    assert report["maximum_single_source_seconds"] == 826.645344
    assert report["acquisition_evidence"]["finalized_sources"] == 1
    assert report["transcription"]["human_reviewed"] == 1
    assert "expression-classification" not in report["missing_gates"]
