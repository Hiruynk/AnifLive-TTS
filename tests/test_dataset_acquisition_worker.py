from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from aniflive_tts import dataset_acquisition_worker as worker


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wav(path: Path, frequency: float = 220.0) -> None:
    timeline = np.arange(16_000, dtype=np.float32) / 16_000
    worker._write_wav(path, 0.2 * np.sin(2 * np.pi * frequency * timeline), 16_000)


def test_media_decode_accepts_video_first_audio_stream(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=10:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=48000:duration=1",
            "-shortest",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-y",
            str(source),
        ],
        check=False,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")

    audio = worker._decode_media_audio(source)

    assert audio.dtype == np.float32
    assert abs(audio.size - 16_000) <= 1_600
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 0.01


def test_media_decode_rejects_unknown_suffix(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"not media")
    with pytest.raises(worker.DatasetAcquisitionWorkerError, match="unsupported media type"):
        worker._decode_media_audio(source)


class _FakeAsr:
    def __init__(self, *, processed_text: str) -> None:
        self.processed_text = processed_text

    def transcribe(self, records, *, stage_root: Path, declared_language=None):
        segments = []
        for record in records:
            name = Path(record["path"]).name
            text = self.processed_text if name == "processed.wav" else "hello world"
            segments.append(
                {
                    "position": int(record["position"]),
                    "text": text,
                    "language": declared_language or "en",
                    "confidence": 0.99,
                    "emotion_suggestion": None,
                }
            )
        return {"schema": "fake-asr-v1", "segments": segments}


def _separation_dependency(tmp_path: Path) -> Path:
    root = tmp_path / "dependency" / "target-speaker-separation"
    clip_root = root / "clips" / "salvage" / "seg_000000_abcdef"
    original = clip_root / "original.wav"
    processed = clip_root / "processed.wav"
    _wav(original, 220)
    _wav(processed, 221)
    report = {
        "schema": worker.TARGET_SEPARATION_SCHEMA,
        "reference_prototype": {
            "schema": "aniflive-speaker-reference-prototype-v1",
            "reference_count": 2,
            "aggregation": "top-k-mean",
            "support_count": 2,
            "segments": [
                {"start_sample": 0, "end_sample": 8_000},
                {"start_sample": 8_000, "end_sample": 16_000},
            ],
        },
        "routing_policy": {
            "minimum_clean_snr_db": 20.0,
            "sensitive_diarization_margin": 0.10,
        },
        "diarization": {"backend": "nemo-speech-sortformer-v2", "clip_count": 1},
        "sensitive_diarization": {"candidate_clip_count": 1, "evidence_clip_count": 1},
        "records": [
            {
                "id": "seg_000000_abcdef",
                "position": 0,
                "path": processed.relative_to(root).as_posix(),
                "sha256": _sha256(processed),
                "source_name": "long-recording.mp4",
                "source_sha256": "a" * 64,
                "source_start_sample": 0,
                "source_end_sample": 16_000,
                "route": "salvage",
                "speaker": {"target_similarity": 0.94},
                "acquisition": {"route": "salvage"},
                "audio": {
                    "original_path": original.relative_to(root).as_posix(),
                    "original_sha256": _sha256(original),
                    "processed_path": processed.relative_to(root).as_posix(),
                    "processed_sha256": _sha256(processed),
                    "selected": "processed",
                },
                "separation": {"backend": "MossFormer2_SS_16K"},
            }
        ],
    }
    (root / "separation-report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return root.parent


def _routing_dependency(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "routing-dependency" / "target-speaker-routing"
    clip_root = root / "clips" / "salvage" / "seg_000000_abcdef"
    original = clip_root / "original.wav"
    reference = tmp_path / "reference.wav"
    _wav(original, 220)
    _wav(reference, 220)
    report = {
        "schema": worker.TARGET_ROUTING_SCHEMA,
        "source_sha256": "a" * 64,
        "reference_sha256": _sha256(reference),
        "source_seconds": 1.0,
        "reference_seconds": 1.0,
        "reference_prototype": {
            "schema": "aniflive-speaker-reference-prototype-v1",
            "reference_count": 2,
            "aggregation": "top-k-mean",
            "support_count": 2,
            "segments": [
                {"start_sample": 0, "end_sample": 8_000},
                {"start_sample": 8_000, "end_sample": 16_000},
            ],
        },
        "routing_policy": {
            "minimum_clean_snr_db": 20.0,
            "sensitive_diarization_margin": 0.10,
        },
        "diarization": {"backend": "nemo-speech-sortformer-v2", "clip_count": 1},
        "sensitive_diarization": {"candidate_clip_count": 1, "evidence_clip_count": 1},
        "records": [
            {
                "id": "seg_000000_abcdef",
                "position": 0,
                "path": original.relative_to(root).as_posix(),
                "sha256": _sha256(original),
                "route": "salvage",
                "route_reasons": ["overlap-evidence"],
                "acquisition": {"route": "salvage"},
                "audio": {
                    "original_path": original.relative_to(root).as_posix(),
                    "original_sha256": _sha256(original),
                    "processed_path": None,
                    "processed_sha256": None,
                    "selected": "original",
                },
            }
        ],
    }
    (root / "routing-report.json").write_text(json.dumps(report), encoding="utf-8")
    return root.parent, reference


@pytest.mark.parametrize(
    ("processed_text", "expected_route", "expected_selected"),
    [
        ("hello world", "salvage", "processed"),
        ("unrelated content", "review", "original"),
    ],
)
def test_transcription_preserves_raw_and_gates_processed_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    processed_text: str,
    expected_route: str,
    expected_selected: str,
) -> None:
    dependency = _separation_dependency(tmp_path)
    monkeypatch.setattr(
        worker,
        "create_workstation_asr_backend",
        lambda *_args, **_kwargs: _FakeAsr(processed_text=processed_text),
    )
    output = tmp_path / "transcription-output"

    report, artifacts = worker.run_target_speaker_transcription(
        dependency=dependency,
        asr_model=tmp_path,
        output=output,
        settings={"declared_language": "en"},
    )

    record = report["records"][0]
    assert record["route"] == expected_route
    assert record["audio"]["selected"] == expected_selected
    assert (output / "target-speaker-transcription" / record["audio"]["original_path"]).is_file()
    assert (output / "target-speaker-transcription" / record["audio"]["processed_path"]).is_file()
    assert any(path.name == "original.wav" for path in artifacts)
    assert any(path.name == "processed.wav" for path in artifacts)
    assert report["reference_prototype"]["aggregation"] == "top-k-mean"
    assert report["routing_policy"]["minimum_clean_snr_db"] == 20.0
    assert report["diarization"]["clip_count"] == 1
    assert report["sensitive_diarization"]["evidence_clip_count"] == 1

    final_output = tmp_path / "final-output"
    final_report, final_artifacts = worker.run_target_speaker_finalize(
        dependency=output,
        output=final_output,
    )
    final = final_report["records"][0]
    assert final["audio"]["selected"] == expected_selected
    assert (final_output / final["audio"]["original_path"]).is_file()
    assert (final_output / final["audio"]["processed_path"]).is_file()
    assert any(path.name == "original.wav" for path in final_artifacts)
    assert any(path.name == "processed.wav" for path in final_artifacts)
    assert final_report["reference_prototype"]["aggregation"] == "top-k-mean"
    assert final_report["routing_policy"]["minimum_clean_snr_db"] == 20.0
    assert final_report["diarization"]["clip_count"] == 1
    assert final_report["sensitive_diarization"]["evidence_clip_count"] == 1


def test_segment_separation_failure_retains_original_for_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency, reference = _routing_dependency(tmp_path)

    class FakeVerifier:
        def __init__(self, _root: Path) -> None:
            pass

        def __call__(self, _audio: np.ndarray, _rate: int) -> np.ndarray:
            return np.asarray([1.0, 0.0], dtype=np.float32)

    class FailingSeparator:
        def __init__(self, _root: Path) -> None:
            pass

        def separate_target(self, *_args, **_kwargs):
            from aniflive_tts.tse_separation import SeparationBackendError

            raise SeparationBackendError("segment could not be separated")

    from aniflive_tts import tse_separation

    monkeypatch.setattr(worker, "WorkstationSpeakerVerifier", FakeVerifier)
    monkeypatch.setattr(
        worker,
        "_decode_media_audio",
        lambda path, **_kwargs: worker._load_audio(path),
    )
    monkeypatch.setattr(tse_separation, "MossFormer2TargetSeparator", FailingSeparator)

    report, artifacts = worker.run_target_speaker_separation(
        dependency=dependency,
        reference=reference,
        speaker_component=tmp_path,
        separation_model=tmp_path,
        output=tmp_path / "separation-output",
        settings={},
    )

    record = report["records"][0]
    assert record["route"] == "review"
    assert record["audio"]["selected"] == "original"
    assert record["audio"]["processed_path"] is None
    assert record["separation"]["failure"]["code"] == "segment-separation-failed"
    assert "separation-failed-original-retained" in record["route_reasons"]
    assert report["counts"]["review"] == 1
    assert report["reference_prototype"]["aggregation"] == "top-k-mean"
    assert report["routing_policy"]["minimum_clean_snr_db"] == 20.0
    assert report["diarization"]["clip_count"] == 1
    assert report["sensitive_diarization"]["evidence_clip_count"] == 1
    assert any(path.name == "original.wav" for path in artifacts)


def test_no_target_clips_finishes_with_explicit_skipped_asr_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dependency = _separation_dependency(tmp_path)
    report_path = dependency / "target-speaker-separation" / "separation-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    record = report["records"][0]
    record["route"] = "reject"
    record["path"] = record["audio"]["original_path"]
    record["sha256"] = record["audio"]["original_sha256"]
    record["audio"]["selected"] = "original"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    class UnusedAsr:
        backend_id = "sensevoice-small-cuda-v1"

        def transcribe(self, *_args, **_kwargs):
            raise AssertionError("ASR must not run when every clip is rejected")

    monkeypatch.setattr(
        worker,
        "create_workstation_asr_backend",
        lambda *_args, **_kwargs: UnusedAsr(),
    )
    output = tmp_path / "transcription-output"

    result, _ = worker.run_target_speaker_transcription(
        dependency=dependency,
        asr_model=tmp_path,
        output=output,
        settings={"declared_language": "ja"},
    )

    assert result["records"][0]["route"] == "reject"
    assert result["asr"]["segments"] == []
    assert result["asr"]["model_qualification"] == "not-executed-no-target-clips"
    assert result["asr"]["review_reasons"] == ["no-target-clips-to-transcribe"]
