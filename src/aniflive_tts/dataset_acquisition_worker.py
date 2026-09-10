from __future__ import annotations

from dataclasses import asdict
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping
import unicodedata

import numpy as np

from .dataset_acquisition import (
    PurityRoute,
    SpeakerPurityEvidence,
    fuse_diarized_speaker_scores,
    post_separation_quality_gate,
    route_speaker_purity,
    stable_clip_id,
)
from .dataset_asr import DATASET_ASR_SCHEMA, create_workstation_asr_backend
from .dataset_quality import analyze_pcm_quality
from .workstation_speaker import (
    SpeakerReferencePrototype,
    WorkstationSpeakerVerifier,
    build_speaker_reference_prototype,
)
from .workstation_diarization import (
    DiarizationPostprocess,
    DiarizationResult,
    SortformerDiarizer,
    interval_diarization_evidence,
)
from .workstation_vad import WorkstationFsmnVad


TARGET_ROUTING_SCHEMA = "aniflive-target-speaker-routing-v1"
TARGET_SEPARATION_SCHEMA = "aniflive-target-speaker-separation-v1"
TARGET_TRANSCRIPTION_SCHEMA = "aniflive-target-speaker-transcription-v1"
TARGET_FINALIZE_SCHEMA = "aniflive-target-speaker-finalize-v1"
TARGET_SAMPLE_RATE = 16_000
TARGET_MAX_SOURCE_SECONDS = 4 * 60 * 60
TARGET_MEDIA_SUFFIXES = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".mp4", ".mkv", ".mov"}
)


class DatasetAcquisitionWorkerError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise DatasetAcquisitionWorkerError("clip audio is empty or non-finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), values, sample_rate, subtype="PCM_16")


def _bounded_stderr(value: bytes, limit: int = 2_000) -> str:
    return value.decode("utf-8", errors="replace").strip()[:limit]


def _decode_media_audio(
    path: Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    max_duration_seconds: float = TARGET_MAX_SOURCE_SECONDS,
) -> np.ndarray:
    source = Path(path).resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise DatasetAcquisitionWorkerError("media input must be a regular file")
    if source.suffix.casefold() not in TARGET_MEDIA_SUFFIXES:
        raise DatasetAcquisitionWorkerError(
            f"unsupported media type for target-speaker acquisition: {source.suffix or '(none)'}"
        )
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "file,pipe",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type,sample_rate,channels,duration:format=duration",
        "-of",
        "json",
        str(source),
    ]
    try:
        probe = subprocess.run(
            probe_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except FileNotFoundError as error:
        raise DatasetAcquisitionWorkerError(
            "ffprobe is not installed in the Linux acquisition worker"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise DatasetAcquisitionWorkerError("ffprobe exceeded its hard timeout") from error
    if probe.returncode != 0:
        raise DatasetAcquisitionWorkerError(
            "ffprobe rejected target-speaker media: " + (_bounded_stderr(probe.stderr) or "unknown error")
        )
    try:
        document = json.loads(probe.stdout.decode("utf-8"))
        streams = document.get("streams")
        stream = streams[0] if isinstance(streams, list) and len(streams) == 1 else None
        format_value = document.get("format")
        format_data = format_value if isinstance(format_value, Mapping) else {}
        duration_value = stream.get("duration") if isinstance(stream, Mapping) else None
        if duration_value in {None, "", "N/A"}:
            duration_value = format_data.get("duration")
        duration = float(duration_value)
        channels = int(stream.get("channels")) if isinstance(stream, Mapping) else 0
    except (AttributeError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetAcquisitionWorkerError("ffprobe returned malformed media metadata") from error
    if (
        not isinstance(stream, Mapping)
        or stream.get("codec_type") != "audio"
        or channels < 1
        or channels > 64
        or not math.isfinite(duration)
        or duration <= 0
    ):
        raise DatasetAcquisitionWorkerError("media has no valid first audio stream")
    if duration > max_duration_seconds:
        raise DatasetAcquisitionWorkerError(
            f"target-speaker source exceeds the {max_duration_seconds / 3600:g}-hour safety limit"
        )
    estimated_samples = int(math.ceil(duration * sample_rate))
    maximum_output_bytes = 65_536 + estimated_samples * 2
    timeout_seconds = max(120.0, min(172_800.0, duration * 4.0 + 60.0))
    with tempfile.TemporaryDirectory(prefix="aniflive-target-decode-") as temporary:
        canonical = Path(temporary) / "canonical.wav"
        command = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-t",
            f"{max_duration_seconds:.9f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            "s16",
            "-c:a",
            "pcm_s16le",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-fflags",
            "+bitexact",
            "-flags:a",
            "+bitexact",
            "-threads",
            "1",
            "-fs",
            str(maximum_output_bytes),
            "-f",
            "wav",
            "-y",
            str(canonical),
        ]
        try:
            decoded = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as error:
            raise DatasetAcquisitionWorkerError(
                "ffmpeg is not installed in the Linux acquisition worker"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise DatasetAcquisitionWorkerError("ffmpeg exceeded its hard timeout") from error
        if decoded.returncode != 0:
            raise DatasetAcquisitionWorkerError(
                "ffmpeg rejected target-speaker media: "
                + (_bounded_stderr(decoded.stderr) or "unknown error")
            )
        values = _load_audio(canonical, sample_rate)
    tolerance = max(64, round(sample_rate * 0.1))
    if abs(values.size - estimated_samples) > tolerance:
        raise DatasetAcquisitionWorkerError(
            "decoded sample count materially disagrees with ffprobe duration"
        )
    return values


def _load_audio(path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf

    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as error:
        raise DatasetAcquisitionWorkerError(f"audio could not be decoded: {path.name}") from error
    values = np.mean(np.asarray(audio, dtype=np.float32), axis=1).reshape(-1)
    if int(rate) != sample_rate:
        from scipy.signal import resample_poly

        common = math.gcd(int(rate), sample_rate)
        values = resample_poly(
            values, sample_rate // common, int(rate) // common
        ).astype(np.float32, copy=False)
    if values.size == 0 or not np.isfinite(values).all():
        raise DatasetAcquisitionWorkerError(f"audio is empty or malformed: {path.name}")
    return np.clip(values, -1.0, 1.0)


def _normalised_transcript(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE)


def _transcript_consistency(before: Any, after: Any) -> tuple[bool, float]:
    left = _normalised_transcript(before)
    right = _normalised_transcript(after)
    if not left or not right:
        return False, 0.0
    score = SequenceMatcher(None, left, right, autojunk=False).ratio()
    return score >= 0.80, float(score)


def _strict_report(root: Path, schema: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in root.rglob("*.json"):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("schema") == schema:
            matches.append((path, value))
    if len(matches) != 1:
        raise DatasetAcquisitionWorkerError(
            f"dependency must contain exactly one {schema} report"
        )
    return matches[0]


def _contained_path(root: Path, report_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise DatasetAcquisitionWorkerError("dependency clip path is malformed")
    candidate = (report_path.parent / value).resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise DatasetAcquisitionWorkerError("dependency clip escaped its artifact tree")
    if not candidate.is_file() or candidate.is_symlink() or candidate.suffix.casefold() != ".wav":
        raise DatasetAcquisitionWorkerError("dependency clip must be a regular WAV file")
    return candidate


def _record_artifacts(root: Path) -> list[Path]:
    return sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.as_posix())


def _prototype_bounds_from_report(
    value: Any,
    *,
    maximum_samples: int,
) -> tuple[tuple[int, int], ...]:
    if value is None:
        # Resume pre-prototype v1.4 research artifacts deterministically. New
        # routing reports always persist the segmented prototype contract.
        return ((0, maximum_samples),)
    if not isinstance(value, Mapping) or value.get("schema") != (
        "aniflive-speaker-reference-prototype-v1"
    ):
        raise DatasetAcquisitionWorkerError("reference prototype report is missing")
    segments = value.get("segments")
    if not isinstance(segments, list) or not 1 <= len(segments) <= 16:
        raise DatasetAcquisitionWorkerError("reference prototype segments are malformed")
    result: list[tuple[int, int]] = []
    for item in segments:
        if not isinstance(item, Mapping):
            raise DatasetAcquisitionWorkerError("reference prototype segment is malformed")
        start = item.get("start_sample")
        end = item.get("end_sample")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end > maximum_samples
        ):
            raise DatasetAcquisitionWorkerError("reference prototype bounds are malformed")
        result.append((start, end))
    if len(result) != value.get("reference_count"):
        raise DatasetAcquisitionWorkerError("reference prototype count does not match")
    return tuple(result)


def _prototype_similarity(
    prototype: SpeakerReferencePrototype,
    embedding: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    profile = prototype.score_embedding(embedding)
    return float(profile["similarity"]), profile


def _score_diarization(
    *,
    clip: np.ndarray,
    result: DiarizationResult,
    sample_rate: int,
    verifier: WorkstationSpeakerVerifier,
    reference_prototype: SpeakerReferencePrototype,
    target_threshold: float,
    review_margin: float,
    distinct_speaker_margin: float,
    minimum_speaker_seconds: float,
    minimum_overlap_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = interval_diarization_evidence(
        result.segments,
        start_sample=0,
        end_sample=clip.size,
        sample_rate=sample_rate,
        minimum_speaker_seconds=minimum_speaker_seconds,
        minimum_overlap_seconds=minimum_overlap_seconds,
    )
    scores: dict[str, float] = {}
    profiles: dict[str, dict[str, Any]] = {}
    for speaker in evidence["speakers"]:
        pieces = [
            clip[item.start_sample : item.end_sample]
            for item in result.segments
            if item.speaker == speaker
        ]
        if not pieces:
            continue
        embedding = verifier(np.concatenate(pieces), sample_rate)
        score, profile = _prototype_similarity(reference_prototype, embedding)
        scores[speaker] = score
        profiles[speaker] = profile
    fusion = fuse_diarized_speaker_scores(
        scores,
        target_threshold=target_threshold,
        review_margin=review_margin,
        distinct_speaker_margin=distinct_speaker_margin,
    )
    return evidence, {
        **fusion,
        "speaker_score_profiles": dict(sorted(profiles.items())),
    }


def run_target_speaker_routing(
    *,
    source: Path,
    reference: Path,
    speaker_component: Path,
    vad_component: Path,
    diarization_component: Path,
    output: Path,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    audio = _decode_media_audio(source)
    reference_audio = _decode_media_audio(reference, max_duration_seconds=30 * 60)
    sample_rate = TARGET_SAMPLE_RATE
    target_threshold = float(settings.get("target_threshold", settings.get("speaker_threshold", 0.72)))
    review_margin = float(settings.get("review_margin", 0.08))
    ambiguity_margin = float(settings.get("ambiguity_margin", 0.03))
    verifier = WorkstationSpeakerVerifier(speaker_component)
    vad = WorkstationFsmnVad(vad_component)
    reference_segments, reference_vad_profile = vad.detect(
        reference_audio,
        sample_rate,
        minimum_speech_ms=float(settings.get("minimum_reference_speech_ms", 160.0)),
        maximum_segment_ms=float(
            settings.get("maximum_reference_segment_ms", 6_000.0)
        ),
        context_ms=0.0,
    )
    reference_prototype = build_speaker_reference_prototype(
        reference_audio,
        sample_rate,
        embedder=verifier,
        segments=tuple(
            (segment.start_sample, segment.end_sample)
            for segment in reference_segments
        ),
        maximum_references=int(settings.get("maximum_reference_prototypes", 8)),
    )
    reference_prototype_profile = {
        **reference_prototype.as_dict(sample_rate),
        "vad_backend": reference_vad_profile.get("backend"),
    }
    segments, vad_profile = vad.detect(
        audio,
        sample_rate,
        minimum_speech_ms=float(settings.get("minimum_speech_ms", 160.0)),
        maximum_segment_ms=float(settings.get("maximum_segment_ms", 30_000.0)),
        context_ms=float(settings.get("context_ms", 40.0)),
    )
    clips = [audio[segment.start_sample : segment.end_sample] for segment in segments]
    diarization = SortformerDiarizer(diarization_component).diarize_clips(
        clips, sample_rate
    )
    source_sha = _sha256_file(source)
    stage = output / "target-speaker-routing"
    minimum_speaker_seconds = float(
        settings.get("minimum_diarized_speaker_seconds", 0.24)
    )
    minimum_overlap_seconds = float(
        settings.get("minimum_diarized_overlap_seconds", 0.16)
    )
    distinct_speaker_margin = float(settings.get("distinct_speaker_margin", 0.10))
    minimum_clean_snr_db = float(settings.get("minimum_clean_snr_db", 20.0))
    sensitive_margin = float(settings.get("sensitive_diarization_margin", 0.10))
    if not math.isfinite(sensitive_margin) or not 0.0 <= sensitive_margin <= 0.30:
        raise DatasetAcquisitionWorkerError(
            "sensitive diarization margin must be between zero and 0.30"
        )
    profiles: list[dict[str, Any]] = []
    for position, (segment, clip, clip_diarization) in enumerate(
        zip(segments, clips, diarization.clips, strict=True)
    ):
        embedding = verifier(clip, sample_rate)
        similarity, reference_score = _prototype_similarity(
            reference_prototype, embedding
        )
        diarization_evidence, diarization_fusion = _score_diarization(
            clip=clip,
            result=clip_diarization,
            sample_rate=sample_rate,
            verifier=verifier,
            reference_prototype=reference_prototype,
            target_threshold=target_threshold,
            review_margin=review_margin,
            distinct_speaker_margin=distinct_speaker_margin,
            minimum_speaker_seconds=minimum_speaker_seconds,
            minimum_overlap_seconds=minimum_overlap_seconds,
        )
        quality = analyze_pcm_quality(clip[:, None], sample_rate).as_dict()
        route = route_speaker_purity(
            SpeakerPurityEvidence(
                target_similarity=similarity,
                # There is no competing separated candidate before MossFormer2.
                # Speaker contamination is supplied by the independent diarizer.
                winner_margin=1.0,
                embedding_variance=0.0,
                estimated_snr_db=quality.get("estimated_snr_db"),
                clipped_ratio=quality.get("clipped_ratio", 0.0),
                vad_quality=1.0,
                overlap_evidence=bool(
                    diarization_evidence["overlap_evidence"]
                    and diarization_fusion["distinct_speaker_evidence"]
                ),
                speaker_change_evidence=bool(
                    diarization_fusion["distinct_speaker_evidence"]
                ),
                diarization_uncertain_evidence=bool(
                    diarization_fusion["uncertain_multi_speaker"]
                ),
            ),
            target_threshold=target_threshold,
            review_margin=review_margin,
            ambiguity_margin=ambiguity_margin,
            minimum_clean_snr_db=minimum_clean_snr_db,
        )
        if segment.forced_split and route.route != "reject":
            route = PurityRoute(
                "review",
                tuple(dict.fromkeys((*route.reasons, "forced-split-not-silence-validated"))),
            )
        profiles.append(
            {
                "position": position,
                "segment": segment,
                "clip": clip,
                "similarity": similarity,
                "reference_score": reference_score,
                "quality": quality,
                "route": route,
                "diarization": diarization_evidence,
                "diarization_fusion": diarization_fusion,
                "sensitive_diarization": None,
                "sensitive_diarization_fusion": None,
            }
        )

    sensitive_indices = [
        position
        for position, profile in enumerate(profiles)
        if profile["route"].route == "clean"
        and target_threshold <= float(profile["similarity"])
        < min(1.0, target_threshold + sensitive_margin)
    ]
    sensitive_postprocess = DiarizationPostprocess(
        onset=float(settings.get("sensitive_diarization_onset", 0.30)),
        offset=float(settings.get("sensitive_diarization_offset", 0.70)),
        pad_onset_seconds=float(
            settings.get("sensitive_diarization_pad_onset_seconds", 0.05)
        ),
        pad_offset_seconds=0.0,
        minimum_on_seconds=float(
            settings.get("sensitive_diarization_minimum_on_seconds", 0.20)
        ),
        minimum_off_seconds=float(
            settings.get("sensitive_diarization_minimum_off_seconds", 0.20)
        ),
    )
    sensitive_diarization = None
    sensitive_evidence_count = 0
    if sensitive_indices:
        sensitive_diarization = SortformerDiarizer(
            diarization_component, postprocess=sensitive_postprocess
        ).diarize_clips([clips[position] for position in sensitive_indices], sample_rate)
        for position, clip_diarization in zip(
            sensitive_indices, sensitive_diarization.clips, strict=True
        ):
            profile = profiles[position]
            evidence, fusion = _score_diarization(
                clip=profile["clip"],
                result=clip_diarization,
                sample_rate=sample_rate,
                verifier=verifier,
                reference_prototype=reference_prototype,
                target_threshold=target_threshold,
                review_margin=review_margin,
                distinct_speaker_margin=distinct_speaker_margin,
                minimum_speaker_seconds=minimum_speaker_seconds,
                minimum_overlap_seconds=minimum_overlap_seconds,
            )
            profile["sensitive_diarization"] = evidence
            profile["sensitive_diarization_fusion"] = fusion
            if not (
                evidence["overlap_evidence"]
                or evidence["speaker_change_evidence"]
            ):
                continue
            sensitive_evidence_count += 1
            route = route_speaker_purity(
                SpeakerPurityEvidence(
                    target_similarity=float(profile["similarity"]),
                    winner_margin=1.0,
                    estimated_snr_db=profile["quality"].get("estimated_snr_db"),
                    clipped_ratio=profile["quality"].get("clipped_ratio", 0.0),
                    vad_quality=1.0,
                    overlap_evidence=bool(
                        evidence["overlap_evidence"]
                        and fusion["distinct_speaker_evidence"]
                    ),
                    speaker_change_evidence=bool(
                        fusion["distinct_speaker_evidence"]
                    ),
                    diarization_uncertain_evidence=bool(
                        fusion["uncertain_multi_speaker"]
                    ),
                ),
                target_threshold=target_threshold,
                review_margin=review_margin,
                ambiguity_margin=ambiguity_margin,
                minimum_clean_snr_db=minimum_clean_snr_db,
            )
            if route.route == "clean":
                route = PurityRoute(
                    "review", ("sensitive-diarization-needs-review",)
                )
            profile["route"] = route

    records: list[dict[str, Any]] = []
    for profile in profiles:
        position = int(profile["position"])
        segment = profile["segment"]
        clip = profile["clip"]
        similarity = float(profile["similarity"])
        reference_score = profile["reference_score"]
        quality = profile["quality"]
        route = profile["route"]
        diarization_evidence = profile["diarization"]
        diarization_fusion = profile["diarization_fusion"]
        clip_id = stable_clip_id(
            source_sha, segment.start_sample, segment.end_sample, position
        )
        clip_path = stage / "clips" / route.route / clip_id / "original.wav"
        _write_wav(clip_path, clip, sample_rate)
        records.append(
            {
                "id": clip_id,
                "position": position,
                "path": clip_path.relative_to(stage).as_posix(),
                "sha256": _sha256_file(clip_path),
                "source_name": source.name,
                "source_sha256": source_sha,
                "source_start_sample": segment.start_sample,
                "source_end_sample": segment.end_sample,
                "boundary": {
                    "silence_validated": segment.silence_validated,
                    "forced_split": segment.forced_split,
                },
                "route": route.route,
                "route_reasons": list(route.reasons),
                "speaker": {
                    "target_similarity": similarity,
                    "reference_score": reference_score,
                    "diarization": diarization_evidence,
                    "diarization_fusion": diarization_fusion,
                    "sensitive_diarization": profile["sensitive_diarization"],
                    "sensitive_diarization_fusion": profile[
                        "sensitive_diarization_fusion"
                    ],
                },
                "quality": quality,
                "acquisition": {
                    "route": route.route,
                    "overlap_evidence": bool(
                        diarization_evidence["overlap_evidence"]
                        or (
                            profile["sensitive_diarization"] is not None
                            and profile["sensitive_diarization"][
                                "overlap_evidence"
                            ]
                        )
                    ),
                    "speaker_change_evidence": bool(
                        diarization_fusion["distinct_speaker_evidence"]
                        or (
                            profile["sensitive_diarization_fusion"] is not None
                            and profile["sensitive_diarization_fusion"][
                                "distinct_speaker_evidence"
                            ]
                        )
                    ),
                    "verifier": "workstation-eres2netv2-tensorrt11",
                    "diarizer": "sortformer-v2-nemo-speech-cpp-cuda",
                },
                "audio": {
                    "original_path": clip_path.relative_to(stage).as_posix(),
                    "original_sha256": _sha256_file(clip_path),
                    "processed_path": None,
                    "processed_sha256": None,
                    "selected": "original",
                },
            }
        )
    report = {
        "schema": TARGET_ROUTING_SCHEMA,
        "source_sha256": source_sha,
        "reference_sha256": _sha256_file(reference),
        "sample_rate": sample_rate,
        "source_seconds": audio.size / sample_rate,
        "reference_seconds": reference_audio.size / sample_rate,
        "reference_prototype": reference_prototype_profile,
        "records": records,
        "counts": {
            route: sum(record["route"] == route for record in records)
            for route in ("clean", "salvage", "review", "reject")
        },
        "speaker_backend": "workstation-eres2netv2-tensorrt11",
        "routing_policy": {
            "target_threshold": target_threshold,
            "review_margin": review_margin,
            "ambiguity_margin": ambiguity_margin,
            "minimum_clean_snr_db": minimum_clean_snr_db,
        },
        "diarization": diarization.as_dict(),
        "sensitive_diarization": {
            "enabled": True,
            "candidate_count": len(sensitive_indices),
            "evidence_count": sensitive_evidence_count,
            "selection": {
                "primary_route": "clean",
                "target_similarity_minimum": target_threshold,
                "target_similarity_maximum_exclusive": min(
                    1.0, target_threshold + sensitive_margin
                ),
            },
            "postprocess": sensitive_postprocess.as_dict(),
            "batch": (
                sensitive_diarization.as_dict()
                if sensitive_diarization is not None
                else None
            ),
        },
        "vad": vad_profile,
        "network_required": False,
    }
    report_path = stage / "routing-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, _record_artifacts(stage)


def run_target_speaker_separation(
    *,
    dependency: Path,
    reference: Path,
    speaker_component: Path,
    separation_model: Path,
    output: Path,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    report_path, routing = _strict_report(dependency, TARGET_ROUTING_SCHEMA)
    records_value = routing.get("records")
    if not isinstance(records_value, list):
        raise DatasetAcquisitionWorkerError("speaker routing report has no records")
    verifier = WorkstationSpeakerVerifier(speaker_component)
    reference_audio = _decode_media_audio(reference, max_duration_seconds=30 * 60)
    if _sha256_file(reference) != routing.get("reference_sha256"):
        raise DatasetAcquisitionWorkerError(
            "reference audio does not match the routing report"
        )
    reference_prototype = build_speaker_reference_prototype(
        reference_audio,
        TARGET_SAMPLE_RATE,
        embedder=verifier,
        segments=_prototype_bounds_from_report(
            routing.get("reference_prototype"),
            maximum_samples=reference_audio.size,
        ),
        maximum_references=16,
    )
    reference_embedding = reference_prototype.centroid_embedding

    def score_embedding(value: np.ndarray) -> float:
        return float(reference_prototype.score_embedding(value)["similarity"])
    from .tse_separation import MossFormer2TargetSeparator, SeparationBackendError

    separator = MossFormer2TargetSeparator(separation_model)
    target_threshold = float(settings.get("target_threshold", settings.get("speaker_threshold", 0.72)))
    ambiguity_margin = float(settings.get("ambiguity_margin", 0.03))
    stage = output / "target-speaker-separation"
    final_records: list[dict[str, Any]] = []
    for record in records_value:
        if not isinstance(record, Mapping):
            raise DatasetAcquisitionWorkerError("speaker routing record is malformed")
        clip = _contained_path(dependency, report_path, record.get("path"))
        if _sha256_file(clip) != record.get("sha256"):
            raise DatasetAcquisitionWorkerError("speaker routing clip failed integrity validation")
        audio = _load_audio(clip, TARGET_SAMPLE_RATE)
        route = str(record.get("route"))
        result_record = dict(record)
        destination_route = route
        separation: dict[str, Any] | None = None
        output_audio = audio
        if route == "salvage":
            before_embedding = verifier(audio, TARGET_SAMPLE_RATE)
            before_similarity, before_score = _prototype_similarity(
                reference_prototype, before_embedding
            )
            try:
                separated = separator.separate_target(
                    audio,
                    TARGET_SAMPLE_RATE,
                    reference_embedding=reference_embedding,
                    embedder=verifier,
                    embedding_scorer=score_embedding,
                    target_threshold=target_threshold,
                    ambiguity_margin=ambiguity_margin,
                )
            except SeparationBackendError as error:
                destination_route = "review"
                result_record["route"] = "review"
                result_record["route_reasons"] = list(
                    dict.fromkeys(
                        [
                            *list(result_record.get("route_reasons") or []),
                            "separation-failed-original-retained",
                        ]
                    )
                )
                result_record["acquisition"] = {
                    **dict(result_record.get("acquisition") or {}),
                    "route": "review",
                    "separator": "MossFormer2_SS_16K",
                }
                separation = {
                    "backend": "MossFormer2_SS_16K",
                    "accepted": False,
                    "failure": {
                        "code": "segment-separation-failed",
                        "message": str(error).strip()[:256],
                    },
                }
                separated = None
            if separated is None:
                output_audio = audio
            else:
                output_audio = separated.audio
                after_embedding = verifier(output_audio, TARGET_SAMPLE_RATE)
                after_similarity, after_score = _prototype_similarity(
                    reference_prototype, after_embedding
                )
                selected_scores = [
                    chunk.candidate_similarities[chunk.selected_candidate]
                    for chunk in separated.chunks
                ]
                loser_scores = [
                    chunk.candidate_similarities[1 - chunk.selected_candidate]
                    for chunk in separated.chunks
                ]
                artifact_score = max(
                    (min(1.0, chunk.reconstruction_rmse * 10.0) for chunk in separated.chunks),
                    default=1.0,
                )
                gate = post_separation_quality_gate(
                    target_similarity=min(selected_scores, default=after_similarity),
                    loser_similarity=max(loser_scores, default=-1.0),
                    similarity_before=before_similarity,
                    asr_consistent=None,
                    finite_audio=bool(np.isfinite(output_audio).all()),
                    exact_sample_length=output_audio.size == audio.size,
                    artifact_score=artifact_score,
                    target_threshold=target_threshold,
                    ambiguity_margin=ambiguity_margin,
                )
                destination_route = "clean" if gate.route == "clean" else gate.route
                separation = {
                    "backend": "MossFormer2_SS_16K",
                    "accepted": gate.route == "clean",
                    "gate": gate.as_dict(),
                    "speaker_similarity_before": before_similarity,
                    "speaker_similarity_after": after_similarity,
                    "speaker_score_before": before_score,
                    "speaker_score_after": after_score,
                    "asr_consistency": "pending-transcription",
                    "chunks": [asdict(chunk) for chunk in separated.chunks],
                }
                result_record["route"] = "salvage" if gate.route == "clean" else gate.route
                result_record["acquisition"] = {
                    **dict(result_record.get("acquisition") or {}),
                    "route": result_record["route"],
                    "separator": "MossFormer2_SS_16K",
                }
        clip_root = stage / "clips" / destination_route / str(record["id"])
        original_destination = clip_root / "original.wav"
        _write_wav(original_destination, audio, TARGET_SAMPLE_RATE)
        processed_destination: Path | None = None
        selected_destination = original_destination
        if separation is not None and separated is not None:
            processed_destination = clip_root / "processed.wav"
            _write_wav(processed_destination, output_audio, TARGET_SAMPLE_RATE)
            if gate.route == "clean":
                selected_destination = processed_destination
        result_record["path"] = selected_destination.relative_to(stage).as_posix()
        result_record["sha256"] = _sha256_file(selected_destination)
        result_record["audio"] = {
            "original_path": original_destination.relative_to(stage).as_posix(),
            "original_sha256": _sha256_file(original_destination),
            "processed_path": (
                processed_destination.relative_to(stage).as_posix()
                if processed_destination is not None
                else None
            ),
            "processed_sha256": (
                _sha256_file(processed_destination)
                if processed_destination is not None
                else None
            ),
            "selected": "processed" if selected_destination == processed_destination else "original",
        }
        result_record["separation"] = separation
        final_records.append(result_record)
    report = {
        "schema": TARGET_SEPARATION_SCHEMA,
        "source_sha256": routing.get("source_sha256"),
        "reference_sha256": routing.get("reference_sha256"),
        "source_seconds": routing.get("source_seconds"),
        "reference_seconds": routing.get("reference_seconds"),
        "reference_prototype": routing.get("reference_prototype"),
        "routing_policy": routing.get("routing_policy"),
        "diarization": routing.get("diarization"),
        "sensitive_diarization": routing.get("sensitive_diarization"),
        "records": final_records,
        "speaker_backend": routing.get("speaker_backend"),
        "vad": routing.get("vad"),
        "counts": {
            route: sum(record["route"] == route for record in final_records)
            for route in ("clean", "salvage", "review", "reject")
        },
        "separator": "MossFormer2_SS_16K",
        "network_required": False,
    }
    path = stage / "separation-report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report, _record_artifacts(stage)


def run_target_speaker_transcription(
    *,
    dependency: Path,
    asr_model: Path,
    output: Path,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    report_path, separated = _strict_report(dependency, TARGET_SEPARATION_SCHEMA)
    records_value = separated.get("records")
    if not isinstance(records_value, list):
        raise DatasetAcquisitionWorkerError("separation report has no records")
    stage = output / "target-speaker-transcription"
    segment_records: list[dict[str, Any]] = []
    original_records: list[dict[str, Any]] = []
    copied_records: list[dict[str, Any]] = []
    for record in records_value:
        if not isinstance(record, Mapping):
            raise DatasetAcquisitionWorkerError("separation record is malformed")
        source = _contained_path(dependency, report_path, record.get("path"))
        if _sha256_file(source) != record.get("sha256"):
            raise DatasetAcquisitionWorkerError("separation clip failed integrity validation")
        route = str(record.get("route"))
        destination_root = stage / "clips" / route / str(record.get("id"))
        audio_value = record.get("audio")
        audio = dict(audio_value) if isinstance(audio_value, Mapping) else {}
        copied_audio: dict[str, Any] = {"selected": audio.get("selected", "original")}
        for kind in ("original", "processed"):
            raw_path = audio.get(f"{kind}_path")
            if raw_path is None:
                copied_audio[f"{kind}_path"] = None
                copied_audio[f"{kind}_sha256"] = None
                continue
            source_audio = _contained_path(dependency, report_path, raw_path)
            expected = audio.get(f"{kind}_sha256")
            if expected is not None and _sha256_file(source_audio) != expected:
                raise DatasetAcquisitionWorkerError(
                    f"separation {kind} clip failed integrity validation"
                )
            destination_audio = destination_root / f"{kind}.wav"
            destination_audio.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_audio, destination_audio)
            copied_audio[f"{kind}_path"] = destination_audio.relative_to(stage).as_posix()
            copied_audio[f"{kind}_sha256"] = _sha256_file(destination_audio)
        selected_kind = str(copied_audio["selected"])
        selected_relative = copied_audio.get(f"{selected_kind}_path")
        if not isinstance(selected_relative, str):
            raise DatasetAcquisitionWorkerError("selected acquisition clip is missing")
        destination = stage / selected_relative
        copied = dict(record)
        copied["path"] = destination.relative_to(stage).as_posix()
        copied["sha256"] = _sha256_file(destination)
        copied["audio"] = copied_audio
        copied_records.append(copied)
        if route != "reject":
            segment_records.append(
                {
                    "position": int(record.get("position", len(segment_records))),
                    "path": destination.relative_to(stage).as_posix(),
                    "sha256": _sha256_file(destination),
                }
            )
            if copied_audio.get("processed_path") is not None:
                original_records.append(
                    {
                        "position": int(record.get("position", len(original_records))),
                        "path": str(copied_audio["original_path"]),
                        "sha256": str(copied_audio["original_sha256"]),
                    }
                )
    backend_name = str(settings.get("asr_backend", "sensevoice-small"))
    backend = create_workstation_asr_backend(backend_name, asr_model)
    if segment_records:
        transcript_report = backend.transcribe(
            segment_records,
            stage_root=stage,
            declared_language=settings.get("declared_language"),
        )
    else:
        transcript_report = {
            "schema": DATASET_ASR_SCHEMA,
            "backend": backend.backend_id,
            "execution_environment": "linux-docker-cuda-only",
            "network_required": False,
            "model": None,
            "model_qualification": "not-executed-no-target-clips",
            "runtime_versions": {},
            "declared_language": settings.get("declared_language"),
            "language": settings.get("declared_language"),
            "text": "",
            "segments": [],
            "review_required": True,
            "review_reasons": ["no-target-clips-to-transcribe"],
        }
    original_report = (
        backend.transcribe(
            original_records,
            stage_root=stage,
            declared_language=settings.get("declared_language"),
        )
        if original_records
        else {"segments": []}
    )
    by_position = {
        int(record["position"]): record for record in transcript_report["segments"]
    }
    originals_by_position = {
        int(record["position"]): record for record in original_report["segments"]
    }
    for record in copied_records:
        position = int(record.get("position", -1))
        transcript = by_position.get(position)
        original_transcript = originals_by_position.get(position)
        separation_value = record.get("separation")
        if isinstance(separation_value, Mapping) and original_transcript is not None:
            consistent, score = _transcript_consistency(
                original_transcript.get("text"),
                transcript.get("text") if isinstance(transcript, Mapping) else None,
            )
            separation = dict(separation_value)
            separation["asr_consistency"] = {
                "passed": consistent,
                "similarity": round(score, 6),
                "original_text": original_transcript.get("text"),
                "processed_text": transcript.get("text") if isinstance(transcript, Mapping) else None,
            }
            record["separation"] = separation
            if not consistent:
                record["route"] = "review"
                audio = dict(record.get("audio") or {})
                audio["selected"] = "original"
                record["audio"] = audio
                record["path"] = str(audio["original_path"])
                record["sha256"] = str(audio["original_sha256"])
                transcript = original_transcript
        record["transcription"] = transcript
        if transcript is not None:
            record["annotations"] = {
                "transcript": transcript.get("text"),
                "language": transcript.get("language"),
                "expression_suggestion": transcript.get("emotion_suggestion"),
                "asr_backend": transcript_report.get("backend"),
                "authoritative": False,
                "review_required": True,
            }
    report = {
        "schema": TARGET_TRANSCRIPTION_SCHEMA,
        "source_sha256": separated.get("source_sha256"),
        "reference_sha256": separated.get("reference_sha256"),
        "source_seconds": separated.get("source_seconds"),
        "reference_seconds": separated.get("reference_seconds"),
        "reference_prototype": separated.get("reference_prototype"),
        "routing_policy": separated.get("routing_policy"),
        "diarization": separated.get("diarization"),
        "sensitive_diarization": separated.get("sensitive_diarization"),
        "records": copied_records,
        "speaker_backend": separated.get("speaker_backend"),
        "vad": separated.get("vad"),
        "separator": separated.get("separator"),
        "asr": transcript_report,
        "human_review_required": True,
        "network_required": False,
    }
    path = stage / "transcription-report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report, _record_artifacts(stage)


def run_target_speaker_finalize(
    *,
    dependency: Path,
    output: Path,
) -> tuple[dict[str, Any], list[Path]]:
    report_path, transcription = _strict_report(dependency, TARGET_TRANSCRIPTION_SCHEMA)
    records_value = transcription.get("records")
    if not isinstance(records_value, list):
        raise DatasetAcquisitionWorkerError("transcription report has no records")
    stage = output / "dataset-acquisition"
    finalized: list[dict[str, Any]] = []
    for record in records_value:
        if not isinstance(record, Mapping):
            raise DatasetAcquisitionWorkerError("transcription record is malformed")
        source = _contained_path(dependency, report_path, record.get("path"))
        if _sha256_file(source) != record.get("sha256"):
            raise DatasetAcquisitionWorkerError("transcribed clip failed integrity validation")
        route = str(record.get("route"))
        destination_root = stage / ("rejected" if route == "reject" else "clips") / str(record.get("id"))
        audio_value = record.get("audio")
        audio = dict(audio_value) if isinstance(audio_value, Mapping) else {}
        copied_audio: dict[str, Any] = {"selected": audio.get("selected", "original")}
        for kind in ("original", "processed"):
            raw_path = audio.get(f"{kind}_path")
            if raw_path is None:
                copied_audio[f"{kind}_path"] = None
                copied_audio[f"{kind}_sha256"] = None
                continue
            source_audio = _contained_path(dependency, report_path, raw_path)
            expected = audio.get(f"{kind}_sha256")
            if expected is not None and _sha256_file(source_audio) != expected:
                raise DatasetAcquisitionWorkerError(
                    f"transcribed {kind} clip failed integrity validation"
                )
            destination_audio = destination_root / f"{kind}.wav"
            destination_audio.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_audio, destination_audio)
            copied_audio[f"{kind}_path"] = destination_audio.relative_to(output).as_posix()
            copied_audio[f"{kind}_sha256"] = _sha256_file(destination_audio)
        selected_kind = str(copied_audio["selected"])
        selected_relative = copied_audio.get(f"{selected_kind}_path")
        if not isinstance(selected_relative, str):
            raise DatasetAcquisitionWorkerError("final selected acquisition clip is missing")
        destination = output / selected_relative
        final = dict(record)
        final["path"] = destination.relative_to(output).as_posix()
        final["sha256"] = _sha256_file(destination)
        final["audio"] = copied_audio
        finalized.append(final)
    report = {
        "schema": TARGET_FINALIZE_SCHEMA,
        "status": "waiting_for_review",
        "source_sha256": transcription.get("source_sha256"),
        "reference_sha256": transcription.get("reference_sha256"),
        "source_seconds": transcription.get("source_seconds"),
        "reference_seconds": transcription.get("reference_seconds"),
        "reference_prototype": transcription.get("reference_prototype"),
        "routing_policy": transcription.get("routing_policy"),
        "diarization": transcription.get("diarization"),
        "sensitive_diarization": transcription.get("sensitive_diarization"),
        "speaker_backend": transcription.get("speaker_backend"),
        "vad": transcription.get("vad"),
        "separator": transcription.get("separator"),
        "transcription_backend": (
            transcription.get("asr", {}).get("backend")
            if isinstance(transcription.get("asr"), Mapping)
            else None
        ),
        "records": finalized,
        "counts": {
            route: sum(record.get("route") == route for record in finalized)
            for route in ("clean", "salvage", "review", "reject")
        },
        "human_actions": [
            "audio-accept-reject",
            "transcript-correction",
            "expression-classification",
            "reference-confirmation",
        ],
        "network_required": False,
    }
    path = stage / "dataset-acquisition-report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report, _record_artifacts(stage)


__all__ = [
    "TARGET_FINALIZE_SCHEMA",
    "TARGET_ROUTING_SCHEMA",
    "TARGET_SEPARATION_SCHEMA",
    "TARGET_TRANSCRIPTION_SCHEMA",
    "DatasetAcquisitionWorkerError",
    "run_target_speaker_finalize",
    "run_target_speaker_routing",
    "run_target_speaker_separation",
    "run_target_speaker_transcription",
]
