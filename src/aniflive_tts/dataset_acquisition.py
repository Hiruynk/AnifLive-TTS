from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


DATASET_ACQUISITION_SCHEMA = "aniflive-dataset-acquisition-v1"
ACQUISITION_MODES = frozenset({"standard", "target-speaker", "gpt-sovits-list"})
DATASET_WORKFLOW_STAGES = (
    "source",
    "speaker",
    "clean",
    "text",
    "review",
    "style",
    "ready",
)
PURITY_ROUTES = frozenset({"clean", "salvage", "review", "reject"})
DATASET_LANGUAGES = frozenset({"yue", "zh", "ja", "en", "ko"})


class DatasetAcquisitionError(ValueError):
    """Raised when a voice-acquisition contract cannot be validated."""


def _finite_number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise DatasetAcquisitionError(f"{field} must be a JSON number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetAcquisitionError(f"{field} must be a JSON number") from error
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise DatasetAcquisitionError(
            f"{field} must be between {minimum:g} and {maximum:g}"
        )
    return result


def _path_text(value: Any, field: str, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DatasetAcquisitionError(f"{field} must be a local path")
    result = value.strip()
    if "\x00" in result or len(result) > 4096:
        raise DatasetAcquisitionError(f"{field} is malformed")
    return result


@dataclass(frozen=True)
class DatasetAcquisitionConfig:
    acquisition_mode: str = "standard"
    sources: tuple[str, ...] = ()
    reference_audio: str | None = None
    speaker_threshold: float = 0.72
    ambiguity_margin: float = 0.03
    review_margin: float = 0.08
    speaker_component: str = "eres2netv2-speaker-verifier"
    vad_component: str = "fsmn-vad"
    diarization_component: str = "sortformer-overlap-diarization"
    separation_component: str = "mossformer2-ss-16k"
    asr_component: str = "sensevoice-small"
    declared_language: str | None = None
    require_expressions: bool = True

    def validated(self) -> "DatasetAcquisitionConfig":
        if self.acquisition_mode not in ACQUISITION_MODES:
            raise DatasetAcquisitionError("acquisition_mode is unsupported")
        if len(self.sources) > 256:
            raise DatasetAcquisitionError("sources is limited to 256 entries")
        sources = tuple(
            _path_text(value, f"sources[{index}]", required=True) or ""
            for index, value in enumerate(self.sources)
        )
        if len(sources) != len(set(sources)):
            raise DatasetAcquisitionError("sources contains duplicates")
        reference = _path_text(
            self.reference_audio,
            "reference_audio",
            required=self.acquisition_mode == "target-speaker",
        )
        if self.acquisition_mode == "target-speaker" and not sources:
            raise DatasetAcquisitionError(
                "target-speaker acquisition requires at least one source"
            )
        if self.acquisition_mode == "gpt-sovits-list":
            if len(sources) != 1 or Path(sources[0]).suffix.casefold() != ".list":
                raise DatasetAcquisitionError(
                    "gpt-sovits-list acquisition requires exactly one .list source"
                )
        if self.acquisition_mode != "target-speaker" and reference is not None:
            raise DatasetAcquisitionError(
                "reference_audio is valid only for target-speaker acquisition"
            )
        for field, value in (
            ("speaker_component", self.speaker_component),
            ("vad_component", self.vad_component),
            ("diarization_component", self.diarization_component),
            ("separation_component", self.separation_component),
            ("asr_component", self.asr_component),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 80
                or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in value)
            ):
                raise DatasetAcquisitionError(f"{field} is malformed")
        declared_language = self.declared_language
        if declared_language is not None:
            if not isinstance(declared_language, str):
                raise DatasetAcquisitionError("declared_language must be a language code")
            declared_language = declared_language.strip().casefold()
            if declared_language not in DATASET_LANGUAGES:
                raise DatasetAcquisitionError("declared_language is unsupported")
        if not isinstance(self.require_expressions, bool):
            raise DatasetAcquisitionError("require_expressions must be boolean")
        return DatasetAcquisitionConfig(
            acquisition_mode=self.acquisition_mode,
            sources=sources,
            reference_audio=reference,
            speaker_threshold=_finite_number(
                self.speaker_threshold, "speaker_threshold", 0.0, 1.0
            ),
            ambiguity_margin=_finite_number(
                self.ambiguity_margin, "ambiguity_margin", 0.0, 0.5
            ),
            review_margin=_finite_number(
                self.review_margin, "review_margin", 0.0, 0.5
            ),
            speaker_component=self.speaker_component,
            vad_component=self.vad_component,
            diarization_component=self.diarization_component,
            separation_component=self.separation_component,
            asr_component=self.asr_component,
            declared_language=declared_language,
            require_expressions=self.require_expressions,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetAcquisitionConfig":
        if not isinstance(value, Mapping):
            raise DatasetAcquisitionError("dataset acquisition config must be an object")
        mode = value.get("acquisition_mode", "standard")
        raw_sources = value.get("sources")
        if raw_sources is None:
            source = value.get("source")
            raw_sources = [] if source in {None, ""} else [source]
        if isinstance(raw_sources, (str, bytes)) or not isinstance(raw_sources, Sequence):
            raise DatasetAcquisitionError("sources must be a list")
        return cls(
            acquisition_mode=str(mode),
            sources=tuple(raw_sources),
            reference_audio=value.get("reference_audio", value.get("reference")),
            speaker_threshold=value.get(
                "speaker_threshold", value.get("target_threshold", 0.72)
            ),
            ambiguity_margin=value.get("ambiguity_margin", 0.03),
            review_margin=value.get("review_margin", 0.08),
            speaker_component=str(
                value.get("speaker_component", "eres2netv2-speaker-verifier")
            ),
            vad_component=str(value.get("vad_component", "fsmn-vad")),
            diarization_component=str(
                value.get(
                    "diarization_component", "sortformer-overlap-diarization"
                )
            ),
            separation_component=str(
                value.get("separation_component", "mossformer2-ss-16k")
            ),
            asr_component=str(value.get("asr_component", "sensevoice-small")),
            declared_language=value.get("declared_language", value.get("language")),
            require_expressions=value.get("require_expressions", True),
        ).validated()

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self.validated())
        result["sources"] = list(result["sources"])
        result["source"] = result["sources"][0] if result["sources"] else None
        result["reference"] = result["reference_audio"]
        return result


@dataclass(frozen=True)
class SpeakerPurityEvidence:
    target_similarity: float
    winner_margin: float
    embedding_variance: float = 0.0
    estimated_snr_db: float | None = None
    clipped_ratio: float = 0.0
    vad_quality: float = 1.0
    overlap_evidence: bool = False
    speaker_change_evidence: bool = False
    diarization_uncertain_evidence: bool = False

    def validated(self) -> "SpeakerPurityEvidence":
        return SpeakerPurityEvidence(
            target_similarity=_finite_number(
                self.target_similarity, "target_similarity", -1.0, 1.0
            ),
            winner_margin=_finite_number(self.winner_margin, "winner_margin", 0.0, 2.0),
            embedding_variance=_finite_number(
                self.embedding_variance, "embedding_variance", 0.0, 2.0
            ),
            estimated_snr_db=(
                None
                if self.estimated_snr_db is None
                else _finite_number(self.estimated_snr_db, "estimated_snr_db", -100.0, 200.0)
            ),
            clipped_ratio=_finite_number(self.clipped_ratio, "clipped_ratio", 0.0, 1.0),
            vad_quality=_finite_number(self.vad_quality, "vad_quality", 0.0, 1.0),
            overlap_evidence=bool(self.overlap_evidence),
            speaker_change_evidence=bool(self.speaker_change_evidence),
            diarization_uncertain_evidence=bool(
                self.diarization_uncertain_evidence
            ),
        )


@dataclass(frozen=True)
class PurityRoute:
    route: str
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"route": self.route, "reasons": list(self.reasons)}


def route_speaker_purity(
    evidence: SpeakerPurityEvidence,
    *,
    target_threshold: float = 0.72,
    review_margin: float = 0.08,
    ambiguity_margin: float = 0.03,
    maximum_clean_variance: float = 0.035,
    minimum_clean_snr_db: float = 20.0,
) -> PurityRoute:
    """Route contamination to recovery instead of treating it as deletion evidence."""

    item = evidence.validated()
    threshold = _finite_number(target_threshold, "target_threshold", 0.0, 1.0)
    review = _finite_number(review_margin, "review_margin", 0.0, 0.5)
    ambiguity = _finite_number(ambiguity_margin, "ambiguity_margin", 0.0, 0.5)
    _finite_number(
        maximum_clean_variance, "maximum_clean_variance", 0.0, 1.0
    )
    clean_snr = _finite_number(
        minimum_clean_snr_db, "minimum_clean_snr_db", -100.0, 200.0
    )
    reasons: list[str] = []
    if item.target_similarity < threshold - review:
        return PurityRoute("reject", ("speaker-similarity-below-review-floor",))
    if item.clipped_ratio > 0.01:
        reasons.append("clipping")
    if item.vad_quality < 0.55:
        reasons.append("weak-vad-evidence")
    if item.winner_margin < ambiguity:
        reasons.append("speaker-ambiguity")
    # Short-window speaker embeddings vary materially with phonetic and emotional
    # content. Keep embedding_variance as diagnostic telemetry, but never route a
    # clip to separation from that value alone.
    if item.overlap_evidence:
        reasons.append("overlap-evidence")
    if item.speaker_change_evidence:
        reasons.append("multiple-speakers")
    if item.diarization_uncertain_evidence:
        reasons.append("diarization-needs-review")
    if item.estimated_snr_db is not None and item.estimated_snr_db < clean_snr:
        reasons.append("low-snr")
    if item.target_similarity < threshold:
        reasons.append("speaker-similarity-needs-review")
    if not reasons:
        return PurityRoute("clean", ())
    salvage_signals = {
        "speaker-ambiguity",
        "overlap-evidence",
        "multiple-speakers",
    }
    if (
        item.target_similarity >= threshold - review
        and salvage_signals.intersection(reasons)
    ):
        return PurityRoute("salvage", tuple(reasons))
    return PurityRoute("review", tuple(reasons))


def fuse_diarized_speaker_scores(
    speaker_scores: Mapping[str, float],
    *,
    target_threshold: float = 0.72,
    review_margin: float = 0.08,
    distinct_speaker_margin: float = 0.10,
) -> dict[str, Any]:
    if not isinstance(speaker_scores, Mapping) or len(speaker_scores) > 32:
        raise DatasetAcquisitionError("diarized speaker scores are malformed")
    threshold = _finite_number(target_threshold, "target_threshold", 0.0, 1.0)
    review = _finite_number(review_margin, "review_margin", 0.0, 0.5)
    distinct_margin = _finite_number(
        distinct_speaker_margin, "distinct_speaker_margin", 0.01, 1.0
    )
    scores: dict[str, float] = {}
    for speaker, value in speaker_scores.items():
        if (
            not isinstance(speaker, str)
            or not speaker
            or len(speaker) > 64
            or speaker in scores
        ):
            raise DatasetAcquisitionError("diarized speaker ID is malformed")
        scores[speaker] = _finite_number(
            value, "diarized_speaker_score", -1.0, 1.0
        )
    ordered = sorted(scores.values())
    minimum = ordered[0] if ordered else None
    maximum = ordered[-1] if ordered else None
    score_range = (
        float(maximum - minimum)
        if minimum is not None and maximum is not None
        else 0.0
    )
    multiple = len(scores) >= 2
    target_present = bool(
        maximum is not None and maximum >= threshold - review
    )
    distinct = bool(
        multiple
        and target_present
        and minimum is not None
        and minimum < threshold
        and score_range >= distinct_margin
    )
    uncertain = multiple and not distinct
    return {
        "schema": "aniflive-diarization-speaker-fusion-v1",
        "speaker_scores": dict(sorted(scores.items())),
        "speaker_count": len(scores),
        "minimum_score": minimum,
        "maximum_score": maximum,
        "score_range": score_range,
        "target_present": target_present,
        "distinct_speaker_evidence": distinct,
        "uncertain_multi_speaker": uncertain,
        "decision": (
            "distinct-speaker"
            if distinct
            else "review"
            if uncertain
            else "single-speaker"
        ),
    }


def stable_clip_id(
    source_sha256: str,
    start_sample: int,
    end_sample: int,
    position: int,
) -> str:
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise DatasetAcquisitionError("source_sha256 is malformed")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (start_sample, end_sample, position)
    ) or end_sample <= start_sample:
        raise DatasetAcquisitionError("clip sample bounds are malformed")
    digest = hashlib.sha256(
        f"{source_sha256}:{start_sample}:{end_sample}".encode("ascii")
    ).hexdigest()[:6]
    return f"seg_{position:06d}_{digest}"


EmbeddingFunction = Callable[[np.ndarray, int], np.ndarray]


def speaker_embedding_trajectory(
    audio: np.ndarray,
    sample_rate: int,
    *,
    embedder: EmbeddingFunction,
    reference_embedding: np.ndarray,
    window_seconds: float = 1.0,
    overlap_ratio: float = 0.5,
) -> dict[str, Any]:
    """Measure stationary speaker drift with one shared verifier.

    The result is evidence for routing only. It never rejects or separates audio by
    itself, and therefore stays valid for all V2ProPlus voices.
    """

    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise DatasetAcquisitionError("audio is empty or non-finite")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate < 1:
        raise DatasetAcquisitionError("sample_rate is malformed")
    window_seconds = _finite_number(window_seconds, "window_seconds", 0.25, 4.0)
    overlap_ratio = _finite_number(overlap_ratio, "overlap_ratio", 0.0, 0.9)
    window = max(1, round(window_seconds * sample_rate))
    step = max(1, round(window * (1.0 - overlap_ratio)))
    reference = np.asarray(reference_embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(reference))
    if reference.size == 0 or not np.isfinite(reference).all() or norm <= 1e-8:
        raise DatasetAcquisitionError("reference_embedding is malformed")
    reference /= norm
    starts = list(range(0, max(1, values.size - window + 1), step))
    last = max(0, values.size - window)
    if not starts or starts[-1] != last:
        starts.append(last)
    embeddings: list[np.ndarray] = []
    scores: list[float] = []
    for start in starts:
        part = values[start : min(values.size, start + window)]
        if part.size < window:
            part = np.pad(part, (0, window - part.size))
        embedding = np.asarray(embedder(part, sample_rate), dtype=np.float32).reshape(-1)
        embedding_norm = float(np.linalg.norm(embedding))
        if (
            embedding.shape != reference.shape
            or not np.isfinite(embedding).all()
            or embedding_norm <= 1e-8
        ):
            raise DatasetAcquisitionError("speaker embedder returned malformed output")
        embedding /= embedding_norm
        embeddings.append(embedding)
        scores.append(float(np.clip(np.dot(reference, embedding), -1.0, 1.0)))
    matrix = np.stack(embeddings)
    centroid = np.mean(matrix, axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
    centroid_distances = 1.0 - np.clip(matrix @ centroid, -1.0, 1.0)
    pairwise = matrix @ matrix.T
    off_diagonal = pairwise[np.triu_indices(pairwise.shape[0], k=1)]
    second_cluster_evidence = bool(
        off_diagonal.size and float(np.min(off_diagonal)) < 0.72
    )
    return {
        "schema": "aniflive-speaker-trajectory-v1",
        "window_seconds": window_seconds,
        "overlap_ratio": overlap_ratio,
        "windows": len(starts),
        "target_scores": scores,
        "minimum_target_score": min(scores),
        "maximum_target_score": max(scores),
        "target_score_range": max(scores) - min(scores),
        "centroid_drift": float(np.max(centroid_distances)),
        "embedding_variance": float(np.mean(centroid_distances)),
        "second_cluster_evidence": second_cluster_evidence,
    }


def stationary_overlap_probe(
    audio: np.ndarray,
    sample_rate: int,
    *,
    embedder: EmbeddingFunction,
    reference_embedding: np.ndarray,
    target_threshold: float = 0.72,
    window_seconds: Sequence[float] = (0.75, 1.0, 1.5),
) -> dict[str, Any]:
    """Fuse multiple speaker trajectories into conservative overlap evidence.

    This probe is a routing hint, not a deletion gate. A positive result sends the
    original clip to the qualified separator; a failed separator preserves the raw
    clip for human review.
    """

    threshold = _finite_number(target_threshold, "target_threshold", 0.0, 1.0)
    if (
        isinstance(window_seconds, (str, bytes))
        or not isinstance(window_seconds, Sequence)
        or not 1 <= len(window_seconds) <= 8
    ):
        raise DatasetAcquisitionError("window_seconds must be a bounded list")
    profiles: list[dict[str, Any]] = []
    votes: list[bool] = []
    for value in window_seconds:
        window = _finite_number(value, "window_seconds", 0.25, 4.0)
        profile = speaker_embedding_trajectory(
            audio,
            sample_rate,
            embedder=embedder,
            reference_embedding=reference_embedding,
            window_seconds=window,
            overlap_ratio=0.5,
        )
        drift = float(profile["centroid_drift"])
        score_range = float(profile["target_score_range"])
        minimum_score = float(profile["minimum_target_score"])
        maximum_score = float(profile["maximum_target_score"])
        vote = bool(
            profile["second_cluster_evidence"]
            and (drift >= 0.05 or score_range >= 0.08)
        ) or bool(
            minimum_score < threshold - 0.06
            and maximum_score >= threshold
            and score_range >= 0.08
        )
        profiles.append(profile)
        votes.append(vote)
    required_votes = 1 if len(votes) == 1 else 2
    return {
        "schema": "aniflive-stationary-overlap-probe-v1",
        "window_seconds": [float(value) for value in window_seconds],
        "profiles": profiles,
        "positive_votes": sum(votes),
        "required_votes": required_votes,
        "overlap_evidence": sum(votes) >= required_votes,
        "decision": "separator-probe" if sum(votes) >= required_votes else "no-probe",
    }


def post_separation_quality_gate(
    *,
    target_similarity: float,
    loser_similarity: float,
    similarity_before: float,
    asr_consistent: bool | None,
    finite_audio: bool,
    exact_sample_length: bool,
    artifact_score: float,
    target_threshold: float = 0.72,
    ambiguity_margin: float = 0.03,
) -> PurityRoute:
    target = _finite_number(target_similarity, "target_similarity", -1.0, 1.0)
    loser = _finite_number(loser_similarity, "loser_similarity", -1.0, 1.0)
    before = _finite_number(similarity_before, "similarity_before", -1.0, 1.0)
    artifact = _finite_number(artifact_score, "artifact_score", 0.0, 1.0)
    threshold = _finite_number(target_threshold, "target_threshold", 0.0, 1.0)
    margin = _finite_number(ambiguity_margin, "ambiguity_margin", 0.0, 0.5)
    reasons: list[str] = []
    if not finite_audio:
        reasons.append("non-finite-audio")
    if not exact_sample_length:
        reasons.append("sample-length-mismatch")
    if target < threshold:
        reasons.append("speaker-threshold")
    if target - loser < margin:
        reasons.append("speaker-ambiguity")
    if target + 0.01 < before:
        reasons.append("speaker-similarity-regression")
    if asr_consistent is False:
        reasons.append("asr-content-regression")
    if artifact > 0.15:
        reasons.append("separation-artifact-risk")
    if not reasons:
        return PurityRoute("clean", ())
    return PurityRoute("review", tuple(reasons))


__all__ = [
    "ACQUISITION_MODES",
    "DATASET_ACQUISITION_SCHEMA",
    "DATASET_WORKFLOW_STAGES",
    "PURITY_ROUTES",
    "DatasetAcquisitionConfig",
    "DatasetAcquisitionError",
    "PurityRoute",
    "SpeakerPurityEvidence",
    "post_separation_quality_gate",
    "fuse_diarized_speaker_scores",
    "route_speaker_purity",
    "speaker_embedding_trajectory",
    "stationary_overlap_probe",
    "stable_clip_id",
]
