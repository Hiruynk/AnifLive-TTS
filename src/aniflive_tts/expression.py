from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import PackageValidationError
from .expression_transition import TransitionCurve
from .model_package import resolve_contained_path, validate_safe_identifier


CANONICAL_EXPRESSION_LANGUAGES = frozenset({"zh", "yue", "en", "ja", "ko"})
DEFAULT_SHORT_PREBUFFER_MS = 32
DEFAULT_LONG_PREBUFFER_MS = 64
MAX_PREBUFFER_MS = 500


class ConditioningPolicy(str, Enum):
    FULL_SWITCH = "full-switch"
    IDENTITY_LOCK = "identity-lock"
    SEMANTIC_STYLE = "semantic-style"
    ACOUSTIC_STYLE = "acoustic-style"
    SV_ONLY = "sv-only"


class BoundaryKind(str, Enum):
    HARD_NATURAL = "hard-natural"
    SOFT_NATURAL = "soft-natural"
    TECHNICAL = "technical"
    EXPLICIT_EXPRESSION = "explicit-expression"


@dataclass(frozen=True)
class ExpressionSegment:
    """One public text segment and its symbolic expression request."""

    text: str
    enabled: bool = False
    profile: str | None = None
    intensity: float = 0.5
    policy: ConditioningPolicy = ConditioningPolicy.FULL_SWITCH


_BOUNDARY_CLOSERS = "\"'\u201d\u2019\u300d\u300f\u300b\u3009\u3011\u3015\u3009"
_HARD_NATURAL_ENDINGS = (".", "\u3002", "!", "\uff01", "?", "\uff1f")
_SOFT_NATURAL_ENDINGS = (",", "\uff0c", "\u3001", ";", "\uff1b", ":", "\uff1a")
_SAFE_EXPRESSION_SWITCH_ENDINGS = frozenset(",.;?!\u3001\uff0c\u3002\uff1f\uff01\uff1b")


def classify_boundary(segment: str, *, expression_switch: bool = False) -> BoundaryKind:
    """Classify a trailing boundary without interpreting arbitrary punctuation as a cut."""

    if expression_switch:
        return BoundaryKind.EXPLICIT_EXPRESSION
    if segment.endswith(("\n\n", "\r\n\r\n")):
        return BoundaryKind.HARD_NATURAL
    ending = segment.rstrip()
    while ending and ending[-1] in _BOUNDARY_CLOSERS:
        ending = ending[:-1].rstrip()
    if ending.endswith(_HARD_NATURAL_ENDINGS):
        return BoundaryKind.HARD_NATURAL
    if ending.endswith(_SOFT_NATURAL_ENDINGS):
        return BoundaryKind.SOFT_NATURAL
    return BoundaryKind.TECHNICAL


def should_bridge_expression_context(
    segment: str, *, expression_switch: bool
) -> bool:
    """Keep linguistic history when a style change splits one utterance."""

    return bool(
        expression_switch
        and classify_boundary(segment, expression_switch=False)
        is BoundaryKind.TECHNICAL
    )


def has_safe_expression_boundary(segment: str) -> bool:
    """Return whether a style switch may safely follow this complete clause."""

    if segment.endswith(("\n\n", "\r\n\r\n")):
        return True
    ending = segment.rstrip()
    while ending and ending[-1] in _BOUNDARY_CLOSERS:
        ending = ending[:-1].rstrip()
    return bool(ending) and ending[-1] in _SAFE_EXPRESSION_SWITCH_ENDINGS


@dataclass(frozen=True)
class ReferenceDescriptor:
    id: str
    emotion: str
    intensity: float
    reference_audio: Path
    reference_text: str
    reference_language: str
    manual_verified: bool
    vad: tuple[float, float, float] | None = None
    preferred_policy: ConditioningPolicy | None = None

    def public_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "emotion": self.emotion,
            "intensity": self.intensity,
            "reference_language": self.reference_language,
            "preferred_policy": (
                self.preferred_policy.value if self.preferred_policy is not None else None
            ),
        }


@dataclass(frozen=True)
class PlaybackPolicy:
    short_prebuffer_ms: int = DEFAULT_SHORT_PREBUFFER_MS
    long_prebuffer_ms: int = DEFAULT_LONG_PREBUFFER_MS

    def public_metadata(self) -> dict[str, int]:
        return {
            "short_prebuffer_ms": self.short_prebuffer_ms,
            "long_prebuffer_ms": self.long_prebuffer_ms,
        }


@dataclass(frozen=True)
class ExpressionRuntimePolicy:
    """Package-calibrated expression streaming settings.

    These values are deliberately package data rather than model-name branches
    or developer-shell defaults.  Packages without expression metadata retain
    the v1.2 runtime path.
    """

    preview_publish_seconds: float = 0.112
    minimum_initial_preview_seconds: float = 0.112
    progressive_refill_tokens: int = 8
    progressive_refill_seconds: float = 0.24
    progressive_refill_count: int = 1
    progressive_refill_min_segments: int = 2
    progressive_refill_min_phonemes: int = 80
    progressive_refill_short_tokens: int = 16
    progressive_refill_short_seconds: float = 0.112
    transition: TransitionCurve = TransitionCurve.HARD_NATURAL
    transition_ms: float = 4.0
    sigmoid_k: float = 12.0
    first_context_tokens: int = 17
    onset_hold_ms: int = 0
    normalize_reference_semantic_onset: bool = False
    reference_semantic_onset_threshold_dbfs: float = -45.0
    reference_semantic_onset_retain_ms: int = 40

    def public_metadata(self) -> dict[str, Any]:
        return {
            "preview_publish_seconds": self.preview_publish_seconds,
            "minimum_initial_preview_seconds": self.minimum_initial_preview_seconds,
            "progressive_refill_tokens": self.progressive_refill_tokens,
            "progressive_refill_seconds": self.progressive_refill_seconds,
            "progressive_refill_count": self.progressive_refill_count,
            "progressive_refill_min_segments": self.progressive_refill_min_segments,
            "progressive_refill_min_phonemes": self.progressive_refill_min_phonemes,
            "progressive_refill_short_tokens": self.progressive_refill_short_tokens,
            "progressive_refill_short_seconds": self.progressive_refill_short_seconds,
            "transition": self.transition.value,
            "transition_ms": self.transition_ms,
            "sigmoid_k": self.sigmoid_k,
            "first_context_tokens": self.first_context_tokens,
            "onset_hold_ms": self.onset_hold_ms,
            "normalize_reference_semantic_onset": (
                self.normalize_reference_semantic_onset
            ),
            "reference_semantic_onset_threshold_dbfs": (
                self.reference_semantic_onset_threshold_dbfs
            ),
            "reference_semantic_onset_retain_ms": (
                self.reference_semantic_onset_retain_ms
            ),
        }


@dataclass(frozen=True)
class ExpressionCatalog:
    default_profile: str
    profiles: tuple[ReferenceDescriptor, ...]
    preferred_policy: ConditioningPolicy = ConditioningPolicy.SEMANTIC_STYLE
    output_gain: float = 1.0
    playback_policy: PlaybackPolicy = PlaybackPolicy()
    runtime_policy: ExpressionRuntimePolicy = ExpressionRuntimePolicy()

    def select(self, *, profile: str, intensity: float, language: str) -> ReferenceDescriptor:
        requested = validate_safe_identifier(profile, "expression.profile")
        candidates = tuple(
            item for item in self.profiles if item.emotion == requested or item.id == requested
        )
        if not candidates:
            raise KeyError(requested)
        same_language = tuple(
            item for item in candidates if item.reference_language == language
        )
        if same_language:
            candidates = same_language
        return min(candidates, key=lambda item: (abs(item.intensity - intensity), item.id))

    def policy_for(
        self, *, profile: str, intensity: float, language: str
    ) -> ConditioningPolicy:
        """Resolve a converter-calibrated policy without model-name branches."""

        selected = self.select(profile=profile, intensity=intensity, language=language)
        return selected.preferred_policy or self.preferred_policy

    def public_metadata(self) -> dict[str, Any]:
        emotions: dict[str, dict[str, Any]] = {}
        for item in self.profiles:
            entry = emotions.setdefault(
                item.emotion,
                {"id": item.emotion, "intensity_levels": set(), "languages": set()},
            )
            entry["intensity_levels"].add(item.intensity)
            entry["languages"].add(item.reference_language)
            entry.setdefault("preferred_policies", set()).add(
                (item.preferred_policy or self.preferred_policy).value
            )
        return {
            "default": self.default_profile,
            "preferred_policy": self.preferred_policy.value,
            "playback_policy": self.playback_policy.public_metadata(),
            "runtime_policy": self.runtime_policy.public_metadata(),
            "profiles": [
                {
                    "id": entry["id"],
                    "intensity_levels": sorted(entry["intensity_levels"]),
                    "languages": sorted(CANONICAL_EXPRESSION_LANGUAGES),
                    "reference_languages": sorted(entry["languages"]),
                    "preferred_policies": sorted(entry["preferred_policies"]),
                }
                for _, entry in sorted(emotions.items())
            ],
        }


@dataclass(frozen=True)
class PreparedReference:
    prompt_semantic: Any
    spectrogram: Any
    speaker_embedding: Any
    phones: list[int]
    bert: Any
    id: str = "default"
    semantic_onset_removed_ms: float = 0.0


@dataclass(frozen=True)
class ConditioningBundle:
    semantic: PreparedReference
    spectrogram: Any
    speaker_embedding: Any
    expression_id: str
    policy: ConditioningPolicy

    @classmethod
    def full_switch(cls, reference: PreparedReference) -> "ConditioningBundle":
        return cls(
            semantic=reference,
            spectrogram=reference.spectrogram,
            speaker_embedding=reference.speaker_embedding,
            expression_id=reference.id,
            policy=ConditioningPolicy.FULL_SWITCH,
        )


class PreparedReferenceBank:
    def __init__(self, *, identity_id: str = "default") -> None:
        self.identity_id = validate_safe_identifier(identity_id, "identity reference id")
        self._references: dict[str, PreparedReference] = {}

    def add(self, reference: PreparedReference) -> None:
        reference_id = validate_safe_identifier(reference.id, "prepared reference id")
        if reference_id in self._references:
            raise ValueError(f"Prepared reference already exists: {reference_id}")
        self._references[reference_id] = reference

    def replace(self, reference: PreparedReference) -> None:
        reference_id = validate_safe_identifier(reference.id, "prepared reference id")
        self._references[reference_id] = reference

    def alias(self, *, alias_id: str, reference_id: str) -> PreparedReference:
        alias_id = validate_safe_identifier(alias_id, "prepared reference alias")
        source = self.get(reference_id)
        alias = PreparedReference(
            prompt_semantic=source.prompt_semantic,
            spectrogram=source.spectrogram,
            speaker_embedding=source.speaker_embedding,
            phones=source.phones,
            bert=source.bert,
            id=alias_id,
            semantic_onset_removed_ms=source.semantic_onset_removed_ms,
        )
        self.add(alias)
        return alias

    def get(self, reference_id: str) -> PreparedReference:
        try:
            return self._references[reference_id]
        except KeyError as error:
            raise KeyError(f"Unknown prepared reference: {reference_id}") from error

    @property
    def identity(self) -> PreparedReference:
        return self.get(self.identity_id)

    @property
    def references(self) -> Mapping[str, PreparedReference]:
        return MappingProxyType(self._references)

    def conditioning(
        self, *, reference_id: str, policy: ConditioningPolicy
    ) -> ConditioningBundle:
        expression = self.get(reference_id)
        identity = self.identity
        if policy is ConditioningPolicy.FULL_SWITCH:
            return ConditioningBundle.full_switch(expression)
        if policy is ConditioningPolicy.IDENTITY_LOCK:
            return ConditioningBundle(
                semantic=expression,
                spectrogram=expression.spectrogram,
                speaker_embedding=identity.speaker_embedding,
                expression_id=expression.id,
                policy=policy,
            )
        if policy is ConditioningPolicy.SEMANTIC_STYLE:
            return ConditioningBundle(
                semantic=expression,
                spectrogram=identity.spectrogram,
                speaker_embedding=identity.speaker_embedding,
                expression_id=expression.id,
                policy=policy,
            )
        if policy is ConditioningPolicy.ACOUSTIC_STYLE:
            return ConditioningBundle(
                semantic=identity,
                spectrogram=expression.spectrogram,
                speaker_embedding=identity.speaker_embedding,
                expression_id=expression.id,
                policy=policy,
            )
        if policy is ConditioningPolicy.SV_ONLY:
            return ConditioningBundle(
                semantic=identity,
                spectrogram=identity.spectrogram,
                speaker_embedding=expression.speaker_embedding,
                expression_id=expression.id,
                policy=policy,
            )
        raise ValueError(f"Unsupported conditioning policy: {policy}")


def _finite_unit_interval(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackageValidationError(f"{label} must be a number from 0 to 1")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise PackageValidationError(f"{label} must be a number from 0 to 1")
    return result


def _parse_conditioning_policy(value: Any, label: str) -> ConditioningPolicy:
    try:
        return ConditioningPolicy(value)
    except (TypeError, ValueError) as error:
        raise PackageValidationError(f"{label} is unsupported") from error


def _finite_output_gain(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackageValidationError(f"{label} must be a finite number from 0.1 to 8")
    result = float(value)
    if not math.isfinite(result) or not 0.1 <= result <= 8.0:
        raise PackageValidationError(f"{label} must be a finite number from 0.1 to 8")
    return result


def _prebuffer_ms(value: Any, label: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise PackageValidationError(
            f"{label} must be an integer from 0 to {MAX_PREBUFFER_MS}"
        )
    if not 0 <= value <= MAX_PREBUFFER_MS:
        raise PackageValidationError(
            f"{label} must be an integer from 0 to {MAX_PREBUFFER_MS}"
        )
    return value


def parse_playback_policy(value: Any) -> PlaybackPolicy:
    if value is None:
        return PlaybackPolicy()
    if not isinstance(value, Mapping):
        raise PackageValidationError("expression.playback_policy must be an object")
    short_ms = _prebuffer_ms(
        value.get("short_prebuffer_ms"),
        "expression.playback_policy.short_prebuffer_ms",
        DEFAULT_SHORT_PREBUFFER_MS,
    )
    long_ms = _prebuffer_ms(
        value.get("long_prebuffer_ms"),
        "expression.playback_policy.long_prebuffer_ms",
        DEFAULT_LONG_PREBUFFER_MS,
    )
    if long_ms < short_ms:
        raise PackageValidationError(
            "expression.playback_policy.long_prebuffer_ms must not be less than "
            "short_prebuffer_ms"
        )
    return PlaybackPolicy(
        short_prebuffer_ms=short_ms,
        long_prebuffer_ms=long_ms,
    )


def _bounded_float(
    value: Any, label: str, default: float, *, minimum: float, maximum: float
) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackageValidationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise PackageValidationError(
            f"{label} must be between {minimum} and {maximum}"
        )
    return result


def _bounded_int(
    value: Any, label: str, default: int, *, minimum: int, maximum: int
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise PackageValidationError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise PackageValidationError(
            f"{label} must be between {minimum} and {maximum}"
        )
    return value


def _boolean(value: Any, label: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise PackageValidationError(f"{label} must be a boolean")
    return value


def parse_expression_runtime_policy(value: Any) -> ExpressionRuntimePolicy:
    if value is None:
        return ExpressionRuntimePolicy()
    if not isinstance(value, Mapping):
        raise PackageValidationError("expression.runtime_policy must be an object")
    defaults = ExpressionRuntimePolicy()
    try:
        transition = TransitionCurve(value.get("transition", defaults.transition.value))
    except (TypeError, ValueError) as error:
        raise PackageValidationError(
            "expression.runtime_policy.transition is unsupported"
        ) from error
    return ExpressionRuntimePolicy(
        preview_publish_seconds=_bounded_float(
            value.get("preview_publish_seconds"),
            "expression.runtime_policy.preview_publish_seconds",
            defaults.preview_publish_seconds,
            minimum=0.0,
            maximum=1.0,
        ),
        minimum_initial_preview_seconds=_bounded_float(
            value.get("minimum_initial_preview_seconds"),
            "expression.runtime_policy.minimum_initial_preview_seconds",
            defaults.minimum_initial_preview_seconds,
            minimum=0.0,
            maximum=1.0,
        ),
        progressive_refill_tokens=_bounded_int(
            value.get("progressive_refill_tokens"),
            "expression.runtime_policy.progressive_refill_tokens",
            defaults.progressive_refill_tokens,
            minimum=0,
            maximum=64,
        ),
        progressive_refill_seconds=_bounded_float(
            value.get("progressive_refill_seconds"),
            "expression.runtime_policy.progressive_refill_seconds",
            defaults.progressive_refill_seconds,
            minimum=0.0,
            maximum=1.0,
        ),
        progressive_refill_count=_bounded_int(
            value.get("progressive_refill_count"),
            "expression.runtime_policy.progressive_refill_count",
            defaults.progressive_refill_count,
            minimum=1,
            maximum=4,
        ),
        progressive_refill_min_segments=_bounded_int(
            value.get("progressive_refill_min_segments"),
            "expression.runtime_policy.progressive_refill_min_segments",
            defaults.progressive_refill_min_segments,
            minimum=1,
            maximum=64,
        ),
        progressive_refill_min_phonemes=_bounded_int(
            value.get("progressive_refill_min_phonemes"),
            "expression.runtime_policy.progressive_refill_min_phonemes",
            defaults.progressive_refill_min_phonemes,
            minimum=1,
            maximum=4096,
        ),
        progressive_refill_short_tokens=_bounded_int(
            value.get("progressive_refill_short_tokens"),
            "expression.runtime_policy.progressive_refill_short_tokens",
            defaults.progressive_refill_short_tokens,
            minimum=0,
            maximum=64,
        ),
        progressive_refill_short_seconds=_bounded_float(
            value.get("progressive_refill_short_seconds"),
            "expression.runtime_policy.progressive_refill_short_seconds",
            defaults.progressive_refill_short_seconds,
            minimum=0.0,
            maximum=1.0,
        ),
        transition=transition,
        transition_ms=_bounded_float(
            value.get("transition_ms"),
            "expression.runtime_policy.transition_ms",
            defaults.transition_ms,
            minimum=0.0,
            maximum=20.0,
        ),
        sigmoid_k=_bounded_float(
            value.get("sigmoid_k"),
            "expression.runtime_policy.sigmoid_k",
            defaults.sigmoid_k,
            minimum=1.0,
            maximum=40.0,
        ),
        first_context_tokens=_bounded_int(
            value.get("first_context_tokens"),
            "expression.runtime_policy.first_context_tokens",
            defaults.first_context_tokens,
            minimum=1,
            maximum=64,
        ),
        onset_hold_ms=_bounded_int(
            value.get("onset_hold_ms"),
            "expression.runtime_policy.onset_hold_ms",
            defaults.onset_hold_ms,
            minimum=0,
            maximum=500,
        ),
        normalize_reference_semantic_onset=_boolean(
            value.get("normalize_reference_semantic_onset"),
            "expression.runtime_policy.normalize_reference_semantic_onset",
            defaults.normalize_reference_semantic_onset,
        ),
        reference_semantic_onset_threshold_dbfs=_bounded_float(
            value.get("reference_semantic_onset_threshold_dbfs"),
            "expression.runtime_policy.reference_semantic_onset_threshold_dbfs",
            defaults.reference_semantic_onset_threshold_dbfs,
            minimum=-80.0,
            maximum=-20.0,
        ),
        reference_semantic_onset_retain_ms=_bounded_int(
            value.get("reference_semantic_onset_retain_ms"),
            "expression.runtime_policy.reference_semantic_onset_retain_ms",
            defaults.reference_semantic_onset_retain_ms,
            minimum=0,
            maximum=250,
        ),
    )


def _parse_vad(value: Any, label: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PackageValidationError(f"{label} must be an object")
    return (
        _finite_unit_interval(value.get("valence"), f"{label}.valence"),
        _finite_unit_interval(value.get("arousal"), f"{label}.arousal"),
        _finite_unit_interval(value.get("dominance"), f"{label}.dominance"),
    )


def load_expression_catalog(
    *, profile_dir: Path, profile_manifest: Mapping[str, Any]
) -> ExpressionCatalog | None:
    raw = profile_manifest.get("expression")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or raw.get("schema") != 1:
        raise PackageValidationError("voice expression metadata must use schema 1")
    default_profile = validate_safe_identifier(
        raw.get("default_profile"), "expression.default_profile"
    )
    try:
        preferred_policy = ConditioningPolicy(
            raw.get("preferred_policy", ConditioningPolicy.SEMANTIC_STYLE.value)
        )
    except (TypeError, ValueError) as error:
        raise PackageValidationError("expression.preferred_policy is unsupported") from error
    output_gain = _finite_output_gain(
        raw.get("output_gain", 1.0), "expression.output_gain"
    )
    playback_policy = parse_playback_policy(raw.get("playback_policy"))
    runtime_policy = parse_expression_runtime_policy(raw.get("runtime_policy"))
    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise PackageValidationError("expression.profiles must be a non-empty array")

    profiles: list[ReferenceDescriptor] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_profiles):
        label = f"expression.profiles[{index}]"
        if not isinstance(item, Mapping):
            raise PackageValidationError(f"{label} must be an object")
        reference_id = validate_safe_identifier(item.get("id"), f"{label}.id")
        emotion = validate_safe_identifier(item.get("emotion"), f"{label}.emotion")
        if reference_id in seen:
            raise PackageValidationError(f"duplicate expression profile id: {reference_id}")
        seen.add(reference_id)
        language = item.get("reference_language")
        if language not in CANONICAL_EXPRESSION_LANGUAGES:
            raise PackageValidationError(f"{label}.reference_language is unsupported")
        text = item.get("reference_text")
        if not isinstance(text, str) or not text.strip():
            raise PackageValidationError(f"{label}.reference_text must not be empty")
        if item.get("manual_verified") is not True:
            raise PackageValidationError(f"{label}.manual_verified must be true")
        audio = resolve_contained_path(
            profile_dir, item.get("reference_audio"), f"{label}.reference_audio"
        )
        if not audio.is_file():
            raise PackageValidationError(f"missing expression reference audio: {reference_id}")
        profiles.append(
            ReferenceDescriptor(
                id=reference_id,
                emotion=emotion,
                intensity=_finite_unit_interval(item.get("intensity"), f"{label}.intensity"),
                reference_audio=audio,
                reference_text=text.strip(),
                reference_language=str(language),
                manual_verified=True,
                vad=_parse_vad(item.get("vad"), f"{label}.vad"),
                preferred_policy=(
                    None
                    if item.get("preferred_policy") is None
                    else _parse_conditioning_policy(
                        item.get("preferred_policy"), f"{label}.preferred_policy"
                    )
                ),
            )
        )
    if default_profile not in {item.emotion for item in profiles} | seen:
        raise PackageValidationError("expression.default_profile is unavailable")
    return ExpressionCatalog(
        default_profile=default_profile,
        profiles=tuple(profiles),
        preferred_policy=preferred_policy,
        output_gain=output_gain,
        playback_policy=playback_policy,
        runtime_policy=runtime_policy,
    )
