"""Typed public value objects for the AnifLive-TTS SDK.

These records contain protocol values returned by the native client. Unknown JSON
fields are retained in each response record's read-only extra mapping.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, TypeVar, cast


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected a mapping")
    return MappingProxyType(dict(value))


def _freeze_extra(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze_mapping(value)


def _freeze_sequence(value: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
    return tuple(value)


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return cast(str, self.value)


class Language(_StringEnum):
    """Languages accepted by the native synthesis endpoint."""

    ZH = "zh"
    YUE = "yue"
    EN = "en"
    JA = "ja"
    KO = "ko"


class ContinuityPolicy(_StringEnum):
    """Stable server continuity-policy identifiers."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class ConditioningPolicy(_StringEnum):
    """Expression conditioning policies accepted by the native endpoint."""

    FULL_SWITCH = "full-switch"
    IDENTITY_LOCK = "identity-lock"
    SEMANTIC_STYLE = "semantic-style"
    ACOUSTIC_STYLE = "acoustic-style"
    SV_ONLY = "sv-only"


class RequiredFeature(_StringEnum):
    """Features that can be required by a service compatibility probe."""

    PCM_STREAMING = "pcm_streaming"
    EXPRESSIONS = "expressions"
    SPEECH_SESSIONS = "speech_sessions"


@dataclass(frozen=True, slots=True)
class Generation:
    """Optional generation controls accepted by the synthesis endpoint."""

    top_k: int | None = None
    top_p: float | None = None
    temperature: float | None = None
    seed: int | None = None
    noise_scale: float | None = None
    speed: float | None = None


@dataclass(frozen=True, slots=True)
class Expression:
    """Optional expression controls for a synthesis request."""

    enabled: bool | None = None
    profile: str | None = None
    intensity: float | None = None
    policy: ConditioningPolicy | str | None = None

    def __post_init__(self) -> None:
        if self.policy is not None:
            object.__setattr__(self, "policy", ConditioningPolicy(self.policy))


@dataclass(frozen=True, slots=True)
class ExpressionSegment:
    """A text segment with optional per-segment expression controls."""

    text: str = field(repr=False)
    expression: Expression | None = None


@dataclass(frozen=True, slots=True)
class SynthesisRequest:
    """Native structured request for complete WAV synthesis."""

    language: Language | str
    text: str | None = field(default=None, repr=False)
    segments: Sequence[ExpressionSegment] | None = field(default=None, repr=False)
    model: str | None = None
    voice_profile: str | None = None
    expression: Expression | None = None
    generation: Generation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", Language(self.language))
        if self.segments is not None:
            object.__setattr__(self, "segments", tuple(self.segments))


@dataclass(frozen=True, slots=True)
class SynthesisMetadata:
    """Parsed and allowlisted metadata from a complete synthesis response."""

    sample_rate: int | None = None
    model: str | None = None
    service_version: str | None = None
    backend: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", _freeze_mapping(self.headers))


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Complete WAV payload and response metadata."""

    audio: bytes = field(repr=False)
    content_type: str
    metadata: SynthesisMetadata

    @property
    def sample_rate(self) -> int | None:
        return self.metadata.sample_rate

    @property
    def model(self) -> str | None:
        return self.metadata.model

    @property
    def service_version(self) -> str | None:
        return self.metadata.service_version

    @property
    def backend(self) -> str | None:
        return self.metadata.backend


@dataclass(frozen=True, slots=True)
class Health:
    """Typed stable fields and optional runtime diagnostics from /health."""

    service: str
    version: str
    ready: bool
    status: str
    backend: str | None = None
    model: str | None = None
    voice: str | None = None
    cuda: object | None = field(default=None, repr=False)
    tensorrt: object | None = field(default=None, repr=False)
    gpu: object | None = field(default=None, repr=False)
    engine_fingerprint: str | None = None
    engine_count: int | None = None
    engines: Mapping[str, bool] | None = field(default=None, repr=False)
    reference: Mapping[str, Any] | None = field(default=None, repr=False)
    expression: Mapping[str, Any] | None = field(default=None, repr=False)
    request_segment_char_limit: int | None = None
    streaming: Mapping[str, Any] | None = field(default=None, repr=False)
    startup_warmup: object | None = field(default=None, repr=False)
    active_requests: int | None = None
    gpu_resource_lease: Mapping[str, Any] | None = field(default=None, repr=False)
    warm_retention: object | None = field(default=None, repr=False)
    switching: bool | None = None
    available_model_count: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "engines",
            "reference",
            "expression",
            "streaming",
            "gpu_resource_lease",
            "extra",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze_mapping(value))


@dataclass(frozen=True, slots=True)
class StreamingCapabilities:
    """Availability of the currently advertised audio response formats."""

    pcm16: bool
    wav: bool
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class ExpressionCapabilities:
    """Expression features advertised by the service."""

    native: bool
    controlled_profiles: bool
    continuous_vector: bool = False
    segmented: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class ContinuityPolicyCapability:
    """One policy entry from the capabilities response."""

    name: str | None = None
    phones_bert: bool | None = None
    semantic_tokens: int | None = None
    expression_aware: bool | None = None
    boundary_adaptive: bool | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class SessionLimits:
    """Server-reported speech-session limits; no client limits are inferred."""

    maximum_open: int
    ttl_seconds: float
    maximum_segments: int
    maximum_characters: int
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class SessionCapabilities:
    """Session feature metadata advertised by the service."""

    committed_segments: bool
    pcm_stream: bool
    cancel: bool = False
    flush: bool = False
    context_mode: str | None = None
    neural_state_continuity: bool | None = None
    acoustic_latent_continuity: bool | None = None
    default_policy: str | None = None
    policies: Mapping[str, ContinuityPolicyCapability] = field(default_factory=dict, repr=False)
    qualification: str | None = None
    limits: SessionLimits | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policies", _freeze_mapping(self.policies))
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class Capabilities:
    """Typed capability fields and unknown top-level response fields."""

    service: str
    version: str
    languages: tuple[str, ...]
    streaming: StreamingCapabilities
    expression: ExpressionCapabilities
    model_family: str | None = None
    backend: str | None = None
    precision: str | None = None
    pytorch_fallback: bool | None = None
    adaptive_punctuation_segments: bool | None = None
    warm_retention_seconds: float | None = None
    speech_sessions: SessionCapabilities | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "languages", _freeze_sequence(self.languages))
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


class SpeechSegmentState(_StringEnum):
    """Known lifecycle states; unknown future state strings remain preserved."""

    QUEUED = "queued"
    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ListResponse(Generic[T]):
    """A typed server list envelope with forward-compatible extra fields."""

    object: str
    data: tuple[T, ...]
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", _freeze_sequence(self.data))
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    object: str
    active: bool
    voice_profiles: tuple[str, ...]
    owned_by: str | None = None
    description: str | None = None
    model_family: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "voice_profiles", _freeze_sequence(self.voice_profiles))
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    id: str
    reference_language: str
    reference_configured: bool
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class ExpressionProfileInfo:
    id: str
    intensity_levels: tuple[float, ...]
    languages: tuple[str, ...]
    reference_languages: tuple[str, ...]
    preferred_policies: tuple[str, ...]
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "intensity_levels",
            "languages",
            "reference_languages",
            "preferred_policies",
        ):
            object.__setattr__(self, name, _freeze_sequence(getattr(self, name)))
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class ExpressionCatalogInfo:
    object: str
    enabled: bool
    default: str | None
    profiles: tuple[ExpressionProfileInfo, ...]
    policies: tuple[str, ...]
    preferred_policy: str | None = None
    playback_policy: Mapping[str, Any] | None = field(default=None, repr=False)
    runtime_policy: Mapping[str, Any] | None = field(default=None, repr=False)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "profiles", _freeze_sequence(self.profiles))
        object.__setattr__(self, "policies", _freeze_sequence(self.policies))
        for name in ("playback_policy", "runtime_policy", "extra"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _freeze_mapping(value))


@dataclass(frozen=True, slots=True)
class ActivationResult:
    code: int
    message: str = field(repr=False)
    changed: bool
    model: str
    voice: str
    backend: str | None = None
    pytorch_fallback: bool | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class CancellationResult:
    cancelled: bool
    model: str
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class StreamMetadata:
    """Validated audio format and safe response metadata for a PCM stream."""

    sample_rate: int
    channels: int
    sample_format: str
    metadata: SynthesisMetadata


@dataclass(frozen=True, slots=True)
class SessionCreate:
    """Optional configuration for a remote speech session."""

    model: str | None = None
    voice_profile: str | None = None
    continuity_policy: ContinuityPolicy | str | None = None

    def __post_init__(self) -> None:
        if self.continuity_policy is not None:
            object.__setattr__(
                self,
                "continuity_policy",
                ContinuityPolicy(self.continuity_policy),
            )


@dataclass(frozen=True, slots=True)
class SegmentRequest:
    """One committed segment in a speech session."""

    segment_id: str = field(repr=False)
    text: str = field(repr=False)
    language: Language | str
    paragraph_id: str | None = None
    pause_after_ms: int | None = None
    expression: Expression | None = None
    generation: Generation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "language", Language(self.language))


@dataclass(frozen=True, slots=True)
class SegmentInfo:
    """Server state for a committed session segment."""

    segment_id: str = field(repr=False)
    sequence: int
    characters: int
    language: str
    paragraph_id: str | None
    pause_after_ms: int
    state: SpeechSegmentState | str
    created_at: str
    updated_at: str
    error: str | None = field(default=None, repr=False)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.state, str):
            try:
                object.__setattr__(self, "state", SpeechSegmentState(self.state))
            except ValueError:
                pass
        else:
            raise TypeError("state must be a string or SpeechSegmentState")
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Server-reported continuity state; values remain opaque to the client."""

    mode: str
    policy: str
    neural_state_continuity: bool
    acoustic_latent_continuity: bool
    committed_segments: int
    previous_language: str | None = None
    previous_expression: str | None = None
    paragraph_id: str | None = None
    previous_pause_ms: int | None = None
    previous_output_samples: int | None = None
    total_output_samples: int | None = None
    hard_boundary: bool | None = None
    retained_phonemes: int | None = None
    retained_semantic_tokens: int | None = None
    retained_neural_bytes: int | None = None
    last_plan: Mapping[str, Any] | None = field(default=None, repr=False)
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.last_plan is not None:
            object.__setattr__(self, "last_plan", _freeze_mapping(self.last_plan))
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Immutable snapshot returned by a session create, fetch, or mutation."""

    id: str = field(repr=False)
    object: str
    state: str
    model: str
    voice_profile: str
    sample_rate: int
    format: str
    continuity_policy: str
    context: SessionContext
    audio_consumer_attached: bool
    created_at: str
    updated_at: str
    segments: tuple[SegmentInfo, ...] = field(repr=False)
    active_audio_cancelled: bool | None = None
    extra: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", _freeze_sequence(self.segments))
        object.__setattr__(self, "extra", _freeze_extra(self.extra))


# Public ergonomic names are aliases, so they share the same validated schema.
SpeechSegment = SegmentRequest
PCMStreamMetadata = StreamMetadata
SpeechSessionContext = SessionContext
SpeechSegmentInfo = SegmentInfo


@dataclass(frozen=True, slots=True)
class ServiceInfo:
    """Combined service identity and capability result from a compatibility probe."""

    health: Health
    capabilities: Capabilities


__all__ = [
    "ActivationResult",
    "CancellationResult",
    "Capabilities",
    "ConditioningPolicy",
    "ContinuityPolicy",
    "ContinuityPolicyCapability",
    "Expression",
    "ExpressionCapabilities",
    "ExpressionCatalogInfo",
    "ExpressionProfileInfo",
    "ExpressionSegment",
    "Generation",
    "Health",
    "Language",
    "ListResponse",
    "ModelInfo",
    "PCMStreamMetadata",
    "RequiredFeature",
    "SegmentInfo",
    "SegmentRequest",
    "ServiceInfo",
    "SpeechSegment",
    "SpeechSegmentInfo",
    "SpeechSegmentState",
    "SessionContext",
    "SpeechSessionContext",
    "SessionCreate",
    "SessionInfo",
    "SessionCapabilities",
    "SessionLimits",
    "SynthesisMetadata",
    "SynthesisRequest",
    "StreamMetadata",
    "SynthesisResult",
    "StreamingCapabilities",
    "VoiceInfo",
]
