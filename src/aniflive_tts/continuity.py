from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
from types import MappingProxyType
from typing import Any, Mapping

from .expression import BoundaryKind, classify_boundary


MAX_CONTEXT_TEXT_CHARS = 512
MAX_CONTEXT_PHONEMES = 75
MAX_CONTEXT_SEMANTIC_TOKENS = 64
EXPRESSION_SWITCH_PHONEMES = 12
EXPRESSION_SWITCH_SEMANTIC_TOKENS = 48


class ContinuityPolicy(str, Enum):
    """Cross-request context policy for committed Speech Session segments."""

    NONE = "A"
    TEXT_BERT = "B"
    SEMANTIC_32 = "C"
    SEMANTIC_64 = "D"
    EXPRESSION_AWARE = "E"
    BOUNDARY_ADAPTIVE = "F"


CONTINUITY_POLICY_CONTRACTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "A": MappingProxyType({"name": "none", "phones_bert": False, "semantic_tokens": 0, "expression_aware": False, "boundary_adaptive": False}),
        "B": MappingProxyType({"name": "text-bert", "phones_bert": True, "semantic_tokens": 0, "expression_aware": False, "boundary_adaptive": False}),
        "C": MappingProxyType({"name": "text-bert-semantic-32", "phones_bert": True, "semantic_tokens": 32, "expression_aware": False, "boundary_adaptive": False}),
        "D": MappingProxyType({"name": "text-bert-semantic-64", "phones_bert": True, "semantic_tokens": 64, "expression_aware": False, "boundary_adaptive": False}),
        "E": MappingProxyType({"name": "expression-aware", "phones_bert": True, "semantic_tokens": 64, "expression_aware": True, "boundary_adaptive": False}),
        "F": MappingProxyType({"name": "boundary-adaptive", "phones_bert": True, "semantic_tokens": 64, "expression_aware": True, "boundary_adaptive": True}),
    }
)


def parse_continuity_policy(value: Any) -> ContinuityPolicy:
    if isinstance(value, ContinuityPolicy):
        return value
    if not isinstance(value, str):
        raise ValueError("continuity_policy must be one of A, B, C, D, E or F")
    try:
        return ContinuityPolicy(value.strip().upper())
    except ValueError as error:
        raise ValueError("continuity_policy must be one of A, B, C, D, E or F") from error


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError("continuation tensor does not expose a shape")
    return tuple(int(item) for item in shape)


def _tensor_bytes(value: Any | None) -> int:
    if value is None:
        return 0
    try:
        return int(value.nelement()) * int(value.element_size())
    except (AttributeError, TypeError, ValueError):
        return 0


def _clone_tail(value: Any, count: int) -> Any:
    detached = value.detach()
    result = detached[..., -count:] if count > 0 else detached[..., :0]
    return result.clone()


@dataclass(frozen=True)
class NeuralContinuationState:
    """Bounded, committed exact neural history retained on the current GPU.

    It contains phones, BERT features and accepted Transformer semantic tokens.
    It deliberately does not contain an acoustic latent.
    """

    model_id: str
    voice_profile: str
    language: str
    expression: str | None
    phones: tuple[int, ...]
    bert: Any
    semantic: Any

    @classmethod
    def capture(
        cls,
        *,
        model_id: str,
        voice_profile: str,
        language: str,
        expression: str | None,
        phones: Any,
        bert: Any,
        semantic: Any,
    ) -> NeuralContinuationState:
        resolved_phones = tuple(int(item) for item in phones)[-MAX_CONTEXT_PHONEMES:]
        if not resolved_phones:
            raise ValueError("continuation phones must not be empty")
        bert_shape = _shape(bert)
        semantic_shape = _shape(semantic)
        if len(bert_shape) != 2 or bert_shape[-1] < len(resolved_phones):
            raise ValueError("continuation BERT tensor must be rank 2 and cover all phones")
        if len(semantic_shape) != 2 or semantic_shape[0] != 1 or semantic_shape[-1] < 1:
            raise ValueError("continuation semantic tensor must have shape [1, tokens]")
        semantic_count = min(MAX_CONTEXT_SEMANTIC_TOKENS, semantic_shape[-1])
        return cls(
            model_id=str(model_id),
            voice_profile=str(voice_profile),
            language=str(language),
            expression=expression,
            phones=resolved_phones,
            bert=_clone_tail(bert, len(resolved_phones)),
            semantic=_clone_tail(semantic, semantic_count),
        )

    @property
    def semantic_tokens(self) -> int:
        return _shape(self.semantic)[-1]

    @property
    def retained_bytes(self) -> int:
        return _tensor_bytes(self.bert) + _tensor_bytes(self.semantic)

    def limited(self, *, phone_limit: int, semantic_limit: int) -> NeuralContinuationInput | None:
        phone_count = min(max(0, int(phone_limit)), len(self.phones))
        if phone_count <= 0:
            return None
        semantic_count = min(max(0, int(semantic_limit)), self.semantic_tokens)
        return NeuralContinuationInput(
            phones=self.phones[-phone_count:],
            bert=self.bert[:, -phone_count:],
            semantic=self.semantic[:, -semantic_count:] if semantic_count > 0 else None,
        )


@dataclass(frozen=True)
class NeuralContinuationInput:
    phones: tuple[int, ...]
    bert: Any
    semantic: Any | None

    @property
    def semantic_tokens(self) -> int:
        return 0 if self.semantic is None else _shape(self.semantic)[-1]


@dataclass(frozen=True)
class ContinuityPlan:
    configured_policy: ContinuityPolicy
    effective_policy: ContinuityPolicy
    reason: str
    neural_input: NeuralContinuationInput | None

    @property
    def enabled(self) -> bool:
        return self.neural_input is not None

    def public_dict(self) -> dict[str, Any]:
        return {
            "configured_policy": self.configured_policy.value,
            "effective_policy": self.effective_policy.value,
            "reason": self.reason,
            "phones": len(self.neural_input.phones) if self.neural_input else 0,
            "semantic_tokens": self.neural_input.semantic_tokens if self.neural_input else 0,
            "acoustic_latent": False,
        }


class ContinuationCapture:
    """Thread-safe handoff for one successfully completed TensorRT stream."""

    def __init__(self, *, model_id: str, voice_profile: str) -> None:
        self.model_id = model_id
        self.voice_profile = voice_profile
        self._state: NeuralContinuationState | None = None
        self._lock = threading.Lock()

    def publish(
        self,
        *,
        language: str,
        expression: str | None,
        phones: Any,
        bert: Any,
        semantic: Any,
    ) -> None:
        state = NeuralContinuationState.capture(
            model_id=self.model_id,
            voice_profile=self.voice_profile,
            language=language,
            expression=expression,
            phones=phones,
            bert=bert,
            semantic=semantic,
        )
        with self._lock:
            self._state = state

    def take(self) -> NeuralContinuationState | None:
        with self._lock:
            state = self._state
            self._state = None
            return state

    def clear(self) -> None:
        with self._lock:
            self._state = None


@dataclass
class SpeechContextState:
    """Committed metadata and bounded neural context for one speech session."""

    policy: ContinuityPolicy = ContinuityPolicy.NONE
    model_id: str | None = None
    voice_profile: str | None = None
    segment_index: int = 0
    previous_text: str = ""
    previous_language: str | None = None
    previous_expression: str | None = None
    paragraph_id: str | None = None
    previous_pause_ms: int = 0
    previous_output_samples: int = 0
    total_output_samples: int = 0
    hard_boundary: bool = True
    neural: NeuralContinuationState | None = None
    last_plan: ContinuityPlan | None = None

    def bind(self, *, model_id: str, voice_profile: str, policy: ContinuityPolicy | str) -> None:
        resolved = parse_continuity_policy(policy)
        if self.model_id is not None and (
            self.model_id != model_id or self.voice_profile != voice_profile
        ):
            self.clear()
        self.model_id = str(model_id)
        self.voice_profile = str(voice_profile)
        self.policy = resolved
        if resolved is ContinuityPolicy.NONE:
            self.clear_neural()

    def plan_for(
        self,
        *,
        text: str,
        language: str,
        expression: str | None,
        paragraph_id: str | None,
    ) -> ContinuityPlan:
        del text
        configured = self.policy
        if configured is ContinuityPolicy.NONE:
            plan = ContinuityPlan(configured, ContinuityPolicy.NONE, "policy-a", None)
            self.last_plan = plan
            return plan
        state = self.neural
        if state is None:
            plan = ContinuityPlan(configured, ContinuityPolicy.NONE, "no-committed-state", None)
            self.last_plan = plan
            return plan
        if state.model_id != self.model_id or state.voice_profile != self.voice_profile:
            self.clear_neural()
            plan = ContinuityPlan(configured, ContinuityPolicy.NONE, "runtime-identity-mismatch", None)
            self.last_plan = plan
            return plan

        effective = configured
        phone_limit = MAX_CONTEXT_PHONEMES
        semantic_limit = int(CONTINUITY_POLICY_CONTRACTS[configured.value]["semantic_tokens"])
        reason = str(CONTINUITY_POLICY_CONTRACTS[configured.value]["name"])
        expression_changed = self.previous_expression != expression
        if configured is ContinuityPolicy.EXPRESSION_AWARE and expression_changed:
            phone_limit = EXPRESSION_SWITCH_PHONEMES
            semantic_limit = EXPRESSION_SWITCH_SEMANTIC_TOKENS
            reason = "expression-switch"
        elif configured is ContinuityPolicy.BOUNDARY_ADAPTIVE:
            paragraph_changed = self.segment_index > 0 and self.paragraph_id != paragraph_id
            language_changed = self.previous_language not in {None, language}
            boundary = classify_boundary(self.previous_text)
            if paragraph_changed:
                effective, reason = ContinuityPolicy.NONE, "paragraph-boundary"
            elif self.previous_pause_ms >= 700:
                effective, reason = ContinuityPolicy.NONE, "long-pause-boundary"
            elif language_changed:
                effective, reason = ContinuityPolicy.NONE, "language-boundary"
            elif boundary is BoundaryKind.HARD_NATURAL:
                effective, reason = ContinuityPolicy.NONE, "hard-natural-boundary"
            elif expression_changed:
                effective = ContinuityPolicy.EXPRESSION_AWARE
                phone_limit = EXPRESSION_SWITCH_PHONEMES
                semantic_limit = EXPRESSION_SWITCH_SEMANTIC_TOKENS
                reason = "expression-switch"
            elif boundary is BoundaryKind.SOFT_NATURAL:
                effective = ContinuityPolicy.SEMANTIC_32
                semantic_limit = 32
                reason = "soft-natural-boundary"
            else:
                effective = ContinuityPolicy.SEMANTIC_64
                semantic_limit = 64
                reason = "technical-boundary"

        neural_input = None if effective is ContinuityPolicy.NONE else state.limited(
            phone_limit=phone_limit, semantic_limit=semantic_limit
        )
        plan = ContinuityPlan(configured, effective, reason, neural_input)
        self.last_plan = plan
        return plan

    def commit(
        self,
        *,
        text: str,
        language: str,
        expression: str | None,
        paragraph_id: str | None,
        pause_after_ms: int,
        output_samples: int,
        neural_state: NeuralContinuationState | None = None,
        require_neural_state: bool = False,
    ) -> None:
        if require_neural_state and neural_state is None:
            raise ValueError("TensorRT stream completed without committed continuation state")
        if neural_state is not None and (
            neural_state.model_id != self.model_id or neural_state.voice_profile != self.voice_profile
        ):
            raise ValueError("continuation state runtime identity does not match the session")
        paragraph_changed = self.segment_index > 0 and self.paragraph_id != paragraph_id
        self.segment_index += 1
        self.previous_text = text[-MAX_CONTEXT_TEXT_CHARS:]
        self.previous_language = language
        self.previous_expression = expression
        self.paragraph_id = paragraph_id
        self.previous_pause_ms = pause_after_ms
        self.previous_output_samples = max(0, int(output_samples))
        self.total_output_samples += self.previous_output_samples
        self.hard_boundary = paragraph_changed or pause_after_ms >= 700
        self.neural = neural_state if self.policy is not ContinuityPolicy.NONE else None

    def clear_neural(self) -> None:
        self.neural = None

    def clear(self) -> None:
        self.segment_index = 0
        self.previous_text = ""
        self.previous_language = None
        self.previous_expression = None
        self.paragraph_id = None
        self.previous_pause_ms = 0
        self.previous_output_samples = 0
        self.total_output_samples = 0
        self.hard_boundary = True
        self.clear_neural()
        self.last_plan = None

    def public_dict(self) -> dict[str, Any]:
        state = self.neural
        return {
            "mode": "committed-neural-v1",
            "policy": self.policy.value,
            "neural_state_continuity": bool(self.policy is not ContinuityPolicy.NONE and state is not None),
            "acoustic_latent_continuity": False,
            "committed_segments": self.segment_index,
            "previous_language": self.previous_language,
            "previous_expression": self.previous_expression,
            "paragraph_id": self.paragraph_id,
            "previous_pause_ms": self.previous_pause_ms,
            "previous_output_samples": self.previous_output_samples,
            "total_output_samples": self.total_output_samples,
            "hard_boundary": self.hard_boundary,
            "retained_phonemes": len(state.phones) if state else 0,
            "retained_semantic_tokens": state.semantic_tokens if state else 0,
            "retained_neural_bytes": state.retained_bytes if state else 0,
            "last_plan": self.last_plan.public_dict() if self.last_plan else None,
        }


__all__ = [
    "CONTINUITY_POLICY_CONTRACTS",
    "EXPRESSION_SWITCH_PHONEMES",
    "EXPRESSION_SWITCH_SEMANTIC_TOKENS",
    "MAX_CONTEXT_PHONEMES",
    "MAX_CONTEXT_SEMANTIC_TOKENS",
    "MAX_CONTEXT_TEXT_CHARS",
    "ContinuationCapture",
    "ContinuityPlan",
    "ContinuityPolicy",
    "NeuralContinuationInput",
    "NeuralContinuationState",
    "SpeechContextState",
    "parse_continuity_policy",
]
