from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SemanticSamplingConfig:
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")


def logits_to_probabilities(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor | None,
    config: SemanticSamplingConfig,
) -> torch.Tensor:
    """Match GPT-SoVITS AR.models.utils.logits_to_probs on a CUDA tensor."""

    if logits.ndim != 2:
        raise ValueError(f"Expected rank-2 semantic logits, got {tuple(logits.shape)}")
    working = logits.clone()
    if previous_tokens is not None and config.repetition_penalty != 1.0:
        history = previous_tokens.to(dtype=torch.long)
        scores = torch.gather(working, dim=1, index=history)
        scores = torch.where(
            scores < 0,
            scores * config.repetition_penalty,
            scores / config.repetition_penalty,
        )
        working.scatter_(dim=1, index=history, src=scores)

    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(working, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative > config.top_p
        sorted_remove[:, 0] = False
        remove = sorted_remove.scatter(
            dim=1,
            index=sorted_indices,
            src=sorted_remove,
        )
        working = working.masked_fill(remove, -float("inf"))

    working = working / max(config.temperature, 1e-5)
    values, _ = torch.topk(working, min(config.top_k, working.shape[-1]))
    threshold = values[:, -1].unsqueeze(-1)
    working = torch.where(working < threshold, -float("inf"), working)
    return torch.softmax(working, dim=-1)


def sample_semantic_token(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor | None,
    config: SemanticSamplingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = logits_to_probabilities(logits, previous_tokens, config)
    exponential = torch.empty_like(probabilities).exponential_(1)
    token = torch.argmax(probabilities / exponential, dim=-1, keepdim=True).to(dtype=torch.int)
    return token, probabilities


__all__ = [
    "SemanticSamplingConfig",
    "logits_to_probabilities",
    "sample_semantic_token",
]
