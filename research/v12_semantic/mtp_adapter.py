from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn


ADAPTER_FORMAT = "aniflive-tts-mtp-adapter-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MTPAdapterSpec:
    base_gpt_sha256: str
    hidden_dim: int
    vocab_size: int
    heads: int

    def __post_init__(self) -> None:
        if len(self.base_gpt_sha256) != 64:
            raise ValueError("base_gpt_sha256 must contain 64 hexadecimal characters")
        int(self.base_gpt_sha256, 16)
        if self.hidden_dim < 1 or self.vocab_size < 2:
            raise ValueError("hidden_dim and vocab_size must be positive")
        if self.heads < 2:
            raise ValueError("MTP requires at least two prediction heads")

    @property
    def future_head_count(self) -> int:
        return self.heads - 1

    def metadata(self) -> dict[str, str]:
        return {
            "format": ADAPTER_FORMAT,
            "base_gpt_sha256": self.base_gpt_sha256,
            "hidden_dim": str(self.hidden_dim),
            "vocab_size": str(self.vocab_size),
            "heads": str(self.heads),
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, str]) -> "MTPAdapterSpec":
        if metadata.get("format") != ADAPTER_FORMAT:
            raise ValueError("Unsupported MTP adapter format")
        return cls(
            base_gpt_sha256=metadata["base_gpt_sha256"],
            hidden_dim=int(metadata["hidden_dim"]),
            vocab_size=int(metadata["vocab_size"]),
            heads=int(metadata["heads"]),
        )


class MTPAdapter(nn.Module):
    """Independent future-token heads over a frozen Transformer hidden state."""

    def __init__(self, spec: MTPAdapterSpec) -> None:
        super().__init__()
        self.spec = spec
        self.weight = nn.Parameter(
            torch.empty(
                spec.future_head_count,
                spec.vocab_size,
                spec.hidden_dim,
            )
        )
        nn.init.normal_(self.weight, mean=0.0, std=spec.hidden_dim**-0.5)

    @torch.no_grad()
    def initialize_from_base_head(self, base_weight: torch.Tensor) -> None:
        expected = (self.spec.vocab_size, self.spec.hidden_dim)
        if tuple(base_weight.shape) != expected:
            raise ValueError(
                f"Base head has shape {tuple(base_weight.shape)}, expected {expected}"
            )
        self.weight.copy_(base_weight.unsqueeze(0).expand_as(self.weight))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.spec.hidden_dim:
            raise ValueError(
                f"Hidden size {hidden.shape[-1]} does not match {self.spec.hidden_dim}"
            )
        return torch.einsum("...d,hvd->...hv", hidden, self.weight)


def future_targets(tokens: torch.Tensor, heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return hidden-row indices and t+2..t+heads targets for one token row."""
    if tokens.ndim != 1:
        raise ValueError("tokens must be a one-dimensional tensor")
    if heads < 2:
        raise ValueError("heads must be at least two")
    usable = int(tokens.numel()) - heads
    if usable < 1:
        raise ValueError("token row is too short for the requested MTP heads")
    rows = torch.arange(usable, dtype=torch.int64, device=tokens.device)
    targets = torch.stack(
        [tokens[offset : offset + usable] for offset in range(2, heads + 1)],
        dim=1,
    )
    return rows, targets


def prompt_conditioned_targets(tokens: torch.Tensor, heads: int) -> torch.Tensor:
    """Return t+2..t+heads labels including the prompt-final hidden row.

    Row zero is the final prompt hidden state, whose original head predicts the
    first semantic token. Subsequent rows are the hidden states after consuming
    token zero, token one, and so on.
    """
    if tokens.ndim != 1:
        raise ValueError("tokens must be a one-dimensional tensor")
    if heads < 2:
        raise ValueError("heads must be at least two")
    rows = int(tokens.numel()) - heads + 1
    if rows < 1:
        raise ValueError("token row is too short for the requested MTP heads")
    return torch.stack(
        [tokens[offset : offset + rows] for offset in range(1, heads)],
        dim=1,
    )


def save_adapter(
    adapter: MTPAdapter,
    path: Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> None:
    combined = adapter.spec.metadata()
    combined.update(dict(metadata or {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"mtp_head_weight": adapter.weight.detach().cpu().contiguous()},
        str(path),
        metadata=combined,
    )


def load_adapter(
    path: Path,
    *,
    expected_base_gpt_sha256: str,
    device: torch.device | str = "cpu",
) -> MTPAdapter:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        spec = MTPAdapterSpec.from_metadata(metadata)
        if spec.base_gpt_sha256 != expected_base_gpt_sha256:
            raise ValueError(
                "MTP adapter checkpoint mismatch: "
                f"adapter={spec.base_gpt_sha256}, expected={expected_base_gpt_sha256}"
            )
        weight = handle.get_tensor("mtp_head_weight")
    adapter = MTPAdapter(spec).to(device)
    expected = (
        spec.future_head_count,
        spec.vocab_size,
        spec.hidden_dim,
    )
    if tuple(weight.shape) != expected:
        raise ValueError(f"MTP adapter weight has shape {tuple(weight.shape)}, expected {expected}")
    with torch.no_grad():
        adapter.weight.copy_(weight.to(device))
    return adapter


__all__ = [
    "ADAPTER_FORMAT",
    "MTPAdapter",
    "MTPAdapterSpec",
    "future_targets",
    "prompt_conditioned_targets",
    "load_adapter",
    "save_adapter",
    "sha256_file",
]
