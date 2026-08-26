from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SOURCE_DIR = Path(__file__).resolve().parents[1] / "minimal_inference"
sys.path.insert(0, str(SOURCE_DIR))

from GPT_SoVITS.AR.models.t2s_model import T2SBlock, T2SMLP, T2STransformer  # noqa: E402


def _block(hidden_dim: int = 16, num_heads: int = 4) -> T2SBlock:
    generator = torch.Generator().manual_seed(1234)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * 0.05

    mlp = T2SMLP(
        randn(hidden_dim * 2, hidden_dim),
        randn(hidden_dim * 2),
        randn(hidden_dim, hidden_dim * 2),
        randn(hidden_dim),
    )
    return T2SBlock(
        num_heads,
        hidden_dim,
        mlp,
        randn(hidden_dim * 3, hidden_dim),
        randn(hidden_dim * 3),
        randn(hidden_dim, hidden_dim),
        randn(hidden_dim),
        torch.ones(hidden_dim),
        torch.zeros(hidden_dim),
        1e-5,
        torch.ones(hidden_dim),
        torch.zeros(hidden_dim),
        1e-5,
    )


@pytest.mark.parametrize("block_size", [2, 4])
def test_causal_block_matches_sequential_cache_updates(block_size: int) -> None:
    hidden_dim = 16
    prefix = 5
    capacity = 16
    transformer = T2STransformer(2, [_block(), _block()])
    generator = torch.Generator().manual_seed(5678)
    inputs = torch.randn(1, block_size, hidden_dim, generator=generator)
    base_k = [torch.zeros(1, capacity, hidden_dim) for _ in range(2)]
    base_v = [torch.zeros(1, capacity, hidden_dim) for _ in range(2)]
    for layer in range(2):
        base_k[layer][:, :prefix] = torch.randn(
            1, prefix, hidden_dim, generator=generator
        )
        base_v[layer][:, :prefix] = torch.randn(
            1, prefix, hidden_dim, generator=generator
        )

    sequential_k = [value.clone() for value in base_k]
    sequential_v = [value.clone() for value in base_v]
    sequential_outputs: list[torch.Tensor] = []
    for offset in range(block_size):
        output, sequential_k, sequential_v = transformer.decode_next_token(
            inputs[:, offset : offset + 1],
            sequential_k,
            sequential_v,
            idx=torch.tensor(prefix + offset),
        )
        sequential_outputs.append(output)

    block_output, block_k, block_v = transformer.decode_token_block(
        inputs,
        [value.clone() for value in base_k],
        [value.clone() for value in base_v],
        idx=torch.tensor(prefix),
    )
    torch.testing.assert_close(block_output, torch.cat(sequential_outputs, dim=1))
    for actual, expected in zip(block_k, sequential_k, strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(block_v, sequential_v, strict=True):
        torch.testing.assert_close(actual, expected)
