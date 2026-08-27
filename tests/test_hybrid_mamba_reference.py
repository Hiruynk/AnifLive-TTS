from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.v12_semantic.hybrid_model import (  # noqa: E402
    Mamba2ReferenceConfig,
    Mamba2ReferenceMixer,
    MambaReplacementBlock,
    block_beg_attention_layers,
    synthetic_t2s_block,
)


def test_v2proplus_contract() -> None:
    config = Mamba2ReferenceConfig()
    assert config.head_dim == 32
    assert config.group_width == 32
    assert config.xbc_dim == 1536
    assert config.projection_dim == 2064
    assert block_beg_attention_layers(24) == tuple(range(0, 24, 2))


def test_sequence_matches_recurrent_steps() -> None:
    torch.manual_seed(1234)
    config = Mamba2ReferenceConfig(
        d_model=32,
        nheads=4,
        ngroups=4,
        d_state=8,
        d_conv=4,
    )
    mixer = Mamba2ReferenceMixer(config).eval()
    inputs = torch.randn(2, 11, config.d_model)
    sequence, sequence_state = mixer(inputs)
    state = mixer.allocate_state(2, device=inputs.device, dtype=inputs.dtype)
    parts = []
    for index in range(inputs.shape[1]):
        output, state = mixer.step(inputs[:, index : index + 1], state)
        parts.append(output)
    recurrent = torch.cat(parts, dim=1)
    torch.testing.assert_close(recurrent, sequence, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(state.conv, sequence_state.conv, atol=5e-7, rtol=5e-7)
    torch.testing.assert_close(state.ssm, sequence_state.ssm, atol=2e-6, rtol=2e-6)


def test_training_scan_matches_reference_sequence() -> None:
    torch.manual_seed(4321)
    config = Mamba2ReferenceConfig(
        d_model=32,
        nheads=4,
        ngroups=4,
        d_state=8,
        d_conv=4,
    )
    mixer = Mamba2ReferenceMixer(config)
    inputs = torch.randn(2, 67, config.d_model)
    mixer.eval()
    reference, reference_state = mixer(inputs)
    mixer.train()
    scanned, scanned_state = mixer(inputs)
    torch.testing.assert_close(scanned, reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        scanned_state.ssm, reference_state.ssm, atol=2e-5, rtol=2e-5
    )


def test_structural_weight_mapping() -> None:
    torch.manual_seed(9)
    config = Mamba2ReferenceConfig(
        d_model=32,
        nheads=4,
        ngroups=4,
        d_state=8,
    )
    source = synthetic_t2s_block(config.d_model)
    mixer = Mamba2ReferenceMixer(config)
    mixer.initialize_from_attention(source)
    q, k, v = source.qkv_w.chunk(3, dim=0)
    z_slice = mixer.in_proj.weight[:32]
    x_slice = mixer.in_proj.weight[32:64]
    b_slice = mixer.in_proj.weight[64:96]
    c_slice = mixer.in_proj.weight[96:128]
    torch.testing.assert_close(z_slice, q)
    torch.testing.assert_close(x_slice, v)
    torch.testing.assert_close(b_slice, k)
    torch.testing.assert_close(c_slice, q)
    torch.testing.assert_close(mixer.out_proj.weight, source.out_w)


def test_replacement_block_sequence_matches_steps() -> None:
    torch.manual_seed(17)
    config = Mamba2ReferenceConfig(
        d_model=32,
        nheads=4,
        ngroups=4,
        d_state=8,
    )
    block = MambaReplacementBlock.from_t2s_block(
        synthetic_t2s_block(config.d_model), config
    ).eval()
    inputs = torch.randn(1, 7, config.d_model)
    sequence, sequence_state = block(inputs)
    state = block.mamba.allocate_state(1, device=inputs.device, dtype=inputs.dtype)
    outputs = []
    for index in range(inputs.shape[1]):
        output, state = block.step(inputs[:, index : index + 1], state)
        outputs.append(output)
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), sequence, atol=3e-6, rtol=3e-6
    )
    torch.testing.assert_close(state.conv, sequence_state.conv, atol=5e-7, rtol=5e-7)
    torch.testing.assert_close(state.ssm, sequence_state.ssm, atol=3e-6, rtol=3e-6)


def test_reference_has_no_mamba_runtime_dependency() -> None:
    source = (
        ROOT / "research" / "v12_semantic" / "hybrid_model.py"
    ).read_text(encoding="utf-8")
    assert "import mamba_ssm" not in source
    assert "from mamba_ssm" not in source


def test_only_mamba_parameters_are_trainable() -> None:
    config = Mamba2ReferenceConfig(
        d_model=32,
        nheads=4,
        ngroups=4,
        d_state=8,
    )
    block = MambaReplacementBlock.from_t2s_block(
        synthetic_t2s_block(config.d_model), config
    )
    trainable = [name for name, value in block.named_parameters() if value.requires_grad]
    assert trainable
    assert all(name.startswith("mamba.") for name in trainable)
