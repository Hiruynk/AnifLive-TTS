"""Research-only Mamba-2 reference for the AnifLive-TTS v1.2 hybrid study.

This module intentionally has no dependency on ``mamba_ssm`` or a PyTorch
production fallback.  It provides a readable training/reference
implementation whose recurrent update is shared with the standalone
TensorRT 11 plugin feasibility experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any, Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Mamba2ReferenceConfig:
    d_model: int = 512
    nheads: int = 16
    ngroups: int = 16
    d_state: int = 32
    d_conv: int = 4

    def __post_init__(self) -> None:
        values = (self.d_model, self.nheads, self.ngroups, self.d_state, self.d_conv)
        if any(value <= 0 for value in values):
            raise ValueError("Mamba-2 dimensions must be positive")
        if self.d_model % self.nheads:
            raise ValueError("d_model must be divisible by nheads")
        if self.nheads % self.ngroups:
            raise ValueError("nheads must be divisible by ngroups")
        if self.group_width != self.d_state:
            raise ValueError(
                "The v1.2 structural mapping requires d_model/ngroups == d_state"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.nheads

    @property
    def group_width(self) -> int:
        return self.d_model // self.ngroups

    @property
    def xbc_dim(self) -> int:
        return self.d_model + 2 * self.ngroups * self.d_state

    @property
    def projection_dim(self) -> int:
        return self.d_model + self.xbc_dim + self.nheads


@dataclass
class Mamba2ReferenceState:
    conv: Tensor
    ssm: Tensor

    def detached(self) -> "Mamba2ReferenceState":
        return Mamba2ReferenceState(self.conv.detach(), self.ssm.detach())


def block_beg_attention_layers(num_layers: int) -> tuple[int, ...]:
    """Return the Transformer layers retained by the 1:1 BlockBeg layout."""

    if num_layers <= 0 or num_layers % 2:
        raise ValueError("BlockBeg 1:1 requires a positive even layer count")
    return tuple(range(0, num_layers, 2))


class Mamba2ReferenceMixer(nn.Module):
    """Transparent selective-state mixer used only for distillation research."""

    def __init__(self, config: Mamba2ReferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, config.projection_dim, bias=True)
        self.conv1d = nn.Conv1d(
            config.xbc_dim,
            config.xbc_dim,
            kernel_size=config.d_conv,
            groups=config.xbc_dim,
            bias=True,
        )
        self.A_log = nn.Parameter(torch.empty(config.nheads, dtype=torch.float32))
        self.D = nn.Parameter(torch.ones(config.nheads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(config.nheads, dtype=torch.float32))
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.conv1d.weight)
        with torch.no_grad():
            self.conv1d.weight[:, 0, -1] = 1.0
        nn.init.zeros_(self.conv1d.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        with torch.no_grad():
            self.A_log.copy_(
                torch.log(torch.linspace(1.0, 16.0, self.config.nheads))
            )
            target_dt = torch.full((self.config.nheads,), 0.01)
            self.dt_bias.copy_(target_dt + torch.log(-torch.expm1(-target_dt)))
            self.D.fill_(1.0)

    def allocate_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Mamba2ReferenceState:
        config = self.config
        return Mamba2ReferenceState(
            conv=torch.zeros(
                batch_size,
                config.xbc_dim,
                config.d_conv,
                device=device,
                dtype=dtype,
            ),
            ssm=torch.zeros(
                batch_size,
                config.nheads,
                config.head_dim,
                config.d_state,
                device=device,
                dtype=dtype,
            ),
        )

    def initialize_from_attention(self, block: Any) -> None:
        """Apply the paper's Q->C, K->B, V->x structural initialization."""

        config = self.config
        q_weight, k_weight, v_weight = block.qkv_w.detach().chunk(3, dim=0)
        q_bias, k_bias, v_bias = block.qkv_b.detach().chunk(3, dim=0)
        if q_weight.shape != (config.d_model, config.d_model):
            raise ValueError("Transformer attention dimensions do not match Mamba config")
        dt_start = config.d_model + config.xbc_dim
        with torch.no_grad():
            self.in_proj.weight.zero_()
            self.in_proj.bias.zero_()
            cursor = 0
            self.in_proj.weight[cursor : cursor + config.d_model].copy_(q_weight)
            self.in_proj.bias[cursor : cursor + config.d_model].copy_(q_bias)
            cursor += config.d_model
            self.in_proj.weight[cursor : cursor + config.d_model].copy_(v_weight)
            self.in_proj.bias[cursor : cursor + config.d_model].copy_(v_bias)
            cursor += config.d_model
            self.in_proj.weight[
                cursor : cursor + config.ngroups * config.d_state
            ].copy_(k_weight)
            self.in_proj.bias[
                cursor : cursor + config.ngroups * config.d_state
            ].copy_(k_bias)
            cursor += config.ngroups * config.d_state
            self.in_proj.weight[
                cursor : cursor + config.ngroups * config.d_state
            ].copy_(q_weight)
            self.in_proj.bias[
                cursor : cursor + config.ngroups * config.d_state
            ].copy_(q_bias)
            self.in_proj.weight[dt_start:].zero_()
            self.in_proj.bias[dt_start:].zero_()
            self.out_proj.weight.copy_(block.out_w.detach())
            self.out_proj.bias.copy_(block.out_b.detach())

    def _split_projection(self, projected: Tensor) -> tuple[Tensor, ...]:
        config = self.config
        group_features = config.ngroups * config.d_state
        return torch.split(
            projected,
            [
                config.d_model,
                config.d_model,
                group_features,
                group_features,
                config.nheads,
            ],
            dim=-1,
        )

    def _expand_groups(self, value: Tensor) -> Tensor:
        repeat = self.config.nheads // self.config.ngroups
        return value if repeat == 1 else value.repeat_interleave(repeat, dim=1)

    def _ssm_step(
        self,
        x: Tensor,
        b: Tensor,
        c: Tensor,
        delta: Tensor,
        z: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        config = self.config
        original_dtype = x.dtype
        x_heads = x.reshape(x.shape[0], config.nheads, config.head_dim).float()
        z_heads = z.reshape(z.shape[0], config.nheads, config.head_dim).float()
        b_groups = b.reshape(b.shape[0], config.ngroups, config.d_state).float()
        c_groups = c.reshape(c.shape[0], config.ngroups, config.d_state).float()
        b_heads = self._expand_groups(b_groups)
        c_heads = self._expand_groups(c_groups)
        dt = F.softplus(delta.float() + self.dt_bias.unsqueeze(0))
        transition = torch.exp(
            dt[:, :, None, None] * -torch.exp(self.A_log)[None, :, None, None]
        )
        new_state = state.float() * transition
        new_state = new_state + (
            dt[:, :, None, None]
            * x_heads[:, :, :, None]
            * b_heads[:, :, None, :]
        )
        output = (new_state * c_heads[:, :, None, :]).sum(dim=-1)
        output = output + self.D[None, :, None] * x_heads
        output = output * F.silu(z_heads)
        return output.reshape(x.shape[0], config.d_model).to(original_dtype), new_state.to(
            original_dtype
        )

    def _ssm_scan_training(
        self,
        x: Tensor,
        b: Tensor,
        c: Tensor,
        delta: Tensor,
        z: Tensor,
        state: Tensor,
        *,
        chunk_size: int = 32,
    ) -> tuple[Tensor, Tensor]:
        """Differentiable chunked scan used to keep research training practical."""

        config = self.config
        original_dtype = x.dtype
        batch, length, _ = x.shape
        x_heads = x.reshape(batch, length, config.nheads, config.head_dim).float()
        z_heads = z.reshape(batch, length, config.nheads, config.head_dim).float()
        b_heads = b.reshape(batch, length, config.ngroups, config.d_state).float()
        c_heads = c.reshape(batch, length, config.ngroups, config.d_state).float()
        repeat = config.nheads // config.ngroups
        if repeat != 1:
            b_heads = b_heads.repeat_interleave(repeat, dim=2)
            c_heads = c_heads.repeat_interleave(repeat, dim=2)
        dt = F.softplus(delta.float() + self.dt_bias[None, None])
        decay = torch.exp(
            dt[:, :, :, None, None]
            * -torch.exp(self.A_log)[None, None, :, None, None]
        )
        update = (
            dt[:, :, :, None, None]
            * x_heads[:, :, :, :, None]
            * b_heads[:, :, :, None, :]
        )
        current = state.float()
        outputs: list[Tensor] = []
        for start in range(0, length, chunk_size):
            stop = min(start + chunk_size, length)
            chunk_decay = decay[:, start:stop]
            chunk_update = update[:, start:stop]
            prefix = torch.cumprod(chunk_decay, dim=1)
            states = prefix * (
                current[:, None]
                + torch.cumsum(chunk_update / prefix.clamp_min(1e-20), dim=1)
            )
            current = states[:, -1]
            output = (
                states * c_heads[:, start:stop, :, None, :]
            ).sum(dim=-1)
            output = output + self.D[None, None, :, None] * x_heads[:, start:stop]
            output = output * F.silu(z_heads[:, start:stop])
            outputs.append(output.reshape(batch, stop - start, config.d_model))
        return torch.cat(outputs, dim=1).to(original_dtype), current.to(original_dtype)

    def forward(
        self,
        hidden: Tensor,
        state: Mamba2ReferenceState | None = None,
    ) -> tuple[Tensor, Mamba2ReferenceState]:
        if hidden.ndim != 3:
            raise ValueError("Mamba sequence input must have shape [batch, length, dim]")
        batch, length, dim = hidden.shape
        if dim != self.config.d_model or length == 0:
            raise ValueError("Mamba sequence input has an invalid shape")
        if state is None:
            state = self.allocate_state(batch, device=hidden.device, dtype=hidden.dtype)
        projected = self.in_proj(hidden)
        z, x, b, c, delta = self._split_projection(projected)
        raw_xbc = torch.cat((x, b, c), dim=-1)
        prefix = state.conv[:, :, 1:]
        convolved = self.conv1d(
            torch.cat((prefix, raw_xbc.transpose(1, 2)), dim=-1)
        ).transpose(1, 2)
        activated = F.silu(convolved)
        x_conv, b_conv, c_conv = torch.split(
            activated,
            [
                self.config.d_model,
                self.config.ngroups * self.config.d_state,
                self.config.ngroups * self.config.d_state,
            ],
            dim=-1,
        )
        current_ssm = state.ssm
        if self.training:
            stacked, current_ssm = self._ssm_scan_training(
                x_conv, b_conv, c_conv, delta, z, current_ssm
            )
        else:
            outputs: list[Tensor] = []
            for index in range(length):
                output, current_ssm = self._ssm_step(
                    x_conv[:, index],
                    b_conv[:, index],
                    c_conv[:, index],
                    delta[:, index],
                    z[:, index],
                    current_ssm,
                )
                outputs.append(output)
            stacked = torch.stack(outputs, dim=1)
        combined_raw = torch.cat((state.conv, raw_xbc.transpose(1, 2)), dim=-1)
        next_conv = combined_raw[:, :, -self.config.d_conv :]
        result = self.out_proj(stacked)
        return result, Mamba2ReferenceState(next_conv, current_ssm)

    def step(
        self,
        hidden: Tensor,
        state: Mamba2ReferenceState,
    ) -> tuple[Tensor, Mamba2ReferenceState]:
        if hidden.ndim != 3 or hidden.shape[1:] != (1, self.config.d_model):
            raise ValueError("Mamba step input must have shape [batch, 1, dim]")
        projected = self.in_proj(hidden[:, 0])
        z, x, b, c, delta = self._split_projection(projected)
        raw_xbc = torch.cat((x, b, c), dim=-1)
        next_conv = torch.cat((state.conv[:, :, 1:], raw_xbc.unsqueeze(-1)), dim=-1)
        convolved = (next_conv * self.conv1d.weight[:, 0].unsqueeze(0)).sum(dim=-1)
        convolved = convolved + self.conv1d.bias.unsqueeze(0)
        activated = F.silu(convolved)
        x_conv, b_conv, c_conv = torch.split(
            activated,
            [
                self.config.d_model,
                self.config.ngroups * self.config.d_state,
                self.config.ngroups * self.config.d_state,
            ],
            dim=-1,
        )
        output, next_ssm = self._ssm_step(
            x_conv, b_conv, c_conv, delta, z, state.ssm
        )
        return self.out_proj(output).unsqueeze(1), Mamba2ReferenceState(
            next_conv, next_ssm
        )


class MambaReplacementBlock(nn.Module):
    """Mamba attention replacement retaining GPT's post-norm FFN contract."""

    def __init__(self, config: Mamba2ReferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.mamba = Mamba2ReferenceMixer(config)
        self.linear1 = nn.Linear(config.d_model, config.d_model * 4)
        self.linear2 = nn.Linear(config.d_model * 4, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    @classmethod
    def from_t2s_block(
        cls,
        block: Any,
        config: Mamba2ReferenceConfig,
    ) -> "MambaReplacementBlock":
        result = cls(config)
        result.mamba.initialize_from_attention(block)
        with torch.no_grad():
            result.linear1.weight.copy_(block.mlp.w1.detach())
            result.linear1.bias.copy_(block.mlp.b1.detach())
            result.linear2.weight.copy_(block.mlp.w2.detach())
            result.linear2.bias.copy_(block.mlp.b2.detach())
            result.norm1.weight.copy_(block.norm_w1.detach())
            result.norm1.bias.copy_(block.norm_b1.detach())
            result.norm2.weight.copy_(block.norm_w2.detach())
            result.norm2.bias.copy_(block.norm_b2.detach())
        result.norm1.eps = float(block.norm_eps1)
        result.norm2.eps = float(block.norm_eps2)
        for module in (result.linear1, result.linear2, result.norm1, result.norm2):
            for parameter in module.parameters():
                parameter.requires_grad = False
        return result

    def forward(
        self,
        hidden: Tensor,
        state: Mamba2ReferenceState | None = None,
    ) -> tuple[Tensor, Mamba2ReferenceState]:
        mixed, next_state = self.mamba(hidden, state)
        hidden = self.norm1(hidden + mixed)
        hidden = self.norm2(hidden + self.linear2(F.relu(self.linear1(hidden))))
        return hidden, next_state

    def step(
        self,
        hidden: Tensor,
        state: Mamba2ReferenceState,
    ) -> tuple[Tensor, Mamba2ReferenceState]:
        mixed, next_state = self.mamba.step(hidden, state)
        hidden = self.norm1(hidden + mixed)
        hidden = self.norm2(hidden + self.linear2(F.relu(self.linear1(hidden))))
        return hidden, next_state


@dataclass
class HybridLayerState:
    kind: str
    k_cache: Tensor | None = None
    v_cache: Tensor | None = None
    mamba: Mamba2ReferenceState | None = None


class HybridT2SBackbone(nn.Module):
    """Interleaved GPT Transformer/Mamba research backbone."""

    def __init__(
        self,
        core: nn.Module,
        *,
        attention_layers: Iterable[int] | None = None,
    ) -> None:
        super().__init__()
        self.core = core
        self.num_layers = int(core.num_layers)
        self.attention_layers = frozenset(
            block_beg_attention_layers(self.num_layers)
            if attention_layers is None
            else attention_layers
        )
        invalid = sorted(index for index in self.attention_layers if not 0 <= index < self.num_layers)
        if invalid:
            raise ValueError(f"Invalid Transformer layer indices: {invalid}")
        config = Mamba2ReferenceConfig(
            d_model=int(core.model_dim),
            nheads=int(core.num_head),
            ngroups=int(core.num_head),
            d_state=int(core.model_dim) // int(core.num_head),
        )
        self.mamba_layers = nn.ModuleDict(
            {
                str(index): MambaReplacementBlock.from_t2s_block(
                    core.t2s_transformer.blocks[index], config
                )
                for index in range(self.num_layers)
                if index not in self.attention_layers
            }
        )
        for parameter in self.core.parameters():
            parameter.requires_grad = False

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def process_prompt(
        self,
        hidden: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, list[HybridLayerState]]:
        states: list[HybridLayerState] = []
        for index in range(self.num_layers):
            if index in self.attention_layers:
                hidden, k_cache, v_cache = self.core.t2s_transformer.blocks[
                    index
                ].process_prompt(hidden, attention_mask, None, True)
                states.append(HybridLayerState("transformer", k_cache, v_cache))
            else:
                hidden, mamba_state = self.mamba_layers[str(index)](hidden)
                states.append(HybridLayerState("mamba", mamba=mamba_state))
        return hidden, states

    def decode_next_token(
        self,
        hidden: Tensor,
        states: list[HybridLayerState],
    ) -> tuple[Tensor, list[HybridLayerState]]:
        if len(states) != self.num_layers:
            raise ValueError("Hybrid state count does not match the backbone")
        next_states: list[HybridLayerState] = []
        for index, state in enumerate(states):
            if state.kind == "transformer":
                if state.k_cache is None or state.v_cache is None:
                    raise ValueError("Transformer layer state is incomplete")
                hidden, k_cache, v_cache = self.core.t2s_transformer.blocks[
                    index
                ].decode_next_token(hidden, state.k_cache, state.v_cache)
                next_states.append(HybridLayerState("transformer", k_cache, v_cache))
            elif state.kind == "mamba":
                if state.mamba is None:
                    raise ValueError("Mamba layer state is incomplete")
                hidden, mamba_state = self.mamba_layers[str(index)].step(
                    hidden, state.mamba
                )
                next_states.append(HybridLayerState("mamba", mamba=mamba_state))
            else:
                raise ValueError(f"Unknown hybrid layer kind: {state.kind}")
        return hidden, next_states


def synthetic_t2s_block(d_model: int) -> Any:
    """Create the minimum T2SBlock-shaped object used by reference tests."""

    return SimpleNamespace(
        qkv_w=nn.Parameter(torch.randn(d_model * 3, d_model) * 0.02),
        qkv_b=nn.Parameter(torch.randn(d_model * 3) * 0.02),
        out_w=nn.Parameter(torch.randn(d_model, d_model) * 0.02),
        out_b=nn.Parameter(torch.randn(d_model) * 0.02),
        mlp=SimpleNamespace(
            w1=nn.Parameter(torch.randn(d_model * 4, d_model) * 0.02),
            b1=nn.Parameter(torch.randn(d_model * 4) * 0.02),
            w2=nn.Parameter(torch.randn(d_model, d_model * 4) * 0.02),
            b2=nn.Parameter(torch.randn(d_model) * 0.02),
        ),
        norm_w1=nn.Parameter(torch.ones(d_model)),
        norm_b1=nn.Parameter(torch.zeros(d_model)),
        norm_eps1=1e-5,
        norm_w2=nn.Parameter(torch.ones(d_model)),
        norm_b2=nn.Parameter(torch.zeros(d_model)),
        norm_eps2=1e-5,
    )


__all__ = [
    "HybridLayerState",
    "HybridT2SBackbone",
    "Mamba2ReferenceConfig",
    "Mamba2ReferenceMixer",
    "Mamba2ReferenceState",
    "MambaReplacementBlock",
    "block_beg_attention_layers",
    "synthetic_t2s_block",
]
