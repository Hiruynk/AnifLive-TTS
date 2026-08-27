from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import torch

from aniflive_tts.backend.semantic_runtime import (
    SemanticBatch,
    TransformerSemanticRuntime,
    TransformerSemanticState,
)
from aniflive_tts.backend.trt_builder import DetailedTensorRTLogger
from aniflive_tts.backend.trt_runtime import TensorRTRunner


EOS_TOKEN = 1024


def _accepted_prefix_length(
    verified: list[int], drafts: list[int], *, limit: int
) -> int:
    accepted = 0
    while (
        accepted < limit
        and accepted < len(verified)
        and accepted < len(drafts)
        and verified[accepted] == drafts[accepted]
    ):
        accepted += 1
    return accepted


def _mtp4_cycle_plan(
    *, index: int, accepted_prefix: int
) -> tuple[int, int | None, int]:
    if not 0 <= accepted_prefix <= 3:
        raise ValueError("MTP-4 accepted prefix must be between zero and three")
    if accepted_prefix == 3:
        return 3, None, index + 4
    return (
        accepted_prefix + 1,
        index + 1 + accepted_prefix,
        index + 2 + accepted_prefix,
    )


def _filtered_topk_distribution(
    values: torch.Tensor,
    indices: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_count = max(1, min(int(top_k), int(values.shape[-1])))
    selected_values = values[..., :candidate_count]
    selected_indices = indices[..., :candidate_count]
    if float(temperature) != 1.0:
        selected_values = selected_values / float(temperature)
    probabilities = torch.softmax(selected_values, dim=-1)
    if float(top_p) < 1.0:
        cumulative = torch.cumsum(probabilities, dim=-1)
        keep = (cumulative - probabilities) <= max(float(top_p), 0.0)
        keep[..., 0] = True
        probabilities = probabilities.masked_fill(~keep, 0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return probabilities, selected_indices


def _output_buffers(
    runner: TensorRTRunner, device: torch.device
) -> dict[str, torch.Tensor]:
    dtypes = {
        "DataType.HALF": torch.float16,
        "DataType.FLOAT": torch.float32,
        "DataType.INT64": torch.int64,
        "DataType.INT32": torch.int32,
    }
    result: dict[str, torch.Tensor] = {}
    for name in runner.output_names:
        if name in {"k_cache_new", "v_cache_new"}:
            continue
        shape = tuple(int(value) for value in runner.engine.get_tensor_shape(name))
        result[name] = torch.empty(
            shape,
            dtype=dtypes[str(runner.engine.get_tensor_dtype(name))],
            device=device,
        )
    return result


class MTPSpeculativeSemanticRuntime:
    """Verified MTP-2 decoding with the original Transformer as verifier."""

    backend_name = "transformer-mtp2-verified"

    def __init__(
        self,
        *,
        base_runtime: TransformerSemanticRuntime,
        h1_engine: Path,
        h2_engine: Path,
        sample_topk: Any,
        engine: Any,
        logger: logging.Logger | None = None,
    ) -> None:
        self.base_runtime = base_runtime
        self.sample_topk = sample_topk
        self.engine = engine
        self.logger = logger or logging.getLogger(__name__)
        trt_logger = DetailedTensorRTLogger(echo=False)
        self.h1 = TensorRTRunner(
            h1_engine, device=engine.device, logger=trt_logger
        )
        self.h2 = TensorRTRunner(
            h2_engine, device=engine.device, logger=trt_logger
        )
        expected_capacity = int(
            engine.model_gpt_step.engine.get_tensor_shape("k_cache")[2]
        )
        for name, runner in (("h1", self.h1), ("h2", self.h2)):
            capacity = int(runner.engine.get_tensor_shape("k_cache")[2])
            if capacity != expected_capacity:
                raise RuntimeError(
                    f"MTP {name} cache capacity {capacity} does not match "
                    f"the active GPT step capacity {expected_capacity}"
                )
        self._buffers = {
            1: _output_buffers(self.h1, torch.device(engine.device)),
            2: _output_buffers(self.h2, torch.device(engine.device)),
        }

    def prepare(
        self, inputs: Mapping[str, Any], **options: Any
    ) -> TransformerSemanticState:
        state = self.base_runtime.prepare(inputs, **options)
        state.sample_cuda_graph_enabled = False
        state.mtp_proposed_tokens = 0
        state.mtp_accepted_tokens = 0
        state.mtp_rejected_tokens = 0
        state.mtp_trace = (
            []
            if os.environ.get("ANIFLIVE_TTS_MTP_TRACE", "0").strip().lower()
            not in {"0", "false", "no", "off"}
            else None
        )
        return state

    def _run(
        self,
        runner: TensorRTRunner,
        block_size: int,
        state: TransformerSemanticState,
        samples: torch.Tensor,
        source_pair: int,
        destination_pair: int,
        index: int,
    ) -> tuple[dict[str, torch.Tensor], int]:
        outputs = self._buffers[block_size]
        outputs["k_cache_new"] = state.cache_pair[destination_pair][0]
        outputs["v_cache_new"] = state.cache_pair[destination_pair][1]
        runner.infer(
            {
                "samples": samples.to(torch.int64),
                "k_cache": state.cache_pair[source_pair][0],
                "v_cache": state.cache_pair[source_pair][1],
                "x_len": state.x_length.to(self.engine.device),
                "y_len": state.y_length.to(self.engine.device),
                "idx": torch.tensor(
                    [index], dtype=torch.int64, device=self.engine.device
                ),
            },
            outputs=outputs,
            stream=self.engine.stream,
            synchronize=False,
            profile=False,
        )
        state.steps += 1
        return outputs, destination_pair

    def _sample(
        self,
        state: TransformerSemanticState,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.sample_topk(
            values,
            indices,
            temperature=state.temperature,
            top_k=state.top_k,
            top_p=state.top_p,
        ).to(self.engine.device)

    @staticmethod
    def _future_top1(outputs: Mapping[str, torch.Tensor], row: int) -> torch.Tensor:
        return outputs["mtp_topk_indices"][:, row, :1].to(torch.int64)

    @staticmethod
    def _base_row(
        outputs: Mapping[str, torch.Tensor], row: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            outputs["base_topk_values"][:, row, :],
            outputs["base_topk_indices"][:, row, :],
        )

    @staticmethod
    def _control_values(
        state: TransformerSemanticState, *tokens: torch.Tensor
    ) -> list[int]:
        started = time.perf_counter()
        values = (
            torch.cat([token.reshape(-1)[:1] for token in tokens])
            .detach()
            .cpu()
            .tolist()
        )
        state.host_sync_count += 1
        state.host_sync_seconds += time.perf_counter() - started
        return [int(value) for value in values]

    def iter_batches(
        self,
        state: TransformerSemanticState,
        *,
        sync_policy: Callable[[int, int, int], bool],
        cancelled: Any | None = None,
    ) -> Iterator[SemanticBatch]:
        pending: list[torch.Tensor] = []
        generated = 0
        last_reported_nfe = 0
        decode_started = time.perf_counter()

        def make_batch(*, eos: bool) -> SemanticBatch:
            nonlocal pending, last_reported_nfe
            if pending:
                tokens = torch.cat(pending, dim=1)
            else:
                tokens = torch.empty(
                    (1, 0), dtype=torch.int64, device=self.engine.device
                )
            count = int(tokens.shape[1])
            batch = SemanticBatch(
                tokens=tokens,
                eos=eos,
                proposed_tokens=count,
                accepted_tokens=count,
                nfe=state.steps - last_reported_nfe,
            )
            pending = []
            last_reported_nfe = state.steps
            if state.first_batch_seconds == 0.0:
                state.first_batch_seconds = time.perf_counter() - decode_started
            return batch

        def should_emit() -> bool:
            return bool(
                pending
                and sync_policy(
                    len(pending), max(0, generated - 1), state.maximum_steps
                )
            )

        current_pair = 0
        h1_outputs, current_pair = self._run(
            self.h1,
            1,
            state,
            state.first_token,
            current_pair,
            1 - current_pair,
            0,
        )
        base_values, base_indices = self._base_row(h1_outputs, 0)
        next_token = self._sample(state, base_values, base_indices)
        draft = self._future_top1(h1_outputs, 0)
        index = 1

        while generated < state.maximum_steps:
            if cancelled is not None and cancelled.is_set():
                return
            next_value = self._control_values(state, next_token)[0]
            if next_value == EOS_TOKEN:
                yield make_batch(eos=True)
                return
            pending.append(next_token)
            generated += 1
            if should_emit():
                yield make_batch(eos=False)
            if generated >= state.maximum_steps:
                break

            proposals = torch.cat((next_token, draft), dim=1)
            h2_outputs, block_pair = self._run(
                self.h2,
                2,
                state,
                proposals,
                current_pair,
                1 - current_pair,
                index,
            )
            base_values, base_indices = self._base_row(h2_outputs, 0)
            verified = self._sample(state, base_values, base_indices)
            verified_value, draft_value = self._control_values(
                state, verified, draft
            )
            state.mtp_proposed_tokens += 1
            if verified_value == EOS_TOKEN:
                current_pair = block_pair
                yield make_batch(eos=True)
                return

            pending.append(verified)
            generated += 1
            accepted = verified_value == draft_value
            if state.mtp_trace is not None:
                state.mtp_trace.append(
                    {
                        "index": index + 1,
                        "draft": draft_value,
                        "verified": verified_value,
                        "accepted": accepted,
                    }
                )
            if accepted:
                state.mtp_accepted_tokens += 1
                current_pair = block_pair
                base_values, base_indices = self._base_row(h2_outputs, 1)
                next_token = self._sample(state, base_values, base_indices)
                draft = self._future_top1(h2_outputs, 1)
            else:
                state.mtp_rejected_tokens += 1
                h1_outputs, current_pair = self._run(
                    self.h1,
                    1,
                    state,
                    verified,
                    block_pair,
                    current_pair,
                    index + 1,
                )
                base_values, base_indices = self._base_row(h1_outputs, 0)
                next_token = self._sample(state, base_values, base_indices)
                draft = self._future_top1(h1_outputs, 0)
            index += 2
            if should_emit():
                yield make_batch(eos=False)

        if pending:
            yield make_batch(eos=False)

    def close(self) -> None:
        self.base_runtime.close()
        self._buffers.clear()


class MTP4SpeculativeSemanticRuntime(MTPSpeculativeSemanticRuntime):
    """Verify three MTP drafts with one four-token Transformer block step."""

    backend_name = "transformer-mtp4-verified"

    def __init__(
        self,
        *,
        base_runtime: TransformerSemanticRuntime,
        h1_engine: Path,
        h4_engine: Path,
        sample_topk: Any,
        engine: Any,
        logger: logging.Logger | None = None,
    ) -> None:
        self.base_runtime = base_runtime
        self.sample_topk = sample_topk
        self.engine = engine
        self.logger = logger or logging.getLogger(__name__)
        trt_logger = DetailedTensorRTLogger(echo=False)
        self.h1 = TensorRTRunner(
            h1_engine, device=engine.device, logger=trt_logger
        )
        self.h4 = TensorRTRunner(
            h4_engine, device=engine.device, logger=trt_logger
        )
        expected_capacity = int(
            engine.model_gpt_step.engine.get_tensor_shape("k_cache")[2]
        )
        for name, runner in (("h1", self.h1), ("h4", self.h4)):
            capacity = int(runner.engine.get_tensor_shape("k_cache")[2])
            if capacity != expected_capacity:
                raise RuntimeError(
                    f"MTP {name} cache capacity {capacity} does not match "
                    f"the active GPT step capacity {expected_capacity}"
                )
        self._buffers = {
            1: _output_buffers(self.h1, torch.device(engine.device)),
            4: _output_buffers(self.h4, torch.device(engine.device)),
        }

    @staticmethod
    def _future_drafts(
        outputs: Mapping[str, torch.Tensor], row: int
    ) -> torch.Tensor:
        indices = outputs["mtp_topk_indices"]
        if indices.ndim != 4 or int(indices.shape[2]) != 3:
            raise RuntimeError(
                "MTP-4 engine must expose three future prediction heads"
            )
        return indices[:, row, :, 0].to(torch.int64).contiguous()

    def _sample_rows(
        self,
        state: TransformerSemanticState,
        outputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        values = outputs["base_topk_values"]
        indices = outputs["base_topk_indices"]
        rows = int(values.shape[1])
        sampled = self._sample(
            state,
            values.reshape(-1, values.shape[-1]),
            indices.reshape(-1, indices.shape[-1]),
        )
        return sampled.reshape(1, rows)

    @staticmethod
    def _tensor_values(
        state: TransformerSemanticState, tensor: torch.Tensor
    ) -> list[int]:
        started = time.perf_counter()
        values = tensor.detach().cpu().reshape(-1).tolist()
        state.host_sync_count += 1
        state.host_sync_seconds += time.perf_counter() - started
        return [int(value) for value in values]

    def iter_batches(
        self,
        state: TransformerSemanticState,
        *,
        sync_policy: Callable[[int, int, int], bool],
        cancelled: Any | None = None,
    ) -> Iterator[SemanticBatch]:
        pending: list[torch.Tensor] = []
        generated = 0
        last_reported_nfe = 0
        decode_started = time.perf_counter()

        def make_batch(*, eos: bool) -> SemanticBatch:
            nonlocal pending, last_reported_nfe
            if pending:
                tokens = torch.cat(pending, dim=1)
            else:
                tokens = torch.empty(
                    (1, 0), dtype=torch.int64, device=self.engine.device
                )
            count = int(tokens.shape[1])
            batch = SemanticBatch(
                tokens=tokens,
                eos=eos,
                proposed_tokens=count,
                accepted_tokens=count,
                nfe=state.steps - last_reported_nfe,
            )
            pending = []
            last_reported_nfe = state.steps
            if state.first_batch_seconds == 0.0:
                state.first_batch_seconds = time.perf_counter() - decode_started
            return batch

        def should_emit() -> bool:
            return bool(
                pending
                and sync_policy(
                    len(pending), max(0, generated - 1), state.maximum_steps
                )
            )

        def append_token(token: torch.Tensor, value: int) -> bool:
            nonlocal generated
            if value == EOS_TOKEN:
                return False
            pending.append(token)
            generated += 1
            return True

        current_pair = 0
        h1_outputs, current_pair = self._run(
            self.h1,
            1,
            state,
            state.first_token,
            current_pair,
            1 - current_pair,
            0,
        )
        next_token = self._sample(state, *self._base_row(h1_outputs, 0))
        drafts = self._future_drafts(h1_outputs, 0)
        index = 1

        while generated < state.maximum_steps:
            if cancelled is not None and cancelled.is_set():
                return
            next_value = self._control_values(state, next_token)[0]
            if not append_token(next_token, next_value):
                yield make_batch(eos=True)
                return
            if should_emit():
                yield make_batch(eos=False)
            if generated >= state.maximum_steps:
                break

            # A four-token block would overrun the cache contract near the tail.
            # Continue with the fitted H1 engine for those final positions.
            if state.maximum_steps - generated < 3:
                h1_outputs, current_pair = self._run(
                    self.h1,
                    1,
                    state,
                    next_token,
                    current_pair,
                    1 - current_pair,
                    index,
                )
                next_token = self._sample(state, *self._base_row(h1_outputs, 0))
                drafts = self._future_drafts(h1_outputs, 0)
                index += 1
                continue

            proposals = torch.cat((next_token, drafts), dim=1)
            h4_outputs, block_pair = self._run(
                self.h4,
                4,
                state,
                proposals,
                current_pair,
                1 - current_pair,
                index,
            )
            verified = self._sample_rows(state, h4_outputs)
            control_values = self._tensor_values(
                state, torch.cat((verified, drafts), dim=1)
            )
            verified_values = control_values[:4]
            draft_values = control_values[4:]
            accepted_prefix = _accepted_prefix_length(
                verified_values, draft_values, limit=3
            )

            state.mtp_proposed_tokens += 3
            state.mtp_accepted_tokens += accepted_prefix
            if accepted_prefix < 3:
                state.mtp_rejected_tokens += 3 - accepted_prefix
            if state.mtp_trace is not None:
                state.mtp_trace.append(
                    {
                        "index": index,
                        "drafts": draft_values,
                        "verified": verified_values[:3],
                        "accepted_prefix": accepted_prefix,
                    }
                )

            commit_count, correction_index, next_index = _mtp4_cycle_plan(
                index=index, accepted_prefix=accepted_prefix
            )
            for offset in range(commit_count):
                token = verified[:, offset : offset + 1]
                if not append_token(token, verified_values[offset]):
                    yield make_batch(eos=True)
                    return
                if generated >= state.maximum_steps:
                    break
            if generated >= state.maximum_steps:
                break

            if accepted_prefix == 3:
                current_pair = block_pair
                next_token = verified[:, 3:4].contiguous()
                drafts = self._future_drafts(h4_outputs, 3)
                index = next_index
            else:
                if correction_index is None:
                    raise RuntimeError("MTP-4 mismatch is missing a correction index")
                corrected = verified[
                    :, accepted_prefix : accepted_prefix + 1
                ].contiguous()
                h1_outputs, current_pair = self._run(
                    self.h1,
                    1,
                    state,
                    corrected,
                    block_pair,
                    current_pair,
                    correction_index,
                )
                next_token = self._sample(
                    state, *self._base_row(h1_outputs, 0)
                )
                drafts = self._future_drafts(h1_outputs, 0)
                index = next_index
            if should_emit():
                yield make_batch(eos=False)

        if pending:
            yield make_batch(eos=False)


class MTP4RejectionSemanticRuntime(MTP4SpeculativeSemanticRuntime):
    """Distribution-preserving MTP-4 rejection sampling experiment."""

    backend_name = "transformer-mtp4-rejection"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        device = torch.device(self.engine.device)
        self._p_dense = torch.empty((1, 1025), dtype=torch.float32, device=device)
        self._q_dense = torch.empty((1, 1025), dtype=torch.float32, device=device)

    @staticmethod
    def _sample_distribution(
        probabilities: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        flat_probabilities = probabilities.reshape(-1, probabilities.shape[-1])
        flat_indices = indices.reshape(-1, indices.shape[-1])
        selected = torch.multinomial(flat_probabilities, num_samples=1)
        return torch.gather(flat_indices, -1, selected).reshape(indices.shape[:-1])

    def _future_proposals(
        self,
        state: TransformerSemanticState,
        outputs: Mapping[str, torch.Tensor],
        row: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities, indices = _filtered_topk_distribution(
            outputs["mtp_topk_values"][:, row, :, :],
            outputs["mtp_topk_indices"][:, row, :, :],
            temperature=state.temperature,
            top_k=state.top_k,
            top_p=state.top_p,
        )
        drafts = self._sample_distribution(probabilities, indices).to(torch.int64)
        return drafts.contiguous(), probabilities, indices

    def _target_rows(
        self,
        state: TransformerSemanticState,
        outputs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _filtered_topk_distribution(
            outputs["base_topk_values"][:, :3, :],
            outputs["base_topk_indices"][:, :3, :],
            temperature=state.temperature,
            top_k=state.top_k,
            top_p=state.top_p,
        )

    @staticmethod
    def _token_probability(
        probabilities: torch.Tensor,
        indices: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        return (
            probabilities
            * (indices == tokens.unsqueeze(-1)).to(probabilities.dtype)
        ).sum(dim=-1)

    def _accepted_mask(
        self,
        *,
        target_probabilities: torch.Tensor,
        target_indices: torch.Tensor,
        draft_probabilities: torch.Tensor,
        draft_indices: torch.Tensor,
        drafts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p_draft = self._token_probability(
            target_probabilities, target_indices, drafts
        )
        q_draft = self._token_probability(
            draft_probabilities, draft_indices, drafts
        ).clamp_min(1e-12)
        ratios = (p_draft / q_draft).clamp(max=1.0)
        return torch.rand_like(ratios) <= ratios, ratios

    def _sample_residual(
        self,
        *,
        target_probabilities: torch.Tensor,
        target_indices: torch.Tensor,
        draft_probabilities: torch.Tensor,
        draft_indices: torch.Tensor,
    ) -> torch.Tensor:
        self._p_dense.zero_()
        self._q_dense.zero_()
        self._p_dense.scatter_add_(
            1,
            target_indices.reshape(1, -1),
            target_probabilities.reshape(1, -1).to(self._p_dense.dtype),
        )
        self._q_dense.scatter_add_(
            1,
            draft_indices.reshape(1, -1),
            draft_probabilities.reshape(1, -1).to(self._q_dense.dtype),
        )
        residual = (self._p_dense - self._q_dense).clamp_min_(0)
        residual_sum = residual.sum(dim=-1, keepdim=True)
        fallback = self._p_dense / self._p_dense.sum(dim=-1, keepdim=True)
        normalized = torch.where(
            residual_sum > 1e-12,
            residual / residual_sum.clamp_min(1e-12),
            fallback,
        )
        return torch.multinomial(normalized, num_samples=1).to(torch.int64)

    def iter_batches(
        self,
        state: TransformerSemanticState,
        *,
        sync_policy: Callable[[int, int, int], bool],
        cancelled: Any | None = None,
    ) -> Iterator[SemanticBatch]:
        pending: list[torch.Tensor] = []
        generated = 0
        last_reported_nfe = 0
        decode_started = time.perf_counter()

        def make_batch(*, eos: bool) -> SemanticBatch:
            nonlocal pending, last_reported_nfe
            tokens = (
                torch.cat(pending, dim=1)
                if pending
                else torch.empty(
                    (1, 0), dtype=torch.int64, device=self.engine.device
                )
            )
            count = int(tokens.shape[1])
            batch = SemanticBatch(
                tokens=tokens,
                eos=eos,
                proposed_tokens=count,
                accepted_tokens=count,
                nfe=state.steps - last_reported_nfe,
            )
            pending = []
            last_reported_nfe = state.steps
            if state.first_batch_seconds == 0.0:
                state.first_batch_seconds = time.perf_counter() - decode_started
            return batch

        def should_emit() -> bool:
            return bool(
                pending
                and sync_policy(
                    len(pending), max(0, generated - 1), state.maximum_steps
                )
            )

        def append_token(token: torch.Tensor, value: int) -> bool:
            nonlocal generated
            if value == EOS_TOKEN:
                return False
            pending.append(token)
            generated += 1
            return True

        current_pair = 0
        h1_outputs, current_pair = self._run(
            self.h1,
            1,
            state,
            state.first_token,
            current_pair,
            1 - current_pair,
            0,
        )
        next_token = self._sample(state, *self._base_row(h1_outputs, 0))
        drafts, q_probabilities, q_indices = self._future_proposals(
            state, h1_outputs, 0
        )
        index = 1

        while generated < state.maximum_steps:
            if cancelled is not None and cancelled.is_set():
                return
            next_value = self._control_values(state, next_token)[0]
            if not append_token(next_token, next_value):
                yield make_batch(eos=True)
                return
            if should_emit():
                yield make_batch(eos=False)
            if generated >= state.maximum_steps:
                break

            if state.maximum_steps - generated < 3:
                h1_outputs, current_pair = self._run(
                    self.h1,
                    1,
                    state,
                    next_token,
                    current_pair,
                    1 - current_pair,
                    index,
                )
                next_token = self._sample(state, *self._base_row(h1_outputs, 0))
                drafts, q_probabilities, q_indices = self._future_proposals(
                    state, h1_outputs, 0
                )
                index += 1
                continue

            proposals = torch.cat((next_token, drafts), dim=1)
            h4_outputs, block_pair = self._run(
                self.h4,
                4,
                state,
                proposals,
                current_pair,
                1 - current_pair,
                index,
            )
            p_probabilities, p_indices = self._target_rows(state, h4_outputs)
            accepted_mask, ratios = self._accepted_mask(
                target_probabilities=p_probabilities,
                target_indices=p_indices,
                draft_probabilities=q_probabilities,
                draft_indices=q_indices,
                drafts=drafts,
            )
            control = self._tensor_values(
                state,
                torch.cat(
                    (
                        accepted_mask.to(torch.int64),
                        drafts,
                    ),
                    dim=1,
                ),
            )
            accepted_flags = control[:3]
            draft_values = control[3:]
            accepted_prefix = 0
            while accepted_prefix < 3 and accepted_flags[accepted_prefix]:
                accepted_prefix += 1

            state.mtp_proposed_tokens += 3
            state.mtp_accepted_tokens += accepted_prefix
            if accepted_prefix < 3:
                state.mtp_rejected_tokens += 3 - accepted_prefix
            if state.mtp_trace is not None:
                state.mtp_trace.append(
                    {
                        "index": index,
                        "drafts": draft_values,
                        "accepted_prefix": accepted_prefix,
                        "acceptance_ratios": [
                            float(value)
                            for value in ratios.detach().cpu().reshape(-1).tolist()
                        ],
                    }
                )

            correction: torch.Tensor | None = None
            if accepted_prefix < 3:
                correction = self._sample_residual(
                    target_probabilities=p_probabilities[
                        :, accepted_prefix, :
                    ],
                    target_indices=p_indices[:, accepted_prefix, :],
                    draft_probabilities=q_probabilities[
                        :, accepted_prefix, :
                    ],
                    draft_indices=q_indices[:, accepted_prefix, :],
                )
                correction_value = self._control_values(state, correction)[0]
            else:
                correction_value = -1

            commit_count, correction_index, next_index = _mtp4_cycle_plan(
                index=index, accepted_prefix=accepted_prefix
            )
            for offset in range(commit_count):
                if offset < accepted_prefix:
                    token = drafts[:, offset : offset + 1]
                    value = draft_values[offset]
                else:
                    if correction is None:
                        raise RuntimeError("MTP-4 rejection correction is missing")
                    token = correction
                    value = correction_value
                if not append_token(token, value):
                    yield make_batch(eos=True)
                    return
                if generated >= state.maximum_steps:
                    break
            if generated >= state.maximum_steps:
                break

            if accepted_prefix == 3:
                current_pair = block_pair
                next_token = self._sample(state, *self._base_row(h4_outputs, 3))
                drafts, q_probabilities, q_indices = self._future_proposals(
                    state, h4_outputs, 3
                )
                index = next_index
            else:
                if correction is None or correction_index is None:
                    raise RuntimeError("MTP-4 rejection repair state is incomplete")
                h1_outputs, current_pair = self._run(
                    self.h1,
                    1,
                    state,
                    correction,
                    block_pair,
                    current_pair,
                    correction_index,
                )
                next_token = self._sample(
                    state, *self._base_row(h1_outputs, 0)
                )
                drafts, q_probabilities, q_indices = self._future_proposals(
                    state, h1_outputs, 0
                )
                index = next_index
            if should_emit():
                yield make_batch(eos=False)

        if pending:
            yield make_batch(eos=False)

    def close(self) -> None:
        super().close()
        self._p_dense = torch.empty(0, device=self.engine.device)
        self._q_dense = torch.empty(0, device=self.engine.device)


class FittedH1SemanticRuntime:
    """Single-token fitted engine used to isolate block decoding numerics."""

    backend_name = "transformer-fitted-h1"

    def __init__(
        self,
        *,
        base_runtime: TransformerSemanticRuntime,
        h1_engine: Path,
        sample_topk: Any,
        engine: Any,
        logger: logging.Logger | None = None,
    ) -> None:
        self.base_runtime = base_runtime
        self.sample_topk = sample_topk
        self.engine = engine
        self.logger = logger or logging.getLogger(__name__)
        self.h1 = TensorRTRunner(
            h1_engine,
            device=engine.device,
            logger=DetailedTensorRTLogger(echo=False),
        )
        expected_capacity = int(
            engine.model_gpt_step.engine.get_tensor_shape("k_cache")[2]
        )
        capacity = int(self.h1.engine.get_tensor_shape("k_cache")[2])
        if capacity != expected_capacity:
            raise RuntimeError(
                f"Fitted H1 cache capacity {capacity} does not match "
                f"the active GPT step capacity {expected_capacity}"
            )
        self._outputs = _output_buffers(self.h1, torch.device(engine.device))

    def prepare(
        self, inputs: Mapping[str, Any], **options: Any
    ) -> TransformerSemanticState:
        state = self.base_runtime.prepare(inputs, **options)
        state.sample_cuda_graph_enabled = False
        return state

    def _sample(
        self,
        state: TransformerSemanticState,
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.sample_topk(
            values,
            indices,
            temperature=state.temperature,
            top_k=state.top_k,
            top_p=state.top_p,
        ).to(self.engine.device)

    def iter_batches(
        self,
        state: TransformerSemanticState,
        *,
        sync_policy: Callable[[int, int, int], bool],
        cancelled: Any | None = None,
    ) -> Iterator[SemanticBatch]:
        pending: list[torch.Tensor] = []
        current = state.first_token
        current_pair = 0
        last_reported_nfe = 0
        decode_started = time.perf_counter()

        for step in range(state.maximum_steps):
            if cancelled is not None and cancelled.is_set():
                return
            destination_pair = 1 - current_pair
            self._outputs["k_cache_new"] = state.cache_pair[destination_pair][0]
            self._outputs["v_cache_new"] = state.cache_pair[destination_pair][1]
            self.h1.infer(
                {
                    "samples": current.to(torch.int64),
                    "k_cache": state.cache_pair[current_pair][0],
                    "v_cache": state.cache_pair[current_pair][1],
                    "x_len": state.x_length.to(self.engine.device),
                    "y_len": state.y_length.to(self.engine.device),
                    "idx": torch.tensor(
                        [step], dtype=torch.int64, device=self.engine.device
                    ),
                },
                outputs=self._outputs,
                stream=self.engine.stream,
                synchronize=False,
                profile=False,
            )
            current_pair = destination_pair
            current = self._sample(
                state,
                self._outputs["base_topk_values"][:, 0, :],
                self._outputs["base_topk_indices"][:, 0, :],
            )
            state.steps += 1
            pending.append(current)
            if not sync_policy(len(pending), step, state.maximum_steps):
                continue

            pending_batch = torch.cat(pending, dim=1)
            sync_started = time.perf_counter() if state.detailed_profile else 0.0
            pending_values = pending_batch.detach().cpu().reshape(-1).tolist()
            if state.detailed_profile:
                state.host_sync_seconds += time.perf_counter() - sync_started
            state.host_sync_count += 1
            if state.first_batch_seconds == 0.0:
                state.first_batch_seconds = time.perf_counter() - decode_started
            eos_offset = next(
                (
                    offset
                    for offset, value in enumerate(pending_values)
                    if int(value) == EOS_TOKEN
                ),
                None,
            )
            accepted_count = len(pending) if eos_offset is None else eos_offset
            yield SemanticBatch(
                tokens=pending_batch[:, :accepted_count],
                eos=eos_offset is not None,
                proposed_tokens=len(pending),
                accepted_tokens=accepted_count,
                nfe=state.steps - last_reported_nfe,
            )
            last_reported_nfe = state.steps
            pending.clear()
            if eos_offset is not None:
                return

    def close(self) -> None:
        self.base_runtime.close()
        self._outputs.clear()


class MTPDirectSemanticRuntime(MTPSpeculativeSemanticRuntime):
    """Direct two-token decoding used only behind an explicit research backend."""

    backend_name = "transformer-mtp2-direct"

    def iter_batches(
        self,
        state: TransformerSemanticState,
        *,
        sync_policy: Callable[[int, int, int], bool],
        cancelled: Any | None = None,
    ) -> Iterator[SemanticBatch]:
        pending: list[torch.Tensor] = []
        generated = 0
        last_reported_nfe = 0
        decode_started = time.perf_counter()

        def make_batch() -> SemanticBatch:
            nonlocal pending, last_reported_nfe
            tokens = torch.cat(pending, dim=1)
            sync_started = time.perf_counter() if state.detailed_profile else 0.0
            values = tokens.detach().cpu().reshape(-1).tolist()
            if state.detailed_profile:
                state.host_sync_seconds += time.perf_counter() - sync_started
            state.host_sync_count += 1
            if state.first_batch_seconds == 0.0:
                state.first_batch_seconds = time.perf_counter() - decode_started
            eos_offset = next(
                (index for index, value in enumerate(values) if int(value) == EOS_TOKEN),
                None,
            )
            accepted_count = len(values) if eos_offset is None else eos_offset
            batch = SemanticBatch(
                tokens=tokens[:, :accepted_count],
                eos=eos_offset is not None,
                proposed_tokens=len(values),
                accepted_tokens=accepted_count,
                nfe=state.steps - last_reported_nfe,
            )
            pending = []
            last_reported_nfe = state.steps
            return batch

        def append_pair(first: torch.Tensor, second: torch.Tensor) -> None:
            nonlocal generated
            pending.append(first)
            generated += 1
            if generated < state.maximum_steps:
                pending.append(second)
                generated += 1
                state.mtp_proposed_tokens += 1
                state.mtp_accepted_tokens += 1

        current_pair = 0
        h1_outputs, current_pair = self._run(
            self.h1,
            1,
            state,
            state.first_token,
            current_pair,
            1 - current_pair,
            0,
        )
        first = self._sample(state, *self._base_row(h1_outputs, 0))
        second = self._future_top1(h1_outputs, 0)
        append_pair(first, second)
        index = 1

        while pending and generated <= state.maximum_steps:
            if cancelled is not None and cancelled.is_set():
                return
            if sync_policy(len(pending), max(0, generated - 1), state.maximum_steps):
                batch = make_batch()
                yield batch
                if batch.eos or generated >= state.maximum_steps:
                    return
            proposals = torch.cat((first, second), dim=1)
            h2_outputs, current_pair = self._run(
                self.h2,
                2,
                state,
                proposals,
                current_pair,
                1 - current_pair,
                index,
            )
            first = self._sample(state, *self._base_row(h2_outputs, 1))
            second = self._future_top1(h2_outputs, 1)
            append_pair(first, second)
            index += 2

        if pending:
            yield make_batch()


__all__ = [
    "FittedH1SemanticRuntime",
    "MTPDirectSemanticRuntime",
    "MTP4SpeculativeSemanticRuntime",
    "MTP4RejectionSemanticRuntime",
    "MTPSpeculativeSemanticRuntime",
]
