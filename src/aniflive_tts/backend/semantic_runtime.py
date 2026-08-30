from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Protocol


@dataclass(frozen=True)
class SemanticBatch:
    tokens: Any
    eos: bool
    proposed_tokens: int
    accepted_tokens: int
    nfe: int


class SemanticRuntime(Protocol):
    backend_name: str

    def prepare(
        self, inputs: Mapping[str, Any], **options: Any
    ) -> "TransformerSemanticState": ...

    def iter_batches(
        self,
        state: "TransformerSemanticState",
        *,
        sync_policy: Callable[[int, int, int], bool],
        cancelled: Any | None = None,
    ) -> Iterator[SemanticBatch]: ...

    def close(self) -> None: ...


def _apply_repetition_penalty_to_topk(
    torch: Any,
    topk_values: Any,
    topk_indices: Any,
    *,
    seen_token_mask: Any | None,
    repetition_penalty: float,
) -> tuple[Any, Any]:
    values = topk_values.detach()
    indices = topk_indices.detach()
    if seen_token_mask is None or repetition_penalty == 1.0:
        return values, indices

    repeated = seen_token_mask.index_select(
        0, indices.reshape(-1).to(torch.int64)
    ).reshape_as(indices)
    penalized = torch.where(
        values < 0,
        values * repetition_penalty,
        values / repetition_penalty,
    )
    values = torch.where(repeated, penalized, values)
    values, order = torch.sort(values, dim=-1, descending=True)
    indices = torch.gather(indices, dim=-1, index=order)
    return values, indices


def _suppress_eos_in_topk(
    torch: Any,
    topk_values: Any,
    topk_indices: Any,
    *,
    eos_token: int = 1024,
) -> tuple[Any, Any]:
    """Keep EOS out of sampling without moving candidates off the GPU."""
    values = topk_values.detach().masked_fill(
        topk_indices.detach() == int(eos_token),
        float("-inf"),
    )
    values, order = torch.sort(values, dim=-1, descending=True)
    indices = torch.gather(topk_indices.detach(), dim=-1, index=order)
    return values, indices


def _resolve_minimum_semantic_tokens(repetition_penalty: float) -> int:
    configured = os.environ.get("ANIFLIVE_TTS_MIN_SEMANTIC_TOKENS")
    minimum = (
        int(configured)
        if configured is not None
        else (11 if repetition_penalty != 1.0 else 0)
    )
    if minimum < 0:
        raise ValueError("ANIFLIVE_TTS_MIN_SEMANTIC_TOKENS must be non-negative")
    return min(minimum, 1000)


class _SampleCudaGraph:
    def __init__(
        self,
        *,
        torch: Any,
        sample_topk: Any,
        stream: Any,
        topk_values: Any,
        topk_indices: Any,
        token_capacity: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> None:
        self.torch = torch
        self.stream = stream
        self.token_capacity = int(token_capacity)
        self.parameters = (float(temperature), int(top_k), float(top_p))
        self.input_addresses = (topk_values.data_ptr(), topk_indices.data_ptr())
        self.token_storage = torch.empty(
            (self.token_capacity, 1),
            dtype=topk_indices.dtype,
            device=topk_values.device,
        )

        stream.synchronize()
        rng_state = torch.cuda.get_rng_state(device=topk_values.device)
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                sampled = sample_topk(
                    topk_values,
                    topk_indices,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
        finally:
            torch.cuda.set_rng_state(rng_state, device=topk_values.device)
        self.graph = graph
        self.sampled = sampled
        stream.synchronize()

    def matches(
        self,
        *,
        topk_values: Any,
        topk_indices: Any,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> bool:
        return self.input_addresses == (
            topk_values.data_ptr(),
            topk_indices.data_ptr(),
        ) and self.parameters == (float(temperature), int(top_k), float(top_p))

    def replay(self, step: int) -> tuple[Any, Any]:
        self.graph.replay()
        stored = self.token_storage[step : step + 1].reshape(1, 1)
        stored.copy_(self.sampled)
        return self.sampled, stored


class _PersistentGPTStepContexts:
    """Pre-bind the two ping-pong GPT contexts without changing model math."""

    def __init__(
        self,
        *,
        torch: Any,
        trt: Any,
        model: Any,
        stream: Any,
        cache_pair: list[tuple[Any, Any]],
        x_length: Any,
        y_length: Any,
    ) -> None:
        self.stream = stream
        self.cache_addresses = tuple(
            tensor.data_ptr() for pair in cache_pair for tensor in pair
        )
        self.length_addresses = (x_length.data_ptr(), y_length.data_ptr())
        self.contexts: list[Any] = []
        self.sample_addresses: list[int | None] = [None, None]

        output_device = {
            name: (
                "cpu"
                if model.tensor_location.get(name, trt.TensorLocation.DEVICE)
                == trt.TensorLocation.HOST
                else model.device
            )
            for name in ("topk_values", "topk_indices")
        }
        self.topk_values = torch.empty(
            tuple(model.engine.get_tensor_shape("topk_values")),
            dtype=model.tensor_dtype["topk_values"],
            device=output_device["topk_values"],
        )
        self.topk_indices = torch.empty(
            tuple(model.engine.get_tensor_shape("topk_indices")),
            dtype=model.tensor_dtype["topk_indices"],
            device=output_device["topk_indices"],
        )
        self.outputs = {
            "topk_values": self.topk_values,
            "topk_indices": self.topk_indices,
        }

        for parity in range(2):
            source_cache = cache_pair[parity]
            destination_cache = cache_pair[(parity + 1) % 2]
            context = model.engine.create_execution_context()
            if context is None:
                raise RuntimeError(
                    "TensorRT could not create a persistent GPT execution context"
                )
            fixed_bindings = {
                "k_cache": source_cache[0],
                "v_cache": source_cache[1],
                "x_len": x_length,
                "y_len": y_length,
                "topk_values": self.topk_values,
                "topk_indices": self.topk_indices,
                "k_cache_new": destination_cache[0],
                "v_cache_new": destination_cache[1],
            }
            for name, tensor in fixed_bindings.items():
                if not context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(
                        f"TensorRT rejected persistent GPT binding {name}"
                    )
            self.contexts.append(context)

    def matches(
        self,
        *,
        cache_pair: list[tuple[Any, Any]],
        x_length: Any,
        y_length: Any,
    ) -> bool:
        return self.cache_addresses == tuple(
            tensor.data_ptr() for pair in cache_pair for tensor in pair
        ) and self.length_addresses == (x_length.data_ptr(), y_length.data_ptr())

    def execute(self, *, step: int, current: Any, index: Any) -> dict[str, Any]:
        parity = step % 2
        context = self.contexts[parity]
        sample_address = int(current.data_ptr())
        if self.sample_addresses[parity] != sample_address:
            if not context.set_tensor_address("samples", sample_address):
                raise RuntimeError("TensorRT rejected persistent GPT sample binding")
            self.sample_addresses[parity] = sample_address
        if not context.set_tensor_address("idx", index.data_ptr()):
            raise RuntimeError("TensorRT rejected persistent GPT index binding")
        if not context.execute_async_v3(stream_handle=self.stream.cuda_stream):
            raise RuntimeError(
                "TensorRT persistent GPT execute_async_v3 returned false"
            )
        return self.outputs


@dataclass
class TransformerSemanticState:
    current: Any
    first_token: Any
    cache_pair: list[tuple[Any, Any]]
    x_length: Any
    y_length: Any
    indices: Any
    outputs: dict[str, Any]
    maximum_steps: int
    encoded_lengths: tuple[int, int]
    encoder_seconds: float
    attention_kv_bytes: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    minimum_semantic_tokens: int
    seen_token_mask: Any | None
    detailed_profile: bool
    steps: int = 0
    host_sync_count: int = 0
    host_sync_seconds: float = 0.0
    first_batch_seconds: float = 0.0
    sample_cuda_graph_cache_hits: int = 0
    sample_cuda_graph_captures: int = 0
    sample_cuda_graph_enabled: bool = False
    persistent_step_contexts: Any | None = None


class TransformerSemanticRuntime:
    """The v1.1 Transformer AR path behind a versioned semantic boundary."""

    backend_name = "transformer"
    use_ping_pong_cache = True

    def __init__(
        self,
        *,
        engine: Any,
        sample_topk: Any,
        torch: Any,
        trt: Any,
        logger: logging.Logger | None = None,
    ) -> None:
        self.engine = engine
        self.sample_topk = sample_topk
        self.torch = torch
        self.trt = trt
        self.logger = logger or logging.getLogger(__name__)
        self._destination_cache: tuple[Any, Any] | None = None
        self._sample_graphs: OrderedDict[tuple[Any, ...], _SampleCudaGraph] = (
            OrderedDict()
        )
        self._sample_graph_disabled = False
        self._full_graph_notice_emitted = False
        self._persistent_step_contexts: _PersistentGPTStepContexts | None = None
        self._seen_token_mask: Any | None = None

    def _prepare_seen_token_mask(self, repetition_penalty: float) -> Any | None:
        if repetition_penalty == 1.0:
            return None
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")
        if (
            self._seen_token_mask is None
            or int(self._seen_token_mask.numel()) != 1025
            or self._seen_token_mask.device != self.engine.device
        ):
            self._seen_token_mask = self.torch.zeros(
                1025,
                dtype=self.torch.bool,
                device=self.engine.device,
            )
        else:
            self._seen_token_mask.zero_()
        return self._seen_token_mask

    def _sample_candidates(
        self,
        topk_values: Any,
        topk_indices: Any,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seen_token_mask: Any | None,
        suppress_eos: bool = False,
    ) -> Any:
        values, indices = _apply_repetition_penalty_to_topk(
            self.torch,
            topk_values,
            topk_indices,
            seen_token_mask=seen_token_mask,
            repetition_penalty=repetition_penalty,
        )
        if suppress_eos:
            values, indices = _suppress_eos_in_topk(
                self.torch,
                values,
                indices,
            )
        sampled = self.sample_topk(
            values,
            indices,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ).to(self.engine.device)
        if seen_token_mask is not None:
            seen_token_mask.scatter_(
                0,
                sampled.reshape(-1).to(self.torch.int64),
                True,
            )
        return sampled

    def _prepare_step_input(self, name: str, tensor: Any) -> Any:
        location = self.engine.model_gpt_step.tensor_location.get(
            name, self.trt.TensorLocation.DEVICE
        )
        if location == self.trt.TensorLocation.HOST:
            return tensor.detach().cpu().to(self.torch.int64)
        return tensor.detach().to(self.engine.device).to(self.torch.int64)

    def prepare(
        self,
        inputs: Mapping[str, Any],
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float | None = None,
        minimum_semantic_tokens: int | None = None,
        detailed_profile: bool = False,
    ) -> TransformerSemanticState:
        if repetition_penalty is None:
            repetition_penalty = float(
                os.environ.get("ANIFLIVE_TTS_REPETITION_PENALTY", "1.0")
            )
        if minimum_semantic_tokens is None:
            minimum_semantic_tokens = _resolve_minimum_semantic_tokens(
                repetition_penalty
            )
        elif minimum_semantic_tokens < 0:
            raise ValueError("minimum_semantic_tokens must be non-negative")
        minimum_semantic_tokens = min(int(minimum_semantic_tokens), 1000)
        started = time.perf_counter()
        encoded = self.engine.model_gpt_enc(dict(inputs))
        encoder_seconds = time.perf_counter() - started
        seen_token_mask = self._prepare_seen_token_mask(repetition_penalty)
        current = self._sample_candidates(
            encoded["topk_values"],
            encoded["topk_indices"],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seen_token_mask=seen_token_mask,
            suppress_eos=minimum_semantic_tokens > 0,
        )

        k_cache = encoded["k_cache"]
        v_cache = encoded["v_cache"]
        encoded_lengths_list = (
            self.torch.stack(
                (encoded["x_len"].reshape(-1)[0], encoded["y_len"].reshape(-1)[0])
            )
            .detach()
            .cpu()
            .tolist()
        )
        encoded_lengths = (int(encoded_lengths_list[0]), int(encoded_lengths_list[1]))
        maximum_steps = max(
            1,
            min(
                1000,
                int(k_cache.shape[2]) - encoded_lengths[0] - encoded_lengths[1] - 1,
            ),
        )
        x_length = self._prepare_step_input("x_len", encoded["x_len"])
        y_length = self._prepare_step_input("y_len", encoded["y_len"])
        step_cache_shape = tuple(
            int(value)
            for value in self.engine.model_gpt_step.engine.get_tensor_shape("k_cache")
        )
        step_cache_capacity = int(step_cache_shape[2])
        required_cache = encoded_lengths[0] + encoded_lengths[1] + 2
        if required_cache > step_cache_capacity:
            raise RuntimeError(
                "GPT step cache profile is too short for this segment: "
                f"requires {required_cache}, engine capacity {step_cache_capacity}"
            )
        maximum_steps = max(
            1,
            min(
                maximum_steps,
                step_cache_capacity - encoded_lengths[0] - encoded_lengths[1] - 1,
            ),
        )
        if int(k_cache.shape[2]) != step_cache_capacity:
            k_cache = k_cache[:, :, :step_cache_capacity, :].contiguous()
            v_cache = v_cache[:, :, :step_cache_capacity, :].contiguous()
        if self.use_ping_pong_cache:
            if (
                self._destination_cache is None
                or self._destination_cache[0].shape != k_cache.shape
                or self._destination_cache[0].dtype != k_cache.dtype
            ):
                self._destination_cache = (
                    self.torch.empty_like(k_cache),
                    self.torch.empty_like(v_cache),
                )
            cache_pair = [(k_cache, v_cache), self._destination_cache]
        else:
            cache_pair = [(k_cache, v_cache)]
        index_location = self.engine.model_gpt_step.tensor_location.get(
            "idx", self.trt.TensorLocation.DEVICE
        )
        index_device = (
            "cpu"
            if index_location == self.trt.TensorLocation.HOST
            else self.engine.device
        )
        indices = self.torch.arange(
            maximum_steps, dtype=self.torch.int64, device=index_device
        )
        if (
            os.environ.get("ANIFLIVE_TTS_CUDA_GRAPH", "0").strip().lower()
            not in {"0", "false", "no", "off"}
            and not self._full_graph_notice_emitted
        ):
            self.logger.warning(
                "Full GPT TensorRT CUDA Graph is disabled: TensorRT's internal "
                "train-station operation rejects stream capture; sampling-only CUDA Graph "
                "remains active"
            )
            self._full_graph_notice_emitted = True
        attention_kv_bytes = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for pair in cache_pair
            for tensor in pair
        )
        persistent_contexts_requested = os.environ.get(
            "ANIFLIVE_TTS_PERSISTENT_GPT_CONTEXTS", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}
        persistent_step_contexts = None
        if persistent_contexts_requested and self.use_ping_pong_cache:
            if (
                self._persistent_step_contexts is None
                or not self._persistent_step_contexts.matches(
                    cache_pair=cache_pair,
                    x_length=x_length,
                    y_length=y_length,
                )
            ):
                self._persistent_step_contexts = _PersistentGPTStepContexts(
                    torch=self.torch,
                    trt=self.trt,
                    model=self.engine.model_gpt_step,
                    stream=self.engine.stream,
                    cache_pair=cache_pair,
                    x_length=x_length,
                    y_length=y_length,
                )
            persistent_step_contexts = self._persistent_step_contexts
        return TransformerSemanticState(
            current=current,
            first_token=current,
            cache_pair=cache_pair,
            x_length=x_length,
            y_length=y_length,
            indices=indices,
            outputs={"k_cache_new": None, "v_cache_new": None},
            maximum_steps=maximum_steps,
            encoded_lengths=encoded_lengths,
            encoder_seconds=encoder_seconds,
            attention_kv_bytes=attention_kv_bytes,
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            minimum_semantic_tokens=minimum_semantic_tokens,
            seen_token_mask=seen_token_mask,
            detailed_profile=bool(detailed_profile),
            persistent_step_contexts=persistent_step_contexts,
        )

    def _execute_step(
        self, state: TransformerSemanticState, step: int
    ) -> dict[str, Any]:
        source_cache = state.cache_pair[step % 2]
        destination_cache = state.cache_pair[(step + 1) % 2]
        state.outputs["k_cache_new"], state.outputs["v_cache_new"] = destination_cache
        if state.persistent_step_contexts is None:
            return self.engine.model_gpt_step(
                {
                    "samples": state.current.to(self.torch.int64),
                    "k_cache": source_cache[0],
                    "v_cache": source_cache[1],
                    "idx": state.indices[step : step + 1],
                    "x_len": state.x_length,
                    "y_len": state.y_length,
                },
                outputs=state.outputs,
                sync=False,
            )
        return state.persistent_step_contexts.execute(
            step=step,
            current=state.current,
            index=state.indices[step : step + 1],
        )

    def iter_batches(
        self,
        state: TransformerSemanticState,
        *,
        sync_policy: Callable[[int, int, int], bool],
        cancelled: Any | None = None,
    ) -> Iterator[SemanticBatch]:
        pending_tokens: list[Any] = []
        pending_storage_start: int | None = None
        sample_graph: _SampleCudaGraph | None = None
        decode_started = time.perf_counter()
        sample_graph_requested = (
            os.environ.get("ANIFLIVE_TTS_SAMPLE_CUDA_GRAPH", "1").strip().lower()
            not in {
                "0",
                "false",
                "no",
                "off",
            }
            and state.repetition_penalty == 1.0
            and state.minimum_semantic_tokens == 0
        )
        try:
            for step in range(state.maximum_steps):
                if cancelled is not None and cancelled.is_set():
                    return
                decoded = self._execute_step(state, step)
                if sample_graph_requested and not self._sample_graph_disabled:
                    try:
                        if sample_graph is None:
                            graph_key = (
                                int(decoded["topk_values"].data_ptr()),
                                int(decoded["topk_indices"].data_ptr()),
                                state.temperature,
                                state.top_k,
                                state.top_p,
                            )
                            sample_graph = self._sample_graphs.get(graph_key)
                            if sample_graph is not None and sample_graph.matches(
                                topk_values=decoded["topk_values"],
                                topk_indices=decoded["topk_indices"],
                                temperature=state.temperature,
                                top_k=state.top_k,
                                top_p=state.top_p,
                            ):
                                self._sample_graphs.move_to_end(graph_key)
                                state.sample_cuda_graph_cache_hits += 1
                            else:
                                sample_graph = _SampleCudaGraph(
                                    torch=self.torch,
                                    sample_topk=self.sample_topk,
                                    stream=self.engine.stream,
                                    topk_values=decoded["topk_values"],
                                    topk_indices=decoded["topk_indices"],
                                    token_capacity=1000,
                                    temperature=state.temperature,
                                    top_k=state.top_k,
                                    top_p=state.top_p,
                                )
                                self._sample_graphs[graph_key] = sample_graph
                                state.sample_cuda_graph_captures += 1
                                while len(self._sample_graphs) > 4:
                                    self._sample_graphs.popitem(last=False)
                        state.current, stored_current = sample_graph.replay(step)
                        if not pending_tokens:
                            pending_storage_start = step
                    except Exception as exc:
                        sample_graph = None
                        self._sample_graphs.clear()
                        self._sample_graph_disabled = True
                        self.logger.warning(
                            "Sampling CUDA Graph disabled after capture failure: %s",
                            exc,
                        )
                if (
                    not sample_graph_requested
                    or sample_graph is None
                    or self._sample_graph_disabled
                ):
                    pending_storage_start = None
                    state.current = self._sample_candidates(
                        decoded["topk_values"],
                        decoded["topk_indices"],
                        temperature=state.temperature,
                        top_k=state.top_k,
                        top_p=state.top_p,
                        repetition_penalty=state.repetition_penalty,
                        seen_token_mask=state.seen_token_mask,
                        suppress_eos=(
                            1 + state.steps < state.minimum_semantic_tokens
                        ),
                    )
                    stored_current = state.current
                state.steps += 1
                pending_tokens.append(stored_current)
                if not sync_policy(len(pending_tokens), step, state.maximum_steps):
                    continue

                if pending_storage_start is not None and sample_graph is not None:
                    pending_batch = sample_graph.token_storage[
                        pending_storage_start : step + 1
                    ].reshape(1, -1)
                else:
                    pending_batch = self.torch.cat(pending_tokens, dim=1)
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
                        if int(value) == 1024
                    ),
                    None,
                )
                accepted_count = (
                    len(pending_tokens) if eos_offset is None else eos_offset
                )
                yield SemanticBatch(
                    tokens=pending_batch[:, :accepted_count],
                    eos=eos_offset is not None,
                    proposed_tokens=len(pending_tokens),
                    accepted_tokens=accepted_count,
                    nfe=len(pending_tokens),
                )
                pending_tokens.clear()
                pending_storage_start = None
                if eos_offset is not None:
                    break
        finally:
            state.sample_cuda_graph_enabled = bool(
                sample_graph is not None and not self._sample_graph_disabled
            )

    def close(self) -> None:
        self._sample_graphs.clear()
        self._destination_cache = None
        self._persistent_step_contexts = None
        self._seen_token_mask = None


__all__ = [
    "SemanticBatch",
    "SemanticRuntime",
    "TransformerSemanticRuntime",
    "TransformerSemanticState",
]
