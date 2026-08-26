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

    def prepare(self, inputs: Mapping[str, Any], **options: Any) -> "TransformerSemanticState": ...

    def iter_batches(
        self,
        state: "TransformerSemanticState",
        *,
        sync_policy: Callable[[int, int, int], bool],
        cancelled: Any | None = None,
    ) -> Iterator[SemanticBatch]: ...

    def close(self) -> None: ...


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
            (self.token_capacity, 1), dtype=topk_indices.dtype, device=topk_values.device
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
        return (
            self.input_addresses == (topk_values.data_ptr(), topk_indices.data_ptr())
            and self.parameters == (float(temperature), int(top_k), float(top_p))
        )

    def replay(self, step: int) -> tuple[Any, Any]:
        self.graph.replay()
        stored = self.token_storage[step : step + 1].reshape(1, 1)
        stored.copy_(self.sampled)
        return self.sampled, stored


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
    detailed_profile: bool
    steps: int = 0
    host_sync_count: int = 0
    host_sync_seconds: float = 0.0
    first_batch_seconds: float = 0.0
    sample_cuda_graph_cache_hits: int = 0
    sample_cuda_graph_captures: int = 0
    sample_cuda_graph_enabled: bool = False


class TransformerSemanticRuntime:
    """The v1.1 Transformer AR path behind a versioned semantic boundary."""

    backend_name = "transformer"

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
        self._sample_graphs: OrderedDict[tuple[Any, ...], _SampleCudaGraph] = OrderedDict()
        self._sample_graph_disabled = False
        self._full_graph_notice_emitted = False

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
        detailed_profile: bool = False,
    ) -> TransformerSemanticState:
        started = time.perf_counter()
        encoded = self.engine.model_gpt_enc(dict(inputs))
        encoder_seconds = time.perf_counter() - started
        current = self.sample_topk(
            encoded["topk_values"].detach(),
            encoded["topk_indices"].detach(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ).to(self.engine.device)

        k_cache = encoded["k_cache"]
        v_cache = encoded["v_cache"]
        encoded_lengths_list = self.torch.stack(
            (encoded["x_len"].reshape(-1)[0], encoded["y_len"].reshape(-1)[0])
        ).detach().cpu().tolist()
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
        if int(k_cache.shape[2]) != step_cache_capacity:
            k_cache = k_cache[:, :, :step_cache_capacity, :].contiguous()
            v_cache = v_cache[:, :, :step_cache_capacity, :].contiguous()
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
        index_location = self.engine.model_gpt_step.tensor_location.get(
            "idx", self.trt.TensorLocation.DEVICE
        )
        index_device = (
            "cpu" if index_location == self.trt.TensorLocation.HOST else self.engine.device
        )
        indices = self.torch.arange(maximum_steps, dtype=self.torch.int64, device=index_device)
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
            detailed_profile=bool(detailed_profile),
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
            not in {"0", "false", "no", "off"}
        )
        try:
            for step in range(state.maximum_steps):
                if cancelled is not None and cancelled.is_set():
                    return
                source_cache = state.cache_pair[step % 2]
                destination_cache = state.cache_pair[(step + 1) % 2]
                state.outputs["k_cache_new"], state.outputs["v_cache_new"] = destination_cache
                decoded = self.engine.model_gpt_step(
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
                            "Sampling CUDA Graph disabled after capture failure: %s", exc
                        )
                if (
                    not sample_graph_requested
                    or sample_graph is None
                    or self._sample_graph_disabled
                ):
                    pending_storage_start = None
                    state.current = self.sample_topk(
                    decoded["topk_values"].detach(),
                    decoded["topk_indices"].detach(),
                    temperature=state.temperature,
                    top_k=state.top_k,
                    top_p=state.top_p,
                    ).to(self.engine.device)
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
                    (offset for offset, value in enumerate(pending_values) if int(value) == 1024),
                    None,
                )
                accepted_count = len(pending_tokens) if eos_offset is None else eos_offset
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


__all__ = [
    "SemanticBatch",
    "SemanticRuntime",
    "TransformerSemanticRuntime",
    "TransformerSemanticState",
]
