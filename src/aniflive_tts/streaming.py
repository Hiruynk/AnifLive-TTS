"""Fixed-reference, low-latency TensorRT streaming for AnifLive-TTS.

The upstream ``api_server_trt.py`` demonstrates token streaming, but its
reference preparation still relies on non-TensorRT helper models and its WAV
response has no final RIFF data length.  This module keeps the complete exported
TensorRT engines in use, prepares the immutable voice reference once at
startup, and exposes PCM chunks to the HTTP facade.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import soundfile as sf
import soxr

from aniflive_tts.backend.semantic_runtime import TransformerSemanticRuntime


INITIAL_LEADING_SILENCE_RETAIN_SECONDS = 0.007
FOLLOWING_LEADING_SILENCE_RETAIN_SECONDS = 0.005
MIN_INITIAL_PREVIEW_PUBLISHED_SECONDS = 0.112
MAX_CACHED_PREVIEW_TOKENS = 32


@dataclass(frozen=True)
class PreparedReference:
    """GPU-resident features extracted from the fixed reference recording."""

    prompt_semantic: Any
    spectrogram: Any
    speaker_embedding: Any
    phones: list[int]
    bert: Any


@dataclass(frozen=True)
class _DecodedTail:
    audio: np.ndarray
    decode_seconds: float
    trailing_silence_seconds: float
    invocations: int


@dataclass(frozen=True)
class _PendingTail:
    future: Future[_DecodedTail]
    segment: str
    has_next: bool


class _GPTStepCudaGraphs:
    """Two fixed-address CUDA graphs for alternating GPT KV cache buffers."""

    def __init__(
        self,
        *,
        torch: Any,
        model: Any,
        stream: Any,
        cache_pair: Sequence[tuple[Any, Any]],
        sample: Any,
        x_length: Any,
        y_length: Any,
    ) -> None:
        self.torch = torch
        self.model = model
        self.stream = stream
        self.sample = torch.empty_like(sample)
        self.x_length = torch.empty_like(x_length)
        self.y_length = torch.empty_like(y_length)
        self.index = torch.zeros((1,), dtype=torch.int64, device=sample.device)
        self.sample.copy_(sample)
        self.x_length.copy_(x_length)
        self.y_length.copy_(y_length)
        self.contexts: list[Any] = []
        self.aux_streams: list[list[Any]] = []
        self.graphs: list[Any] = []
        self.outputs: list[dict[str, Any]] = []
        self.cache_addresses = tuple(tensor.data_ptr() for pair in cache_pair for tensor in pair)

        backup = (torch.empty_like(cache_pair[0][0]), torch.empty_like(cache_pair[0][1]))
        backup[0].copy_(cache_pair[0][0])
        backup[1].copy_(cache_pair[0][1])
        stream.synchronize()
        try:
            for parity in range(2):
                source_cache = cache_pair[parity]
                destination_cache = cache_pair[(parity + 1) % 2]
                context = model.engine.create_execution_context()
                if context is None:
                    raise RuntimeError("TensorRT could not create a CUDA Graph execution context")
                context_aux_streams = [
                    torch.cuda.Stream(device=sample.device)
                    for _ in range(int(model.engine.num_aux_streams))
                ]
                if context_aux_streams:
                    context.set_aux_streams(
                        [aux_stream.cuda_stream for aux_stream in context_aux_streams]
                    )
                topk_values = torch.empty(
                    tuple(model.engine.get_tensor_shape("topk_values")),
                    dtype=model.tensor_dtype["topk_values"],
                    device=sample.device,
                )
                topk_indices = torch.empty(
                    tuple(model.engine.get_tensor_shape("topk_indices")),
                    dtype=model.tensor_dtype["topk_indices"],
                    device=sample.device,
                )
                bindings = {
                    "samples": self.sample,
                    "k_cache": source_cache[0],
                    "v_cache": source_cache[1],
                    "x_len": self.x_length,
                    "y_len": self.y_length,
                    "idx": self.index,
                    "topk_values": topk_values,
                    "topk_indices": topk_indices,
                    "k_cache_new": destination_cache[0],
                    "v_cache_new": destination_cache[1],
                }
                for name, tensor in bindings.items():
                    if not context.set_tensor_address(name, tensor.data_ptr()):
                        raise RuntimeError(f"TensorRT rejected CUDA Graph binding {name}")

                # NVIDIA requires one uncaptured enqueue after context setup.
                if not context.execute_async_v3(stream_handle=stream.cuda_stream):
                    raise RuntimeError("TensorRT CUDA Graph warmup enqueue failed")
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    if not context.execute_async_v3(stream_handle=stream.cuda_stream):
                        raise RuntimeError("TensorRT CUDA Graph capture enqueue failed")
                    self.index.add_(1)
                self.contexts.append(context)
                self.aux_streams.append(context_aux_streams)
                self.graphs.append(graph)
                self.outputs.append(
                    {"topk_values": topk_values, "topk_indices": topk_indices}
                )
        finally:
            # Uncaptured warmup writes over both cache destinations. Restore
            # the encoder state even when graph construction fails midway.
            cache_pair[0][0].copy_(backup[0])
            cache_pair[0][1].copy_(backup[1])
            self.index.zero_()
            self.sample.copy_(sample)
            stream.synchronize()

    def matches(self, cache_pair: Sequence[tuple[Any, Any]]) -> bool:
        return self.cache_addresses == tuple(
            tensor.data_ptr() for pair in cache_pair for tensor in pair
        )

    def prepare(self, *, sample: Any, x_length: Any, y_length: Any) -> None:
        self.sample.copy_(sample)
        self.x_length.copy_(x_length)
        self.y_length.copy_(y_length)
        self.index.zero_()

    def replay(self, step: int) -> dict[str, Any]:
        parity = step % 2
        self.graphs[parity].replay()
        return self.outputs[parity]

    def update_sample(self, sample: Any) -> None:
        self.sample.copy_(sample)


class _SampleCudaGraph:
    """Capture GPU top-k sampling while preserving eager RNG semantics."""

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


class TensorRTFixedReferenceStreamer:
    """Generate cross-faded PCM chunks without reprocessing the voice reference.

    Instances are deliberately used by only one producer thread at a time.
    The public API owns that serialization because all calls share the base
    minimal-inference CUDA stream and TensorRT execution contexts.
    """

    def __init__(
        self,
        engine: Any,
        *,
        reference_wav: Path,
        reference_text: str,
        reference_language: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self.engine = engine
        self.reference_wav = Path(reference_wav)
        self.reference_text = reference_text
        self.reference_language = reference_language
        self.logger = logger or logging.getLogger(__name__)
        self.torch = importlib.import_module("torch")
        self.trt = importlib.import_module("tensorrt")
        source = importlib.import_module("run_trt_inference")
        self.sample_topk = getattr(source, "sample_topk")
        self.semantic_runtime = TransformerSemanticRuntime(
            engine=self.engine,
            sample_topk=self.sample_topk,
            torch=self.torch,
            trt=self.trt,
            logger=self.logger,
        )
        self.reference: PreparedReference | None = None
        self._mute_matrix: Any | None = None
        self._mute_matrix_checked = False
        self._gpt_destination_cache: tuple[Any, Any] | None = None
        self._gpt_step_graphs: _GPTStepCudaGraphs | None = None
        self._gpt_graph_disabled = False
        self._gpt_graph_notice_emitted = False
        self._sample_graphs: OrderedDict[tuple[Any, ...], _SampleCudaGraph] = OrderedDict()
        self._sample_graph_disabled = False
        self._tail_stream: Any | None = None
        self._tail_sovits: Any | None = None
        self._tail_executor: ThreadPoolExecutor | None = None
        self._warm_input_ids: Any | None = None
        self._preview_token_hint: int | None = None
        self.last_profile: dict[str, Any] | None = None
        self._configure_overlap_decoder(source)

    def _configure_overlap_decoder(self, source: Any) -> None:
        enabled = os.environ.get("ANIFLIVE_TTS_PIPELINE_OVERLAP", "1").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            self.logger.info("Dedicated TensorRT SoVITS tail context is disabled")
            return
        try:
            self._tail_stream = self.torch.cuda.Stream(device=self.engine.device)
            runtime_class = getattr(source, "TRTModule")
            self._tail_sovits = runtime_class(
                str(Path(self.engine.trt_dir) / "sovits.engine"),
                self.engine.device,
                self._tail_stream,
            )
            self._tail_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="aniflive-tts-sovits-tail"
            )
            self.logger.info("Dedicated TensorRT SoVITS tail context is ready")
        except Exception:
            self._tail_stream = None
            self._tail_sovits = None
            self._tail_executor = None
            self.logger.warning(
                "Dedicated TensorRT SoVITS tail context could not be initialized; using serialized decode",
                exc_info=True,
            )

    def close(self) -> None:
        if self._tail_executor is not None:
            self._tail_executor.shutdown(wait=True, cancel_futures=False)
            self._tail_executor = None
        self._tail_sovits = None
        self._tail_stream = None
        self._sample_graphs.clear()
        self.semantic_runtime.close()

    @property
    def sample_rate(self) -> int:
        return int(self.engine.hps["data"]["sampling_rate"])

    def _load_audio(self, sample_rate: int) -> np.ndarray:
        audio, source_rate = sf.read(str(self.reference_wav), dtype="float32", always_2d=True)
        mono = audio.mean(axis=1, dtype=np.float32)
        if int(source_rate) == int(sample_rate):
            return mono
        return soxr.resample(mono, int(source_rate), int(sample_rate), quality="HQ").astype(
            np.float32, copy=False
        )

    def prepare_reference(self) -> PreparedReference:
        """Run the fixed reference through SSL/VQ/spec/SV engines once."""

        torch = self.torch
        with torch.cuda.stream(self.engine.stream):
            self.logger.info("Preparing reference: decode 16 kHz audio")
            wav16k = self._load_audio(16000)
            wav16k_tensor = torch.from_numpy(wav16k).to(self.engine.device).to(self.engine.precision)
            silence = torch.zeros(
                int(16000 * 0.3), device=self.engine.device, dtype=self.engine.precision
            )
            ssl_audio = torch.cat([wav16k_tensor, silence])[None, :]
            self.logger.info("Preparing reference: TensorRT SSL enqueue")
            ssl_content = self.engine.model_ssl({"audio": ssl_audio})["last_hidden_state"]
            self.logger.info("Preparing reference: TensorRT VQ enqueue")
            codes = self.engine.model_vq({"ssl_content": ssl_content})["codes"]
            prompt_semantic = codes[0, 0][None, :]

            self.logger.info("Preparing reference: TensorRT spectrogram enqueue")
            wav_ref = self._load_audio(self.sample_rate)
            spectrogram = self.engine.model_spectrogram(
                {
                    "audio": torch.from_numpy(wav_ref)[None, :]
                    .to(self.engine.device)
                    .to(self.engine.precision)
                }
            )["spectrogram"]

            self.logger.info("Preparing reference: TensorRT speaker embedding enqueue")
            sv_wav = wav16k
            speaker_embedding = self.engine.model_sv_embedding(
                {
                    "audio": torch.from_numpy(sv_wav)[None, :]
                    .to(self.engine.device)
                    .to(self.engine.precision)
                }
            )["sv_embedding"]
            expected_sv_size = 20480 if "Pro" in self.engine.version else 512
            if int(speaker_embedding.shape[-1]) != expected_sv_size:
                padded = torch.zeros(
                    (1, expected_sv_size), device=self.engine.device, dtype=torch.float32
                )
                copied = min(int(speaker_embedding.shape[-1]), expected_sv_size)
                padded[:, :copied] = speaker_embedding[:, :copied]
                speaker_embedding = padded
            speaker_embedding = speaker_embedding.to(self.engine.precision)

            self.logger.info("Preparing reference: multilingual text frontend and TensorRT BERT")
            phones, bert, _ = self.engine.get_phones_and_bert(
                self.reference_text, self.reference_language, self.engine.version
            )

        self.logger.info("Preparing reference: synchronize CUDA stream")
        torch.cuda.synchronize(self.engine.stream)
        self.reference = PreparedReference(
            prompt_semantic=prompt_semantic,
            spectrogram=spectrogram,
            speaker_embedding=speaker_embedding,
            phones=list(phones),
            bert=bert,
        )
        self._warm_input_ids = torch.zeros(
            (1, 32), dtype=torch.int64, device=self.engine.device
        )
        self.logger.info(
            "Prepared fixed voice reference once: prompt=%d tokens, spectrogram=%s",
            int(prompt_semantic.shape[-1]),
            tuple(int(value) for value in spectrogram.shape),
        )
        return self.reference

    def warm_frontends(self) -> None:
        """Resolve every advertised language frontend before accepting traffic."""

        samples = (
            ("你好。", "all_zh"),
            ("你好。", "all_yue"),
            ("Hello.", "en"),
            ("こんにちは。", "all_ja"),
            ("안녕하세요.", "all_ko"),
        )
        started = time.perf_counter()
        with self.torch.cuda.stream(self.engine.stream):
            for text, language in samples:
                self.engine.get_phones_and_bert(
                    text,
                    language,
                    self.engine.version,
                    default_lang=self.reference_language,
                )
        self.torch.cuda.synchronize(self.engine.stream)
        self.logger.info(
            "Prepared all five language frontends in %.3fs", time.perf_counter() - started
        )

    def keepwarm_pulse(self) -> float:
        """Run bounded TensorRT work without sampling tokens or producing audio."""

        if self._warm_input_ids is None:
            raise RuntimeError("TensorRT warm-retention inputs are not prepared")
        started = time.perf_counter()
        inputs = {
            "input_ids": self._warm_input_ids,
            "attention_mask": self._warm_input_ids,
            "token_type_ids": self._warm_input_ids,
        }
        with self.torch.cuda.stream(self.engine.stream):
            self.engine.model_bert(inputs, sync=False)
            self.engine.model_bert(inputs, sync=False)
        self.engine.stream.synchronize()
        return time.perf_counter() - started

    def _load_mute_matrix(self) -> Any | None:
        if self._mute_matrix_checked:
            return self._mute_matrix
        self._mute_matrix_checked = True
        # The engine directory normally lives in a named volume, so derive the
        # canonical source path from the imported minimal-inference module.
        source_file = Path(importlib.import_module("run_trt_inference").__file__).resolve()
        path = source_file.parent / "GPT_SoVITS" / "pretrained_models" / "gpts1_mute_emb_sim_matrix.pt"
        if not path.is_file():
            self.logger.info("Mute-boundary matrix is not bundled; using fixed token chunks")
            return None
        self._mute_matrix = self.torch.load(
            path, map_location=self.engine.device, weights_only=True
        )
        return self._mute_matrix

    def _prepare_step_input(self, name: str, tensor: Any) -> Any:
        location = self.engine.model_gpt_step.tensor_location.get(
            name, self.trt.TensorLocation.DEVICE
        )
        if location == self.trt.TensorLocation.HOST:
            return tensor.detach().cpu().to(self.torch.int64)
        return tensor.detach().to(self.engine.device).to(self.torch.int64)

    def complete_wav_chunk_length(self) -> int:
        """Return the largest safe semantic chunk for a complete WAV response.

        PCM streaming deliberately keeps chunks short so a client can begin
        playback quickly.  A complete WAV cannot be sent until its final RIFF
        data length is known, so using the same tiny chunks there only creates
        unnecessary SoVITS invocations.  Leave room for the overlap decoder's
        32-token history and 16-token lookahead when the engine reports a
        bounded semantic profile.
        """

        maximum = getattr(self.engine, "sovits_max_sem_len", None)
        if maximum is None:
            # Unknown profiles retain the proved low-latency conservative path.
            return 24
        # The mute-boundary path can inspect two extra buffered tokens before
        # selecting a split, so reserve those too.  This keeps every later
        # history + chunk + lookahead decode inside the TRT maximum.
        return max(8, int(maximum) - 32 - 16 - 2)

    def streaming_chunk_length(self) -> int:
        """Return a balanced, profile-safe PCM chunk size.

        The optional mute-boundary matrix is not part of the portable model package.
        Without it the previous 24-token fallback made a SoVITS TensorRT call
        roughly four times more often than necessary.  Ninety-six tokens still
        gives streaming clients bounded first-audio latency while leaving the
        overlap decoder safely within every fitted engine profile.
        """

        maximum = getattr(self.engine, "sovits_max_sem_len", None)
        if maximum is None:
            return 24
        return max(8, min(96, self.complete_wav_chunk_length()))

    def _decode_chunk(
        self,
        *,
        tokens: Any,
        history: Any | None,
        lookahead: Any | None,
        phones: Sequence[int],
        noise_scale: float,
        speed: float,
        history_tokens: int,
        lookahead_tokens: int,
        model: Any | None = None,
        stream: Any | None = None,
    ) -> np.ndarray:
        reference = self.reference
        if reference is None:
            raise RuntimeError("The fixed voice reference has not been prepared")

        selected_model = model or self.engine.model_sovits
        selected_stream = stream or self.engine.stream
        with self.torch.cuda.stream(selected_stream):
            inputs = []
            if history is not None:
                inputs.append(history[:, -history_tokens:])
            inputs.append(tokens)
            if lookahead is not None:
                inputs.append(lookahead[:, :lookahead_tokens])
            all_tokens = self.torch.cat(inputs, dim=1)
            semantic = all_tokens[:, None, :]
            text_seq = self.torch.tensor(
                phones, dtype=self.torch.int64, device=self.engine.device
            )[None, :]
            sovits_inputs = {
                "pred_semantic": semantic.to(self.torch.int64),
                "text_seq": text_seq,
                "refer_spec": reference.spectrogram,
                "sv_emb": reference.speaker_embedding,
                "noise_scale": self.torch.tensor(
                    [noise_scale], dtype=self.torch.float32, device=self.engine.device
                ),
                "speed": self.torch.tensor(
                    [speed], dtype=self.torch.float32, device=self.engine.device
                ),
            }
            sovits_inputs = {
                name: value
                for name, value in sovits_inputs.items()
                if name in selected_model.input_names
            }
            audio = selected_model(sovits_inputs)["audio"].flatten()
            samples_per_token = float(audio.numel()) / max(1, int(all_tokens.shape[-1]))
            history_count = (
                min(int(history.shape[-1]), history_tokens) if history is not None else 0
            )
            begin = min(int(history_count * samples_per_token), int(audio.numel()))
            count = max(1, int(int(tokens.shape[-1]) * samples_per_token))
            end = min(begin + count, int(audio.numel()))
            result = audio[begin:end].detach().float().cpu().numpy()
        if result.size == 0 or not np.isfinite(result).all():
            raise RuntimeError("TensorRT SoVITS produced an invalid streaming audio chunk")
        return result.astype(np.float32, copy=False)

    def _decode_native_stream_chunk(
        self,
        *,
        cumulative_tokens: Any,
        new_token_count: int,
        phones: Sequence[int],
        noise_scale: float,
        speed: float,
        previous_latent: Any | None,
        acoustic_noise: Any,
    ) -> tuple[np.ndarray, Any, int]:
        """Decode a model-aligned V2ProPlus chunk with TensorRT only."""

        reference = self.reference
        if reference is None:
            raise RuntimeError("The fixed voice reference has not been prepared")
        model = self.engine.model_sovits_stream
        maximum = int(model.input_max_shapes.get("pred_semantic", (1, 1, 250))[-1])
        if int(cumulative_tokens.shape[-1]) > maximum:
            cumulative_tokens = cumulative_tokens[:, -maximum:]

        overlap_frames = int(model.engine.get_tensor_shape("overlap_frames")[-1])
        if overlap_frames <= 0 or overlap_frames % 2:
            raise RuntimeError(
                f"TensorRT streaming SoVITS has invalid overlap shape: {overlap_frames}"
            )
        overlap_semantic_tokens = overlap_frames // 2
        has_overlap = previous_latent is not None
        result_tokens = int(new_token_count) + (
            int(overlap_semantic_tokens) if has_overlap else 0
        )
        result_tokens = max(
            1, min(result_tokens, int(cumulative_tokens.shape[-1]))
        )
        result_frames = result_tokens * 2
        hop_length = int(self.engine.hps["data"]["hop_length"])
        if has_overlap:
            overlap = previous_latent[:, :, -overlap_frames:].contiguous()
        else:
            overlap = self.torch.zeros(
                (1, 192, overlap_frames),
                dtype=self.engine.precision,
                device=self.engine.device,
            )

        text_seq = self.torch.tensor(
            phones, dtype=self.torch.int64, device=self.engine.device
        )[None, :]
        inputs = {
            "pred_semantic": cumulative_tokens[:, None, :].to(self.torch.int64),
            "text_seq": text_seq,
            "refer_spec": reference.spectrogram,
            "sv_emb": reference.speaker_embedding,
            "noise_scale": self.torch.tensor(
                [noise_scale], dtype=self.torch.float32, device=self.engine.device
            ),
            "result_length": self.torch.tensor(
                [result_tokens], dtype=self.torch.int64, device=self.engine.device
            ),
            "overlap_frames": overlap,
            "overlap_enabled": self.torch.tensor(
                [1.0 if has_overlap else 0.0],
                dtype=self.torch.float32,
                device=self.engine.device,
            ),
            "acoustic_noise": acoustic_noise,
        }
        inputs = {
            name: value for name, value in inputs.items() if name in model.input_names
        }
        output_dtype = model.tensor_dtype["audio"]
        outputs = {
            "audio": self.torch.empty(
                (1, 1, result_frames * hop_length),
                dtype=output_dtype,
                device=self.engine.device,
            ),
            "latent": self.torch.empty(
                (1, 192, result_frames),
                dtype=model.tensor_dtype["latent"],
                device=self.engine.device,
            ),
            "latent_mask": self.torch.empty(
                (1, 1, result_frames),
                dtype=model.tensor_dtype["latent_mask"],
                device=self.engine.device,
            ),
        }
        with self.torch.cuda.stream(self.engine.stream):
            decoded = model(inputs, outputs=outputs)
            audio = decoded["audio"].flatten().detach().float().cpu().numpy()
            latent_tail = decoded["latent"][:, :, -overlap_frames:].detach().clone()
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("TensorRT streaming SoVITS produced invalid audio")
        return (
            audio.astype(np.float32, copy=False),
            latent_tail,
            overlap_frames * hop_length,
        )

    @staticmethod
    def _native_stream_shape_is_safe(
        *, new_token_count: int, overlap_frames: int, has_overlap: bool
    ) -> bool:
        """Return whether the fitted stream graph can broadcast its overlap input."""

        return bool(has_overlap) or int(new_token_count) * 2 >= int(overlap_frames)

    @staticmethod
    def _preview_target_for_segment(
        *, base_tokens: int, cached_tokens: int | None, segment_index: int
    ) -> int:
        """Apply the cross-request TTFA hint only to a request's first segment."""

        if int(segment_index) != 0 or cached_tokens is None:
            return int(base_tokens)
        return max(
            int(base_tokens),
            min(int(cached_tokens), MAX_CACHED_PREVIEW_TOKENS),
        )

    @staticmethod
    def _remember_preview_target(
        *, cached_tokens: int | None, successful_tokens: int
    ) -> int:
        """Keep a fast baseline without letting one long text poison it."""

        bounded = min(int(successful_tokens), MAX_CACHED_PREVIEW_TOKENS)
        if cached_tokens is None:
            return bounded
        return min(int(cached_tokens), bounded)

    @staticmethod
    def _sola_merge(
        previous_tail: np.ndarray,
        current_audio: np.ndarray,
        *,
        overlap_samples: int,
        search_samples: int = 0,
    ) -> np.ndarray:
        """Crossfade model-aligned native chunks without duplicating speech.

        The native V2ProPlus decoder already aligns both chunks through the
        shared latent overlap and acoustic-noise frame indices. A second
        waveform search can lock onto a neighbouring pitch period and delete
        up to one search window from otherwise aligned speech.
        """

        overlap = min(
            int(overlap_samples), int(previous_tail.size), int(current_audio.size)
        )
        if overlap <= 0:
            return np.concatenate((previous_tail, current_audio))
        reference = previous_tail[-overlap:].astype(np.float64, copy=False)
        candidate = current_audio[
            : min(int(current_audio.size), overlap + max(0, int(search_samples)))
        ].astype(np.float64, copy=False)
        offset = 0
        if candidate.size > overlap:
            correlation = np.correlate(candidate, reference, mode="valid")
            squared = np.square(candidate, dtype=np.float64)
            cumulative = np.concatenate(([0.0], np.cumsum(squared)))
            energy = cumulative[overlap:] - cumulative[:-overlap]
            scores = correlation / np.sqrt(np.maximum(energy, 1e-8))
            offset = int(np.argmax(scores))
        aligned = current_audio[offset:].copy()
        overlap = min(overlap, int(aligned.size))
        if overlap <= 0:
            return previous_tail.copy()
        window = np.hanning(overlap * 2).astype(np.float32, copy=False)
        aligned[:overlap] = (
            window[:overlap] * aligned[:overlap]
            + window[overlap:] * previous_tail[-overlap:]
        )
        return aligned

    @classmethod
    def _refill_from_full_context(
        cls,
        previous_tail: np.ndarray,
        full_audio: np.ndarray,
        *,
        expected_start: int,
        sample_rate: int,
    ) -> tuple[np.ndarray, int, float]:
        """Join a low-latency preview to a full-context decode by waveform match."""

        if previous_tail.size == 0 or full_audio.size == 0:
            start = min(max(0, int(expected_start)), int(full_audio.size))
            return full_audio[start:].copy(), start, 0.0
        match_samples = min(
            int(previous_tail.size),
            max(256, int(round(max(1, sample_rate) * 0.050))),
        )
        reference = previous_tail[:match_samples].astype(np.float64, copy=False)
        reference = reference - float(np.mean(reference))
        reference_energy = float(np.dot(reference, reference))
        expected = min(
            max(0, int(expected_start)),
            max(0, int(full_audio.size) - match_samples),
        )
        if reference_energy <= 1e-10:
            start = expected
            score = 0.0
        else:
            # The decoder supplies an architecture-derived expected offset.
            # Correlation is only a local phase refinement: a wide search can
            # lock onto a neighbouring pitch period or repeated syllable and
            # reinsert tens of milliseconds of speech at the refill boundary.
            radius = max(32, int(round(max(1, sample_rate) * 0.002)))
            low = max(0, expected - radius)
            high = min(int(full_audio.size) - match_samples, expected + radius)
            stride = 4
            reference_ds = reference[::stride]
            candidate = full_audio[low : high + match_samples : stride].astype(
                np.float64, copy=False
            )
            candidate = candidate - float(np.mean(candidate))
            correlation = np.correlate(candidate, reference_ds, mode="valid")
            squared = np.square(candidate, dtype=np.float64)
            cumulative = np.concatenate(([0.0], np.cumsum(squared)))
            width = int(reference_ds.size)
            energy = cumulative[width:] - cumulative[:-width]
            scores = correlation / np.sqrt(
                np.maximum(energy * float(np.dot(reference_ds, reference_ds)), 1e-12)
            )
            coarse = int(np.argmax(scores))
            coarse_start = low + coarse * stride
            refine_low = max(low, coarse_start - stride)
            refine_high = min(high, coarse_start + stride)
            start = coarse_start
            score = float(scores[coarse])
            for candidate_start in range(refine_low, refine_high + 1):
                window = full_audio[
                    candidate_start : candidate_start + match_samples
                ].astype(np.float64, copy=False)
                window = window - float(np.mean(window))
                denominator = np.sqrt(
                    max(float(np.dot(window, window)) * reference_energy, 1e-12)
                )
                candidate_score = float(np.dot(window, reference) / denominator)
                if candidate_score > score:
                    start = candidate_start
                    score = candidate_score
            if score < 0.15:
                start = expected
        crossfade_samples = min(
            match_samples,
            max(128, int(round(max(1, sample_rate) * 0.015))),
        )
        merged = cls._sola_merge(
            previous_tail[:crossfade_samples],
            full_audio[start:],
            overlap_samples=crossfade_samples,
        )
        return merged, start, score

    @staticmethod
    def _trailing_silence_seconds(audio: np.ndarray, sample_rate: int) -> float:
        if audio.size == 0 or sample_rate <= 0:
            return 0.0
        frame = max(1, int(round(sample_rate * 0.010)))
        threshold = 10.0 ** (-45.0 / 20.0)
        last_active = int(audio.size)
        for end in range(int(audio.size), 0, -frame):
            begin = max(0, end - frame)
            rms = float(np.sqrt(np.mean(np.square(audio[begin:end], dtype=np.float64))))
            if rms > threshold:
                last_active = end
                break
            last_active = begin
        return max(0.0, float(audio.size - last_active) / float(sample_rate))

    @classmethod
    def _split_trailing_silence(
        cls, audio: np.ndarray, sample_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        trailing_seconds = cls._trailing_silence_seconds(audio, sample_rate)
        trailing_samples = min(
            int(audio.size), int(round(trailing_seconds * max(0, sample_rate)))
        )
        if trailing_samples <= 0:
            return audio, np.empty(0, dtype=np.float32)
        if trailing_samples >= int(audio.size):
            return np.empty(0, dtype=np.float32), audio
        return audio[:-trailing_samples], audio[-trailing_samples:]

    @staticmethod
    def _leading_silence_seconds(audio: np.ndarray, sample_rate: int) -> float:
        if audio.size == 0 or sample_rate <= 0:
            return 0.0
        frame = max(1, int(round(sample_rate * 0.010)))
        threshold = 10.0 ** (-45.0 / 20.0)
        first_active = 0
        for begin in range(0, int(audio.size), frame):
            end = min(int(audio.size), begin + frame)
            rms = float(np.sqrt(np.mean(np.square(audio[begin:end], dtype=np.float64))))
            if rms > threshold:
                first_active = begin
                break
            first_active = end
        return max(0.0, float(first_active) / float(sample_rate))

    @classmethod
    def _contains_active_audio(cls, audio: np.ndarray, sample_rate: int) -> bool:
        """Return whether a PCM block contains a frame above the audibility gate."""

        if audio.size == 0 or sample_rate <= 0:
            return False
        leading_silence = cls._leading_silence_seconds(audio, sample_rate)
        return leading_silence < float(audio.size) / float(sample_rate)

    @staticmethod
    def _has_natural_boundary(segment: str) -> bool:
        if segment.endswith(("\n\n", "\r\n\r\n")):
            return True
        ending = segment.rstrip().rstrip("\"'”’」』）》】〕〉")
        return ending.endswith((",", ".", ";", "?", "!", "、", "，", "。", "；", "？", "！"))

    @staticmethod
    def _trim_excess_leading_silence(
        audio: np.ndarray,
        *,
        sample_rate: int,
        leading_silence_seconds: float,
        retained_silence_seconds: float,
    ) -> tuple[np.ndarray, float]:
        if audio.size == 0 or sample_rate <= 0:
            return audio, 0.0
        excess_seconds = max(
            0.0, float(leading_silence_seconds) - max(0.0, retained_silence_seconds)
        )
        trim_samples = min(int(audio.size), int(round(excess_seconds * sample_rate)))
        if trim_samples <= 0:
            return audio, 0.0
        return audio[trim_samples:], float(trim_samples) / float(sample_rate)

    @staticmethod
    def _target_pause_seconds(segment: str, sentence_pause: float) -> float:
        if sentence_pause <= 0:
            return 0.0
        if segment.endswith("\n\n") or segment.endswith("\r\n\r\n"):
            return min(0.800, sentence_pause * (320.0 / 220.0))
        ending = segment.rstrip().rstrip("\"'”’」』）》】〕〉")
        if ending.endswith((",", "，", "、")):
            return sentence_pause * 0.5
        if ending.endswith((";", "；")):
            return sentence_pause * (150.0 / 220.0)
        if ending.endswith((".", "。")):
            return sentence_pause
        if ending.endswith(("!", "！", "?", "？")):
            return sentence_pause * (200.0 / 220.0)
        return min(0.030, sentence_pause * 0.10)

    def _pause_chunk(
        self, *, segment: str, sentence_pause: float, trailing_silence_seconds: float
    ) -> np.ndarray | None:
        target = self._target_pause_seconds(segment, sentence_pause)
        missing = max(0.0, target - max(0.0, trailing_silence_seconds))
        samples = int(round(self.sample_rate * missing))
        if samples <= 0:
            return None
        return np.zeros(samples, dtype=np.float32)

    @staticmethod
    def _trim_excess_trailing_silence(
        audio: np.ndarray,
        *,
        sample_rate: int,
        trailing_silence_seconds: float,
        target_pause_seconds: float,
    ) -> tuple[np.ndarray, float, float]:
        """Keep at most the requested pause in a non-final segment tail."""

        if audio.size == 0 or sample_rate <= 0:
            return audio, 0.0, 0.0
        excess_seconds = max(
            0.0, float(trailing_silence_seconds) - max(0.0, target_pause_seconds)
        )
        trim_samples = min(int(audio.size), int(round(excess_seconds * sample_rate)))
        if trim_samples <= 0:
            return audio, max(0.0, trailing_silence_seconds), 0.0
        retained_seconds = max(
            0.0, float(trailing_silence_seconds) - float(trim_samples) / sample_rate
        )
        return audio[:-trim_samples], retained_seconds, float(trim_samples) / sample_rate

    def _decode_tail(
        self,
        *,
        ready_event: Any,
        full_tokens: Any,
        phones: Sequence[int],
        noise_scale: float,
        speed: float,
        streamed_samples: int,
        prior_fade: np.ndarray | None,
        fade_samples: int,
        history_tokens: int,
        lookahead_tokens: int,
    ) -> _DecodedTail:
        if self._tail_stream is None or self._tail_sovits is None:
            raise RuntimeError("The overlap SoVITS execution context is unavailable")
        started = time.perf_counter()
        self._tail_stream.wait_event(ready_event)
        full_audio, invocations = self._decode_complete_profile_safe(
            full_tokens=full_tokens,
            phones=phones,
            noise_scale=noise_scale,
            speed=speed,
            history_tokens=history_tokens,
            lookahead_tokens=lookahead_tokens,
            fade_samples=fade_samples,
            model=self._tail_sovits,
            stream=self._tail_stream,
        )
        suffix = full_audio[min(streamed_samples, int(full_audio.size)) :].copy()
        if prior_fade is not None and suffix.size:
            count = min(fade_samples, int(suffix.size), int(prior_fade.size))
            fade_in = np.linspace(0.0, 1.0, count, dtype=np.float32)
            suffix[:count] = suffix[:count] * fade_in + prior_fade[-count:] * (
                1.0 - fade_in
            )
        return _DecodedTail(
            audio=suffix,
            decode_seconds=time.perf_counter() - started,
            trailing_silence_seconds=self._trailing_silence_seconds(
                full_audio, self.sample_rate
            ),
            invocations=invocations,
        )

    def _decode_complete_profile_safe(
        self,
        *,
        full_tokens: Any,
        phones: Sequence[int],
        noise_scale: float,
        speed: float,
        history_tokens: int,
        lookahead_tokens: int,
        fade_samples: int,
        model: Any,
        stream: Any,
    ) -> tuple[np.ndarray, int]:
        semantic_limit = int(getattr(self.engine, "sovits_max_sem_len", 0) or 0)
        total_tokens = int(full_tokens.shape[-1])
        if semantic_limit <= 0 or total_tokens <= semantic_limit:
            return (
                self._decode_chunk(
                    tokens=full_tokens,
                    history=None,
                    lookahead=None,
                    phones=phones,
                    noise_scale=noise_scale,
                    speed=speed,
                    history_tokens=history_tokens,
                    lookahead_tokens=lookahead_tokens,
                    model=model,
                    stream=stream,
                ),
                1,
            )

        chunk_tokens = max(
            8,
            min(
                self.complete_wav_chunk_length(),
                semantic_limit - history_tokens - lookahead_tokens,
            ),
        )
        position = 0
        history: Any | None = None
        prior_fade: np.ndarray | None = None
        pieces: list[np.ndarray] = []
        invocations = 0
        while position < total_tokens:
            count = min(chunk_tokens, total_tokens - position)
            current = full_tokens[:, position : position + count]
            following = full_tokens[
                :, position + count : position + count + lookahead_tokens
            ]
            audio = self._decode_chunk(
                tokens=current,
                history=history,
                lookahead=following if int(following.shape[-1]) else None,
                phones=phones,
                noise_scale=noise_scale,
                speed=speed,
                history_tokens=history_tokens,
                lookahead_tokens=lookahead_tokens,
                model=model,
                stream=stream,
            )
            invocations += 1
            if prior_fade is not None and audio.size:
                overlap = min(fade_samples, int(prior_fade.size), int(audio.size))
                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                audio[:overlap] = audio[:overlap] * fade_in + prior_fade[-overlap:] * (
                    1.0 - fade_in
                )
            final = position + count >= total_tokens
            if not final and audio.size:
                overlap = min(fade_samples, int(audio.size))
                pieces.append(audio[:-overlap] if overlap else audio)
                prior_fade = audio[-overlap:].copy() if overlap else None
            else:
                pieces.append(audio)
                prior_fade = None
            history = (
                current
                if history is None
                else self.torch.cat([history, current], dim=1)[:, -history_tokens:]
            )
            position += count
        return np.concatenate(pieces).astype(np.float32, copy=False), invocations

    def iter_audio(
        self,
        *,
        segments: Sequence[str],
        text_language: str,
        top_k: int,
        top_p: float,
        temperature: float,
        noise_scale: float,
        speed: float,
        pause_length: float,
        request_seed: int,
        chunk_length: int | None = None,
        cancelled: threading.Event | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield mono float32 chunks while retaining the reference on the GPU."""

        reference = self.reference
        if reference is None:
            raise RuntimeError("The fixed voice reference has not been prepared")
        if chunk_length is None:
            chunk_length = self.streaming_chunk_length()
        if chunk_length < 8:
            raise ValueError("chunk_length must be at least 8")

        request_started = time.perf_counter()
        profile: dict[str, Any] = {
            "text_processing_seconds": 0.0,
            "gpt_encoder_seconds": 0.0,
            "gpt_decode_seconds": 0.0,
            "sovits_seconds": 0.0,
            "semantic_tokens": 0,
            "gpt_steps": 0,
            "semantic_backend": self.semantic_runtime.backend_name,
            "semantic_nfe": 0,
            "semantic_tokens_per_nfe": 0.0,
            "first_preview_semantic_seconds": 0.0,
            "host_sync_count": 0,
            "host_sync_seconds": 0.0,
            "attention_kv_bytes": 0,
            "mamba_state_bytes": 0,
            "mtp_proposed_tokens": 0,
            "mtp_accepted_tokens": 0,
            "mtp_acceptance_rate": 0.0,
            "sovits_invocations": 0,
            "segments": len(segments),
            "request_seed": int(request_seed),
            "first_audio_seconds": 0.0,
            "sample_cuda_graph_cache_hits": 0,
            "sample_cuda_graph_captures": 0,
            "pipeline_overlap_enabled": 0,
            "dedicated_tail_context_enabled": int(self._tail_executor is not None),
            "segment_profiles": [],
        }
        self.last_profile = None

        torch = self.torch
        detailed_profile = (
            os.environ.get("ANIFLIVE_TTS_DETAILED_PROFILE", "0").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        mute_matrix = self._load_mute_matrix()
        history_tokens, lookahead_tokens, fade_samples = 32, 16, 256
        pending_tail: _PendingTail | None = None
        pending_technical_tail: np.ndarray | None = None
        technical_crossfade_samples = max(fade_samples, int(round(self.sample_rate * 0.020)))
        continuation_phones: list[int] = []
        continuation_bert: Any | None = None
        continuation_tokens: Any | None = None
        previous_was_technical = False
        # Do not publish silent preview chunks as if speech had started.  The
        # same conservative head trim is used for the first segment and every
        # later natural sentence, retaining a short guard against clipped
        # consonants or a hard waveform edge.
        trim_natural_segment_head = True
        first_audio_emitted = False

        def note_audio(audio: np.ndarray, segment_index: int) -> np.ndarray:
            nonlocal first_audio_emitted
            if audio.size and not first_audio_emitted:
                profile["first_audio_seconds"] = time.perf_counter() - request_started
                profile["first_audio_segment"] = segment_index
                first_audio_emitted = True
            return audio

        def prepare_segment_head(audio: np.ndarray) -> np.ndarray:
            nonlocal pending_technical_tail, trim_natural_segment_head
            if audio.size == 0:
                return audio
            if pending_technical_tail is None and trim_natural_segment_head:
                leading_silence = self._leading_silence_seconds(audio, self.sample_rate)
                if not self._contains_active_audio(audio, self.sample_rate):
                    profile["discarded_silent_preview_seconds"] = float(
                        profile.get("discarded_silent_preview_seconds", 0.0)
                    ) + float(audio.size) / float(self.sample_rate)
                    return np.empty(0, dtype=np.float32)
                retained_silence_seconds = (
                    INITIAL_LEADING_SILENCE_RETAIN_SECONDS
                    if segment_index == 0
                    else FOLLOWING_LEADING_SILENCE_RETAIN_SECONDS
                )
                audio, trimmed_seconds = self._trim_excess_leading_silence(
                    audio,
                    sample_rate=self.sample_rate,
                    leading_silence_seconds=leading_silence,
                    retained_silence_seconds=retained_silence_seconds,
                )
                profile["trimmed_natural_leading_silence_seconds"] = float(
                    profile.get("trimmed_natural_leading_silence_seconds", 0.0)
                ) + trimmed_seconds
                # A segment can begin with one or more entirely silent PCM
                # chunks. Keep trimming until the first audible samples arrive.
                trim_natural_segment_head = audio.size == 0
                return audio
            if pending_technical_tail is None:
                return audio
            leading_silence = self._leading_silence_seconds(audio, self.sample_rate)
            audio, trimmed_seconds = self._trim_excess_leading_silence(
                audio,
                sample_rate=self.sample_rate,
                leading_silence_seconds=leading_silence,
                retained_silence_seconds=0.005,
            )
            tail = pending_technical_tail
            pending_technical_tail = None
            if audio.size == 0:
                return tail
            overlap = min(technical_crossfade_samples, int(tail.size), int(audio.size))
            if overlap <= 0:
                return np.concatenate((tail, audio))
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            blended = tail[-overlap:] * (1.0 - fade_in) + audio[:overlap] * fade_in
            profile["technical_crossfades"] = int(profile.get("technical_crossfades", 0)) + 1
            profile["trimmed_leading_silence_seconds"] = float(
                profile.get("trimmed_leading_silence_seconds", 0.0)
            ) + trimmed_seconds
            return np.concatenate((tail[:-overlap], blended, audio[overlap:]))

        def consume_pending_tail() -> list[np.ndarray]:
            nonlocal pending_tail
            if pending_tail is None:
                return []
            wait_started = time.perf_counter()
            was_ready = pending_tail.future.done()
            decoded = pending_tail.future.result()
            wait_elapsed = time.perf_counter() - wait_started
            if not was_ready:
                profile["pipeline_wait_seconds"] = float(
                    profile.get("pipeline_wait_seconds", 0.0)
                ) + wait_elapsed
            profile["sovits_seconds"] += decoded.decode_seconds
            profile["sovits_invocations"] += decoded.invocations
            chunks: list[np.ndarray] = []
            tail_audio = decoded.audio
            retained_silence = decoded.trailing_silence_seconds
            if pending_tail.has_next:
                target_pause = self._target_pause_seconds(
                    pending_tail.segment, pause_length
                )
                tail_audio, retained_silence, trimmed_seconds = (
                    self._trim_excess_trailing_silence(
                        tail_audio,
                        sample_rate=self.sample_rate,
                        trailing_silence_seconds=decoded.trailing_silence_seconds,
                        target_pause_seconds=target_pause,
                    )
                )
                profile["trimmed_trailing_silence_seconds"] = float(
                    profile.get("trimmed_trailing_silence_seconds", 0.0)
                ) + trimmed_seconds
            if tail_audio.size:
                chunks.append(tail_audio)
            if pending_tail.has_next:
                pause = self._pause_chunk(
                    segment=pending_tail.segment,
                    sentence_pause=pause_length,
                    trailing_silence_seconds=retained_silence,
                )
                if pause is not None:
                    chunks.append(pause)
                    profile["inserted_pause_seconds"] = float(
                        profile.get("inserted_pause_seconds", 0.0)
                    ) + float(pause.size) / float(self.sample_rate)
            pending_tail = None
            return chunks

        with torch.cuda.stream(self.engine.stream):
            for segment_index, raw_text in enumerate(segments):
                if cancelled is not None and cancelled.is_set():
                    if pending_tail is not None:
                        pending_tail.future.result()
                    return
                text = raw_text.strip()
                segment_profile: dict[str, Any] = {
                    "index": segment_index,
                    "characters": len(text),
                    "boundary": text[-1:] if text else "",
                }
                technical_continuation = (
                    segment_index + 1 < len(segments)
                    and not self._has_natural_boundary(raw_text)
                )
                natural_continuation = (
                    segment_index + 1 < len(segments) and not technical_continuation
                )
                pending_natural_silence = np.empty(0, dtype=np.float32)

                def hold_natural_trailing_silence(audio: np.ndarray) -> list[np.ndarray]:
                    nonlocal pending_natural_silence
                    if not natural_continuation or audio.size == 0:
                        return [audio] if audio.size else []
                    active, trailing = self._split_trailing_silence(
                        audio, self.sample_rate
                    )
                    if active.size == 0:
                        if trailing.size:
                            pending_natural_silence = np.concatenate(
                                (pending_natural_silence, trailing)
                            )
                        return []
                    if pending_natural_silence.size:
                        active = np.concatenate((pending_natural_silence, active))
                        pending_natural_silence = np.empty(0, dtype=np.float32)
                    if trailing.size:
                        pending_natural_silence = trailing.copy()
                    return [active]

                segment_profile["technical_continuation"] = int(technical_continuation)
                profile["segment_profiles"].append(segment_profile)
                segment_seed = (int(request_seed) + segment_index * 1_000_003) % (
                    (1 << 63) - 1
                )
                torch.manual_seed(segment_seed)
                torch.cuda.manual_seed_all(segment_seed)
                segment_profile["seed"] = segment_seed
                text_started = time.perf_counter()
                phones, bert, _ = self.engine.get_phones_and_bert(
                    text, text_language, self.engine.version, default_lang=self.reference_language
                )
                text_elapsed = time.perf_counter() - text_started
                profile["text_processing_seconds"] += text_elapsed
                segment_profile["text_processing_seconds"] = text_elapsed
                segment_profile["phonemes"] = len(phones)
                max_text = self.engine.sovits_max_text_len
                if max_text is not None and len(phones) > int(max_text):
                    raise ValueError(
                        f"A text segment requires {len(phones)} phonemes, exceeding this TensorRT profile limit of {max_text}."
                    )
                history_phones: list[int] = []
                history_bert: Any | None = None
                history_semantic: Any | None = None
                if previous_was_technical and continuation_tokens is not None:
                    encoder_max_phones = int(
                        self.engine.model_gpt_enc.input_max_shapes.get(
                            "phoneme_ids", (1, 256)
                        )[-1]
                    )
                    available_phones = max(
                        0,
                        encoder_max_phones - len(reference.phones) - len(phones),
                    )
                    history_phone_count = min(75, available_phones, len(continuation_phones))
                    if history_phone_count > 0 and continuation_bert is not None:
                        history_phones = continuation_phones[-history_phone_count:]
                        history_bert = continuation_bert[:, -history_phone_count:]
                        encoder_max_prompts = int(
                            self.engine.model_gpt_enc.input_max_shapes.get(
                                "prompts", (1, 300)
                            )[-1]
                        )
                        available_prompts = max(
                            0,
                            encoder_max_prompts - int(reference.prompt_semantic.shape[-1]),
                        )
                        history_token_count = min(
                            125,
                            available_prompts,
                            int(continuation_tokens.shape[-1]),
                        )
                        if history_token_count > 0:
                            history_semantic = continuation_tokens[:, -history_token_count:]
                bert_parts = [reference.bert]
                if history_bert is not None:
                    bert_parts.append(history_bert)
                bert_parts.append(bert)
                merged_bert = torch.cat(bert_parts, dim=1)[None, :, :].to(
                    self.engine.precision
                )
                phoneme_ids = torch.tensor(
                    reference.phones + history_phones + list(phones),
                    dtype=torch.int64,
                    device=self.engine.device,
                )[None, :]
                phoneme_length = torch.tensor(
                    [phoneme_ids.shape[-1]], dtype=torch.int64, device=self.engine.device
                )
                prompt_semantic = reference.prompt_semantic.to(torch.int64)
                if history_semantic is not None and int(history_semantic.shape[-1]) > 0:
                    prompt_semantic = torch.cat(
                        [prompt_semantic, history_semantic.to(torch.int64)], dim=1
                    )
                segment_profile["continuation_phonemes"] = len(history_phones)
                segment_profile["continuation_semantic_tokens"] = (
                    int(history_semantic.shape[-1]) if history_semantic is not None else 0
                )

                semantic_state = self.semantic_runtime.prepare(
                    {
                        "phoneme_ids": phoneme_ids,
                        "phoneme_ids_len": phoneme_length,
                        "prompts": prompt_semantic,
                        "bert_feature": merged_bert,
                    },
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    detailed_profile=detailed_profile,
                )
                encoder_elapsed = semantic_state.encoder_seconds
                profile["gpt_encoder_seconds"] += encoder_elapsed
                segment_profile["gpt_encoder_seconds"] = encoder_elapsed
                current = semantic_state.first_token
                maximum_steps = semantic_state.maximum_steps
                stream_overlap_frames = int(
                    self.engine.model_sovits_stream.engine.get_tensor_shape(
                        "overlap_frames"
                    )[-1]
                )
                acoustic_generator = torch.Generator(device=self.engine.device)
                acoustic_generator.manual_seed(
                    (segment_seed ^ 0x41C64E6D) % ((1 << 63) - 1)
                )
                acoustic_noise = torch.randn(
                    (1, 192, 2 * (maximum_steps + 1)),
                    dtype=self.engine.precision,
                    device=self.engine.device,
                    generator=acoustic_generator,
                )
                profile["cuda_graph_enabled"] = 0
                queue: list[Any] = []
                token_buffer: list[Any] = [current]
                segment_token_parts: list[Any] = [current]
                buffered_count = 1
                native_semantic_history: Any | None = None
                native_previous_latent: Any | None = None
                native_previous_audio_tail: np.ndarray | None = None
                native_preview_audio_samples = 0
                native_acoustic_frame_cursor = 0
                emitted_chunks = 0
                low_latency_stream = chunk_length <= self.streaming_chunk_length()
                first_chunk_tokens = (
                    int(os.environ.get("ANIFLIVE_TTS_FIRST_CHUNK_TOKENS", "9"))
                    if low_latency_stream
                    else chunk_length
                )
                first_lookahead_tokens = (
                    int(os.environ.get("ANIFLIVE_TTS_FIRST_LOOKAHEAD_TOKENS", "8"))
                    if low_latency_stream
                    else lookahead_tokens
                )
                # The quality gate requires the complete 9+8 semantic prefix.
                # It is committed as one native streaming chunk rather than
                # proportionally slicing a shorter waveform preview.
                first_preview_tokens = first_chunk_tokens + first_lookahead_tokens
                preview_target_tokens = self._preview_target_for_segment(
                    base_tokens=first_preview_tokens,
                    cached_tokens=self._preview_token_hint,
                    segment_index=segment_index,
                )
                preview_retry_tokens = max(
                    2,
                    int(os.environ.get("ANIFLIVE_TTS_PREVIEW_RETRY_TOKENS", "2")),
                )

                def emit(tokens: Any, next_tokens: Any | None, *, final: bool) -> np.ndarray:
                    del next_tokens
                    nonlocal native_semantic_history
                    nonlocal native_previous_latent, native_previous_audio_tail
                    nonlocal native_preview_audio_samples
                    nonlocal native_acoustic_frame_cursor
                    decode_started = time.perf_counter()
                    native_semantic_history = (
                        tokens
                        if native_semantic_history is None
                        else torch.cat((native_semantic_history, tokens), dim=1)
                    )
                    new_frames = int(tokens.shape[-1]) * 2
                    noise_start = native_acoustic_frame_cursor
                    if native_previous_latent is not None:
                        noise_start -= stream_overlap_frames
                    noise_end = native_acoustic_frame_cursor + new_frames
                    chunk_noise = acoustic_noise[:, :, noise_start:noise_end].contiguous()
                    new_token_count = int(tokens.shape[-1])
                    if self._native_stream_shape_is_safe(
                        new_token_count=new_token_count,
                        overlap_frames=stream_overlap_frames,
                        has_overlap=native_previous_latent is not None,
                    ):
                        (
                            audio,
                            native_previous_latent,
                            overlap_samples,
                        ) = self._decode_native_stream_chunk(
                            cumulative_tokens=native_semantic_history,
                            new_token_count=new_token_count,
                            phones=phones,
                            noise_scale=noise_scale,
                            speed=speed,
                            previous_latent=native_previous_latent,
                            acoustic_noise=chunk_noise,
                        )
                    else:
                        # A punctuation-delimited segment may end with fewer
                        # semantic frames than the engine's static overlap.
                        # TensorRT validates both broadcast branches even when
                        # overlap_enabled is false, so route only that tiny,
                        # final standalone chunk through the already-loaded
                        # full TensorRT SoVITS engine.
                        audio = self._decode_chunk(
                            tokens=native_semantic_history,
                            history=None,
                            lookahead=None,
                            phones=phones,
                            noise_scale=noise_scale,
                            speed=speed,
                            history_tokens=history_tokens,
                            lookahead_tokens=lookahead_tokens,
                        )
                        native_previous_latent = None
                        overlap_samples = 0
                        profile["short_native_shape_fallbacks"] = int(
                            profile.get("short_native_shape_fallbacks", 0)
                        ) + 1
                    native_acoustic_frame_cursor += new_frames
                    profile["sovits_seconds"] += time.perf_counter() - decode_started
                    profile["sovits_invocations"] += 1
                    if native_previous_audio_tail is not None:
                        audio = self._sola_merge(
                            native_previous_audio_tail,
                            audio,
                            overlap_samples=overlap_samples,
                        )
                    if final:
                        emitted = audio
                        native_previous_audio_tail = None
                    else:
                        if emitted_chunks == 0:
                            native_preview_audio_samples = int(audio.size)
                            profile["preview_decoded_samples"] = int(audio.size)
                        count = min(overlap_samples, int(audio.size))
                        if count:
                            emitted = audio[:-count]
                            native_previous_audio_tail = audio[-count:].copy()
                        else:
                            emitted = audio
                            native_previous_audio_tail = None
                        if emitted_chunks == 0:
                            profile["preview_held_samples"] = int(count)
                            profile["preview_emitted_samples_raw"] = int(
                                emitted.size
                            )
                    return emitted

                decode_started = time.perf_counter()
                segment_tokens = 1
                segment_steps = 0
                # Every punctuation-delimited segment owns a deterministic RNG
                # seed. Speculative draws after EOS therefore cannot perturb the
                # following segment, so multi-segment requests retain the same
                # reduced host-synchronization path as a single segment.
                steady_eos_sync_interval = max(
                    1, min(8, int(os.environ.get("ANIFLIVE_TTS_EOS_SYNC_INTERVAL", "2")))
                )
                preview_eos_sync_interval = max(
                    1,
                    min(
                        32,
                        int(os.environ.get("ANIFLIVE_TTS_PREVIEW_EOS_SYNC_INTERVAL", "16")),
                    ),
                )
                segment_profile["eos_sync_interval"] = steady_eos_sync_interval
                segment_profile["preview_eos_sync_interval"] = preview_eos_sync_interval

                def semantic_sync_policy(
                    pending_count: int, step: int, maximum: int
                ) -> bool:
                    active_eos_sync_interval = (
                        preview_eos_sync_interval
                        if emitted_chunks == 0
                        else steady_eos_sync_interval
                    )
                    preview_ready = emitted_chunks == 0 and (
                        buffered_count + pending_count >= preview_target_tokens
                    )
                    return bool(
                        preview_ready
                        or pending_count >= active_eos_sync_interval
                        or step + 1 >= maximum
                    )

                semantic_batches = self.semantic_runtime.iter_batches(
                    semantic_state,
                    sync_policy=semantic_sync_policy,
                    cancelled=cancelled,
                )
                for semantic_batch in semantic_batches:
                    for offset in range(semantic_batch.accepted_tokens):
                        accepted = semantic_batch.tokens[:, offset : offset + 1]
                        token_buffer.append(accepted)
                        segment_token_parts.append(accepted)
                        buffered_count += 1
                        segment_tokens += 1

                        split = False
                        if mute_matrix is not None and buffered_count >= chunk_length + 2:
                            recent = torch.cat(token_buffer, dim=1).flatten()
                            scores = mute_matrix[recent].clone() - 0.3
                            if scores.numel() > 1:
                                scores[:-1] += scores[1:]
                            split_index = int(torch.argmax(scores).item())
                            if (
                                float(scores[split_index].item()) >= 0.0
                                and split_index + 1 >= chunk_length
                            ):
                                split_at = split_index + 1
                                queue.append(torch.cat(token_buffer[:split_at], dim=1))
                                token_buffer = token_buffer[split_at:]
                                buffered_count -= split_at
                                split = True
                        elif mute_matrix is None:
                            target = (
                                preview_target_tokens
                                if emitted_chunks == 0
                                else chunk_length
                            )
                            if buffered_count >= target:
                                current_chunk = torch.cat(token_buffer[:target], dim=1)
                                emitted = emit(current_chunk, None, final=False)
                                trimmed_before_preview = float(
                                    profile.get(
                                        "trimmed_natural_leading_silence_seconds", 0.0
                                    )
                                )
                                if emitted.size:
                                    emitted = prepare_segment_head(emitted)
                                short_initial_preview = bool(
                                    emitted.size
                                    and segment_index == 0
                                    and target < maximum_steps
                                    and emitted.size
                                    < int(
                                        round(
                                            self.sample_rate
                                            * MIN_INITIAL_PREVIEW_PUBLISHED_SECONDS
                                        )
                                    )
                                )
                                if short_initial_preview:
                                    # A model with a large acoustic overlap can
                                    # leave too little playable PCM in its first
                                    # preview. Publishing it would underrun before
                                    # the full-context refill arrives. Retry with
                                    # the same semantic prefix plus a few tokens;
                                    # do not conceal the gap with synthetic audio.
                                    emitted = np.empty(0, dtype=np.float32)
                                    trim_natural_segment_head = True
                                    profile[
                                        "trimmed_natural_leading_silence_seconds"
                                    ] = trimmed_before_preview
                                    profile["short_preview_attempts"] = int(
                                        profile.get("short_preview_attempts", 0)
                                    ) + 1
                                profile["preview_published_samples"] = int(
                                    emitted.size
                                )
                                if emitted.size:
                                    if segment_index == 0:
                                        self._preview_token_hint = self._remember_preview_target(
                                            cached_tokens=self._preview_token_hint,
                                            successful_tokens=target,
                                        )
                                    profile["preview_target_tokens"] = int(target)
                                    token_buffer = token_buffer[target:]
                                    buffered_count -= target
                                    emitted_chunks += 1
                                    for pending in consume_pending_tail():
                                        yield note_audio(
                                            pending, max(0, segment_index - 1)
                                        )
                                    for ready in hold_natural_trailing_silence(emitted):
                                        yield note_audio(ready, segment_index)
                                else:
                                    native_semantic_history = None
                                    native_previous_latent = None
                                    native_previous_audio_tail = None
                                    native_preview_audio_samples = 0
                                    native_acoustic_frame_cursor = 0
                                    if not short_initial_preview:
                                        profile["silent_preview_attempts"] = int(
                                            profile.get("silent_preview_attempts", 0)
                                        ) + 1
                                    preview_target_tokens = min(
                                        maximum_steps,
                                        target + preview_retry_tokens,
                                    )

                        while split and len(queue) > 1:
                            current_chunk = queue.pop(0)
                            emitted = emit(current_chunk, queue[0], final=False)
                            if emitted.size:
                                emitted = prepare_segment_head(emitted)
                                for pending in consume_pending_tail():
                                    yield note_audio(pending, max(0, segment_index - 1))
                                for ready in hold_natural_trailing_silence(emitted):
                                    yield note_audio(ready, segment_index)

                if cancelled is not None and cancelled.is_set():
                    if pending_tail is not None:
                        pending_tail.future.result()
                    return
                decode_elapsed = time.perf_counter() - decode_started
                segment_steps = semantic_state.steps
                profile["gpt_decode_seconds"] += decode_elapsed
                segment_profile["gpt_decode_seconds"] = decode_elapsed
                profile["sample_cuda_graph_enabled"] = int(
                    semantic_state.sample_cuda_graph_enabled
                )
                profile["sample_cuda_graph_cache_hits"] += int(
                    semantic_state.sample_cuda_graph_cache_hits
                )
                profile["sample_cuda_graph_captures"] += int(
                    semantic_state.sample_cuda_graph_captures
                )
                profile["semantic_tokens"] += segment_tokens
                profile["gpt_steps"] += segment_steps
                profile["semantic_nfe"] += segment_steps
                profile["host_sync_count"] += semantic_state.host_sync_count
                profile["host_sync_seconds"] += semantic_state.host_sync_seconds
                profile["attention_kv_bytes"] = max(
                    int(profile["attention_kv_bytes"]),
                    semantic_state.attention_kv_bytes,
                )
                if profile["first_preview_semantic_seconds"] == 0.0:
                    profile["first_preview_semantic_seconds"] = (
                        semantic_state.first_batch_seconds
                    )
                segment_profile["semantic_tokens"] = segment_tokens
                segment_profile["gpt_steps"] = segment_steps
                segment_profile["semantic_nfe"] = segment_steps
                segment_profile["host_sync_count"] = semantic_state.host_sync_count
                segment_profile["host_sync_seconds"] = semantic_state.host_sync_seconds
                segment_profile["attention_kv_bytes"] = semantic_state.attention_kv_bytes

                use_full_context_refill = (
                    low_latency_stream
                    and emitted_chunks == 1
                    and native_previous_audio_tail is not None
                    and bool(segment_token_parts)
                )
                if use_full_context_refill:
                    full_tokens = torch.cat(segment_token_parts, dim=1)
                    full_token_count = int(full_tokens.shape[-1])
                    semantic_limit = int(
                        self.engine.model_sovits_stream.input_max_shapes.get(
                            "pred_semantic", (1, 1, 250)
                        )[-1]
                    )
                    if full_token_count <= semantic_limit:
                        refill_started = time.perf_counter()
                        full_audio, _, _ = self._decode_native_stream_chunk(
                            cumulative_tokens=full_tokens,
                            new_token_count=full_token_count,
                            phones=phones,
                            noise_scale=noise_scale,
                            speed=speed,
                            previous_latent=None,
                            acoustic_noise=acoustic_noise[
                                :, :, : full_token_count * 2
                            ].contiguous(),
                        )
                        profile["sovits_seconds"] += (
                            time.perf_counter() - refill_started
                        )
                        profile["sovits_invocations"] += 1
                        expected_start = max(
                            0,
                            native_preview_audio_samples
                            - int(native_previous_audio_tail.size),
                        )
                        emitted, refill_start, refill_score = (
                            self._refill_from_full_context(
                                native_previous_audio_tail,
                                full_audio,
                                expected_start=expected_start,
                                sample_rate=self.sample_rate,
                            )
                        )
                        native_previous_audio_tail = None
                        native_previous_latent = None
                        token_buffer = []
                        queue.clear()
                        queue.append(emitted)
                        profile["full_context_refills"] = int(
                            profile.get("full_context_refills", 0)
                        ) + 1
                        segment_profile["refill_start_sample"] = int(refill_start)
                        segment_profile["refill_match_score"] = float(refill_score)
                    else:
                        use_full_context_refill = False
                if not use_full_context_refill and token_buffer:
                    queue.append(torch.cat(token_buffer, dim=1))
                while queue:
                    current_chunk = queue.pop(0)
                    if isinstance(current_chunk, np.ndarray):
                        emitted = current_chunk
                    else:
                        emitted = emit(
                            current_chunk,
                            queue[0] if queue else None,
                            final=not queue,
                        )
                    if emitted.size:
                        emitted = prepare_segment_head(emitted)
                        is_segment_tail = not queue and segment_index + 1 < len(segments)
                        if is_segment_tail and technical_continuation:
                            detected_silence = self._trailing_silence_seconds(
                                emitted, self.sample_rate
                            )
                            emitted, _, trimmed_seconds = (
                                self._trim_excess_trailing_silence(
                                    emitted,
                                    sample_rate=self.sample_rate,
                                    trailing_silence_seconds=detected_silence,
                                    target_pause_seconds=0.005,
                                )
                            )
                            profile["trimmed_trailing_silence_seconds"] = float(
                                profile.get("trimmed_trailing_silence_seconds", 0.0)
                            ) + trimmed_seconds
                            if emitted.size:
                                reserve = min(technical_crossfade_samples, int(emitted.size))
                                pending_technical_tail = emitted[-reserve:].copy()
                                emitted = emitted[:-reserve]
                        if emitted.size:
                            for ready in hold_natural_trailing_silence(emitted):
                                yield note_audio(ready, segment_index)
                if segment_index + 1 < len(segments) and not technical_continuation:
                    target_pause_seconds = self._target_pause_seconds(
                        raw_text, pause_length
                    )
                    target_pause_samples = int(
                        round(self.sample_rate * target_pause_seconds)
                    )
                    retained_samples = min(
                        target_pause_samples, int(pending_natural_silence.size)
                    )
                    pause_parts: list[np.ndarray] = []
                    if retained_samples > 0:
                        pause_parts.append(pending_natural_silence[:retained_samples])
                    missing_samples = max(0, target_pause_samples - retained_samples)
                    if missing_samples > 0:
                        pause_parts.append(np.zeros(missing_samples, dtype=np.float32))
                        profile["inserted_pause_seconds"] = float(
                            profile.get("inserted_pause_seconds", 0.0)
                        ) + float(missing_samples) / float(self.sample_rate)
                    trimmed_samples = max(
                        0, int(pending_natural_silence.size) - retained_samples
                    )
                    profile["trimmed_trailing_silence_seconds"] = float(
                        profile.get("trimmed_trailing_silence_seconds", 0.0)
                    ) + float(trimmed_samples) / float(self.sample_rate)
                    if pause_parts:
                        yield note_audio(np.concatenate(pause_parts), segment_index)
                    pending_natural_silence = np.empty(0, dtype=np.float32)
                    trim_natural_segment_head = True

                if technical_continuation and segment_token_parts:
                    continuation_phones = list(phones)
                    continuation_bert = bert.detach()
                    continuation_tokens = (
                        torch.cat(segment_token_parts, dim=1).detach().clone()
                    )
                    previous_was_technical = True
                else:
                    continuation_phones = []
                    continuation_bert = None
                    continuation_tokens = None
                    previous_was_technical = False

            for pending in consume_pending_tail():
                yield note_audio(pending, max(0, len(segments) - 1))
            if pending_technical_tail is not None:
                yield note_audio(pending_technical_tail, max(0, len(segments) - 1))
        semantic_nfe = int(profile["semantic_nfe"])
        if semantic_nfe > 0:
            profile["semantic_tokens_per_nfe"] = float(profile["semantic_tokens"]) / float(
                semantic_nfe
            )
        mtp_proposed = int(profile["mtp_proposed_tokens"])
        if mtp_proposed > 0:
            profile["mtp_acceptance_rate"] = float(
                profile["mtp_accepted_tokens"]
            ) / float(mtp_proposed)
        profile["total_pipeline_seconds"] = time.perf_counter() - request_started
        self.last_profile = profile
        self.logger.info(
            "TensorRT request completed: segments=%d tokens=%d gpt=%.3fs "
            "sovits=%.3fs first_audio=%.3fs preview=%d/%d/%d "
            "target=%d attempts=%d total=%.3fs",
            int(profile["segments"]),
            int(profile["semantic_tokens"]),
            float(profile["gpt_decode_seconds"]),
            float(profile["sovits_seconds"]),
            float(profile["first_audio_seconds"]),
            int(profile.get("preview_decoded_samples", 0)),
            int(profile.get("preview_held_samples", 0)),
            int(profile.get("preview_published_samples", 0)),
            int(profile.get("preview_target_tokens", 0)),
            int(profile.get("silent_preview_attempts", 0)),
            float(profile["total_pipeline_seconds"]),
        )
        self.logger.debug("TensorRT detailed request profile: %s", profile)
