"""Fixed-reference, low-latency TensorRT streaming for AnifLive-TTS.

The upstream ``api_server_trt.py`` demonstrates token streaming, but its
reference preparation still relies on non-TensorRT helper models and its WAV
response has no final RIFF data length.  This module keeps all eight exported
TensorRT engines in use, prepares the immutable voice reference once at
startup, and exposes PCM chunks to the HTTP facade.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import soundfile as sf
import soxr


@dataclass(frozen=True)
class PreparedReference:
    """GPU-resident features extracted from the fixed reference recording."""

    prompt_semantic: Any
    spectrogram: Any
    speaker_embedding: Any
    phones: list[int]
    bert: Any


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
        maximum_steps: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> None:
        self.torch = torch
        self.stream = stream
        self.maximum_steps = int(maximum_steps)
        self.parameters = (float(temperature), int(top_k), float(top_p))
        self.input_addresses = (topk_values.data_ptr(), topk_indices.data_ptr())
        self.token_storage = torch.empty(
            (self.maximum_steps, 1), dtype=topk_indices.dtype, device=topk_values.device
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
        maximum_steps: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> bool:
        return (
            self.input_addresses == (topk_values.data_ptr(), topk_indices.data_ptr())
            and self.maximum_steps == int(maximum_steps)
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
        self.reference: PreparedReference | None = None
        self._mute_matrix: Any | None = None
        self._mute_matrix_checked = False
        self._gpt_destination_cache: tuple[Any, Any] | None = None
        self._gpt_step_graphs: _GPTStepCudaGraphs | None = None
        self._gpt_graph_disabled = False
        self._gpt_graph_notice_emitted = False
        self._sample_graph: _SampleCudaGraph | None = None
        self._sample_graph_disabled = False
        self.last_profile: dict[str, float | int] | None = None

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
        self.logger.info(
            "Prepared fixed voice reference once: prompt=%d tokens, spectrogram=%s",
            int(prompt_semantic.shape[-1]),
            tuple(int(value) for value in spectrogram.shape),
        )
        return self.reference

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
    ) -> np.ndarray:
        reference = self.reference
        if reference is None:
            raise RuntimeError("The fixed voice reference has not been prepared")

        inputs = []
        if history is not None:
            inputs.append(history[:, -history_tokens:])
        inputs.append(tokens)
        if lookahead is not None:
            inputs.append(lookahead[:, :lookahead_tokens])
        all_tokens = self.torch.cat(inputs, dim=1)
        semantic = all_tokens[:, None, :]
        text_seq = self.torch.tensor(phones, dtype=self.torch.int64, device=self.engine.device)[None, :]
        sovits_inputs = {
            "pred_semantic": semantic.to(self.torch.int64),
            "text_seq": text_seq,
            "refer_spec": reference.spectrogram,
            "sv_emb": reference.speaker_embedding,
            "noise_scale": self.torch.tensor([noise_scale], dtype=self.torch.float32, device=self.engine.device),
            "speed": self.torch.tensor([speed], dtype=self.torch.float32, device=self.engine.device),
        }
        sovits_inputs = {
            name: value
            for name, value in sovits_inputs.items()
            if name in self.engine.model_sovits.input_names
        }
        audio = self.engine.model_sovits(sovits_inputs)["audio"].flatten()
        samples_per_token = float(audio.numel()) / max(1, int(all_tokens.shape[-1]))
        history_count = min(int(history.shape[-1]), history_tokens) if history is not None else 0
        begin = min(int(history_count * samples_per_token), int(audio.numel()))
        count = max(1, int(int(tokens.shape[-1]) * samples_per_token))
        end = min(begin + count, int(audio.numel()))
        result = audio[begin:end].detach().float().cpu().numpy()
        if result.size == 0 or not np.isfinite(result).all():
            raise RuntimeError("TensorRT SoVITS produced an invalid streaming audio chunk")
        return result.astype(np.float32, copy=False)

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
        chunk_length: int | None = None,
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
        profile: dict[str, float | int] = {
            "text_processing_seconds": 0.0,
            "gpt_encoder_seconds": 0.0,
            "gpt_decode_seconds": 0.0,
            "sovits_seconds": 0.0,
            "semantic_tokens": 0,
            "gpt_steps": 0,
            "sovits_invocations": 0,
            "segments": len(segments),
        }
        self.last_profile = None

        torch = self.torch
        mute_matrix = self._load_mute_matrix()
        history_tokens, lookahead_tokens, fade_samples = 32, 16, 256
        prior_fade: np.ndarray | None = None

        with torch.cuda.stream(self.engine.stream):
            for segment_index, text in enumerate(segments):
                text_started = time.perf_counter()
                phones, bert, _ = self.engine.get_phones_and_bert(
                    text, text_language, self.engine.version, default_lang=self.reference_language
                )
                profile["text_processing_seconds"] += time.perf_counter() - text_started
                max_text = self.engine.sovits_max_text_len
                if max_text is not None and len(phones) > int(max_text):
                    raise ValueError(
                        f"A text segment requires {len(phones)} phonemes, exceeding this TensorRT profile limit of {max_text}."
                    )
                merged_bert = torch.cat([reference.bert, bert], dim=1)[None, :, :].to(self.engine.precision)
                phoneme_ids = torch.tensor(
                    reference.phones + list(phones), dtype=torch.int64, device=self.engine.device
                )[None, :]
                phoneme_length = torch.tensor(
                    [phoneme_ids.shape[-1]], dtype=torch.int64, device=self.engine.device
                )

                encoder_started = time.perf_counter()
                encoded = self.engine.model_gpt_enc(
                    {
                        "phoneme_ids": phoneme_ids,
                        "phoneme_ids_len": phoneme_length,
                        "prompts": reference.prompt_semantic.to(torch.int64),
                        "bert_feature": merged_bert,
                    }
                )
                profile["gpt_encoder_seconds"] += time.perf_counter() - encoder_started
                # TensorRT exposes these tensors on CUDA.  Keep the complete
                # sampling operation there: forcing a GPU -> CPU -> GPU round
                # trip for every autoregressive token serializes the stream
                # and was the largest avoidable throughput loss versus the
                # older TensorRT 11 implementation.
                current = self.sample_topk(
                    encoded["topk_values"].detach(),
                    encoded["topk_indices"].detach(),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                ).to(self.engine.device)

                k_cache = encoded["k_cache"]
                v_cache = encoded["v_cache"]
                encoded_lengths = torch.stack(
                    (encoded["x_len"].reshape(-1)[0], encoded["y_len"].reshape(-1)[0])
                ).detach().cpu().tolist()
                maximum_steps = max(
                    1,
                    min(
                        1000,
                        int(k_cache.shape[2])
                        - int(encoded_lengths[0])
                        - int(encoded_lengths[1])
                        - 1,
                    ),
                )
                x_length = self._prepare_step_input("x_len", encoded["x_len"])
                y_length = self._prepare_step_input("y_len", encoded["y_len"])
                step_cache_shape = tuple(
                    int(value)
                    for value in self.engine.model_gpt_step.engine.get_tensor_shape(
                        "k_cache"
                    )
                )
                step_cache_capacity = int(step_cache_shape[2])
                required_cache = int(encoded_lengths[0]) + int(encoded_lengths[1]) + 2
                if required_cache > step_cache_capacity:
                    raise RuntimeError(
                        "GPT step cache profile is too short for this segment: "
                        f"requires {required_cache}, engine capacity {step_cache_capacity}"
                    )
                if int(k_cache.shape[2]) != step_cache_capacity:
                    # A fitted short-step engine avoids rewriting the unused
                    # tail of the 1000-token encoder cache at every AR step.
                    # Materialize the compact layout once per request because
                    # a sliced view retains the original 1000-token stride.
                    k_cache = k_cache[:, :, :step_cache_capacity, :].contiguous()
                    v_cache = v_cache[:, :, :step_cache_capacity, :].contiguous()
                # ``model_gpt_step`` writes both cache outputs in full.  The
                # encoder cache can therefore be used directly, and the
                # alternating destination cache need not be zero-filled.  It
                # avoids two large device-to-device copies plus a memset per
                # request without changing TensorRT I/O ownership.
                if (
                    self._gpt_destination_cache is None
                    or self._gpt_destination_cache[0].shape != k_cache.shape
                    or self._gpt_destination_cache[0].dtype != k_cache.dtype
                ):
                    self._gpt_destination_cache = (
                        torch.empty_like(k_cache),
                        torch.empty_like(v_cache),
                    )
                cache_pair = [(k_cache, v_cache), self._gpt_destination_cache]
                index_location = self.engine.model_gpt_step.tensor_location.get(
                    "idx", self.trt.TensorLocation.DEVICE
                )
                index_device = "cpu" if index_location == self.trt.TensorLocation.HOST else self.engine.device
                cuda_graphs: _GPTStepCudaGraphs | None = None
                full_graph_configured = (
                    os.environ.get("ANIFLIVE_TTS_CUDA_GRAPH", "0").strip().lower()
                    not in {"0", "false", "no", "off"}
                )
                if full_graph_configured and not self._gpt_graph_notice_emitted:
                    self.logger.warning(
                        "Full GPT TensorRT CUDA Graph is disabled: TensorRT's internal "
                        "train-station operation rejects stream capture even when the engine "
                        "is built with max_aux_streams=0; sampling-only CUDA Graph remains active"
                    )
                    self._gpt_graph_notice_emitted = True
                cuda_graph_requested = False
                if cuda_graph_requested and not self._gpt_graph_disabled:
                    try:
                        if (
                            self._gpt_step_graphs is None
                            or not self._gpt_step_graphs.matches(cache_pair)
                        ):
                            self._gpt_step_graphs = _GPTStepCudaGraphs(
                                torch=torch,
                                model=self.engine.model_gpt_step,
                                stream=self.engine.stream,
                                cache_pair=cache_pair,
                                sample=current.to(torch.int64),
                                x_length=x_length,
                                y_length=y_length,
                            )
                        else:
                            self._gpt_step_graphs.prepare(
                                sample=current.to(torch.int64),
                                x_length=x_length,
                                y_length=y_length,
                            )
                        cuda_graphs = self._gpt_step_graphs
                    except Exception as exc:
                        self._gpt_step_graphs = None
                        self._gpt_graph_disabled = True
                        self.logger.warning(
                            "GPT step CUDA Graph disabled after capture failure: %s", exc
                        )
                indices = (
                    None
                    if cuda_graphs is not None
                    else torch.arange(maximum_steps, dtype=torch.int64, device=index_device)
                )
                profile["cuda_graph_enabled"] = int(cuda_graphs is not None)
                outputs = {"k_cache_new": None, "v_cache_new": None}
                queue: list[Any] = []
                token_buffer: list[Any] = [current]
                segment_token_buffer: list[Any] = [current]
                buffered_count = 1
                history: Any | None = None
                emitted_chunks = 0
                streamed_samples = 0
                low_latency_stream = chunk_length <= self.streaming_chunk_length()
                first_chunk_tokens = (
                    int(os.environ.get("ANIFLIVE_TTS_FIRST_CHUNK_TOKENS", "8"))
                    if low_latency_stream
                    else chunk_length
                )
                first_lookahead_tokens = (
                    int(os.environ.get("ANIFLIVE_TTS_FIRST_LOOKAHEAD_TOKENS", "8"))
                    if low_latency_stream
                    else lookahead_tokens
                )

                def emit(tokens: Any, next_tokens: Any | None, *, final: bool) -> np.ndarray:
                    nonlocal prior_fade, history
                    decode_started = time.perf_counter()
                    audio = self._decode_chunk(
                        tokens=tokens,
                        history=history,
                        lookahead=next_tokens,
                        phones=phones,
                        noise_scale=noise_scale,
                        speed=speed,
                        history_tokens=history_tokens,
                        lookahead_tokens=lookahead_tokens,
                    )
                    profile["sovits_seconds"] += time.perf_counter() - decode_started
                    profile["sovits_invocations"] += 1
                    if prior_fade is not None:
                        count = min(fade_samples, int(audio.size), int(prior_fade.size))
                        if count:
                            fade_in = np.linspace(0.0, 1.0, count, dtype=np.float32)
                            audio[:count] = audio[:count] * fade_in + prior_fade[-count:] * (1.0 - fade_in)
                    if final:
                        emitted = audio
                        prior_fade = None
                    else:
                        count = min(fade_samples, int(audio.size))
                        if count:
                            emitted = audio[:-count]
                            prior_fade = audio[-count:].copy()
                        else:
                            emitted = audio
                            prior_fade = None
                    history = tokens if history is None else torch.cat([history, tokens], dim=1)[:, -history_tokens:]
                    return emitted

                decode_started = time.perf_counter()
                segment_tokens = 1
                segment_steps = 0
                # Queue several TensorRT steps before reading EOS back to the
                # CPU. This removes most per-token CUDA synchronization while
                # retaining the exact accepted token sequence. Multi-segment
                # requests stay at interval 1 because speculative draws after
                # EOS would otherwise change the RNG state of the next segment.
                eos_sync_interval = 1 if len(segments) > 1 else max(
                    1, min(8, int(os.environ.get("ANIFLIVE_TTS_EOS_SYNC_INTERVAL", "2")))
                )
                pending_tokens: list[Any] = []
                pending_storage_start: int | None = None
                for step in range(maximum_steps):
                    if cuda_graphs is not None:
                        decoded = cuda_graphs.replay(step)
                    else:
                        source_cache = cache_pair[step % 2]
                        destination_cache = cache_pair[(step + 1) % 2]
                        outputs["k_cache_new"], outputs["v_cache_new"] = destination_cache
                        decoded = self.engine.model_gpt_step(
                            {
                                "samples": current.to(torch.int64),
                                "k_cache": source_cache[0],
                                "v_cache": source_cache[1],
                                "idx": indices[step : step + 1],
                                "x_len": x_length,
                                "y_len": y_length,
                            },
                            outputs=outputs,
                            sync=False,
                        )
                    sample_graph_requested = (
                        os.environ.get("ANIFLIVE_TTS_SAMPLE_CUDA_GRAPH", "1")
                        .strip()
                        .lower()
                        not in {"0", "false", "no", "off"}
                    )
                    if sample_graph_requested and not self._sample_graph_disabled:
                        try:
                            if (
                                self._sample_graph is None
                                or not self._sample_graph.matches(
                                    topk_values=decoded["topk_values"],
                                    topk_indices=decoded["topk_indices"],
                                    maximum_steps=maximum_steps,
                                    temperature=temperature,
                                    top_k=top_k,
                                    top_p=top_p,
                                )
                            ):
                                self._sample_graph = _SampleCudaGraph(
                                    torch=torch,
                                    sample_topk=self.sample_topk,
                                    stream=self.engine.stream,
                                    topk_values=decoded["topk_values"],
                                    topk_indices=decoded["topk_indices"],
                                    maximum_steps=maximum_steps,
                                    temperature=temperature,
                                    top_k=top_k,
                                    top_p=top_p,
                                )
                            current, stored_current = self._sample_graph.replay(step)
                            if not pending_tokens:
                                pending_storage_start = step
                        except Exception as exc:
                            self._sample_graph = None
                            self._sample_graph_disabled = True
                            self.logger.warning(
                                "Sampling CUDA Graph disabled after capture failure: %s", exc
                            )
                    if (
                        not sample_graph_requested
                        or self._sample_graph is None
                        or self._sample_graph_disabled
                    ):
                        pending_storage_start = None
                        current = self.sample_topk(
                            decoded["topk_values"].detach(),
                            decoded["topk_indices"].detach(),
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                        ).to(self.engine.device)
                        stored_current = current
                    if cuda_graphs is not None:
                        cuda_graphs.update_sample(current.to(torch.int64))
                    segment_steps += 1
                    pending_tokens.append(stored_current)
                    if len(pending_tokens) < eos_sync_interval and step + 1 < maximum_steps:
                        continue

                    if pending_storage_start is not None and self._sample_graph is not None:
                        pending_batch = self._sample_graph.token_storage[
                            pending_storage_start : step + 1
                        ].reshape(1, -1)
                    else:
                        pending_batch = torch.cat(pending_tokens, dim=1)
                    pending_values = pending_batch.detach().cpu().reshape(-1).tolist()
                    eos_offset = next(
                        (offset for offset, value in enumerate(pending_values) if int(value) == 1024),
                        None,
                    )
                    accepted_tokens = (
                        pending_tokens if eos_offset is None else pending_tokens[:eos_offset]
                    )
                    for accepted in accepted_tokens:
                        token_buffer.append(accepted)
                        segment_token_buffer.append(accepted)
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
                            target = first_chunk_tokens if emitted_chunks == 0 else chunk_length
                            lookahead = (
                                first_lookahead_tokens
                                if emitted_chunks == 0
                                else lookahead_tokens
                            )
                            if buffered_count >= target + lookahead:
                                current_chunk = torch.cat(token_buffer[:target], dim=1)
                                next_tokens = torch.cat(
                                    token_buffer[target : target + lookahead], dim=1
                                )
                                token_buffer = token_buffer[target:]
                                buffered_count -= target
                                emitted = emit(current_chunk, next_tokens, final=False)
                                emitted_chunks += 1
                                if emitted.size:
                                    streamed_samples += int(emitted.size)
                                    yield emitted

                        while split and len(queue) > 1:
                            current_chunk = queue.pop(0)
                            emitted = emit(current_chunk, queue[0], final=False)
                            if emitted.size:
                                yield emitted

                    pending_tokens.clear()
                    pending_storage_start = None
                    if eos_offset is not None:
                        break

                profile["gpt_decode_seconds"] += time.perf_counter() - decode_started
                profile["sample_cuda_graph_enabled"] = int(
                    self._sample_graph is not None and not self._sample_graph_disabled
                )
                profile["semantic_tokens"] += segment_tokens
                profile["gpt_steps"] += segment_steps

                if low_latency_stream and emitted_chunks:
                    # The preview chunk minimizes TTFA.  Once the semantic
                    # sequence is complete, decode it once with full context
                    # and continue from the already played sample.  This keeps
                    # the long tail identical to the complete-WAV decoder and
                    # avoids accumulating voice drift from short SoVITS calls.
                    full_decode_started = time.perf_counter()
                    full_tokens = torch.cat(segment_token_buffer, dim=1)
                    full_audio = self._decode_chunk(
                        tokens=full_tokens,
                        history=None,
                        lookahead=None,
                        phones=phones,
                        noise_scale=noise_scale,
                        speed=speed,
                        history_tokens=history_tokens,
                        lookahead_tokens=lookahead_tokens,
                    )
                    profile["sovits_seconds"] += time.perf_counter() - full_decode_started
                    profile["sovits_invocations"] += 1
                    suffix = full_audio[min(streamed_samples, int(full_audio.size)) :].copy()
                    if prior_fade is not None and suffix.size:
                        count = min(fade_samples, int(suffix.size), int(prior_fade.size))
                        fade_in = np.linspace(0.0, 1.0, count, dtype=np.float32)
                        suffix[:count] = (
                            suffix[:count] * fade_in
                            + prior_fade[-count:] * (1.0 - fade_in)
                        )
                    prior_fade = None
                    if suffix.size:
                        yield suffix
                else:
                    if token_buffer:
                        queue.append(torch.cat(token_buffer, dim=1))
                    while queue:
                        current_chunk = queue.pop(0)
                        emitted = emit(current_chunk, queue[0] if queue else None, final=not queue)
                        if emitted.size:
                            yield emitted

                if segment_index + 1 < len(segments) and pause_length > 0:
                    prior_fade = None
                    yield np.zeros(int(round(self.sample_rate * pause_length)), dtype=np.float32)
        profile["total_pipeline_seconds"] = time.perf_counter() - request_started
        self.last_profile = profile
