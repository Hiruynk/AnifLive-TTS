"""Opt-in fixed-shape v5 CFM CUDA graphs over the existing TensorRT DiT engine.

Call prepare() only under the runtime's exclusive lifecycle lock and with an
AR-idle predicate. Inference never captures: an unprepared shape deliberately
uses the existing eager CFM loop and increments the miss counter. Capture or
execution failures propagate; they are never retried through another path.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import copy
import logging
from pathlib import Path
import threading
import time
from types import SimpleNamespace


LOGGER = logging.getLogger(__name__)


def _precision_snapshot(torch):
    """Read operator-specific precision without legacy mixed-flag getters."""
    return (str(torch.backends.fp32_precision),
            str(torch.backends.cuda.matmul.fp32_precision),
            str(torch.backends.cudnn.conv.fp32_precision),
            str(torch.backends.cudnn.rnn.fp32_precision))


class CFMGraphAdapter:
    def __init__(self, cfm, plan, *, ar_is_idle=None, collect_misses=False, positive_only=False,
                 shape_plans=None):
        from .trt_dit import TensorRTDiT

        if ar_is_idle is not None and not callable(ar_is_idle):
            raise TypeError("ar_is_idle must be callable or None")
        if type(collect_misses) is not bool:
            raise TypeError("collect_misses must be a boolean")
        if type(positive_only) is not bool:
            raise TypeError("positive_only must be a boolean")
        if shape_plans is not None and not isinstance(shape_plans, Mapping):
            raise TypeError('shape_plans must map integer frame counts to plan paths')
        if not isinstance(cfm.estimator, TensorRTDiT):
            raise ValueError("CFM graph requires the existing TensorRT DiT estimator")
        if Path(cfm.estimator.runner.engine_path).resolve() != Path(plan).resolve():
            raise ValueError("CFM graph plan must match the installed TensorRT DiT engine")
        self.cfm = cfm
        self.original_inference = cfm.inference
        self.eager_estimator = cfm.estimator
        self.native = self.eager_estimator.native
        self.ar_is_idle = ar_is_idle
        self.collect_misses = collect_misses
        self.positive_only = positive_only
        self.shape_plans = {}
        for frames, path in (shape_plans or {}).items():
            if type(frames) is not int or not 128 <= frames <= 1000:
                raise ValueError('Specialized CFM frame counts must be integers from 128 to 1000')
            self.shape_plans[frames] = Path(path).resolve(strict=True)
        self._static_runners = {}
        self.specialized_captures = {}
        self.specialized_hits = {}
        self.contexts = OrderedDict()
        self.pending = OrderedDict()
        self.pending_error = None
        self._lock = threading.RLock()
        self._last_eager_done = None
        self.enabled = False
        self.closed = False
        self.hits = 0
        self.misses = 0
        self.captures = 0
        self.evictions = 0
        self.prepare_seconds = 0.0
        self._check_cfm()

    def _check_cfm(self):
        if (self.cfm.estimator is not self.eager_estimator or self.cfm.in_channels != 100
                or self.cfm.noise_temperature != 0.875 or self.cfm.use_step_embedding
                or not self.cfm.use_conditioner_cache or not self.cfm.use_static_cache):
            raise RuntimeError("CFM graph requires the original v5 Turbo static-cache TensorRT DiT configuration")
        if not self.native.text_embed.extra_modeling:
            raise RuntimeError("CFM graph requires the v5 positional/ConvNeXt text path")

    def _validate(self, mu, lengths, prompt):
        import torch
        if any(not isinstance(x, torch.Tensor) for x in (mu, lengths, prompt)):
            raise TypeError("CFM inputs must be Torch tensors")
        device = torch.device("cuda:0")
        if any(x.device != device for x in (mu, lengths, prompt)):
            raise ValueError("CFM inputs must all be on cuda:0")
        if mu.dtype != torch.float16 or prompt.dtype != torch.float16 or lengths.dtype != torch.int64:
            raise ValueError("CFM mu/prompt must be FP16 and lengths int64")
        if mu.ndim != 3 or mu.shape[0] != 1 or mu.shape[2] != 512:
            raise ValueError("CFM mu must have shape [1,T,512]")
        if prompt.ndim != 3 or tuple(prompt.shape[:2]) != (1, 100) or tuple(lengths.shape) != (1,):
            raise ValueError("CFM prompt/lengths must have shapes [1,100,P] and [1]")
        frames, prompt_frames = int(mu.shape[1]), int(prompt.shape[-1])
        if not 128 <= frames <= 1000 or not 0 <= prompt_frames <= frames:
            raise ValueError("CFM context is outside the TensorRT DiT profile")
        return frames, prompt_frames

    def _static_cache(self, cond0, lengths, text0, position_embedding):
        import torch
        from module.commons import sequence_mask
        text, cond = text0.transpose(2, 1), cond0.transpose(2, 1)
        if text.shape[:2] != position_embedding.shape[:2]:
            raise RuntimeError("Fixed positional embedding shape changed")
        embedded = self.native.text_embed.text_blocks(text + position_embedding)
        projection = self.native.input_embed.proj
        mel_dim = cond.shape[-1]
        positive = torch.nn.functional.linear(torch.cat((cond, embedded), -1),
                                              projection.weight[:, mel_dim:], projection.bias)
        # This adapter accepts CFG 0 only. The DiT step reads the positive
        # condition; the unconditioned projection is unused in that flow.
        negative = None
        if not self.positive_only:
            negative = torch.nn.functional.linear(torch.cat((torch.zeros_like(cond), embedded), -1),
                                                  projection.weight[:, mel_dim:], projection.bias)
        cache = {"condition": positive,
                 "mask": sequence_mask(lengths, max_length=cond.shape[1]).to(cond.device),
                 "rope": self.native.rotary_embed.forward_from_seq_len(cond.shape[1])}
        if negative is not None:
            cache['negative_condition'] = negative
        return cache

    def _flow(self, entry):
        import torch
        x = entry.initial.clone()
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :entry.prompt_frames] = entry.prompt
        text0 = entry.mu.transpose(2, 1)
        cache = self._static_cache(prompt_x, entry.lengths, text0, entry.position_embedding)
        for timestep in entry.times:
            velocity, _, _ = entry.estimator(x, prompt_x, entry.lengths, timestep, text0=text0,
                infer=True, static_cache=cache, drop_audio_cond=False, drop_text=False)
            x = x + 0.25 * velocity.transpose(2, 1)
            x[..., :entry.prompt_frames] = 0
        return x

    def _new_estimator(self, frames):
        import torch
        # Keep the same ICudaEngine/runtime/weights; each captured shape gets an
        # independent execution context and fresh binding metadata/stream.
        estimator = copy.copy(self.eager_estimator)
        if frames in self.shape_plans:
            from aniflive_tts.backend.trt_runtime import TensorRTRunner
            import tensorrt as trt
            source = self._static_runners.get(frames)
            if source is None:
                source = TensorRTRunner(self.shape_plans[frames], device='cuda:0')
                shapes = {'x': (1, 100, frames), 'time': (1,), 'condition': (1, frames, 1024),
                          'mask': (1, frames), 'rope': (1, frames, 64), 'velocity': (1, frames, 100)}
                types = {'x': trt.float16, 'time': trt.float16, 'condition': trt.float16,
                         'mask': trt.bool, 'rope': trt.float16, 'velocity': trt.float16}
                if (set(source.input_names) != set(shapes) - {'velocity'} or source.output_names != ('velocity',)
                        or int(source.engine.num_aux_streams) != 0 or any(
                            tuple(source.engine.get_tensor_shape(name)) != shape
                            or source.engine.get_tensor_dtype(name) != types[name] for name, shape in shapes.items())):
                    raise ValueError(f'Specialized CFM plan has unexpected fixed I/O for {frames} frames')
                self._static_runners[frames] = source
        else:
            source = self.eager_estimator.runner
        estimator.runner = copy.copy(source)
        estimator.runner.context = estimator.runner.engine.create_execution_context()
        if estimator.runner.context is None:
            raise RuntimeError("TensorRT could not create an execution context for the CFM graph")
        estimator.runner._shape_signature = None
        estimator.runner._output_specs = {}
        estimator.stream = torch.cuda.Stream(device="cuda:0")
        estimator.outputs = {}
        return estimator

    def prepare(self, mu, lengths, prompt):
        """Capture one shape while the caller holds its exclusive engine lifecycle lock."""
        import torch
        from GPT_SoVITS.f5_tts.model.modules import get_pos_embed_indices

        with self._lock, torch.inference_mode():
            if self.closed:
                raise RuntimeError("CFM graph adapter is closed")
            if self.pending_error is not None:
                raise RuntimeError(f"CFM graph preparation previously failed; restart or clear: {self.pending_error}")
            self._check_cfm()
            if self.ar_is_idle is None:
                raise RuntimeError("CFM graph preparation requires an AR-idle predicate")
            if not self.ar_is_idle():
                raise RuntimeError("CFM graph preparation requires an idle AR worker")
            key = self._validate(mu, lengths, prompt)
            if key in self.contexts:
                self.contexts.move_to_end(key)
                return False
            # Preparation is a cold lifecycle operation; drain any previously
            # enqueued CFM work before creating a CUDA graph.
            started = time.perf_counter()
            torch.cuda.synchronize(mu.device)
            if len(self.contexts) == 3:
                _, evicted = self.contexts.popitem(last=False)
                if evicted.done is not None:
                    evicted.done.synchronize()
                evicted.estimator.stream.synchronize()
                self.evictions += 1
                del evicted
            frames, prompt_frames = key
            if frames in self.shape_plans and not torch.equal(lengths, torch.full_like(lengths, frames)):
                raise ValueError('Specialized CFM graph requires the native full-length mask invariant')
            rng_cpu = torch.random.get_rng_state()
            rng_cuda = torch.cuda.get_rng_state(mu.device)
            try:
                entry = SimpleNamespace(prompt_frames=prompt_frames, done=None)
                entry.trace_events = None
                entry.conditioning_events = entry.step_events = None
                entry.mu, entry.lengths, entry.prompt = mu.contiguous().clone(), lengths.contiguous().clone(), prompt.contiguous().clone()
                entry.initial = torch.zeros((1, 100, frames), device=mu.device, dtype=mu.dtype)
                entry.times = [torch.full((1,), j * 0.25, device=mu.device, dtype=mu.dtype) for j in range(4)]
                entry.position_indices = get_pos_embed_indices(torch.zeros((1,), dtype=torch.long), frames,
                    max_pos=self.native.text_embed.precompute_max_pos).to(mu.device)
                entry.position_embedding = self.native.text_embed.freqs_cis[entry.position_indices].clone()
                entry.estimator = self._new_estimator(frames)
                entry.specialized = frames in self.shape_plans
                prompt_x = torch.zeros_like(entry.initial)
                prompt_x[..., :prompt_frames] = entry.prompt
                text0 = entry.mu.transpose(2, 1)
                expected = self.native.prepare_static_cache(prompt_x, entry.lengths, text0)
                fixed = self._static_cache(prompt_x, entry.lengths, text0, entry.position_embedding)
                names = ("condition", "mask") if self.positive_only else ("condition", "negative_condition", "mask")
                for name in names:
                    if not torch.equal(expected[name], fixed[name]):
                        raise RuntimeError(f"Graph conditioning differs from native: {name}")
                if (not torch.equal(expected["rope"][0], fixed["rope"][0])
                        or expected["rope"][1] != fixed["rope"][1]):
                    raise RuntimeError("Graph RoPE differs from native")
                for _ in range(2):
                    self._flow(entry)
                torch.cuda.synchronize(mu.device)
                eager = self._flow(entry).clone()
                torch.cuda.synchronize(mu.device)
                entry.capture_stream = torch.cuda.Stream(device=mu.device)
                entry.capture_stream.wait_stream(torch.cuda.current_stream(mu.device))
                entry.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(entry.graph, stream=entry.capture_stream):
                    entry.output = self._flow(entry)
                torch.cuda.synchronize(mu.device)
                entry.graph.replay()
                torch.cuda.synchronize(mu.device)
                if not torch.equal(eager, entry.output):
                    raise RuntimeError("Captured CFM graph output differs from its eager TensorRT flow")
                entry.capture_seconds = time.perf_counter() - started
                entry.precision = _precision_snapshot(torch)
                self.contexts[key] = entry
                self.captures += 1
                if entry.specialized:
                    self.specialized_captures[frames] = self.specialized_captures.get(frames, 0) + 1
                return True
            finally:
                torch.cuda.synchronize(mu.device)
                torch.random.set_rng_state(rng_cpu)
                torch.cuda.set_rng_state(rng_cuda, mu.device)
                torch.cuda.synchronize(mu.device)
                self.prepare_seconds += time.perf_counter() - started

    def inference(self, mu, x_lens, prompt, n_timesteps, temperature=0.875,
                  inference_cfg_rate=1.30, generator=None):
        """Use the prepared graph, or explicitly use the original eager loop on a shape miss."""
        import torch
        with self._lock, torch.inference_mode():
            if self.closed:
                raise RuntimeError("CFM graph adapter is closed")
            if self.pending_error is not None:
                raise RuntimeError(f"CFM graph preparation previously failed; restart or clear: {self.pending_error}")
            self._check_cfm()
            if int(n_timesteps) != 4 or float(inference_cfg_rate) != 0.0:
                raise ValueError("CFM graph supports native four-step Turbo with CFG 0 only")
            key = self._validate(mu, x_lens, prompt)
            current = torch.cuda.current_stream(mu.device)
            entry = self.contexts.get(key)
            if entry is None:
                self.misses += 1
                LOGGER.info("CFM graph shape %s is not prepared; using the explicit eager CFM path", key)
                if self._last_eager_done is not None:
                    current.wait_event(self._last_eager_done)
                output = self.original_inference(mu, x_lens, prompt, n_timesteps,
                    temperature=temperature, inference_cfg_rate=inference_cfg_rate, generator=generator)
                if self.collect_misses and key not in self.pending and len(self.pending) < 3:
                    snapshots = tuple(value.detach().contiguous().clone() for value in (mu, x_lens, prompt))
                    ready = torch.cuda.Event()
                    ready.record(current)
                    self.pending[key] = (*snapshots, ready)
                done = torch.cuda.Event()
                done.record(current)
                self._last_eager_done = done
                return output
            if entry.precision != _precision_snapshot(torch):
                raise RuntimeError("FP32 precision settings changed after CFM graph capture")
            self.contexts.move_to_end(key)
            if entry.done is not None:
                current.wait_event(entry.done)
            entry.mu.copy_(mu)
            entry.lengths.copy_(x_lens)
            entry.prompt.copy_(prompt)
            noise = torch.randn((1, 100, key[0]), device=mu.device, dtype=mu.dtype, generator=generator) * 0.875
            noise[..., :key[1]] = 0
            entry.initial.copy_(noise)
            entry.position_embedding.copy_(self.native.text_embed.freqs_cis[entry.position_indices])
            entry.graph.replay()
            # Callers own their output; later replays cannot overwrite it.
            output = entry.output.clone()
            done = torch.cuda.Event()
            done.record(current)
            entry.done = done
            self.hits += 1
            if entry.specialized:
                self.specialized_hits[key[0]] = self.specialized_hits.get(key[0], 0) + 1
            return output

    def prepare_pending(self):
        """Prepare observed shapes only after the serialized AR worker is idle."""
        with self._lock:
            if self.closed:
                raise RuntimeError("CFM graph adapter is closed")
            if self.pending_error is not None:
                raise RuntimeError(f"CFM graph preparation previously failed; restart or clear: {self.pending_error}")
            if self.ar_is_idle is None or not self.ar_is_idle():
                raise RuntimeError("CFM graph preparation requires an idle AR worker")
            for key in list(self.pending):
                mu, lengths, prompt, ready = self.pending[key]
                try:
                    ready.synchronize()
                    # Preparation uses the same bounded, synchronized LRU
                    # eviction as explicit preparation when all slots are full.
                    if key not in self.contexts:
                        self.prepare(mu, lengths, prompt)
                except Exception as error:
                    self.pending_error = f'{type(error).__name__}: {error}'
                    raise
                del self.pending[key]

    def enable(self):
        with self._lock:
            if self.closed:
                raise RuntimeError("CFM graph adapter is closed")
            if self.enabled:
                if self.cfm.inference != self.inference:
                    raise RuntimeError("CFM inference binding changed externally")
                return
            if self.cfm.inference != self.original_inference:
                raise RuntimeError("Original CFM inference binding changed externally")
            self.cfm.inference = self.inference
            self.enabled = True

    def disable(self):
        with self._lock:
            if not self.enabled:
                return
            if self.cfm.inference != self.inference:
                raise RuntimeError("CFM inference binding changed externally")
            self.cfm.inference = self.original_inference
            self.enabled = False

    def clear(self):
        import torch
        with self._lock:
            if self.closed:
                raise RuntimeError("CFM graph adapter is closed")
            torch.cuda.synchronize("cuda:0")
            if self.shape_plans:
                for entry in self.contexts.values():
                    entry.graph.reset()
            self.contexts.clear()
            self._static_runners.clear()
            self.pending.clear()
            self.pending_error = None
            self._last_eager_done = None

    def close(self):
        import torch
        with self._lock:
            if self.closed:
                return
            self.disable()
            torch.cuda.synchronize("cuda:0")
            if self.shape_plans:
                for entry in self.contexts.values():
                    entry.graph.reset()
            self.contexts.clear()
            self._static_runners.clear()
            self.pending.clear()
            self._last_eager_done = None
            self.closed = True


def install(cfm, plan, *, ar_is_idle=None, collect_misses=False, shape_plans=None, positive_only=False):
    """Enable the adapter; prewarm only under an exclusive idle-engine lifecycle lock."""
    adapter = CFMGraphAdapter(cfm, plan, ar_is_idle=ar_is_idle, collect_misses=collect_misses,
                              shape_plans=shape_plans, positive_only=positive_only)
    adapter.enable()
    return adapter
