"""Opt-in, explicitly prepared CUDA graphs over a TensorRT target encoder.

The caller serializes the engine and supplies an AR-idle predicate. No graph
is captured during inference. A shape miss dispatches to the existing target
encoder; reference preparation and non-unit speed retain its native paths.
Disable this wrapper before toggling its underlying encoder adapter.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import threading
import time
from types import SimpleNamespace


class EncoderGraphAdapter:
    def __init__(self, base, *, ar_is_idle, collect_misses=False):
        from .trt_encoder import TensorRTTargetEncoder
        if not isinstance(base, TensorRTTargetEncoder):
            raise TypeError('Encoder graph requires the installed TensorRT target encoder')
        if not callable(ar_is_idle):
            raise TypeError('Encoder graph requires an AR-idle predicate')
        if type(collect_misses) is not bool:
            raise TypeError('collect_misses must be a boolean')
        if not base.enabled or base.model.decode_encp != base.decode_encp:
            raise RuntimeError('Enable the underlying target encoder before installing the graph')
        self.base = base
        self.model = base.model
        self.ar_is_idle = ar_is_idle
        self.collect_misses = collect_misses
        self.contexts = OrderedDict()
        self.pending = OrderedDict()
        self.pending_error = None
        self._lock = threading.RLock()
        self.enabled = False
        self.closed = False
        self.hits = 0
        self.misses = 0
        self.captures = 0
        self.evictions = 0
        self.prepare_seconds = 0.0

    def _idle(self):
        if not self.ar_is_idle():
            raise RuntimeError('Encoder graph lifecycle requires an idle AR worker')

    def _validate(self, codes, phones, ge):
        import torch
        if not all(isinstance(value, torch.Tensor) for value in (codes, phones, ge)):
            raise TypeError('Target codes, phones and ge must be Torch tensors')
        if any(value.device != torch.device('cuda:0') for value in (codes, phones, ge)):
            raise ValueError('Target codes, phones and ge must all be on cuda:0')
        if codes.dtype != torch.int64 or phones.dtype != torch.int64 or ge.dtype != torch.float16:
            raise ValueError('Target codes/phones must be int64 and ge must be FP16')
        if (codes.ndim != 3 or tuple(codes.shape[:2]) != (1, 1)
                or phones.ndim != 2 or phones.shape[0] != 1 or tuple(ge.shape) != (1, 512, 1)):
            raise ValueError('Expected codes [1,1,S], phones [1,P], and ge [1,512,1]')
        key = (int(codes.shape[2]), int(phones.shape[1]))
        if not 1 <= key[0] <= 1536 or not 1 <= key[1] <= 2048:
            raise ValueError('Target encoder profile requires 1<=S<=1536 and 1<=P<=2048')
        return key

    @staticmethod
    def _release(entry):
        if entry.done is not None:
            entry.done.synchronize()
        entry.stream.synchronize()
        entry.graph.reset()

    def prepare(self, codes, phones, ge):
        """Capture one actual input shape while the caller owns the idle engine."""
        import torch
        with self._lock, torch.cuda.device('cuda:0'), torch.inference_mode():
            if self.closed:
                raise RuntimeError('Encoder graph adapter is closed')
            if self.pending_error is not None:
                raise RuntimeError(f'Encoder graph preparation previously failed; restart: {self.pending_error}')
            self._idle()
            key = self._validate(codes, phones, ge)
            if key in self.contexts:
                self.contexts.move_to_end(key)
                return False
            if len(self.contexts) == 2:
                _, evicted = self.contexts.popitem(last=False)
                self._release(evicted)
                del evicted
                self.evictions += 1
            started = time.perf_counter()
            self.base.stream.synchronize()
            if self.base._last_copy_done is not None:
                self.base._last_copy_done.synchronize()
            entry = SimpleNamespace(done=None)
            entry.runner = copy.copy(self.base.runner)
            entry.runner.context = entry.runner.engine.create_execution_context()
            if entry.runner.context is None:
                raise RuntimeError('TensorRT could not create an encoder execution context')
            entry.runner._shape_signature = None
            entry.runner._output_specs = {}
            entry.inputs = {'codes': codes.contiguous().clone(), 'phones': phones.contiguous().clone(),
                            'ge': ge.contiguous().clone()}
            entry.output = torch.empty((1, 512, 4 * key[0]), dtype=torch.float16, device='cuda:0')
            entry.stream = torch.cuda.Stream(device='cuda:0')
            entry.stream.wait_stream(torch.cuda.current_stream('cuda:0'))
            with torch.cuda.stream(entry.stream):
                for _ in range(3):
                    entry.runner.infer(entry.inputs, outputs={'features': entry.output},
                                       stream=entry.stream, synchronize=False, profile=False)
                expected = entry.output.clone()
            entry.stream.synchronize()
            entry.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(entry.graph, stream=entry.stream):
                if not entry.runner.context.execute_async_v3(stream_handle=entry.stream.cuda_stream):
                    raise RuntimeError('TensorRT encoder enqueue failed during graph capture')
            with torch.cuda.stream(entry.stream):
                entry.graph.replay()
            entry.stream.synchronize()
            if not torch.equal(expected, entry.output):
                raise RuntimeError('Captured encoder graph differs from normal TensorRT output')
            self.contexts[key] = entry
            self.captures += 1
            self.prepare_seconds += time.perf_counter() - started
            return True

    def decode_encp(self, codes, text, refer, ge=None, speed=1):
        import torch
        if ge is None or speed != 1:
            return self.base.decode_encp(codes, text, refer, ge, speed)
        # Native full TTS may pass int32 semantic tokens from the AR worker;
        # the existing TensorRT plan and graph bindings require int64.
        if isinstance(codes, torch.Tensor) and codes.dtype == torch.int32:
            codes = codes.to(dtype=torch.int64)
        if isinstance(text, torch.Tensor) and text.dtype == torch.int32:
            text = text.to(dtype=torch.int64)
        with self._lock, torch.cuda.device('cuda:0'), torch.inference_mode():
            if self.closed:
                raise RuntimeError('Encoder graph adapter is closed')
            if self.pending_error is not None:
                raise RuntimeError(f'Encoder graph preparation previously failed; restart: {self.pending_error}')
            key = self._validate(codes, text, ge)
            entry = self.contexts.get(key)
            if entry is None:
                self.misses += 1
                result = self.base.decode_encp(codes, text, refer, ge, speed)
                if (self.collect_misses and key not in self.pending
                        and len(self.contexts) + len(self.pending) < 2):
                    snapshots = tuple(value.detach().contiguous().clone() for value in (codes, text, ge))
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream('cuda:0'))
                    self.pending[key] = (*snapshots, ready)
                return result
            self.contexts.move_to_end(key)
            current = torch.cuda.current_stream('cuda:0')
            if entry.done is not None:
                current.wait_event(entry.done)
            for tensor in (codes, text, ge):
                tensor.record_stream(current)
            entry.inputs['codes'].copy_(codes)
            entry.inputs['phones'].copy_(text)
            entry.inputs['ge'].copy_(ge)
            entry.graph.replay()
            features = entry.output.clone()
            entry.done = torch.cuda.Event()
            entry.done.record(current)
            self.hits += 1
            return features, ge

    def prepare_pending(self):
        """Prepare observed shapes only after the serialized AR worker is idle."""
        with self._lock:
            if self.closed:
                raise RuntimeError('Encoder graph adapter is closed')
            if self.pending_error is not None:
                raise RuntimeError(f'Encoder graph preparation previously failed; restart: {self.pending_error}')
            self._idle()
            for key in list(self.pending):
                codes, phones, ge, ready = self.pending[key]
                try:
                    ready.synchronize()
                    # Explicit preparation may have filled the remaining slot
                    # since this miss; automatic work must not evict a graph.
                    if key not in self.contexts and len(self.contexts) < 2:
                        self.prepare(codes, phones, ge)
                except Exception as error:
                    self.pending_error = f'{type(error).__name__}: {error}'
                    raise
                del self.pending[key]

    def enable(self):
        with self._lock:
            self._idle()
            if self.closed:
                raise RuntimeError('Encoder graph adapter is closed')
            if self.enabled:
                if self.model.decode_encp != self.decode_encp:
                    raise RuntimeError('Encoder graph binding changed externally')
                return
            if not self.base.enabled or self.model.decode_encp != self.base.decode_encp:
                raise RuntimeError('The underlying target encoder must be enabled first')
            self.model.decode_encp = self.decode_encp
            self.enabled = True

    def disable(self):
        with self._lock:
            self._idle()
            if self.enabled:
                if self.model.decode_encp != self.decode_encp:
                    raise RuntimeError('Encoder graph binding changed externally')
                self.model.decode_encp = self.base.decode_encp
                self.enabled = False

    def close(self):
        with self._lock:
            if self.closed:
                return
            self.disable()
            for entry in self.contexts.values():
                self._release(entry)
            self.contexts.clear()
            for _codes, _phones, _ge, ready in self.pending.values():
                ready.synchronize()
            self.pending.clear()
            self.closed = True


def install(base, *, ar_is_idle, collect_misses=False):
    """Install over an enabled target encoder; prepare shapes separately."""
    adapter = EncoderGraphAdapter(base, ar_is_idle=ar_is_idle, collect_misses=collect_misses)
    adapter.enable()
    return adapter
