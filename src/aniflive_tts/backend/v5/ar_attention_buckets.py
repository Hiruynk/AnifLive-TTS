"""Experimental batch-one Torch SDPA attention buckets over the native full KV cache.

No cache history is discarded: only the masked attention view is bounded. The
native 2048 graph is the declared profile for positions above 1024. Install
and switch only while the runner is idle; no model or GPU work occurs on import.
"""
from __future__ import annotations

import time
import weakref


class AttentionBuckets:
    def __init__(self, runner, *, on_select=None, compact_host_state=False, additive_mask=None,
                 compile_neural=False, eager_rounding=False, native_norm=False, native_projections=False):
        import torch

        if on_select is not None and not callable(on_select):
            raise TypeError("on_select must be callable or None")
        if type(compact_host_state) is not bool:
            raise TypeError("compact_host_state must be a boolean")
        if additive_mask is not None:
            from .ar_additive_mask import AdditiveAttentionMask
            if not isinstance(additive_mask, AdditiveAttentionMask) or additive_mask.runner is not runner:
                raise ValueError("Additive mask must belong to this AR runner")
        decoder = runner.decoder
        if (runner.max_batch_size != 1 or runner.device.type != "cuda"
                or runner.xy_pos_buf.dtype != torch.float16 or decoder.max_seq_length != 2048):
            raise ValueError("Attention buckets require one FP16 CUDA slot and native 2048-token KV")
        if not decoder.__class__.__module__.endswith(".torch_static_cuda_graph"):
            raise ValueError("Attention buckets require the native Torch static CUDA graph decoder")
        if runner.bound_sessions or runner.waiting_bind or any(runner.slot_sessions):
            raise RuntimeError("Capture attention buckets only when the runner is idle")
        if not runner._should_use_graph() or 1 not in runner.graph_cache:
            raise RuntimeError("The native 2048 neural CUDA graph is required")
        mask = runner.extra_buffers.get("attn_mask")
        expected_mask = (1, decoder.n_head, 1, 2048)
        if mask is None or tuple(mask.shape) != expected_mask or mask.dtype != torch.bool:
            raise ValueError("Unexpected native static attention mask")
        expected_kv = (1, decoder.n_head, 2048, decoder.head_dim)
        for key, value in runner.kv_cache_buf:
            if (tuple(key.shape) != expected_kv or tuple(value.shape) != expected_kv
                    or key.dtype != torch.float16 or value.dtype != torch.float16
                    or key.device != runner.device or value.device != runner.device):
                raise ValueError("Unexpected native HND KV cache")

        self.runner = runner
        self.original_acquire_graph = runner._acquire_graph
        self.on_select = on_select
        self.additive_mask = additive_mask
        self.compact_host_state = compact_host_state
        if compact_host_state:
            self._validated_session = None
            self._text_length = None
            self._prompt_length = None
        self.counts = {256: 0, 512: 0, 1024: 0, 2048: 0}
        self.graphs = {}
        self.compiled_graphs = {}
        self.use_compiled = False
        self.views = {}
        self.enabled = False
        changed = [runner.xy_pos_buf, runner.xy_dec_buf, runner.input_pos_buf, mask]
        if additive_mask is not None:
            changed.append(additive_mask.mask)
        for key, value in runner.kv_cache_buf:
            changed.extend((key, value))
        started = time.perf_counter()

        with torch.cuda.device(runner.device), torch.inference_mode():
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(runner.device)
            saved = [buffer.clone() for buffer in changed]
            try:
                # Match the native graph capture's valid dummy position/mask.
                runner.input_pos_buf.fill_(1)
                mask.zero_()
                mask[..., 0] = True
                if additive_mask is not None:
                    additive_mask.mask.fill_(float('-inf'))
                    additive_mask.mask[..., 0] = 0
                sizes = (256, 512, 1024, 2048) if additive_mask is not None else (256, 512, 1024)
                for size in sizes:
                    kv_views = [(key[..., :size, :], value[..., :size, :])
                                for key, value in runner.kv_cache_buf]
                    mask_view = (additive_mask.mask if additive_mask is not None else mask)[..., :size]
                    # Keep the Python views alive for the CUDA graph lifetime.
                    self.views[size] = {"kv": kv_views, "mask": mask_view}
                    # Separate graph pools avoid assumptions about replay order.
                    self.graphs[size] = decoder.capture(
                        runner.input_pos_buf, runner.xy_pos_buf, runner.xy_dec_buf,
                        kv_views, pool=None, attn_mask=mask_view)
                if compile_neural:
                    if additive_mask is not None:
                        raise ValueError('Compiled neural probe requires the original boolean mask')
                    options={'triton.cudagraphs':False}
                    if eager_rounding:
                        options.update(emulate_precision_casts=True,emulate_divison_rounding=True)
                    from contextlib import ExitStack
                    with ExitStack() as scope:
                        if native_norm:
                            from .ar_native_norm import native_norms
                            scope.enter_context(native_norms(decoder.h))
                        if native_projections:
                            from .ar_native_norm import native_projections as projections
                            scope.enter_context(projections(decoder.h))
                        self.compiled_neural = torch.compile(decoder.h,dynamic=False,fullgraph=True,options=options)
                        capture_stream=torch.cuda.Stream(device=runner.device)
                        capture_stream.wait_stream(torch.cuda.current_stream(runner.device))
                        with torch.cuda.stream(capture_stream):
                            for size in sizes:
                                inputs=self.views[size]
                                for _ in range(2):
                                    self.compiled_neural(runner.xy_pos_buf,runner.input_pos_buf,
                                                         inputs['kv'],attn_mask=inputs['mask'])
                                capture_stream.synchronize()
                                graph=torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph,stream=capture_stream):
                                    runner.xy_dec_buf.copy_(self.compiled_neural(
                                        runner.xy_pos_buf,runner.input_pos_buf,inputs['kv'],attn_mask=inputs['mask']))
                                self.compiled_graphs[size]=graph
            finally:
                torch.cuda.synchronize(runner.device)
                for target, snapshot in zip(changed, saved):
                    target.copy_(snapshot)
                torch.random.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, runner.device)
                torch.cuda.synchronize(runner.device)
        self.capture_seconds = time.perf_counter() - started

    def acquire_graph(self, batch_size):
        runner = self.runner
        if batch_size != 1 or len(runner.bound_sessions) != 1 or runner.waiting_bind:
            raise ValueError("Attention buckets support exactly one active session")
        if self.additive_mask is not None:
            self.additive_mask.require_active_producer()
        session = runner.slot_sessions[0]
        if session is None or session not in runner.bound_sessions:
            raise ValueError("Attention bucket session must occupy slot zero")
        if self.compact_host_state:
            if len(session.slot_indices) != 1 or session.slot_indices[0] != 0:
                raise ValueError("Attention bucket session must occupy slot zero")
        elif session.slot_indices != [0]:
            raise ValueError("Attention bucket session must occupy slot zero")
        request = session.request
        if session.bsz != 1:
            raise ValueError("Attention buckets support batch size one")
        new_session = (not self.compact_host_state or self._validated_session is None
                       or self._validated_session() is not session)
        if new_session:
            if (not request.use_cuda_graph or len(request.x) != 1 or request.x[0].ndim != 1
                    or request.x_lens.device.type != "cpu" or tuple(request.x_lens.shape) != (1,)):
                raise ValueError("Unsupported attention bucket request layout")
            text_length = int(request.x_lens[0].item())  # CPU tensor, never a GPU readback
            if text_length != request.x[0].numel() or session.prompt_len != request.prompts.shape[-1]:
                raise RuntimeError("Native single-slot prefill length invariant is not satisfied")
            if self.compact_host_state:
                # T2SSession converts the prompt before binding; these request
                # lengths remain immutable throughout the owned session.
                self._validated_session = weakref.ref(session)
                self._text_length = text_length
                self._prompt_length = int(session.prompt_len)
        else:
            text_length = self._text_length
            if session.prompt_len != self._prompt_length:
                raise RuntimeError("Native session prompt length changed")
        bad_lengths = (len(session.output_lengths) != 1 or session.output_lengths[0] != session.step_count
                       if self.compact_host_state else session.output_lengths != [session.step_count])
        if session.step_count <= 0 or bad_lengths or session.total_tokens != session.step_count:
            raise RuntimeError("Native single-slot input-position invariant is not satisfied")
        prompt_length = self._prompt_length if self.compact_host_state else int(session.prompt_len)
        position = text_length + prompt_length + int(session.step_count)
        if not 1 <= position <= 2048:
            raise ValueError("Native AR input position is outside the 2048-token cache")
        size = next(size for size in (256, 512, 1024, 2048) if position <= size)
        self.counts[size] += 1
        if self.on_select is not None:
            self.on_select(position, size)
        if size == 2048 and self.additive_mask is None:
            return self.original_acquire_graph(batch_size)
        return (self.compiled_graphs if self.use_compiled else self.graphs)[size]

    def select_compiled(self, enabled):
        if self.runner.bound_sessions or self.runner.waiting_bind or any(self.runner.slot_sessions):
            raise RuntimeError('Select compiled neural graphs only while idle')
        if enabled and not self.compiled_graphs:
            raise RuntimeError('Compiled neural graphs were not prepared')
        self.use_compiled=bool(enabled)

    def enable(self):
        if self.runner.bound_sessions or self.runner.waiting_bind or any(self.runner.slot_sessions):
            raise RuntimeError("Switch attention graphs only while the runner is idle")
        if self.enabled:
            if self.runner._acquire_graph != self.acquire_graph:
                raise RuntimeError("Attention graph binding changed externally")
            return
        if self.runner._acquire_graph != self.original_acquire_graph:
            raise RuntimeError("Native attention graph binding changed externally")
        self.runner._acquire_graph = self.acquire_graph
        self.enabled = True

    def disable(self):
        if self.runner.bound_sessions or self.runner.waiting_bind or any(self.runner.slot_sessions):
            raise RuntimeError("Switch attention graphs only while the runner is idle")
        if not self.enabled:
            return
        if self.runner._acquire_graph != self.acquire_graph:
            raise RuntimeError("Attention graph binding changed externally")
        self.runner._acquire_graph = self.original_acquire_graph
        self.enabled = False


def install_attention_buckets(runner, *, on_select=None, compact_host_state=False, additive_mask=None,
                              compile_neural=False, eager_rounding=False, native_norm=False,native_projections=False):
    """Capture, install and return the reversible experimental graph handle."""
    handle = AttentionBuckets(runner, on_select=on_select, compact_host_state=compact_host_state,
                              additive_mask=additive_mask,compile_neural=compile_neural,eager_rounding=eager_rounding,
                              native_norm=native_norm,native_projections=native_projections)
    handle.enable()
    return handle
