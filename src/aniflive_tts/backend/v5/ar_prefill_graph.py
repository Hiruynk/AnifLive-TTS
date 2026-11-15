"""Exact shape-specialized native prefill graphs; never cache target content.

Prepare only while idle. Every hit copies fresh embeddings and attention mask,
executes all 24 original layers, and writes fresh prefix KV into the session.
Unprepared lengths use the original prefill and are counted explicitly.
"""
from types import SimpleNamespace
import threading
import time
import weakref


class PrefillGraph:
    def __init__(self, runner, *, idle_guard, collect_misses=False, persistent_workspace=False,
                 batch_bind_copy=False, persistent_bind_index=False, trt_shape_plans=None):
        if runner.max_batch_size != 1 or not runner.decoder.__class__.__module__.endswith('.torch_static_cuda_graph'):
            raise ValueError('Prefill graphs require the single-slot Torch-static decoder')
        self.runner = runner
        self.original = runner.decoder.h.prefill
        self.idle_guard = idle_guard
        if type(collect_misses) is not bool:
            raise TypeError('collect_misses must be a boolean')
        if type(persistent_workspace) is not bool:
            raise TypeError('persistent_workspace must be a boolean')
        if type(batch_bind_copy) is not bool:
            raise TypeError('batch_bind_copy must be a boolean')
        if batch_bind_copy and not persistent_workspace:
            raise ValueError('Batched binding requires persistent prefill workspaces')
        if type(persistent_bind_index) is not bool:
            raise TypeError('persistent_bind_index must be a boolean')
        if persistent_bind_index and (not persistent_workspace or batch_bind_copy):
            raise ValueError('Persistent bind index requires workspace and excludes batched KV binding')
        if trt_shape_plans and not persistent_workspace:
            raise ValueError('TensorRT prefill plans require persistent full-KV workspaces')
        if trt_shape_plans is not None and (not isinstance(trt_shape_plans, dict) or any(
                type(length) is not int or not 1 <= length <= 2048 or not path
                for length, path in trt_shape_plans.items())):
            raise ValueError('TensorRT prefill plans require positive fixed lengths and paths')
        self.trt_shape_plans = dict(trt_shape_plans or {})
        self.trt_hits = {}
        self.collect_misses = collect_misses
        self.persistent_workspace = persistent_workspace
        self.batch_bind_copy = batch_bind_copy
        self.persistent_bind_index = persistent_bind_index
        self._bind_index = None
        self._bind_index_ready = None
        self.original_runner_prefill = None
        self.original_runner_bind = None
        self.workspace_owner_thread = None
        self.workspace_hits = self.workspace_dispatches = 0
        self.batch_bind_hits = 0
        self.index_bind_hits = 0
        self.entries = {}
        self.pending = {}
        self.enabled = False
        self.hits = self.misses = 0
        self.prepare_seconds = 0.0
        if batch_bind_copy:
            import triton
            import triton.language as tl
            @triton.jit
            def copy_kv(source_addresses, target_addresses, BLOCK: tl.constexpr):
                index = tl.program_id(0)
                offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
                source = tl.load(source_addresses + index).to(tl.pointer_type(tl.float16))
                target = tl.load(target_addresses + index).to(tl.pointer_type(tl.float16))
                tl.store(target + offset, tl.load(source + offset))
            self._copy_kv = copy_kv

    def _idle(self):
        if not self.idle_guard():
            raise RuntimeError('Prefill graph lifecycle requires an idle AR worker')

    def _key(self, x, mask):
        import torch
        if (x.device != self.runner.device or x.dtype != torch.float16 or x.ndim != 3
                or tuple(x.shape[::2]) != (1, 512) or mask.device != x.device
                or mask.dtype != torch.bool or tuple(mask.shape) != (1, 1, x.shape[1], x.shape[1])):
            raise ValueError('Unexpected native prefill embedding/mask layout')
        return int(x.shape[1])

    def prepare(self, x, mask):
        import torch
        from .cfm_cuda_graph import _precision_snapshot
        self._idle()
        length = self._key(x, mask)
        if length in self.entries:
            return
        if len(self.entries) == 4:
            raise ValueError('At most four prefill shapes may be prepared')
        started = time.perf_counter()
        with torch.inference_mode():
            if self.persistent_bind_index and self._bind_index is None:
                self._bind_index = torch.zeros((1,), device=x.device, dtype=torch.long)
                self._bind_index_ready = torch.cuda.Event()
                self._bind_index_ready.record(torch.cuda.current_stream(x.device))
            entry = SimpleNamespace(x=x.contiguous().clone(), mask=mask.contiguous().clone(), done=None)
            shape = (1, 16, self.runner.decoder.max_seq_length if self.persistent_workspace else length, 32)
            entry.kv = [(torch.zeros(shape,device=x.device,dtype=x.dtype),
                         torch.zeros(shape,device=x.device,dtype=x.dtype)) for _ in range(24)]
            entry.session = None
            if self.batch_bind_copy:
                expected = (1, 16, 2048, 32)
                source_tensors = tuple(t for pair in entry.kv for t in pair)
                target_tensors = tuple(t for pair in self.runner.kv_cache_buf for t in pair)
                if (len(source_tensors) != 48 or len(target_tensors) != 48 or any(
                        tuple(t.shape) != expected or t.dtype != torch.float16 or t.device != x.device
                        or not t.is_contiguous() for t in (*source_tensors, *target_tensors))):
                    raise ValueError('Batched binding requires contiguous full FP16 single-slot HND caches')
                entry.bind_targets = target_tensors
                entry.source_addresses = torch.tensor([t.data_ptr() for t in source_tensors], device=x.device, dtype=torch.int64)
                entry.target_addresses = torch.tensor([t.data_ptr() for t in target_tensors], device=x.device, dtype=torch.int64)
            stream = torch.cuda.Stream(device=x.device)
            stream.wait_stream(torch.cuda.current_stream(x.device))
            with torch.cuda.stream(stream):
                if self.batch_bind_copy:
                    # Compile on an independent two-megabyte scratch buffer;
                    # cold preparation must not mutate the live runner cache.
                    scratch = torch.empty_like(entry.kv[0][0])
                    scratch_address = torch.tensor([scratch.data_ptr()], device=x.device, dtype=torch.int64)
                    self._copy_kv[(1, 256)](entry.source_addresses, scratch_address, BLOCK=4096, num_warps=4)
                    stream.synchronize()
                for _ in range(2):
                    self.original(entry.x, entry.kv, entry.mask)
                stream.synchronize()
                entry.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(entry.graph, stream=stream):
                    entry.hidden = self.original(entry.x, entry.kv, entry.mask)
                # Alter the actual embedding, poison outputs, then prove replay
                # recomputes instead of returning the capture's saved content.
                entry.x.mul_(0.99).add_(0.001)
                oracle_kv = [(torch.zeros_like(k),torch.zeros_like(v)) for k,v in entry.kv]
                oracle = self.original(entry.x, oracle_kv, entry.mask).clone()
                entry.hidden.fill_(float('nan'))
                for k,v in entry.kv:
                    k[..., :length, :].fill_(float('nan'))
                    v[..., :length, :].fill_(float('nan'))
                entry.graph.replay()
                stream.synchronize()
                if (not torch.equal(oracle, entry.hidden) or
                        any(not torch.equal(a,b) for pair,expected in zip(entry.kv,oracle_kv)
                            for a,b in zip(pair,expected))):
                    entry.graph.reset()
                    raise RuntimeError('Prefill graph differs from native on changed embeddings')
            torch.cuda.current_stream(x.device).wait_stream(stream)
            entry.alternative = None
            if length in self.trt_shape_plans:
                raise ValueError('TensorRT prefill plans are not supported by this runtime')
            entry.precision = _precision_snapshot(torch)
            self.entries[length] = entry
        self.prepare_seconds += time.perf_counter()-started

    def prefill(self, x, kv, mask):
        import torch
        from .cfm_cuda_graph import _precision_snapshot
        length = self._key(x, mask)
        entry = self.entries.get(length)
        if entry is None:
            self.misses += 1
            if (self.collect_misses and length not in self.pending
                    and len(self.entries) + len(self.pending) < 4):
                # The AR worker owns these copies until the ready event; the
                # service consumes them only after it has become idle.
                saved_x = x.detach().contiguous().clone()
                saved_mask = mask.detach().contiguous().clone()
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(x.device))
                self.pending[length] = (saved_x, saved_mask, ready)
            return self.original(x, kv, mask)
        if entry.precision != _precision_snapshot(torch):
            raise RuntimeError('Precision changed after prefill graph capture')
        current = torch.cuda.current_stream(x.device)
        if entry.done is not None:
            current.wait_event(entry.done)
        if entry.alternative is not None and kv is entry.kv:
            hidden = entry.alternative.prefill(x, mask)
            self.trt_hits[length] = self.trt_hits.get(length, 0) + 1
            self.workspace_hits += 1
        else:
            entry.x.copy_(x)
            entry.mask.copy_(mask)
            entry.graph.replay()
            if kv is entry.kv:
                if not self.persistent_workspace:
                    raise RuntimeError('Prefill workspace identity is unexpected')
                self.workspace_hits += 1
            else:
                for (k,v),(saved_k,saved_v) in zip(kv,entry.kv):
                    k[..., :length, :].copy_(saved_k[..., :length, :])
                    v[..., :length, :].copy_(saved_v[..., :length, :])
            hidden = entry.hidden.clone()
        entry.done = torch.cuda.Event()
        entry.done.record(current)
        self.hits += 1
        return hidden

    def workspace_prefill(self, request):
        """Temporarily lend one prepared full cache until the session binds.

        Reuse only on the same worker after bind and with no queued or active
        slots. An unbound live session retains its lease, including cancellation.
        """
        import torch
        runner = self.runner
        length = None
        if (len(request.x) == 1 and isinstance(request.x_lens, torch.Tensor)
                and request.x_lens.device.type == 'cpu' and tuple(request.x_lens.shape) == (1,)
                and isinstance(request.prompts, torch.Tensor) and request.prompts.ndim == 2
                and request.prompts.shape[0] == 1):
            text_length = int(request.x_lens[0])
            if text_length > 0 and request.x[0].numel() == text_length:
                length = text_length + request.prompts.shape[-1]
        entry = self.entries.get(length)
        owner = threading.get_ident()
        prior = entry.session() if entry is not None and entry.session is not None else None
        leased = prior is not None and not prior.slot_indices
        if (entry is None or leased or runner.waiting_bind or runner.bound_sessions
                or any(runner.slot_sessions) or (self.workspace_owner_thread is not None and self.workspace_owner_thread != owner)):
            return self.original_runner_prefill(request)
        self.workspace_owner_thread = owner
        decoder = runner.decoder
        _original_init = decoder.init_cache
        had_instance_init = 'init_cache' in decoder.__dict__
        previous_instance_init = decoder.__dict__.get('init_cache')
        used = False
        def workspace_init(batch_size=0):
            nonlocal used
            if used or batch_size != 1:
                raise RuntimeError('Unexpected scoped prefill cache request')
            used = True
            return entry.kv
        decoder.init_cache = workspace_init
        try:
            session = self.original_runner_prefill(request)
        finally:
            if had_instance_init:
                decoder.init_cache = previous_instance_init
            else:
                del decoder.init_cache
        if not used or session.kv_cache is not entry.kv:
            raise RuntimeError('Prefill did not retain its selected workspace')
        entry.session = weakref.ref(session)
        self.workspace_dispatches += 1
        return session

    def workspace_bind(self, session):
        """Use one explicitly selected single-slot bind optimization."""
        runner = self.runner
        entry = next((item for item in self.entries.values() if session.kv_cache is item.kv), None)
        waiting = runner.waiting_bind
        if (entry is None or self.workspace_owner_thread != threading.get_ident()
                or entry.session is None or entry.session() is not session or session.bsz != 1
                or session.slot_indices or runner.bound_sessions or any(runner.slot_sessions)
                or (waiting and (len(waiting) != 1 or waiting[0] is not session))):
            return self.original_runner_bind(session)
        if self.persistent_bind_index:
            return self._bind_with_persistent_index(session)
        targets = tuple(t for pair in runner.kv_cache_buf for t in pair)
        if len(targets) != len(entry.bind_targets) or any(a is not b for a, b in zip(targets, entry.bind_targets)):
            raise RuntimeError('Runner KV buffer identity changed after workspace preparation')
        self._copy_kv[(48, 256)](entry.source_addresses, entry.target_addresses, BLOCK=4096, num_warps=4)
        original_cache = session.kv_cache
        session.kv_cache = ()
        try:
            result = self.original_runner_bind(session)
        finally:
            session.kv_cache = original_cache
        if not result:
            raise RuntimeError('Native bind rejected the verified free single slot')
        self.batch_bind_hits += 1
        return result

    def _bind_with_persistent_index(self, session):
        """Native single-slot bind with only the immutable CUDA index reused."""
        import torch
        runner = self.runner
        slots = runner._assign_slots(session.bsz)
        if slots != [0] or self._bind_index is None or self._bind_index_ready is None:
            raise RuntimeError('Persistent bind index requires the verified free slot zero')
        for slot_idx in slots:
            runner.slot_sessions[slot_idx] = session
        torch.cuda.current_stream(runner.device).wait_event(self._bind_index_ready)
        idx_tensor = self._bind_index
        take = slice(0, len(slots))
        runner.input_pos_buf.index_copy_(0, idx_tensor, session.input_pos[take])
        runner.step_buf.index_fill_(0, idx_tensor, 0)
        runner.xy_dec_buf.index_copy_(0, idx_tensor, session.prefill_hidden[take])
        for (k_buf, v_buf), (k_src, v_src) in zip(runner.kv_cache_buf, session.kv_cache):
            k_buf.index_copy_(0, idx_tensor, k_src[take])
            v_buf.index_copy_(0, idx_tensor, v_src[take])
        runner.top_k_buf.index_fill_(0, idx_tensor, session.request.top_k)
        runner.top_p_buf.index_fill_(0, idx_tensor, session.request.top_p)
        runner.temperature_buf.index_fill_(0, idx_tensor, session.request.temperature)
        runner.repetition_penalty_buf.index_fill_(0, idx_tensor, session.request.repetition_penalty)
        runner.prefill_len_buf.index_copy_(0, idx_tensor, session.prefill_len[take])
        runner.x_len_buf.index_copy_(0, idx_tensor, session.x_lens[take])
        runner.prompt_len_buf.index_fill_(0, idx_tensor, session.prompt_len)
        max_steps = (session.max_decode_steps if session.request.early_stop_num == -1
                     else min(session.max_decode_steps, session.request.early_stop_num))
        runner.max_step_buf.index_fill_(0, idx_tensor, max_steps)
        runner.completed_buf.index_fill_(0, idx_tensor, False)
        runner.y_buf.index_fill_(0, idx_tensor, runner.decoder.EOS)
        for i, slot_idx in enumerate(slots):
            runner.y_buf[slot_idx, :session.prompt_len] = session.y[i, :session.prompt_len]
        runner.session_slots[session.id] = slots
        session.slot_indices = slots
        runner.decoder.bind_session_hook(session, runner)
        runner.bound_sessions.append(session)
        self.index_bind_hits += 1
        return True

    def prepare_pending(self):
        """Prepare observed missing lengths only under the caller's idle lock."""
        self._idle()
        for length in list(self.pending):
            x, mask, ready = self.pending[length]
            ready.synchronize()
            self.prepare(x, mask)
            del self.pending[length]

    def enable(self):
        self._idle()
        if self.enabled:
            if self.runner.decoder.h.prefill != self.prefill:
                raise RuntimeError('Prefill binding changed externally')
            if (self.batch_bind_copy or self.persistent_bind_index) and self.runner.bind_session != self.workspace_bind:
                raise RuntimeError('Runner bind binding changed externally')
            return
        if self.runner.decoder.h.prefill != self.original:
            raise RuntimeError('Prefill binding changed externally')
        self.runner.decoder.h.prefill = self.prefill
        if self.persistent_workspace:
            # Install any profiling wrapper first so it is preserved on disable.
            self.original_runner_prefill = self.runner.prefill
            self.runner.prefill = self.workspace_prefill
        if self.batch_bind_copy or self.persistent_bind_index:
            self.original_runner_bind = self.runner.bind_session
            self.runner.bind_session = self.workspace_bind
        self.enabled = True

    def disable(self):
        self._idle()
        if self.enabled:
            if self.runner.decoder.h.prefill != self.prefill:
                raise RuntimeError('Prefill binding changed externally')
            if (self.batch_bind_copy or self.persistent_bind_index) and self.runner.bind_session != self.workspace_bind:
                raise RuntimeError('Runner bind binding changed externally')
            if self.persistent_workspace:
                if self.runner.prefill != self.workspace_prefill:
                    raise RuntimeError('Runner prefill binding changed; install profiling before workspace')
                if self.batch_bind_copy or self.persistent_bind_index:
                    self.runner.bind_session = self.original_runner_bind
                    self.original_runner_bind = None
                self.runner.prefill = self.original_runner_prefill
                self.original_runner_prefill = None
                # A later profiling/probe lifecycle may use another thread.
                # Drain bind copies before permitting that owner handoff.
                import torch
                torch.cuda.synchronize(self.runner.device)
                self.workspace_owner_thread = None
            self.runner.decoder.h.prefill = self.original
            self.enabled = False

    def stats(self):
        return dict(hits=self.hits,misses=self.misses,shapes=list(self.entries),
                    pending_shapes=list(self.pending), prepare_seconds=self.prepare_seconds,
                    target_content_cache=False, persistent_workspace=self.persistent_workspace,
                    workspace_hits=self.workspace_hits, workspace_dispatches=self.workspace_dispatches,
                    trt_shape_plans=self.trt_shape_plans, trt_hits=self.trt_hits,
                    batch_bind_copy=self.batch_bind_copy, batch_bind_hits=self.batch_bind_hits,
                    persistent_bind_index=self.persistent_bind_index, index_bind_hits=self.index_bind_hits)

    def close(self):
        import torch
        self.disable()
        torch.cuda.synchronize(self.runner.device)
        for entry in self.entries.values():
            if entry.alternative is not None:
                entry.alternative.close()
            entry.graph.reset()
        self.entries.clear()
        self.pending.clear()
        self._bind_index = self._bind_index_ready = None
