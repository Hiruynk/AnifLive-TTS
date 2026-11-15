"""Experimental single-slot CUDA sampling/post-step graph for the native AR runner.

Install only after the native runner has completed its warmups and is idle.
The scheduler, prefill, KV cache and existing neural-decoder graph remain native.
"""
from __future__ import annotations

import time
import weakref
import os
import torch


@torch.jit.script
def native_sample_ratio(probs: torch.Tensor) -> torch.Tensor:
    return probs / -torch.log(torch.rand_like(probs))


class SamplingGraph:
    def __init__(self, runner, *, fuse_attention_mask=False, compact_host_state=False, fast_topk=False,
                 additive_mask=None, deterministic_repetition=False, fused_post=False, eos_pinned_poll=False,
                 eos_native_poll_library=None, capture_for_joint=False, exact_topk_threshold=False):
        import torch
        import torch.nn.functional as F
        from GPT_SoVITS.Accel.PyTorch.AR.sample_funcs import apply_repetition_penalty, apply_top_k

        if type(deterministic_repetition) is not bool:
            raise TypeError("deterministic_repetition must be a boolean")
        if deterministic_repetition:
            from .repetition_penalty import apply_repetition_penalty
        if type(fuse_attention_mask) is not bool:
            raise TypeError("fuse_attention_mask must be a boolean")
        if type(compact_host_state) is not bool:
            raise TypeError("compact_host_state must be a boolean")
        if type(fused_post) is not bool:
            raise TypeError("fused_post must be a boolean")
        if type(eos_pinned_poll) is not bool:
            raise TypeError("eos_pinned_poll must be a boolean")
        if eos_native_poll_library is not None and not eos_pinned_poll:
            raise ValueError("Native EOS polling requires eos_pinned_poll")
        if fused_post and (not fuse_attention_mask or additive_mask is not None):
            raise ValueError("Fused post-sampling requires the fused boolean attention mask")
        if additive_mask is not None:
            from .ar_additive_mask import AdditiveAttentionMask
            if not isinstance(additive_mask, AdditiveAttentionMask) or additive_mask.runner is not runner:
                raise ValueError("Additive mask must belong to this AR runner")
            if not fuse_attention_mask:
                raise ValueError("Additive attention requires fused native mask generation")
            if additive_mask.producer is not None:
                raise RuntimeError("Additive mask already has a producer")
        if runner.max_batch_size != 1 or runner.device.type != "cuda":
            raise ValueError("Sampling graph requires exactly one CUDA slot")
        if runner.xy_dec_buf.dtype != torch.float16:
            raise ValueError("Sampling graph requires the native FP16 sampler")
        if runner.bound_sessions or runner.waiting_bind or any(runner.slot_sessions):
            raise RuntimeError("Install only when the native AR runner is idle")
        if not runner._should_use_graph() or 1 not in runner.graph_cache:
            raise RuntimeError("The existing neural-decoder CUDA graph is required")

        self.runner = runner
        self.original_decode_step = runner.decode_step
        self.enabled = False
        self.steps = 0
        self.done_readbacks = 0
        self.graph = torch.cuda.CUDAGraph()
        self.joint_handle = None
        decoder = runner.decoder
        self.fuse_attention_mask = fuse_attention_mask
        self.additive_mask = additive_mask
        self.compact_host_state = compact_host_state
        if exact_topk_threshold and not fast_topk:
            raise ValueError('Exact threshold requires the qualified top_k=15 path')
        self.exact_topk_threshold = exact_topk_threshold
        self.deterministic_repetition = deterministic_repetition
        self.fused_post = fused_post
        self.eos_pinned_poll = eos_pinned_poll
        self.eos_poll_queries = 0
        self.eos_poll_wall_ns = 0
        self.eos_poll_cpu_ns = 0
        self.eos_native_poller = None
        if eos_pinned_poll:
            # One owner thread and one CUDA stream consume this scalar before
            # the next replay; neither the host buffer nor event is shared.
            self.eos_device_scalar = runner.completed_buf[0]
            self.eos_host_scalar = torch.empty((), dtype=runner.completed_buf.dtype, device='cpu', pin_memory=True)
            self.eos_event = torch.cuda.Event()
            if eos_native_poll_library is not None:
                from .eos_event_poll import NativeEventPoll
                self.eos_native_poller = NativeEventPoll(eos_native_poll_library)
        if compact_host_state:
            # Immutable views avoid recreating a Tensor.select each step. The
            # completion vectors are read-only to the custom runner and are
            # never passed to the native runner for in-place modification.
            self.completed_scalar = runner.completed_buf[0]
            self.step_scalar = runner.step_buf[0]
            self.completed_false = torch.zeros((1,), device=runner.device, dtype=torch.bool)
            self.completed_true = torch.ones((1,), device=runner.device, dtype=torch.bool)
            self._validated_session = None
        self.native_pre_forward_hook = decoder.pre_forward_slots_hook
        if fuse_attention_mask:
            # Policy B temporarily patches this hook to mask text history. That
            # path must disable this sampling graph; do not silently ignore it.
            if (decoder.max_seq_length != 2048
                    or not decoder.__class__.__module__.endswith(".torch_static_cuda_graph")
                    or getattr(self.native_pre_forward_hook, "__func__", None)
                    is not getattr(type(decoder), "pre_forward_slots_hook", None)):
                raise ValueError("Fused attention mask requires an unpatched Torch-static decoder")
            self.attention_mask = runner.extra_buffers.get("attn_mask")
            if (self.attention_mask is None or self.attention_mask.dtype != torch.bool
                    or self.attention_mask.device != runner.device
                    or tuple(self.attention_mask.shape) != (1, decoder.n_head, 1, decoder.max_seq_length)):
                raise ValueError("Unexpected native Torch-static attention mask")
            self.mask_positions = torch.arange(decoder.max_seq_length, device=runner.device)
        if fused_post:
            from .sampling_post import SamplingPost
            self.post = SamplingPost(runner, self.attention_mask)
        # CUDA graph replay retains raw addresses, so externally allocated
        # inputs must remain live for the entire lifetime of the graph.
        self.batch_index = torch.arange(1, device=runner.device)
        changed = (runner.y_buf, runner.xy_pos_buf, runner.input_pos_buf,
                   runner.step_buf, runner.completed_buf)
        if fuse_attention_mask:
            changed += (self.attention_mask,)
        if additive_mask is not None:
            changed += (additive_mask.mask,)
        started = time.perf_counter()

        def graph_body():
            # Keep logits_all in-place: the native repetition penalty mutates it,
            # and native EOS argmax observes those penalized (pre-top-k) logits.
            logits_all = decoder.ar_predict_layer(runner.xy_dec_buf[:1].squeeze(1))
            logits_all[:, -1] = torch.where(runner.step_buf[:1] == 0,
                                             float("-inf"), logits_all[:, -1])
            penalized = apply_repetition_penalty(
                logits_all, runner.y_buf[:1],
                runner.prompt_len_buf[:1] + runner.step_buf[:1],
                runner.repetition_penalty_buf[:1])
            if fast_topk:
                # This sampler is explicitly restricted to top_k=15. Keep
                # all ties at the identical fifteenth-largest value.
                if exact_topk_threshold:
                    from .ar_topk_threshold import threshold
                    top_logits = threshold(penalized)
                else:
                    pivot=torch.topk(penalized,15,dim=-1,largest=True,sorted=True).values[:,-1:]
                    top_logits=torch.where(penalized < pivot,-float('inf'),penalized)
            else:
                top_logits = apply_top_k(penalized, runner.top_k_buf[:1])
            probs = F.softmax(top_logits, dim=-1)
            ratio = native_sample_ratio(probs)
            if os.environ.get('QUALIFICATION_CAPTURE_SAMPLER_RATIO') == '1':
                self._qa_sampling_tensors = (logits_all, top_logits, probs, ratio)
            if self.fused_post:
                self.post(ratio, logits_all)
            else:
                samples = torch.argmax(ratio, dim=-1, keepdim=True).to(dtype=torch.int32)
                argmax_token = torch.argmax(logits_all, dim=-1)
                eos = (argmax_token == decoder.EOS) | (samples.squeeze(1) == decoder.EOS)
                y_emb = decoder.ar_audio_embedding(samples.clamp_max_(decoder.vocab_size - 2))
                pos_offset = runner.step_buf[:1] + runner.prompt_len_buf[:1] + 1
                # Preserve the native FP16 embedding and positional operations.
                position = decoder.ar_audio_position
                pe = position.pe[self.batch_index, pos_offset - 1]
                next_xy = y_emb * position.x_scale + position.alpha * pe.unsqueeze(1)
                write_pos = (runner.prompt_len_buf[:1] + runner.step_buf[:1]).long().unsqueeze(1)
                write = (~runner.completed_buf[:1]) & (~eos)
                previous = runner.y_buf[:1].gather(1, write_pos)
                runner.y_buf[:1].scatter_(1, write_pos, torch.where(write.unsqueeze(1), samples, previous))
                next_step = runner.step_buf[:1] + write.to(runner.step_buf.dtype)
                completed = runner.completed_buf[:1] | eos | (next_step >= runner.max_step_buf[:1])
                next_pos = runner.input_pos_buf[:1] + write.to(runner.input_pos_buf.dtype)
                runner.input_pos_buf[:1].copy_(torch.where(completed, torch.ones_like(next_pos), next_pos))
                runner.step_buf[:1].copy_(next_step)
                runner.completed_buf[:1].copy_(completed)
                runner.xy_pos_buf[:1].copy_(torch.where((~completed).view(1, 1, 1), next_xy, runner.xy_pos_buf[:1]))
                if self.fuse_attention_mask:
                    valid = (self.mask_positions[None, :] < runner.input_pos_buf[:1, None])[:, None, None, :]
                    self.attention_mask[:1].copy_(valid)
                    if self.additive_mask is not None:
                        torch.where(valid, self.additive_mask.zero, self.additive_mask.negative_inf,
                                    out=self.additive_mask.mask)

        if capture_for_joint:
            self._joint_capture_body = graph_body
            self._joint_changed = changed
        # Capturing and the one warmup touch persistent runner buffers and RNG;
        # restore both before the next real request. No active session may exist.
        with torch.cuda.device(runner.device), torch.inference_mode():
            state = torch.cuda.get_rng_state(runner.device)
            saved = [buffer.clone() for buffer in changed]
            side = torch.cuda.Stream(device=runner.device)
            side.wait_stream(torch.cuda.current_stream(runner.device))
            try:
                with torch.cuda.stream(side):
                    warm_probs = torch.full((1, 1025), 1.0 / 1025, dtype=torch.float16, device=runner.device)
                    for _ in range(8):
                        native_sample_ratio(warm_probs)
                    if 'prim::TensorExprGroup' not in str(torch.jit.last_executed_optimized_graph()):
                        raise RuntimeError('Native sampling ratio did not acquire its TorchScript fusion')
                    graph_body()
                    for target, snapshot in zip(changed, saved):
                        target.copy_(snapshot)
                side.synchronize()
                with torch.cuda.graph(self.graph, stream=side):
                    graph_body()
            finally:
                torch.cuda.synchronize(runner.device)
                for target, snapshot in zip(changed, saved):
                    target.copy_(snapshot)
                torch.cuda.set_rng_state(state, runner.device)
                torch.cuda.synchronize(runner.device)
        self.capture_seconds = time.perf_counter() - started
        if additive_mask is not None:
            additive_mask.bind_producer(self)

    @staticmethod
    def _check_request(session):
        request = session.request
        if (session.bsz != 1 or request.top_k != 15 or request.top_p != 1.0
                or request.temperature != 1.0 or request.repetition_penalty != 1.35
                or not request.use_cuda_graph or request.debug):
            raise ValueError("Sampling graph supports one slot, top_k=15, top_p=1, temperature=1, repetition_penalty=1.35, CUDA graph, debug=false only")
        if session.slot_indices != [0]:
            raise ValueError("Sampling graph requires the sole active slot to be zero")

    def _validate_session(self, session):
        if not self.compact_host_state:
            self._check_request(session)
            return
        if self._validated_session is None or self._validated_session() is not session:
            # T2SSession has converted prompts before it can be bound. The
            # owner thread treats request sampling parameters as immutable.
            self._check_request(session)
            self._validated_session = weakref.ref(session)
        if (session.bsz != 1 or len(session.slot_indices) != 1 or session.slot_indices[0] != 0
                or session.step_count < 0 or len(session.output_lengths) != 1
                or session.output_lengths[0] != session.step_count
                or session.total_tokens != session.step_count):
            raise RuntimeError("Native single-slot sampling counters changed unexpectedly")

    def decode_step(self):
        import torch
        runner = self.runner
        if not runner.bound_sessions and not runner.waiting_bind:
            return []
        with torch.inference_mode():
            joint_replayed = False
            if len(runner.bound_sessions) > 1:
                raise ValueError("Sampling graph does not support concurrent sessions")
            if (runner.bound_sessions and self.fuse_attention_mask
                    and runner.decoder.pre_forward_slots_hook != self.native_pre_forward_hook):
                raise RuntimeError("Fused attention mask is incompatible with text-history masking; disable the sampling graph for policy B")
            if runner.bound_sessions and self.additive_mask is not None:
                self.additive_mask.require_active_consumer()
            forward_slots = []
            if runner.bound_sessions:
                session = runner.bound_sessions[0]
                self._validate_session(session)
                if runner.waiting_bind:
                    raise ValueError("Sampling graph does not support concurrent sessions")
                if session.step_count:
                    forward_slots = [0]
                    if not runner._should_use_graph():
                        raise RuntimeError("Native neural-decoder graph became unavailable")
                    if not self.fuse_attention_mask:
                        runner.decoder.pre_forward_slots_hook(forward_slots, runner)
                    neural_graph = runner._acquire_graph(1)
                    joint = self.joint_handle
                    if joint is not None and joint.enabled and neural_graph is joint.neural.graph:
                        joint.replay()
                        joint_replayed = True
                    else:
                        neural_graph.replay()
            if runner.waiting_bind and runner.free_slots() > 0:
                runner._bind_waiting_sessions()
            if len(runner.bound_sessions) != 1 or runner.waiting_bind:
                raise ValueError("Sampling graph requires exactly one active session")
            session = runner.bound_sessions[0]
            self._validate_session(session)
            if self.fuse_attention_mask and runner.decoder.pre_forward_slots_hook != self.native_pre_forward_hook:
                raise RuntimeError("Fused attention mask is incompatible with text-history masking; disable the sampling graph for policy B")
            if self.additive_mask is not None:
                self.additive_mask.require_active_consumer()
            if not joint_replayed:
                self.graph.replay()
            self.steps += 1
            # One explicit device-to-host completion check per token. Only an
            # ending step needs an extra length read to distinguish EOS/limit.
            if self.eos_pinned_poll:
                started_poll = time.perf_counter_ns()
                started_cpu = time.thread_time_ns()
                self.eos_host_scalar.copy_(self.eos_device_scalar, non_blocking=True)
                self.eos_event.record(torch.cuda.current_stream(runner.device))
                if self.eos_native_poller is not None:
                    queries = self.eos_native_poller.poll(self.eos_event)
                else:
                    queries = 0
                    while True:
                        queries += 1
                        if self.eos_event.query():
                            break
                done = bool(self.eos_host_scalar.item())
                self.eos_poll_queries += queries
                self.eos_poll_wall_ns += time.perf_counter_ns() - started_poll
                self.eos_poll_cpu_ns += time.thread_time_ns() - started_cpu
            else:
                done = bool((self.completed_scalar if self.compact_host_state else runner.completed_buf[0]).item())
            self.done_readbacks += 1
            length = (int((self.step_scalar if self.compact_host_state else runner.step_buf[0]).item())
                      if done else session.step_count + 1)
            session.output_lengths = [length]
            session.step_count = length
            session.total_tokens = length
            session.completed = ((self.completed_true if done else self.completed_false)
                                 if self.compact_host_state else runner.completed_buf[:1].clone())
            finished = []
            if done:
                session.y_results = [runner.y_buf[0, session.prompt_len:session.prompt_len + length].clone()]
                finished = [session]
                runner.unbind_session(session)
            runner.decoder.post_forward_slots_hook(forward_slots, runner)
            return finished

    def enable(self):
        if self.runner.bound_sessions or self.runner.waiting_bind:
            raise RuntimeError("Switch sampling only when the AR runner is idle")
        self.runner.decode_step = self.decode_step
        self.enabled = True

    def disable(self):
        if self.runner.bound_sessions or self.runner.waiting_bind:
            raise RuntimeError("Switch sampling only when the AR runner is idle")
        self.runner.decode_step = self.original_decode_step
        self.enabled = False
