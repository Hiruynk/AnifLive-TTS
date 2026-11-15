"""Scoped v5 policy-B text-history mask for the native Torch-static AR decoder."""
from __future__ import annotations

from contextlib import contextmanager
import time


@contextmanager
def text_history_mask(acceleration, reference_phones, history_phones, prompt_tokens):
    """Allow text queries to use history while preventing audio queries reading it.

    The caller must own the research engine exclusively and close the semantic
    iterator before exiting this scope. No semantic tokens or audio are trimmed.
    """
    if (acceleration is None or acceleration.backend != "torch_static_cuda_graph"
            or not acceleration.graph_enabled or acceleration.engine is None):
        raise RuntimeError("Text-only history requires the native Torch-static CUDA graph backend")
    engine = acceleration.engine
    runner = engine.model_runner
    decoder = runner.decoder
    if (runner.max_batch_size != 1 or not decoder.__class__.__module__.endswith(".torch_static_cuda_graph")
            or not runner._should_use_graph()):
        raise RuntimeError("Text-only history requires the single-slot Torch-static runner")
    if (runner.bound_sessions or runner.waiting_bind or any(runner.slot_sessions)
            or engine.handles or not engine.pending_requests.empty()):
        raise RuntimeError("Text-history mask must be installed while the AR worker is idle")
    if (type(reference_phones) is not int or type(history_phones) is not int
            or type(prompt_tokens) is not int or reference_phones < 0
            or history_phones <= 0 or prompt_tokens <= 0):
        raise ValueError("Invalid text-history mask bounds")
    start, end = reference_phones, reference_phones + history_phones
    original_prefill = decoder.h.prefill
    original_hook = decoder.pre_forward_slots_hook

    def prefill(x, kv, mask):
        text_length = x.shape[1] - prompt_tokens
        if (x.shape[0] != 1 or text_length < end or mask.ndim != 4
                or tuple(mask.shape[-2:]) != (x.shape[1], x.shape[1])):
            raise RuntimeError("Native AR prefill does not match text-history mask bounds")
        changed = mask.clone()
        changed[..., text_length:, start:end] = False
        return original_prefill(x, kv, changed)

    def pre_forward(slots, active_runner):
        if active_runner is not runner or slots != [0]:
            raise RuntimeError("Text-history mask encountered another AR runner/slot")
        result = original_hook(slots, active_runner)
        mask = result.get("attn_mask")
        base = runner.extra_buffers.get("attn_mask")
        if (mask is None or base is None or mask.data_ptr() != base.data_ptr()
                or mask.ndim != 4 or end > mask.shape[-1]):
            raise RuntimeError("Native Torch-static attention mask binding changed")
        mask[..., start:end] = False
        return result

    decoder.h.prefill = prefill
    decoder.pre_forward_slots_hook = pre_forward
    try:
        yield
    finally:
        try:
            # infer_stream.close() cancels and waits for the owner thread. On
            # worker errors, unbinding may finish just after the result arrives.
            deadline = time.monotonic() + 5.0
            while (runner.bound_sessions or runner.waiting_bind or engine.handles
                   or not engine.pending_requests.empty()):
                if time.monotonic() >= deadline:
                    engine._stop_event.set()
                    worker = engine._loop_thread
                    if worker is not None:
                        worker.join(timeout=5.0)
                    acceleration._text_history_mask_failed = True
                    raise RuntimeError("AR worker did not quiesce; attention-mask runtime requires restart")
                time.sleep(0.001)
        finally:
            decoder.h.prefill = original_prefill
            decoder.pre_forward_slots_hook = original_hook
