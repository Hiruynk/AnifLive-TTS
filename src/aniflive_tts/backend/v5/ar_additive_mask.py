"""Persistent additive SDPA mask for the opt-in single-slot research path.

This is a hypothesis under measurement, not a qualified replacement. Policy B
history masking requires native boolean-mask dispatch, so disable both this
mask's SamplingGraph producer and its AttentionBuckets consumer for B.
"""
from __future__ import annotations


class AdditiveAttentionMask:
    def __init__(self, runner):
        import torch
        decoder = runner.decoder
        if (runner.max_batch_size != 1 or runner.device.type != "cuda"
                or runner.xy_pos_buf.dtype != torch.float16 or decoder.max_seq_length != 2048
                or not decoder.__class__.__module__.endswith(".torch_static_cuda_graph")):
            raise ValueError("Additive mask requires the single-slot FP16 Torch-static AR runner")
        if runner.bound_sessions or runner.waiting_bind or any(runner.slot_sessions):
            raise RuntimeError("Create the additive mask only while the runner is idle")
        native = runner.extra_buffers.get("attn_mask")
        if (native is None or native.dtype != torch.bool or native.device != runner.device
                or tuple(native.shape) != (1, decoder.n_head, 1, 2048)):
            raise ValueError("Unexpected native boolean attention mask")
        self.runner = runner
        self.mask = torch.full((1, 1, 1, 2048), float("-inf"), dtype=torch.float16, device=runner.device)
        self.mask[..., 0] = 0
        self.zero = torch.zeros((), dtype=torch.float16, device=runner.device)
        self.negative_inf = torch.full((), float("-inf"), dtype=torch.float16, device=runner.device)
        self.producer = None

    def bind_producer(self, sampling_graph):
        if sampling_graph.runner is not self.runner or not sampling_graph.fuse_attention_mask:
            raise ValueError("Additive mask requires its matching fused SamplingGraph")
        if self.producer is not None and self.producer is not sampling_graph:
            raise RuntimeError("Additive mask is already bound to a SamplingGraph")
        self.producer = sampling_graph

    def require_active_producer(self):
        producer = self.producer
        if producer is not None and self.runner.decoder.pre_forward_slots_hook != producer.native_pre_forward_hook:
            raise RuntimeError("Policy B text-history masking requires native boolean-mask dispatch")
        if (producer is None or not producer.enabled
                or self.runner.decode_step != producer.decode_step):
            raise RuntimeError("Additive attention requires its active fused SamplingGraph producer")

    def require_active_consumer(self):
        buckets = getattr(self.runner._acquire_graph, "__self__", None)
        if (buckets is None or not getattr(buckets, "enabled", False)
                or getattr(buckets, "additive_mask", None) is not self):
            raise RuntimeError("Additive sampling requires the matching active AttentionBuckets consumer")
