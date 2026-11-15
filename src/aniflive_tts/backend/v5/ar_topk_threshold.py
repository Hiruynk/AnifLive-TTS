"""Experimental exact top-15 cutoff for one FP16 AR logit row."""
import torch
import triton
import triton.language as tl


@triton.jit
def _threshold(X, OUT, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    raw = tl.load(X + offsets, mask=offsets < 1025, other=-float('inf'))
    # torch.topk treats NaNs as largest. If its fifteenth value is NaN,
    # comparison with the pivot is false for every value, so nothing is masked.
    isnan = (raw != raw) & (offsets < 1025)
    nan_count = tl.sum(isnan.to(tl.int32), axis=0)
    keys = tl.where(isnan, float('inf'), raw.to(tl.float32))
    ordered = tl.sort(keys, descending=True)
    pivot = tl.gather(ordered, tl.full((1,), 14, tl.int32), axis=0)
    bits = raw.to(tl.uint16, bitcast=True)
    result_bits = tl.where((nan_count < 15) & (raw.to(tl.float32) < pivot),
                           tl.full((BLOCK,), 0xfc00, tl.uint16), bits)
    tl.store(OUT + offsets, result_bits.to(tl.float16, bitcast=True), mask=offsets < 1025)


def threshold(logits, out=None):
    if (not isinstance(logits, torch.Tensor) or logits.device.type != 'cuda'
            or logits.dtype != torch.float16 or tuple(logits.shape) != (1, 1025)
            or not logits.is_contiguous()):
        raise ValueError('Expected contiguous CUDA FP16 logits with shape [1, 1025]')
    if out is None:
        out = torch.empty_like(logits)
    if (out.device != logits.device or out.dtype != logits.dtype or out.shape != logits.shape
            or not out.is_contiguous() or out.data_ptr() == logits.data_ptr()):
        raise ValueError('Output must be a separate contiguous CUDA FP16 [1, 1025] tensor')
    _threshold[(1,)](logits, out, BLOCK=2048, num_warps=8)
    return out
