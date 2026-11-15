"""Isolated fixed-shape TensorRT prefill experiment; no runtime installation."""
import torch
from torch.nn import functional as F
import triton
import triton.language as tl


class PrefillWrapper(torch.nn.Module):
    """Use the frozen native layers; return their fresh prefix K/V explicitly."""
    def __init__(self, decoder):
        super().__init__()
        self.h = decoder.h
        if len(self.h.layers) != 24 or decoder.n_head != 16 or decoder.hidden_dim != 512:
            raise ValueError('Expected the 24-layer, 16-head native decoder')

    def forward(self, x, mask):
        prefixes = []
        for layer in self.h.layers:
            batch, length, _ = x.shape
            q, k, v = layer.attention.in_proj(x).chunk(3, dim=-1)
            q, k, v = (t.contiguous().view(batch, length, 16, 32) for t in (q, k, v))
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            prefixes.extend((k.contiguous(), v.contiguous()))
            attention = F.scaled_dot_product_attention(q, k, v, mask)
            attention = attention.transpose(1, 2).contiguous().view(batch, length, 512)
            h = layer.attention_norm(x + layer.attention.out_proj(attention))
            x = layer.ffn_norm(h + layer.feed_forward(h))
        return (x, *prefixes)


@triton.jit
def _copy_prefix(SOURCES, TARGETS, LENGTH: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(SOURCES + index).to(tl.pointer_type(tl.float16))
    target = tl.load(TARGETS + index).to(tl.pointer_type(tl.float16))
    valid = offsets < 16 * LENGTH * 32
    heads = offsets // (LENGTH * 32)
    in_head = offsets % (LENGTH * 32)
    values = tl.load(source + offsets, mask=valid, other=0)
    tl.store(target + heads * 2048 * 32 + in_head, values, mask=valid)


class TensorRTPrefillGraph:
    """One idle-prepared shape and full workspace; caller serializes all use."""
    def __init__(self, plan, x, mask, full_kv):
        import tensorrt as trt
        from aniflive_tts.backend.trt_runtime import TensorRTRunner
        if (x.device.type != 'cuda' or x.dtype != torch.float16 or x.ndim != 3
                or tuple(x.shape[::2]) != (1, 512)):
            raise ValueError('Expected CUDA FP16 x [1,S,512]')
        self.length = int(x.shape[1])
        if (mask.device != x.device or mask.dtype != torch.bool
                or tuple(mask.shape) != (1, 1, self.length, self.length)):
            raise ValueError('Expected native CUDA bool mask [1,1,S,S]')
        targets = [t for pair in full_kv for t in pair]
        if len(targets) != 48 or any(t.device != x.device or t.dtype != torch.float16
                or tuple(t.shape) != (1, 16, 2048, 32) or not t.is_contiguous() for t in targets):
            raise ValueError('Expected 48 full contiguous native HND caches')
        self.runner = TensorRTRunner(plan, device=x.device)
        self.names = [name for i in range(24) for name in (f'key_{i:02d}', f'value_{i:02d}')]
        if (set(self.runner.input_names) != {'x', 'mask'}
                or set(self.runner.output_names) != {'hidden', *self.names}
                or self.runner.engine.num_aux_streams != 0
                or self.runner.engine.name != f'v5-prefill-fixed-{self.length}'):
            raise ValueError('Unexpected prefill TensorRT plan I/O or auxiliary streams')
        expected = {'x': ((1, self.length, 512), trt.float16),
                    'mask': ((1, 1, self.length, self.length), trt.bool),
                    'hidden': ((1, self.length, 512), trt.float16)}
        expected.update({name: ((1, 16, self.length, 32), trt.float16) for name in self.names})
        if any(tuple(self.runner.engine.get_tensor_shape(name)) != shape
               or self.runner.engine.get_tensor_dtype(name) != dtype
               for name, (shape, dtype) in expected.items()):
            raise ValueError('Prefill TensorRT plan has unexpected fixed I/O shape or dtype')
        self.inputs = {'x': torch.empty_like(x, memory_format=torch.contiguous_format),
                       'mask': torch.empty_like(mask, memory_format=torch.contiguous_format)}
        self.outputs = {'hidden': torch.empty_like(x, memory_format=torch.contiguous_format)}
        self.outputs.update({name: torch.empty((1, 16, self.length, 32), dtype=x.dtype, device=x.device)
                             for name in self.names})
        self.targets = targets  # Tensor addresses remain valid for graph lifetime.
        self.source_ptrs = torch.tensor([self.outputs[name].data_ptr() for name in self.names],
                                        dtype=torch.int64, device=x.device)
        self.target_ptrs = torch.tensor([t.data_ptr() for t in targets], dtype=torch.int64, device=x.device)
        current = torch.cuda.current_stream(x.device)
        side = torch.cuda.Stream(device=x.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            self.inputs['x'].copy_(x)
            self.inputs['mask'].copy_(mask)
            for _ in range(2):
                self.runner.infer(self.inputs, outputs=self.outputs, stream=side,
                                  synchronize=False, profile=False)
                self._writeback()
            side.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=side):
                if not self.runner.context.execute_async_v3(stream_handle=side.cuda_stream):
                    raise RuntimeError('TensorRT prefill enqueue failed during graph capture')
                self._writeback()
            side.synchronize()
        current.wait_stream(side)

    def _writeback(self):
        _copy_prefix[(48, triton.cdiv(16 * self.length * 32, 1024))](
            self.source_ptrs, self.target_ptrs, LENGTH=self.length, BLOCK=1024, num_warps=4)

    def prefill(self, x, mask):
        if (x.device != self.inputs['x'].device or x.dtype != torch.float16
                or tuple(x.shape) != tuple(self.inputs['x'].shape)
                or mask.device != x.device or mask.dtype != torch.bool
                or tuple(mask.shape) != tuple(self.inputs['mask'].shape)):
            raise ValueError('Inputs do not match prepared prefill shape/device/dtypes')
        self.inputs['x'].copy_(x)
        self.inputs['mask'].copy_(mask)
        self.graph.replay()
        return self.outputs['hidden'].clone()

    def close(self):
        torch.cuda.synchronize(self.inputs['x'].device)
        self.graph.reset()
