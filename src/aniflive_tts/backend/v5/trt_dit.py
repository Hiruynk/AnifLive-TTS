"""V5 CFG-0 static-condition DiT export boundary and TensorRT step adapter."""
import torch
from aniflive_tts.backend.trt_runtime import TensorRTRunner


class DiTStep(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        if native.d_embed is not None:
            raise ValueError('V5 DiT boundary requires no step embedding')
        self.native = native

    def forward(self, x, time, condition, mask, rope):
        # cond/text transposes are unused in the native static-cache branch.
        cache = {'condition': condition, 'mask': mask, 'rope': (rope, 1.0)}
        return self.native(x, x, None, time, text0=x, infer=True, static_cache=cache)[0]


class TensorRTDiT(torch.nn.Module):
    def __init__(self, native, plan):
        super().__init__()
        self.native = native
        self.runner = TensorRTRunner(plan, device='cuda:0')
        self.stream = torch.cuda.Stream(device='cuda:0')
        self.outputs = {}

    def prepare_static_cache(self, cond, lengths, mu):
        return self.native.prepare_static_cache(cond, lengths, mu)

    def forward(self, x, cond, lengths, time, *, text0=None, infer=False,
                static_cache=None, drop_audio_cond=False, drop_text=False, **_):
        if not infer or static_cache is None or drop_audio_cond or drop_text:
            raise ValueError('TRT v5 DiT requires native positive static conditioning and CFG 0')
        frames = x.shape[-1]
        if not 128 <= frames <= 1000:
            raise ValueError('DiT engine profile supports 128–1000 total frames')
        rope, scale = static_cache['rope']
        if isinstance(scale, torch.Tensor) or scale not in (1, 1.0, None):
            raise ValueError('This DiT engine requires native unscaled RoPE')
        current = torch.cuda.current_stream(x.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            if frames not in self.outputs:
                self.outputs[frames] = torch.empty((1, frames, 100), device=x.device, dtype=x.dtype)
            inputs = {'x': x.contiguous(), 'time': time.contiguous(),
                      'condition': static_cache['condition'].contiguous(),
                      'mask': static_cache['mask'].contiguous(), 'rope': rope.contiguous()}
            result = self.runner.infer(inputs, outputs={'velocity': self.outputs[frames]},
                                       stream=self.stream, synchronize=False, profile=False)
        current.wait_stream(self.stream)
        return result.outputs['velocity'], None, None
