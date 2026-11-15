"""Qualified-profile vocoder candidate; CUDA tensors stay on the GPU."""
import torch
from aniflive_tts.backend.trt_runtime import TensorRTRunner


class V5TRTVocoder:
    def __init__(self, plan):
        self.runner = TensorRTRunner(plan, device='cuda:0')
        self.stream = torch.cuda.Stream(device='cuda:0')
        self.outputs = {}

    def __call__(self, mel):
        if mel.dtype != torch.float16 or not mel.is_cuda or tuple(mel.shape[:2]) != (1, 100):
            raise ValueError('Vocoder engine requires CUDA FP16 [1,100,T]')
        frames = mel.shape[-1]
        if not 16 <= frames <= 1024:
            raise ValueError('Vocoder candidate profile supports 16–1024 mel frames')
        source_stream = torch.cuda.current_stream(mel.device)
        self.stream.wait_stream(source_stream)
        with torch.cuda.stream(self.stream):
            if frames not in self.outputs:
                self.outputs[frames] = torch.empty((1, 1, frames*480), dtype=torch.float16, device=mel.device)
            result = self.runner.infer({'mel': mel.contiguous()}, outputs={'audio': self.outputs[frames]},
                                       stream=self.stream, synchronize=False, profile=False)
        source_stream.wait_stream(self.stream)
        return result.outputs['audio']
