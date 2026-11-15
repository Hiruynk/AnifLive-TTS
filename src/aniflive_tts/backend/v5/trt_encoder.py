"""Experimental TensorRT v5 target encoder; native reference preparation remains intact."""
from __future__ import annotations

from collections import OrderedDict
import threading


class TensorRTTargetEncoder:
    """Replace speed-1 target decoding while retaining the native reference path.

    The returned feature tensor owns its storage. The cached TensorRT output is
    copied on the caller's stream, and an event protects that copy before reuse.
    """

    def __init__(self, model, plan):
        import torch
        from aniflive_tts.backend.trt_runtime import TensorRTRunner

        if model.version != "v5turbo" or model.semantic_frame_rate != "25hz":
            raise ValueError("Target encoder requires the 25 Hz v5 Turbo model")
        self.model = model
        self.native_decode = model.decode_encp
        self.runner = TensorRTRunner(plan, device="cuda:0")
        if set(self.runner.input_names) != {"codes", "phones", "ge"} or self.runner.output_names != ("features",):
            raise ValueError("Unexpected TensorRT target encoder I/O")
        if int(self.runner.engine.num_aux_streams) != 0:
            raise ValueError("Target encoder requires the verified zero-auxiliary-stream plan")
        expected_profiles = {"codes": ((1, 1, 1), (1, 1, 80), (1, 1, 1536)),
                             "phones": ((1, 1), (1, 96), (1, 2048))}
        for name, expected in expected_profiles.items():
            actual = tuple(map(tuple, self.runner.engine.get_tensor_profile_shape(name, 0)))
            if actual != expected:
                raise ValueError(f"Unexpected TensorRT {name} optimization profile: {actual}")
        self.stream = torch.cuda.Stream(device="cuda:0")
        self.outputs = OrderedDict()
        self._last_copy_done = None
        self._lock = threading.Lock()
        self.enabled = False

    def decode_encp(self, codes, text, refer, ge=None, speed=1):
        import torch

        if ge is None or speed != 1:
            # Reference encoding includes the native style encoder, which is not
            # part of the target TensorRT engine. Non-unit speed deliberately
            # retains the original implementation; the exported speed-1 engine
            # is not invoked or retried for those existing API requests.
            return self.native_decode(codes, text, refer, ge, speed)
        # Match the graph adapter: widen native int32 token IDs for the plan,
        # while preserving the existing strict int64/FP16 binding validation.
        if isinstance(codes, torch.Tensor) and codes.dtype == torch.int32:
            codes = codes.to(dtype=torch.int64)
        if isinstance(text, torch.Tensor) and text.dtype == torch.int32:
            text = text.to(dtype=torch.int64)
        if not all(isinstance(value, torch.Tensor) for value in (codes, text, ge)):
            raise TypeError("Target codes, phones and ge must be Torch tensors")
        device = torch.device("cuda:0")
        if any(value.device != device for value in (codes, text, ge)):
            raise ValueError("Target codes, phones and ge must all be on cuda:0")
        if codes.dtype != torch.int64 or text.dtype != torch.int64 or ge.dtype != torch.float16:
            raise ValueError("Target codes/phones must be int64 and ge must be FP16")
        if codes.ndim != 3 or tuple(codes.shape[:2]) != (1, 1):
            raise ValueError("Target codes must have shape [1,1,S]")
        if text.ndim != 2 or text.shape[0] != 1 or tuple(ge.shape) != (1, 512, 1):
            raise ValueError("Target phones/ge must have shapes [1,P] and [1,512,1]")
        semantic_frames, phone_count = codes.shape[2], text.shape[1]
        if not 1 <= semantic_frames <= 1536 or not 1 <= phone_count <= 2048:
            raise ValueError("Target encoder profile requires 1<=S<=1536 and 1<=P<=2048")

        with self._lock, torch.inference_mode():
            current = torch.cuda.current_stream(device)
            self.stream.wait_stream(current)
            if self._last_copy_done is not None:
                self.stream.wait_event(self._last_copy_done)
            with torch.cuda.stream(self.stream):
                if semantic_frames not in self.outputs:
                    if len(self.outputs) == 4:
                        self.outputs.popitem(last=False)
                    self.outputs[semantic_frames] = torch.empty(
                        (1, 512, 4 * semantic_frames), dtype=torch.float16, device=device)
                self.outputs.move_to_end(semantic_frames)
                persistent = self.outputs[semantic_frames]
                result = self.runner.infer(
                    {"codes": codes.contiguous(), "phones": text.contiguous(), "ge": ge.contiguous()},
                    outputs={"features": persistent}, stream=self.stream, synchronize=False, profile=False)
            current.wait_stream(self.stream)
            # Copy before this shape's engine buffer can be overwritten; return
            # storage remains valid for later windows and other source streams.
            with torch.cuda.stream(current):
                persistent.record_stream(current)
                features = result.outputs["features"].clone()
                copied = torch.cuda.Event()
                copied.record(current)
            self._last_copy_done = copied
            return features, ge

    def enable(self):
        with self._lock:
            if self.enabled:
                if self.model.decode_encp != self.decode_encp:
                    raise RuntimeError("Target encoder binding changed externally")
                return
            if self.model.decode_encp != self.native_decode:
                raise RuntimeError("Native target encoder binding changed externally")
            self.model.decode_encp = self.decode_encp
            self.enabled = True

    def disable(self):
        with self._lock:
            if not self.enabled:
                return
            if self.model.decode_encp != self.decode_encp:
                raise RuntimeError("Target encoder binding changed externally")
            self.stream.synchronize()
            if self._last_copy_done is not None:
                self._last_copy_done.synchronize()
            self.model.decode_encp = self.native_decode
            self.enabled = False


def install_target_encoder(model, plan):
    """Install the experimental adapter and return its enable/disable handle."""
    adapter = TensorRTTargetEncoder(model, plan)
    adapter.enable()
    return adapter
