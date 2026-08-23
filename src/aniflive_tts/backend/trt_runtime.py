from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import tensorrt as trt
import torch

from .compatibility import ensure_tensorrt11
from .trt_builder import DetailedTensorRTLogger


@dataclass(frozen=True)
class InferenceResult:
    outputs: dict[str, torch.Tensor]
    latency_ms: float | None
    profile_index: int
    stream_handle: int


class TensorRTRunner:
    """TensorRT 11 runner using name-based I/O and zero-copy CUDA tensors only."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | str = "cuda:0",
        profile_index: int = 0,
        logger: DetailedTensorRTLogger | None = None,
        logger_path: str | Path | None = None,
    ) -> None:
        self.tensorrt_version = ensure_tensorrt11()
        self.engine_path = Path(engine_path).resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(
                f"TensorRT engine is required and no fallback is allowed: "
                f"{self.engine_path}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; TensorRT execution cannot continue")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("TensorRTRunner only accepts a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.logger = logger or DetailedTensorRTLogger(min_severity=trt.ILogger.INFO)
        self._logger_path = Path(logger_path).resolve() if logger_path else None

        with torch.cuda.device(self.device):
            self._runtime = trt.Runtime(self.logger)
            engine_data = self.engine_path.read_bytes()
            self.engine = self._runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                self._persist_log()
                raise RuntimeError(
                    f"TensorRT failed to deserialize engine: {self.engine_path}"
                )
            if not 0 <= profile_index < self.engine.num_optimization_profiles:
                raise ValueError(
                    f"profile_index {profile_index} is outside [0, "
                    f"{self.engine.num_optimization_profiles})"
                )
            self.profile_index = int(profile_index)
            self.context = self.engine.create_execution_context()
            if self.context is None:
                self._persist_log()
                raise RuntimeError("TensorRT failed to create an execution context")

        self.input_names = tuple(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.INPUT
        )
        self.output_names = tuple(
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == trt.TensorIOMode.OUTPUT
        )
        self._shape_signature: tuple[tuple[str, tuple[int, ...]], ...] | None = None
        self._output_specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        self._persist_log()

    def infer(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        outputs: Mapping[str, torch.Tensor] | None = None,
        stream: torch.cuda.Stream | None = None,
        synchronize: bool = True,
        profile: bool = True,
    ) -> InferenceResult:
        if profile and not synchronize:
            raise ValueError("CUDA event profiling requires synchronize=True")
        supplied_inputs = set(inputs)
        expected_inputs = set(self.input_names)
        if supplied_inputs != expected_inputs:
            missing = sorted(expected_inputs - supplied_inputs)
            extra = sorted(supplied_inputs - expected_inputs)
            raise ValueError(f"TensorRT input mismatch; missing={missing}, extra={extra}")

        supplied_outputs = dict(outputs or {})
        extra_outputs = set(supplied_outputs) - set(self.output_names)
        if extra_outputs:
            raise ValueError(f"Unknown TensorRT outputs: {sorted(extra_outputs)}")

        with torch.cuda.device(self.device):
            execution_stream = stream or torch.cuda.current_stream(self.device)
            if execution_stream.device != self.device:
                raise ValueError(
                    f"CUDA stream is on {execution_stream.device}, expected {self.device}"
                )
            stream_handle = int(execution_stream.cuda_stream)
            if self.profile_index != 0 and not self.context.set_optimization_profile_async(
                self.profile_index, stream_handle
            ):
                raise RuntimeError(
                    f"TensorRT rejected optimization profile {self.profile_index}"
                )

            shape_signature = tuple(
                (name, tuple(int(dim) for dim in inputs[name].shape))
                for name in self.input_names
            )
            shapes_changed = shape_signature != self._shape_signature
            for name in self.input_names:
                tensor = inputs[name]
                self._validate_tensor(name, tensor)
                if shapes_changed and not self.context.set_input_shape(
                    name, tuple(tensor.shape)
                ):
                    raise ValueError(
                        f"TensorRT rejected runtime shape {tuple(tensor.shape)} "
                        f"for input {name!r}"
                    )
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"TensorRT failed to bind input {name!r}")
                tensor.record_stream(execution_stream)

            if shapes_changed:
                insufficient = tuple(self.context.infer_shapes())
                if insufficient:
                    raise ValueError(
                        "TensorRT shape inference has insufficient inputs: "
                        + ", ".join(insufficient)
                    )
                self._output_specs = {
                    name: (
                        tuple(
                            int(dim)
                            for dim in self.context.get_tensor_shape(name)
                        ),
                        _torch_dtype(self.engine.get_tensor_dtype(name)),
                    )
                    for name in self.output_names
                }
                self._shape_signature = shape_signature

            result_outputs: dict[str, torch.Tensor] = {}
            for name in self.output_names:
                shape, expected_dtype = self._output_specs[name]
                if any(dim < 0 for dim in shape):
                    raise RuntimeError(
                        f"TensorRT output {name!r} has data-dependent shape {shape}; "
                        "an explicit output allocator is required"
                    )
                tensor = supplied_outputs.get(name)
                if tensor is None:
                    tensor = torch.empty(shape, dtype=expected_dtype, device=self.device)
                else:
                    self._validate_tensor(name, tensor, expected_shape=shape)
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"TensorRT failed to bind output {name!r}")
                tensor.record_stream(execution_stream)
                result_outputs[name] = tensor

            start_event = torch.cuda.Event(enable_timing=True) if profile else None
            end_event = torch.cuda.Event(enable_timing=True) if profile else None
            if start_event is not None:
                start_event.record(execution_stream)
            if not self.context.execute_async_v3(stream_handle=stream_handle):
                self._persist_log()
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            if end_event is not None:
                end_event.record(execution_stream)
            if synchronize:
                execution_stream.synchronize()
            latency_ms = (
                float(start_event.elapsed_time(end_event))
                if start_event is not None and end_event is not None
                else None
            )
            return InferenceResult(
                outputs=result_outputs,
                latency_ms=latency_ms,
                profile_index=self.profile_index,
                stream_handle=stream_handle,
            )

    def write_engine_inspector(self, path: str | Path) -> Path:
        inspector = self.engine.create_engine_inspector()
        inspector.execution_context = self.context
        information = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
        try:
            json.loads(information)
        except json.JSONDecodeError as error:
            raise RuntimeError("TensorRT engine inspector returned invalid JSON") from error
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(information, encoding="utf-8")
        return target

    def io_description(self) -> tuple[dict[str, object], ...]:
        items: list[dict[str, object]] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            item: dict[str, object] = {
                "name": name,
                "mode": str(self.engine.get_tensor_mode(name)),
                "dtype": str(self.engine.get_tensor_dtype(name)),
                "engine_shape": [
                    int(dim) for dim in self.engine.get_tensor_shape(name)
                ],
            }
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                profile_shapes = self.engine.get_tensor_profile_shape(
                    name, self.profile_index
                )
                item["profile_shapes"] = [
                    [int(dim) for dim in shape] for shape in profile_shapes
                ]
            items.append(item)
        return tuple(items)

    def _validate_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        expected_shape: tuple[int, ...] | None = None,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"TensorRT I/O {name!r} must be a torch.Tensor")
        if tensor.device != self.device:
            raise ValueError(
                f"TensorRT I/O {name!r} is on {tensor.device}, expected {self.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(
                f"TensorRT I/O {name!r} must be contiguous for zero-copy binding"
            )
        expected_dtype = _torch_dtype(self.engine.get_tensor_dtype(name))
        if tensor.dtype != expected_dtype:
            raise TypeError(
                f"TensorRT I/O {name!r} has dtype {tensor.dtype}, "
                f"expected {expected_dtype}"
            )
        if expected_shape is not None and tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"TensorRT output {name!r} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shape}"
            )

    def _persist_log(self) -> None:
        if self._logger_path is not None:
            self.logger.write(self._logger_path)


def _torch_dtype(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.bfloat16: torch.bfloat16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.int64: torch.int64,
        trt.bool: torch.bool,
        trt.uint8: torch.uint8,
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise TypeError(
            f"TensorRT dtype {dtype} cannot be represented by a torch CUDA tensor"
        ) from error
