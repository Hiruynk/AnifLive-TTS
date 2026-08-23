from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import tensorrt as trt

from .compatibility import ensure_tensorrt11


@dataclass(frozen=True)
class ShapeRange:
    """Minimum, optimum, and maximum shapes for one TensorRT input."""

    min_shape: tuple[int, ...]
    opt_shape: tuple[int, ...]
    max_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_shape", tuple(int(v) for v in self.min_shape))
        object.__setattr__(self, "opt_shape", tuple(int(v) for v in self.opt_shape))
        object.__setattr__(self, "max_shape", tuple(int(v) for v in self.max_shape))
        ranks = {len(self.min_shape), len(self.opt_shape), len(self.max_shape)}
        if len(ranks) != 1:
            raise ValueError("Profile min/opt/max shapes must have the same rank")
        for low, optimum, high in zip(
            self.min_shape, self.opt_shape, self.max_shape, strict=True
        ):
            if low < 0 or not low <= optimum <= high:
                raise ValueError(
                    "Profile dimensions must satisfy 0 <= min <= opt <= max"
                )


@dataclass(frozen=True)
class BuildResult:
    engine_path: str
    engine_bytes: int
    build_time_ms: float
    timing_cache_path: str | None
    inspector_path: str | None
    logger_path: str | None
    num_layers: int
    num_optimization_profiles: int
    tf32_allowed: bool
    builder_optimization_level: int
    configured_max_aux_streams: int | None
    engine_num_aux_streams: int
    io_tensors: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DetailedTensorRTLogger(trt.ILogger):
    """Thread-safe TensorRT logger that can persist complete build/runtime logs."""

    def __init__(
        self,
        min_severity: trt.ILogger.Severity = trt.ILogger.VERBOSE,
        *,
        echo: bool = True,
    ) -> None:
        trt.ILogger.__init__(self)
        self.min_severity = min_severity
        self.echo = echo
        self._records: list[str] = []
        self._lock = threading.Lock()

    def log(self, severity: trt.ILogger.Severity, msg: str) -> None:
        if int(severity) > int(self.min_severity):
            return
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        severity_name = str(severity).rsplit(".", 1)[-1]
        line = f"{timestamp} [TensorRT:{severity_name}] {msg.rstrip()}"
        with self._lock:
            self._records.append(line)
            if self.echo:
                print(line, file=sys.stderr, flush=True)

    @property
    def records(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def write(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self.records)
        if content:
            content += "\n"
        target.write_text(content, encoding="utf-8")
        return target


class TensorRTEngineBuilder:
    """Build TensorRT 11 strongly typed engines from ONNX models."""

    def __init__(
        self,
        *,
        workspace_bytes: int = 2 << 30,
        allow_tf32: bool = True,
        builder_optimization_level: int = 3,
        max_aux_streams: int | None = None,
        logger: DetailedTensorRTLogger | None = None,
    ) -> None:
        self.tensorrt_version = ensure_tensorrt11()
        if workspace_bytes <= 0:
            raise ValueError("workspace_bytes must be positive")
        if not 0 <= builder_optimization_level <= 5:
            raise ValueError("builder_optimization_level must be between 0 and 5")
        if max_aux_streams is not None and max_aux_streams < 0:
            raise ValueError("max_aux_streams must be non-negative")
        self.workspace_bytes = int(workspace_bytes)
        self.allow_tf32 = bool(allow_tf32)
        self.builder_optimization_level = int(builder_optimization_level)
        self.max_aux_streams = (
            None if max_aux_streams is None else int(max_aux_streams)
        )
        self.logger = logger or DetailedTensorRTLogger()

    def build(
        self,
        onnx_path: str | Path,
        engine_path: str | Path,
        *,
        profiles: Sequence[Mapping[str, ShapeRange]] = (),
        timing_cache_path: str | Path | None = None,
        inspector_path: str | Path | None = None,
        logger_path: str | Path | None = None,
    ) -> BuildResult:
        source = Path(onnx_path).resolve()
        target = Path(engine_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.logger.clear()

        started = time.perf_counter()
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse_from_file(str(source)):
            errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
            detail = "\n".join(errors) if errors else "Unknown ONNX parser error"
            self._persist_log(logger_path)
            raise RuntimeError(f"TensorRT failed to parse {source}:\n{detail}")

        config = builder.create_builder_config()
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        config.builder_optimization_level = self.builder_optimization_level
        if self.max_aux_streams is not None:
            config.max_aux_streams = self.max_aux_streams
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace_bytes)
        if not self.allow_tf32:
            config.clear_flag(trt.BuilderFlag.TF32)
        self._add_profiles(builder, network, config, profiles)

        timing_cache_target = (
            Path(timing_cache_path).resolve() if timing_cache_path is not None else None
        )
        cache_data = (
            timing_cache_target.read_bytes()
            if timing_cache_target is not None and timing_cache_target.is_file()
            else b""
        )
        timing_cache = config.create_timing_cache(cache_data)
        if timing_cache is None or not config.set_timing_cache(
            timing_cache, ignore_mismatch=False
        ):
            self._persist_log(logger_path)
            raise RuntimeError("TensorRT rejected the timing cache for this device")

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            self._persist_log(logger_path)
            raise RuntimeError("TensorRT build_serialized_network returned None")
        engine_data = memoryview(serialized).tobytes()
        self._write_bytes(target, engine_data)

        if timing_cache_target is not None:
            cache = config.get_timing_cache()
            if cache is None:
                raise RuntimeError("TensorRT did not return the configured timing cache")
            cache_memory = cache.serialize()
            self._write_bytes(timing_cache_target, memoryview(cache_memory).tobytes())

        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            self._persist_log(logger_path)
            raise RuntimeError("TensorRT could not deserialize the newly built engine")

        inspector_target = (
            Path(inspector_path).resolve() if inspector_path is not None else None
        )
        if inspector_target is not None:
            inspector = engine.create_engine_inspector()
            inspector_json = inspector.get_engine_information(
                trt.LayerInformationFormat.JSON
            )
            try:
                json.loads(inspector_json)
            except json.JSONDecodeError as error:
                raise RuntimeError("TensorRT engine inspector returned invalid JSON") from error
            inspector_target.parent.mkdir(parents=True, exist_ok=True)
            inspector_target.write_text(inspector_json, encoding="utf-8")

        io_tensors = tuple(self._describe_io(engine))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        persisted_log = self._persist_log(logger_path)
        return BuildResult(
            engine_path=str(target),
            engine_bytes=len(engine_data),
            build_time_ms=elapsed_ms,
            timing_cache_path=(
                str(timing_cache_target) if timing_cache_target is not None else None
            ),
            inspector_path=(
                str(inspector_target) if inspector_target is not None else None
            ),
            logger_path=str(persisted_log) if persisted_log is not None else None,
            num_layers=int(engine.num_layers),
            num_optimization_profiles=int(engine.num_optimization_profiles),
            tf32_allowed=self.allow_tf32,
            builder_optimization_level=self.builder_optimization_level,
            configured_max_aux_streams=self.max_aux_streams,
            engine_num_aux_streams=int(engine.num_aux_streams),
            io_tensors=io_tensors,
        )

    def _add_profiles(
        self,
        builder: trt.Builder,
        network: trt.INetworkDefinition,
        config: trt.IBuilderConfig,
        profiles: Sequence[Mapping[str, ShapeRange]],
    ) -> None:
        inputs = {
            network.get_input(index).name: network.get_input(index)
            for index in range(network.num_inputs)
        }
        dynamic_names = {
            name
            for name, tensor in inputs.items()
            if tensor.is_shape_tensor or any(int(dim) == -1 for dim in tensor.shape)
        }
        if dynamic_names and not profiles:
            names = ", ".join(sorted(dynamic_names))
            raise ValueError(f"Dynamic ONNX inputs require optimization profiles: {names}")
        if not profiles:
            return

        for profile_index, values in enumerate(profiles):
            missing = dynamic_names - values.keys()
            unknown = values.keys() - inputs.keys()
            if missing:
                raise ValueError(
                    f"Optimization profile {profile_index} is missing: "
                    + ", ".join(sorted(missing))
                )
            if unknown:
                raise ValueError(
                    f"Optimization profile {profile_index} has unknown inputs: "
                    + ", ".join(sorted(unknown))
                )
            profile = builder.create_optimization_profile()
            for name, shape_range in values.items():
                tensor = inputs[name]
                if tensor.is_shape_tensor:
                    profile.set_shape_input(
                        name,
                        shape_range.min_shape,
                        shape_range.opt_shape,
                        shape_range.max_shape,
                    )
                else:
                    self._validate_execution_shapes(tensor, shape_range, profile_index)
                    profile.set_shape(
                        name,
                        shape_range.min_shape,
                        shape_range.opt_shape,
                        shape_range.max_shape,
                    )
            added_index = config.add_optimization_profile(profile)
            if added_index < 0:
                raise RuntimeError(
                    f"TensorRT failed to add optimization profile {profile_index}"
                )

    @staticmethod
    def _validate_execution_shapes(
        tensor: trt.ITensor, shape_range: ShapeRange, profile_index: int
    ) -> None:
        declared = tuple(int(dim) for dim in tensor.shape)
        if len(shape_range.min_shape) != len(declared):
            raise ValueError(
                f"Profile {profile_index} input {tensor.name!r} rank mismatch: "
                f"ONNX rank {len(declared)}, profile rank {len(shape_range.min_shape)}"
            )
        for axis, static_dim in enumerate(declared):
            if static_dim == -1:
                if shape_range.min_shape[axis] <= 0:
                    raise ValueError(
                        f"Dynamic execution dimension must be positive: "
                        f"{tensor.name}[{axis}]"
                    )
                continue
            supplied = (
                shape_range.min_shape[axis],
                shape_range.opt_shape[axis],
                shape_range.max_shape[axis],
            )
            if supplied != (static_dim, static_dim, static_dim):
                raise ValueError(
                    f"Profile {profile_index} changes static dimension "
                    f"{tensor.name}[{axis}]={static_dim}: {supplied}"
                )

    @staticmethod
    def _describe_io(engine: trt.ICudaEngine) -> list[dict[str, object]]:
        description: list[dict[str, object]] = []
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            description.append(
                {
                    "name": name,
                    "mode": str(engine.get_tensor_mode(name)),
                    "dtype": str(engine.get_tensor_dtype(name)),
                    "shape": [int(dim) for dim in engine.get_tensor_shape(name)],
                }
            )
        return description

    def _persist_log(self, path: str | Path | None) -> Path | None:
        return self.logger.write(path) if path is not None else None

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
