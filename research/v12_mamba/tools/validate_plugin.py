#!/usr/bin/env python3
"""Build, execute, and validate the standalone TensorRT 11 Mamba-2 plugin."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import numpy as np

try:
    import tensorrt as trt
except ModuleNotFoundError:
    # The isolated development image pins NVIDIA's bindings and libraries
    # directly. Loading the library wheel first provides the same API without
    # installing the separate top-level loader package.
    import tensorrt_libs  # noqa: F401
    import tensorrt_bindings as trt


PLUGIN_NAME = "AnifLive-TTS-Mamba2Update"
PLUGIN_VERSION = "1"
PLUGIN_NAMESPACE = "aniflive_tts"


class CudaRuntime:
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so.12")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.lib.cudaStreamCreateWithFlags.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
        self.lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaEventCreate.restype = ctypes.c_int
        self.lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaEventRecord.restype = ctypes.c_int
        self.lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventSynchronize.restype = ctypes.c_int
        self.lib.cudaEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.cudaEventElapsedTime.restype = ctypes.c_int
        self.lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventDestroy.restype = ctypes.c_int
        self.lib.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaRuntimeGetVersion.restype = ctypes.c_int

    @staticmethod
    def _check(code: int, operation: str) -> None:
        if code != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {code}")

    def runtime_version(self) -> int:
        value = ctypes.c_int()
        self._check(self.lib.cudaRuntimeGetVersion(ctypes.byref(value)), "cudaRuntimeGetVersion")
        return int(value.value)

    def stream(self) -> ctypes.c_void_p:
        value = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreateWithFlags(ctypes.byref(value), 1), "cudaStreamCreateWithFlags")
        return value

    def event(self) -> ctypes.c_void_p:
        value = ctypes.c_void_p()
        self._check(self.lib.cudaEventCreate(ctypes.byref(value)), "cudaEventCreate")
        return value

    def record(self, event: ctypes.c_void_p, stream: ctypes.c_void_p) -> None:
        self._check(self.lib.cudaEventRecord(event, stream), "cudaEventRecord")

    def event_elapsed_ms(self, start: ctypes.c_void_p, end: ctypes.c_void_p) -> float:
        self._check(self.lib.cudaEventSynchronize(end), "cudaEventSynchronize")
        elapsed = ctypes.c_float()
        self._check(self.lib.cudaEventElapsedTime(ctypes.byref(elapsed), start, end), "cudaEventElapsedTime")
        return float(elapsed.value)

    def synchronize(self, stream: ctypes.c_void_p) -> None:
        self._check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def copy_to_device(self, destination: int, source: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(source)
        self._check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(contiguous.ctypes.data),
                contiguous.nbytes,
                self.HOST_TO_DEVICE,
            ),
            "cudaMemcpy(H2D)",
        )

    def copy_to_host(self, destination: np.ndarray, source: int) -> None:
        if not destination.flags.c_contiguous:
            raise ValueError("Destination must be contiguous")
        self._check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(destination.ctypes.data),
                ctypes.c_void_p(source),
                destination.nbytes,
                self.DEVICE_TO_HOST,
            ),
            "cudaMemcpy(D2H)",
        )


class DeviceBuffer:
    def __init__(self, cuda: CudaRuntime, nbytes: int) -> None:
        self.cuda = cuda
        self.nbytes = nbytes
        pointer = ctypes.c_void_p()
        cuda._check(cuda.lib.cudaMalloc(ctypes.byref(pointer), nbytes), "cudaMalloc")
        self.pointer = pointer

    @property
    def address(self) -> int:
        if self.pointer.value is None:
            raise RuntimeError("Device buffer was released")
        return int(self.pointer.value)

    def close(self) -> None:
        if self.pointer.value is not None:
            self.cuda._check(self.cuda.lib.cudaFree(self.pointer), "cudaFree")
            self.pointer = ctypes.c_void_p()


class TrtLogger(trt.ILogger):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def log(self, severity: trt.ILogger.Severity, message: str) -> None:
        if severity <= trt.ILogger.Severity.INFO:
            line = f"[{severity.name}] {message}"
            self.messages.append(line)
            print(line, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], percentage: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentage))


def gpu_description() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,compute_cap",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def initialize_plugin(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    initializer = library.initAnifLiveTTSMambaPlugins
    initializer.argtypes = []
    initializer.restype = ctypes.c_bool
    if not initializer():
        raise RuntimeError("initAnifLiveTTSMambaPlugins returned false")
    creator = trt.get_plugin_registry().get_creator(PLUGIN_NAME, PLUGIN_VERSION, PLUGIN_NAMESPACE)
    if creator is None:
        raise RuntimeError("TensorRT did not register the Mamba-2 plugin creator")
    return library


def make_plugin(
    creator: trt.IPluginCreatorV3One,
    dim: int,
    nheads: int,
    ngroups: int,
    dstate: int,
) -> trt.IPluginV3:
    values = {
        "dim": np.asarray([dim], dtype=np.int32),
        "nheads": np.asarray([nheads], dtype=np.int32),
        "ngroups": np.asarray([ngroups], dtype=np.int32),
        "dstate": np.asarray([dstate], dtype=np.int32),
        "delta_softplus": np.asarray([1], dtype=np.int32),
    }
    fields = [
        trt.PluginField(name, value, trt.PluginFieldType.INT32)
        for name, value in values.items()
    ]
    plugin = creator.create_plugin(
        "aniflive_mamba2_update",
        trt.PluginFieldCollection(fields),
        trt.TensorRTPhase.BUILD,
    )
    if plugin is None:
        raise RuntimeError("Plugin creator rejected the fixed Mamba-2 configuration")
    return plugin


def build_engine(
    logger: TrtLogger,
    engine_path: Path,
    batch: int,
    dim: int,
    nheads: int,
    ngroups: int,
    dstate: int,
) -> float:
    creator = trt.get_plugin_registry().get_creator(PLUGIN_NAME, PLUGIN_VERSION, PLUGIN_NAMESPACE)
    if creator is None:
        raise RuntimeError("Plugin creator is unavailable during build")

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 5
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024 * 1024)

    head_dim = dim // nheads
    inputs = [
        network.add_input("x", trt.float16, (batch, dim)),
        network.add_input("state", trt.float16, (batch, nheads, dstate, head_dim)),
        network.add_input("delta", trt.float16, (batch, nheads)),
        network.add_input("delta_bias", trt.float32, (nheads,)),
        network.add_input("A", trt.float32, (nheads,)),
        network.add_input("B", trt.float16, (batch, ngroups, dstate)),
        network.add_input("C", trt.float16, (batch, ngroups, dstate)),
        network.add_input("D", trt.float32, (nheads,)),
        network.add_input("z", trt.float16, (batch, dim)),
    ]
    if any(value is None for value in inputs):
        raise RuntimeError("TensorRT rejected an input tensor")

    plugin = make_plugin(creator, dim, nheads, ngroups, dstate)
    layer = network.add_plugin_v3(inputs, [], plugin)
    if layer is None:
        raise RuntimeError("TensorRT rejected the IPluginV3 layer")
    layer.name = "mamba2_update"
    layer.get_output(0).name = "y"
    layer.get_output(1).name = "state_out"
    network.mark_output(layer.get_output(0))
    network.mark_output(layer.get_output(1))

    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if plan is None:
        raise RuntimeError("TensorRT failed to build the Mamba-2 engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(plan))
    return build_seconds


def reference_step(
    state: np.ndarray,
    x: np.ndarray,
    delta: np.ndarray,
    delta_bias: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch, nheads, dstate, head_dim = state.shape
    ngroups = b.shape[1]
    x_heads = x.astype(np.float32).reshape(batch, nheads, head_dim)
    z_heads = z.astype(np.float32).reshape(batch, nheads, head_dim)
    state_float = state.astype(np.float32)
    dt_raw = delta.astype(np.float32) + delta_bias[None, :]
    dt = np.where(dt_raw <= 20.0, np.log1p(np.exp(dt_raw)), dt_raw).astype(np.float32)
    transition = np.exp(a[None, :] * dt).astype(np.float32)
    group_index = np.arange(nheads, dtype=np.int32) // (nheads // ngroups)
    b_heads = b.astype(np.float32)[:, group_index, :]
    c_heads = c.astype(np.float32)[:, group_index, :]
    new_state = (
        state_float * transition[:, :, None, None]
        + b_heads[:, :, :, None] * dt[:, :, None, None] * x_heads[:, :, None, :]
    )
    output = d[None, :, None] * x_heads
    output += np.sum(new_state * c_heads[:, :, :, None], axis=2)
    silu_z = z_heads / (1.0 + np.exp(-z_heads))
    output *= silu_z
    return output.reshape(batch, nheads * head_dim).astype(np.float16), new_state.astype(np.float16)


def comparison(actual: np.ndarray, expected: np.ndarray, atol: float, cosine_gate: float) -> dict[str, Any]:
    actual_f = actual.astype(np.float32).reshape(-1)
    expected_f = expected.astype(np.float32).reshape(-1)
    delta = actual_f - expected_f
    denominator = float(np.linalg.norm(actual_f) * np.linalg.norm(expected_f))
    cosine = float(np.dot(actual_f, expected_f) / denominator) if denominator > 0.0 else 1.0
    maximum = float(np.max(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    return {
        "max_abs": maximum,
        "rmse": rmse,
        "cosine": cosine,
        "atol_gate": atol,
        "cosine_gate": cosine_gate,
        "passed": maximum <= atol and cosine >= cosine_gate,
    }


def bind_context(
    context: trt.IExecutionContext,
    buffers: dict[str, DeviceBuffer],
    state_in: DeviceBuffer,
    state_out: DeviceBuffer,
) -> None:
    for name in ("x", "delta", "delta_bias", "A", "B", "C", "D", "z", "y"):
        if not context.set_tensor_address(name, buffers[name].address):
            raise RuntimeError(f"set_tensor_address failed for {name}")
    if not context.set_tensor_address("state", state_in.address):
        raise RuntimeError("set_tensor_address failed for state")
    if not context.set_tensor_address("state_out", state_out.address):
        raise RuntimeError("set_tensor_address failed for state_out")


def execute_steps(
    contexts: tuple[trt.IExecutionContext, trt.IExecutionContext],
    stream: ctypes.c_void_p,
    steps: int,
) -> None:
    for index in range(steps):
        if not contexts[index & 1].execute_async_v3(int(stream.value)):
            raise RuntimeError(f"TensorRT enqueueV3 failed at recurrent step {index}")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.dstate <= 128:
        raise ValueError("dstate must be in [1, 128]")
    if args.dim % args.nheads != 0 or args.nheads % args.ngroups != 0:
        raise ValueError("dim/nheads and nheads/ngroups must divide evenly")

    plugin_path = Path(args.plugin).resolve()
    engine_path = Path(args.engine).resolve()
    report_path = Path(args.report).resolve()
    if not plugin_path.is_file():
        raise FileNotFoundError(plugin_path)

    logger = TrtLogger()
    _plugin_library = initialize_plugin(plugin_path)
    build_seconds = build_engine(
        logger,
        engine_path,
        args.batch,
        args.dim,
        args.nheads,
        args.ngroups,
        args.dstate,
    )

    started = time.perf_counter()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    deserialize_seconds = time.perf_counter() - started
    if engine is None:
        raise RuntimeError("TensorRT could not deserialize the generated engine")
    contexts = (engine.create_execution_context(), engine.create_execution_context())
    if contexts[0] is None or contexts[1] is None:
        raise RuntimeError("TensorRT could not create both persistent execution contexts")

    rng = np.random.default_rng(args.seed)
    head_dim = args.dim // args.nheads
    host = {
        "x": (rng.normal(0.0, 0.05, (args.batch, args.dim))).astype(np.float16),
        "state": (rng.normal(0.0, 0.03, (args.batch, args.nheads, args.dstate, head_dim))).astype(np.float16),
        "delta": (rng.normal(0.0, 0.1, (args.batch, args.nheads))).astype(np.float16),
        "delta_bias": (rng.uniform(-4.0, -2.0, args.nheads)).astype(np.float32),
        "A": (-rng.uniform(0.5, 1.5, args.nheads)).astype(np.float32),
        "B": (rng.normal(0.0, 0.05, (args.batch, args.ngroups, args.dstate))).astype(np.float16),
        "C": (rng.normal(0.0, 0.05, (args.batch, args.ngroups, args.dstate))).astype(np.float16),
        "D": (rng.normal(0.0, 0.1, args.nheads)).astype(np.float32),
        "z": (rng.normal(0.0, 0.1, (args.batch, args.dim))).astype(np.float16),
    }

    cuda = CudaRuntime()
    stream = cuda.stream()
    buffers: dict[str, DeviceBuffer] = {}
    events: list[ctypes.c_void_p] = []
    try:
        for name in ("x", "delta", "delta_bias", "A", "B", "C", "D", "z"):
            buffers[name] = DeviceBuffer(cuda, host[name].nbytes)
            cuda.copy_to_device(buffers[name].address, host[name])
        buffers["y"] = DeviceBuffer(cuda, host["x"].nbytes)
        state_buffers = (DeviceBuffer(cuda, host["state"].nbytes), DeviceBuffer(cuda, host["state"].nbytes))
        buffers["state0"] = state_buffers[0]
        buffers["state1"] = state_buffers[1]
        bind_context(contexts[0], buffers, state_buffers[0], state_buffers[1])
        bind_context(contexts[1], buffers, state_buffers[1], state_buffers[0])

        # One-step FP16 parity.
        cuda.copy_to_device(state_buffers[0].address, host["state"])
        execute_steps(contexts, stream, 1)
        cuda.synchronize(stream)
        gpu_y = np.empty_like(host["x"])
        gpu_state = np.empty_like(host["state"])
        cuda.copy_to_host(gpu_y, buffers["y"].address)
        cuda.copy_to_host(gpu_state, state_buffers[1].address)
        ref_y, ref_state = reference_step(
            host["state"], host["x"], host["delta"], host["delta_bias"], host["A"], host["B"], host["C"], host["D"], host["z"]
        )
        one_step = {
            "output": comparison(gpu_y, ref_y, args.one_step_atol, 0.99999),
            "state": comparison(gpu_state, ref_state, args.one_step_atol, 0.99999),
        }

        # Recurrent reference parity with FP16 state quantization after each step.
        recurrent_ref_state = host["state"].copy()
        recurrent_ref_y = ref_y
        for _ in range(args.reference_steps):
            recurrent_ref_y, recurrent_ref_state = reference_step(
                recurrent_ref_state, host["x"], host["delta"], host["delta_bias"], host["A"], host["B"], host["C"], host["D"], host["z"]
            )
        cuda.copy_to_device(state_buffers[0].address, host["state"])
        execute_steps(contexts, stream, args.reference_steps)
        cuda.synchronize(stream)
        recurrent_gpu_y = np.empty_like(host["x"])
        recurrent_gpu_state = np.empty_like(host["state"])
        cuda.copy_to_host(recurrent_gpu_y, buffers["y"].address)
        recurrent_final = state_buffers[args.reference_steps & 1].address
        cuda.copy_to_host(recurrent_gpu_state, recurrent_final)
        recurrent = {
            "steps": args.reference_steps,
            "output": comparison(recurrent_gpu_y, recurrent_ref_y, args.recurrent_atol, 0.9999),
            "state": comparison(recurrent_gpu_state, recurrent_ref_state, args.recurrent_atol, 0.9999),
        }

        # Long-loop stability and aggregate GPU execution time.
        cuda.copy_to_device(state_buffers[0].address, host["state"])
        start_event = cuda.event()
        end_event = cuda.event()
        events.extend([start_event, end_event])
        cuda.record(start_event, stream)
        execute_steps(contexts, stream, args.stability_steps)
        cuda.record(end_event, stream)
        stability_ms = cuda.event_elapsed_ms(start_event, end_event)
        stability_y = np.empty_like(host["x"])
        stability_state = np.empty_like(host["state"])
        cuda.copy_to_host(stability_y, buffers["y"].address)
        stability_final = state_buffers[args.stability_steps & 1].address
        cuda.copy_to_host(stability_state, stability_final)
        stability = {
            "steps": args.stability_steps,
            "gpu_total_ms": stability_ms,
            "gpu_us_per_step": stability_ms * 1000.0 / args.stability_steps,
            "output_finite": bool(np.isfinite(stability_y).all()),
            "state_finite": bool(np.isfinite(stability_state).all()),
            "output_peak_abs": float(np.max(np.abs(stability_y.astype(np.float32)))),
            "state_peak_abs": float(np.max(np.abs(stability_state.astype(np.float32)))),
        }

        # Grouped measurements avoid per-step synchronization contaminating the kernel time.
        gpu_us: list[float] = []
        wall_us: list[float] = []
        for group in range(args.benchmark_groups):
            start = cuda.event()
            end = cuda.event()
            events.extend([start, end])
            wall_start = time.perf_counter()
            cuda.record(start, stream)
            for offset in range(args.benchmark_steps_per_group):
                index = group * args.benchmark_steps_per_group + offset
                if not contexts[index & 1].execute_async_v3(int(stream.value)):
                    raise RuntimeError("TensorRT enqueueV3 failed during microbenchmark")
            cuda.record(end, stream)
            elapsed_ms = cuda.event_elapsed_ms(start, end)
            wall_elapsed = time.perf_counter() - wall_start
            gpu_us.append(elapsed_ms * 1000.0 / args.benchmark_steps_per_group)
            wall_us.append(wall_elapsed * 1_000_000.0 / args.benchmark_steps_per_group)

        microbenchmark = {
            "groups": args.benchmark_groups,
            "steps_per_group": args.benchmark_steps_per_group,
            "gpu_us_per_step_p50": percentile(gpu_us, 50),
            "gpu_us_per_step_p95": percentile(gpu_us, 95),
            "gpu_us_per_step_mean": statistics.fmean(gpu_us),
            "wall_us_per_step_p50": percentile(wall_us, 50),
            "wall_us_per_step_p95": percentile(wall_us, 95),
            "wall_us_per_step_mean": statistics.fmean(wall_us),
        }

        passed = (
            all(section["passed"] for section in one_step.values())
            and recurrent["output"]["passed"]
            and recurrent["state"]["passed"]
            and stability["output_finite"]
            and stability["state_finite"]
        )
        report = {
            "schema": "aniflive-mamba2-plugin-feasibility-v1",
            "status": "pass" if passed else "fail",
            "scope": "standalone TensorRT 11 Mamba-2 recurrent update; not production integration",
            "environment": {
                "gpu": gpu_description(),
                "cuda_runtime_version": cuda.runtime_version(),
                "tensorrt_version": trt.__version__,
            },
            "configuration": {
                "batch": args.batch,
                "dim": args.dim,
                "nheads": args.nheads,
                "ngroups": args.ngroups,
                "dstate": args.dstate,
                "head_dim": head_dim,
                "dtype": "FP16 activations/state; FP32 A/D/delta_bias",
            },
            "artifacts": {
                "plugin": str(plugin_path),
                "plugin_sha256": sha256(plugin_path),
                "engine": str(engine_path),
                "engine_sha256": sha256(engine_path),
                "engine_bytes": engine_path.stat().st_size,
            },
            "build_seconds": build_seconds,
            "deserialize_seconds": deserialize_seconds,
            "one_step_parity": one_step,
            "recurrent_parity": recurrent,
            "long_loop_stability": stability,
            "microbenchmark": microbenchmark,
            "tensorrt_log_tail": logger.messages[-80:],
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not passed:
            raise RuntimeError(f"Mamba-2 feasibility gates failed; see {report_path}")
        return report
    finally:
        for event in events:
            cuda._check(cuda.lib.cudaEventDestroy(event), "cudaEventDestroy")
        for buffer in buffers.values():
            buffer.close()
        cuda._check(cuda.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--ngroups", type=int, default=1)
    parser.add_argument("--dstate", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reference-steps", type=int, default=64)
    parser.add_argument("--stability-steps", type=int, default=4096)
    parser.add_argument("--benchmark-groups", type=int, default=40)
    parser.add_argument("--benchmark-steps-per-group", type=int, default=128)
    parser.add_argument("--one-step-atol", type=float, default=0.005)
    parser.add_argument("--recurrent-atol", type=float, default=0.03)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate(args)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
