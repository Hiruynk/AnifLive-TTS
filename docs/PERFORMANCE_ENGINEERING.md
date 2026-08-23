# AnifLive-TTS v1 Performance Engineering

## Ten-Session Local Result

Only AnifLive-TTS was measured locally for this release refresh. The workload
used a fixed external V2ProPlus test asset, short text, seed, and sampling
configuration. Each session received 10 warmups, 100 complete-WAV requests,
and 100 streaming requests over new local HTTP/1.1 connections.

| Metric | Session median | Session range |
|---|---:|---:|
| Complete WAV wall P50 | 252.862 ms | 249.456-263.883 ms |
| Complete WAV wall P95 | 311.778 ms | 287.871-346.559 ms |
| Server P50 | 230.533 ms | 227.369-244.072 ms |
| RTF P50 | 0.085426 | 0.084276-0.089150 |
| Streaming TTFA P50 | 78.964 ms | 77.106-81.385 ms |
| Streaming TTFA P95 | 104.440 ms | 90.404-127.497 ms |
| GPU busy-time P50 / P95 | 47.0% / 51.5% | 46-48% / 51-53% |
| VRAM P50 | 4554 MiB | 4543-4776 MiB |

Ten sessions produced 1000 complete-WAV and 1000 streaming measurements.
Every response reported TensorRT 11 and no PyTorch model fallback. See
`benchmarks/README_BENCHMARK_SUMMARY.json` for the machine-readable summary.

## Quality Gates

The current output and accepted deterministic regression output produced
waveform correlation 1.0, log-mel cosine 1.0, log-mel MAE 0, TensorRT speaker
cosine 1.0, and zero duration difference.
No INT8, reduced steps, changed sampler, or PyTorch model fallback is present.

Quality reports are retained as external acceptance artifacts and are not
included in the source tree.

## Accepted Optimizations

- Persistent TensorRT contexts, bindings, output shapes, and reference features.
- Reused GPT destination KV buffers and one persistent index tensor.
- One device-to-host transfer for encoder lengths.
- CUDA-resident sampling with exact PyTorch RNG semantics.
- Sampling-only CUDA Graph for softmax, multinomial, gather, and token storage.
- EOS readback every two steps for one-segment requests; interval one for multiple segments.
- HTTP/1.1 keep-alive and startup-only warmup.

Intervals 1, 2, 3, 4, 5, 6, and 8 were measured. Interval 2 was the best
validated tradeoff for the fixed 74-step workload. Larger intervals reduce
synchronization but may execute unnecessary steps after EOS.

## GPU Utilization

The ten-session median GPU busy-time was 47% P50 and 51.5% P95. VRAM was
4554 MiB P50. This is not an SM occupancy metric. The bottleneck is the
dependency chain of 74 one-token GPT enqueues: token N+1 cannot run before
sampling token N. GPT decode measured 218.319 ms P50 while SoVITS measured
7.416 ms P50.

More concurrent requests could raise aggregate utilization, but would increase
single-request latency and change the API concurrency contract. It is not used
to inflate the release benchmark.

## Rejected Experiments

### Full GPT-Step CUDA Graph

Capture was tested with the original seven TensorRT auxiliary streams,
explicitly supplied auxiliary streams, thread-local capture mode, and a separately built engine with
`max_aux_streams=0`. All variants failed inside TensorRT's Myelin
`executeTrainStation` with `cudaError 900: operation not permitted when stream
is capturing`. A failed capture also invalidates the process capture state, so
the release hard-disables this path and keeps sampling-only capture.

### Short KV Engine

An engine with a 256-token KV capacity reduced memory traffic but changed the
fixed-seed semantic sequence. TF32-disabled and TF32-enabled variants both
failed the exact-output gate and were rejected.

### C++ Hot Path

The host and runtime image do not contain `nvcc`; the TensorRT Python wheel does
not contain C++ headers. A real native loop additionally needs a deterministic
CUDA sampler/plugin, a CUDA devel build stage, a new package ABI, and full
quality validation. A C++ wrapper around the same per-token enqueue would not
remove the TensorRT/Myelin capture limitation. v1 therefore records this as
an attempted research path, not a completed feature.

### CUDA 12.1 On Blackwell

The cu121 image built all eight TensorRT 11 engines and deserialized them.
Startup then failed at the first PyTorch CUDA tensor transfer because the
PyTorch 2.5.1+cu121 wheel has no `sm_120` kernel. NVIDIA lists CUDA 12.8 as the
first toolkit support for Blackwell. cu121 is not claimed as GPU-E2E verified
on RTX 5070 Ti.

## References

- https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html
- https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
- https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html
- https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html
