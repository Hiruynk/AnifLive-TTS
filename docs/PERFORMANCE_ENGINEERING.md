# AnifLive-TTS Performance Engineering Records

## v1.3 Canonical Result

The v1.3 release benchmark uses the external Roxy V2ProPlus package only. Ten
sessions each run 10 warmups, 100 complete-WAV requests, 100 new-connection
streaming requests, and 100 keep-alive streaming requests at concurrency 1.
Audible TTFA is measured from client request send to the first audible PCM
sample above -45 dBFS in the earliest active 10 ms frame, constrained by the
arrival time of its server-emitted chunk. Device playback latency is excluded.

| Metric | Median across 10 sessions | Session range |
|---|---:|---:|
| Complete WAV wall P50 / P95 | 153.177 / 189.548 ms | 146.508-166.100 / 158.981-201.439 ms |
| Server inference P50 | 148.591 ms | 142.121-161.275 ms |
| RTF P50 | 0.087680 | 0.083863-0.095077 |
| New-connection first-packet P50 / P95 | 67.077 / 90.853 ms | 64.866-74.544 / 71.753-120.376 ms |
| Keep-alive first-packet P50 / P95 | 67.953 / 92.019 ms | 65.456-76.345 / 81.640-136.052 ms |
| New-connection audible TTFA P50 / P95 | 73.702 / 97.478 ms | 71.491-81.169 / 78.378-127.001 ms |
| Keep-alive audible TTFA P50 / P95 | 74.578 / 98.644 ms | 72.081-82.970 / 88.265-142.677 ms |
| GPU busy-time P50 / P95 | 53.0% / 56.0% | 50.5-54% / 56-57% |

All 3,000 formal requests reported TensorRT 11 and no PyTorch neural fallback.
The machine-readable source of truth is
`benchmarks/README_BENCHMARK_SUMMARY.json`. Expression references are prepared
at model activation and remain GPU-resident; the public WebUI reuses the active
API and does not start another neural runtime.

## v1.1 Historical Result

## Local Result

The release workload uses two independent V2ProPlus voice packages. Each voice
runs ten sessions with 10 warmups, 100 complete-WAV requests, 100 new-connection
streaming requests, and 100 keep-alive streaming requests per session. Formal
requests use concurrency 1. The two transport modes are reported separately.

Audible TTFA is measured from request send until the first playable active PCM
frame arrives. Detection uses a -45 dBFS threshold and a 10 ms analysis frame.
Device output latency is not included.

| Metric | Median across 20 model-sessions | Model-session range |
|---|---:|---:|
| Complete WAV wall P50 | 245.769 ms | 212.274-286.741 ms |
| Complete WAV wall P95 | 289.129 ms | 246.759-356.899 ms |
| Server P50 | 224.624 ms | 189.121-267.346 ms |
| RTF P50 | 0.110447 | 0.090111-0.133120 |
| New-connection first-packet P50 / P95 | 97.081 / 121.201 ms | 89.033-102.441 / 106.683-145.568 ms |
| Keep-alive first-packet P50 / P95 | 85.854 / 111.460 ms | 78.242-93.290 / 98.316-131.664 ms |
| New-connection audible TTFA P50 / P95 | 127.081 / 151.201 ms | 119.033-132.441 / 136.683-175.568 ms |
| Keep-alive audible TTFA P50 / P95 | 115.854 / 141.460 ms | 108.242-123.290 / 128.316-161.664 ms |
| GPU busy-time P50 / P95 | 43.0% / 47.0% | 41-45% / 46-49% |
| VRAM P50 | 6810 MiB | 6700-7064 MiB |

The run produced 2,000 complete-WAV, 2,000 new-connection streaming, and 2,000
keep-alive streaming measurements. Every response reported TensorRT 11 with no
PyTorch model fallback. The canonical
machine-readable report is `benchmarks/README_BENCHMARK_SUMMARY.json`.

## Quality Gates

Streaming and complete-WAV paths use the same seed and sampling settings.

| Voice package | Waveform correlation | Log-mel cosine | Speaker cosine | SI-SDR | Duration difference |
|---|---:|---:|---:|---:|---:|
| Miku V2ProPlus | 0.775697 | 0.995100 | 0.987893 | 1.792 dB | 0.000% |
| Roxy V2ProPlus | 0.995861 | 0.993346 | 0.983652 | 20.794 dB | 0.629% |

The release gates are log-mel cosine `>=0.99`, speaker cosine `>=0.98`, and
duration difference `<=3%`. No INT8, reduced generation steps, or PyTorch
model fallback is used. Speaker embeddings remove transport edge silence before
comparison so the initial anti-stutter safety padding does not bias identity.

## Accepted Optimizations

- Persistent TensorRT contexts, bindings, output shapes, and reference features.
- Reused GPT KV buffers, index tensors, and CUDA-resident sampling.
- Sampling CUDA Graph for softmax, multinomial, gather, and token storage.
- EOS readback every two steps after fixed-seed validation.
- Native TensorRT SoVITS preview decoding with latent and acoustic-noise continuity.
- Punctuation-delimited segments are submitted as soon as prior PCM is handed to the HTTP stream; serving never waits for client-side playback to finish.
- Profile-safe 32-character technical segmentation with crossfade continuity.
- Punctuation-aware natural pauses and bounded removal of excess model silence.
- A telemetry-guarded 25-second warm-retention window.
- Startup preparation of all five language frontends.
- Atomic single-model switching that unloads the active voice before loading its replacement.

## GPU Utilization

The 20 model-session median GPU busy-time was 43% P50 and 47% P95. This is an
interval sample from `nvidia-smi`, not an SM occupancy measurement. At
concurrency 1, the dependency chain of one-token GPT autoregressive enqueues
remains the main utilization limit. More concurrent requests could improve
aggregate utilization but would change the interactive single-request latency
contract, so it is not used for the release measurement.

## Rejected Experiments

### Full GPT-Step CUDA Graph

Capture was tested with TensorRT auxiliary streams, explicit auxiliary streams,
thread-local capture, and an engine built with `max_aux_streams=0`. TensorRT's
Myelin execution rejected capture with `cudaError 900`. The release retains the
validated sampling-only graph path.

### Short KV Engine

A 256-token KV engine reduced memory traffic but changed the fixed-seed
semantic sequence. TF32-disabled and TF32-enabled variants were rejected by the
quality gate.

### C++ Hot Path

A native loop requires a CUDA development stage, TensorRT headers, a
deterministic CUDA sampler or plugin, a package ABI, and full quality
revalidation. A wrapper around the same per-token enqueue would not remove the
TensorRT/Myelin capture limitation, so v1.1 keeps the validated Python control
plane and TensorRT hot path.

### CUDA 12.6 Compatibility Profile

The `cu126` profile uses PyTorch 2.10 and removes the vulnerable PyTorch 2.5.1
compatibility stack. CUDA 12.8 remains the locally GPU-validated Blackwell
profile. The `cu126` tag is not a release artifact until its image passes the
same build, vulnerability, offline-startup, and target-host validation gates.

## References

- https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html
- https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html
- https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html
- https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html
