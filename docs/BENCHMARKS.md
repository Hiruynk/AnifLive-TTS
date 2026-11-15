# Benchmarks

## v1.5 HTTP measurements

The AnifLive-TTS runs use matched reference audio, transcript, text, seed and sampling settings on RTX 5070 Ti and Mac Studio M5 Max. Each backend/device configuration has ten sessions, 1,000 measurements per mode (3,000 total per configuration), 100 warmups and ten connection primes. The local gsv.cpp v0.1.3 Vulkan run has three sessions and 300 measurements per mode (900 total), plus 30 warmups and three primes.

Requests are serial HTTP/1.1. Audible TTFA uses 10 ms analysis frames and a -45 dBFS threshold, constrained by PCM arrival; device playback latency is excluded. First packet, audible output and full-response time are recorded separately.

CUDA／TensorRT（Docker） uses FP16; v5Turbo retains 4-step CFM/CFG 0. Apple Silicon（macOS／MLX／Metal） uses FP16 with Float32 opmath on M5 Max (40-core GPU, 128 GB, macOS 27.0). v2ProPlus outputs 32 kHz; v5Turbo outputs 48 kHz.

| Model | Condition | First body / PCM packet P50 / P95 (ms) | Audible TTFA P50 / P95 (ms) | Full-response wall P50 / P95 (ms) | RTF P50 / P95 | Requests |
|---|---|---:|---:|---:|---:|---:|
| v5Turbo | Complete WAV | 199.80 / 210.27 | — | 206.76 / 219.39 | 0.10549 / 0.11193 | 1,000 |
| v5Turbo | PCM, new connection | 68.18 / 71.71 | 82.22 / 85.75 | 73.37 / 79.04 | 0.04566 / 0.04918 | 1,000 |
| v5Turbo | PCM, keep-alive | 66.89 / 70.29 | **80.93 / 84.33** | 75.98 / 79.93 | 0.04728 / 0.04974 | 1,000 |
| v2ProPlus | Complete WAV | 176.78 / 227.03 | — | 180.98 / 231.12 | 0.10922 / 0.13948 | 1,000 |
| v2ProPlus | PCM, new connection | 66.79 / 90.74 | 76.79 / 100.74 | 189.21 / 234.94 | 0.09480 / 0.11771 | 1,000 |
| v2ProPlus | PCM, keep-alive | 65.70 / 89.37 | 75.70 / 99.37 | 189.27 / 238.13 | 0.09483 / 0.11931 | 1,000 |
| v5Turbo (M5 Max) | Complete WAV | — | — | 231.09 / 236.38 | 0.13435 / 0.13743 | 1,000 |
| v5Turbo (M5 Max) | PCM, new connection | 274.13 / 279.84 | 404.67 / 410.38 | 542.37 / 560.29 | 0.32342 / 0.33410 | 1,000 |
| v5Turbo (M5 Max) | PCM, keep-alive | 274.15 / 280.24 | 404.69 / 410.78 | 542.63 / 561.18 | 0.32357 / 0.33464 | 1,000 |

The gsv.cpp rows below use Vulkan NV_coopmat2, F16 AR/F32 vocoder, 4-step CFM/CFG 0 and cut=0. The first body packet is a RIFF header; first PCM and audible output have separate timestamps. All observed reconnects remain included.

| Condition | First body/header packet P50 / P95 (ms) | First PCM packet P50 / P95 (ms) | Audible TTFA P50 / P95 (ms) | Full-response wall P50 / P95 (ms) | RTF P50 / P95 | Requests |
|---|---:|---:|---:|---:|---:|---:|
| Complete WAV | 237.35 / 247.31 | — | — | 242.23 / 252.46 | 0.13763 / 0.14345 | 300 |
| PCM, new connection | 2.31 / 2.68 | 172.72 / 180.30 | 352.72 / 360.30 | 194.88 / 202.96 | 0.11072 / 0.11532 | 300 |
| PCM, keep-alive | 3.99 / 4.35 | 174.45 / 181.30 | 354.45 / 361.30 | 197.06 / 204.35 | 0.11197 / 0.11611 | 300 |

## GPT-SoVITS family and acceleration

First-output values keep their published measurement type. Local audible TTFA uses the P50 first audible PCM time; upstream first-packet, first-token and first-output measurements are not interchangeable. Rows are sorted by the displayed latency.

| System | Metric | First output | Test conditions | Source |
|---|---|---:|---|---|
| AnifLive-TTS v1.3 (historical) | Audible TTFA P50 | 74.578 ms | RTX 5070 Ti; persistent HTTP/1.1; 10 Roxy sessions | [Historical local measurement](../benchmarks/README_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5 v2ProPlus | Audible TTFA P50 | 75.70 ms | Windows; RTX 5070 Ti; TensorRT 11 FP16; keep-alive PCM; 10 sessions | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5 v5Turbo | Audible TTFA P50 | 80.93 ms | Windows; RTX 5070 Ti; TensorRT 11 / Triton FP16; keep-alive PCM; 10 sessions | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| gsv.cpp v0.1.3 · v5Turbo | Audible TTFA P50 | 354.45 ms | RTX 5070 Ti; Vulkan; keep-alive PCM; 3-session exploratory run | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5 v5Turbo (M5 Max) | Audible TTFA P50 | 404.69 ms | Mac Studio M5 Max; macOS 27.0; Rust / MLX / Metal FP16; keep-alive PCM; 10 sessions | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| GPT-SoVITS C++ TRT Stream | First packet | 460 ms | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference ONNX Stream | First token | 1,000 ms | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT fitted | First token | 2,022 ms | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

RTF is reported with each source's workload and conditions. Rows are sorted by the displayed RTF.

| System | RTF | Backend / model | Test conditions | Source |
|---|---:|---|---|---|
| GPT-SoVITS v2ProPlus | 0.014 | PyTorch parallel inference | RTX 4090; about four minutes of output | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| GPT-SoVITS v2ProPlus | 0.028 | PyTorch parallel inference | RTX 4060 Ti | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| AnifLive-TTS v1.5 v5Turbo | 0.04728 | TensorRT 11 / Triton FP16 | Windows; RTX 5070 Ti; keep-alive PCM; 10 sessions | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.3 (historical) | 0.087680 | TensorRT 11 FP16 | RTX 5070 Ti; 10 Roxy sessions | [Historical local measurement](../benchmarks/README_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5 v2ProPlus | 0.09483 | TensorRT 11 FP16 | Windows; RTX 5070 Ti; keep-alive PCM; 10 sessions | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| GPT-SoVITS C++ TRT | 0.1020 | TensorRT | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| gsv.cpp v0.1.3 · v5Turbo | 0.11197 | Vulkan; F16 AR / F32 vocoder | RTX 5070 Ti; 3-session exploratory run | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| GPT-SoVITS Minimal Inference TRT fitted | 0.2096 | TensorRT fitted | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |
| AnifLive-TTS v1.5 v5Turbo (M5 Max) | 0.32357 | Rust / MLX / Metal FP16 | Mac Studio M5 Max; macOS 27.0; keep-alive PCM; 10 sessions | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |

## Other mainstream TTS

Third-party values are reported by their cited sources, not from a controlled comparison. GPU, input, first-packet size, concurrency, and measurement protocol differ. The historical v1.3 local row is retained with its own workload; current Windows v1.5 rows use the matching HTTP campaign. Rows are sorted within each metric table.

| System | First-audio metric | First audio | Statistic | Test conditions | Source |
|---|---|---:|---|---|---|
| AnifLive-TTS v1.3 (historical) | Audible TTFA | 74.578 ms | P50 | RTX 5070 Ti; persistent HTTP/1.1; 10 Roxy sessions | [Historical local measurement](../benchmarks/README_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5 v2ProPlus | Audible TTFA | 75.70 ms | P50 | Windows; RTX 5070 Ti; keep-alive PCM; 10 sessions | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5 v5Turbo | Audible TTFA | 80.93 ms | P50 | Windows; RTX 5070 Ti; keep-alive PCM; 10 sessions | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| Qwen3-TTS-12Hz-0.6B | First-packet latency | 97 ms | Concurrency 1 | Single accelerator; 320 ms speech packet | [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) |
| Fish Audio S2 | TTFA | About 100 ms | Project-published value | H200; single GPU | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| Chatterbox-Flash (D=32, α=0.75) | TTFP | 103 ms | Concurrency 1; 50 utterances | H100 | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| Chatterbox-Flash (default D=16, α=0.5) | TTFP | 118 ms | Concurrency 1; 50 utterances | H100 | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| CosyVoice2 | First chunk | 196.13 ms | P50 | L20; concurrency 1; client-server | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |

IndexTTS 2.0/2.5 and VoxCPM2 do not publish first-audio latency under a comparable protocol.

| System | RTF | Runtime / model | Test conditions | Source |
|---|---:|---|---|---|
| AnifLive-TTS v1.5 v5Turbo | 0.04728 | TensorRT 11 / Triton FP16 | Windows; RTX 5070 Ti; 10 sessions | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| Chatterbox-Flash (D=32, α=0.75) | 0.076 | Block diffusion | H100; concurrency 1; 50 utterances | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| AnifLive-TTS v1.3 (historical) | 0.087680 | TensorRT 11 FP16 | RTX 5070 Ti; 10 Roxy sessions | [Historical local measurement](../benchmarks/README_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5 v2ProPlus | 0.09483 | TensorRT 11 FP16 | Windows; RTX 5070 Ti; 10 sessions | [v1.5 local measurement](../benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| Chatterbox-Flash (default D=16, α=0.5) | 0.107 | Block diffusion | H100; concurrency 1; 50 utterances | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| CosyVoice3 | 0.1091 | TRT-LLM; offline batch 1 | L20 | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice3.md#benchmark-with-offline-inference-mode) |
| CosyVoice2 | 0.1228 | TRT-LLM | L20; concurrency 1; client-server | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |
| VoxCPM2 | About 0.13 | Nano-vLLM / vLLM-Omni | RTX 4090 | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM#-highlights) |
| Fish Audio S2 | 0.195 | SGLang-based inference engine | H200; single GPU | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| IndexTTS 2.5 | 0.2065 | 2.5 BF16; KV cache | RTX 4090; overall | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |
| Qwen3-TTS-12Hz-0.6B | 0.288 | vLLM V0; concurrency 1 | Single accelerator; CUDA Graph | [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) |
| IndexTTS 2.0 | 0.3257 | 2.0 FP16; KV cache | RTX 4090; overall | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |

## Data and reproduction

- [Pooled AnifLive-TTS and local gsv.cpp values](../benchmarks/V1_5_BENCHMARK_SUMMARY.json).
- [Windows/gsv.cpp request records](../benchmarks/v1.5/MEASUREMENTS.csv) and [record manifest](../benchmarks/v1.5/MEASUREMENTS_MANIFEST.json).
- [Mac pooled and per-session records](../benchmarks/v1.5/MACOS_MATCHED_SUMMARY.json).
- [HTTP measurement client](../scripts/benchmark_release.py), with full WAV, new connection and keep-alive modes.
- [Historical v1.3 data](../benchmarks/README_BENCHMARK_SUMMARY.json) and [historical table generator](../scripts/benchmark_readme.py).

The formal table records the measured release-source build. Later functional-change screens and earlier frozen snapshots retain their own source/counts in the machine-readable summary. The Mac record contains aggregates and per-round statistics; Windows/gsv.cpp CSV contains request-level rows.

## Earlier frozen v5Turbo result

| Condition | Audible TTFA P50 / P95 | RTF P50 |
|---|---:|---:|
| v5Turbo, PCM keep-alive | 80.31 / 84.88 ms | 0.04655 |
| v5Turbo, PCM new connection | 81.74 / 86.38 ms | 0.04503 |

## Historical quality and engineering evidence

The following v2ProPlus results belong to the v1.3 workload and its v1.2 reference. Current backend/host evidence is recorded in the [platform matrix](V1_5_PLATFORM_MATRIX.md) and [Mac functional record](../benchmarks/v1.5/MACOS_FINAL_FUNCTIONAL.json).

The v2ProPlus quality metrics in this section reuse its previously accepted historical qualification; they were not re-run for this performance campaign. They do not establish v5Turbo or macOS audio-quality qualification.

> [!NOTE]
> **Quality gate**　Each controlled-expression stream is compared with its complete-WAV output at the same seed and settings. Neutral complete WAV, streaming PCM, and semantic outputs are also checked against the immutable v1.2 baseline. These objective checks do not replace subjective listening.

| Voice package | Japanese expression rows | Minimum log-mel | Minimum speaker cosine | Maximum duration difference | Underruns |
|---|---:|---:|---:|---:|---:|
| Miku v2ProPlus | 18/18 | 0.999454 | 0.989519 | 0.302% | 0 |
| Roxy v2ProPlus | 18/18 | 0.994325 | 0.987721 | 0.300% | 0 |


The post-policy five-language long/short matrix passed 20/20 rows per model. Its controlled-expression minima were `0.999660` log-mel and `0.988638` speaker cosine for Miku, and `0.998827` and `0.993495` for Roxy. Neutral complete WAV, streaming PCM, complete semantics, and streaming semantics remain exactly equal to v1.2 for both packages across `zh`, `yue`, `en`, `ja`, and `ko`.

Six independent short, long, and mixed-expression blind pairs produced five no-difference decisions, one preference for the Roxy complete output, and no reported artifact. The hard gates remain log-mel cosine `>=0.99`, speaker cosine `>=0.98`, and duration difference `<=3%`.

See [Performance Engineering](PERFORMANCE_ENGINEERING.md) for the v1.3/session ranges, older HTTP results, optimizations and rejected experiments. The [v1.2](research/v1.2-semantic-experiments.md) and [v1.3](research/v1.3-latency-experiments.md) research records retain their original version scope.
