<div align="center">

<img src="assets/everynight_dance.gif" alt="Evernight dance" width="260">

# AnifLive-TTS

**A low-latency, high-quality multilingual voice-cloning TTS runtime with first-class Cantonese support**

[![Release](https://img.shields.io/badge/release-v1.1.0-2563eb?style=flat-square)](RELEASE_NOTES_v1.1.0.md)
[![TensorRT](https://img.shields.io/badge/TensorRT-11.2.1.2-76b900?style=flat-square&logo=nvidia)](https://docs.nvidia.com/deeplearning/tensorrt/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Model](https://img.shields.io/badge/GPT--SoVITS-V2ProPlus-0f766e?style=flat-square)](https://github.com/RVC-Boss/GPT-SoVITS)
[![License](https://img.shields.io/badge/license-PolyForm_Noncommercial_1.0.0-22c55e?style=flat-square)](LICENSING.md)

[繁體中文](README_ZH_HK.md) · **English** · [简体中文](README_ZH_CN.md)

</div>

## About the Project

AnifLive-TTS is the first-party TTS for AnifEngine-Voice, born from a practical need: while developing AnifEngine-Voice, I could not find a TTS solution that combined **Cantonese**, **low latency**, **high audio quality**, and self-hosted control. 😮‍💨

I therefore chose GPT-SoVITS as the foundation, reworked its inference internals, and made **low latency** and **high audio quality** the core goals of AnifLive-TTS. 🤓👆

v1 launches with complete V2ProPlus support; future releases will extend the same API and model-package contract to more GPT-SoVITS model generations.

## Highlights

- All nine neural stages use TensorRT 11 `execute_async_v3()`.
- Putonghua/Mandarin (`zh`), Cantonese (`yue`), English (`en`), Japanese (`ja`), and Korean (`ko`), plus GPT-SoVITS flat and OpenAI-compatible adapters.
- Supports GPT-SoVITS V2ProPlus voice-cloning models, packaging custom GPT/SoVITS checkpoints and reference audio as independent voice profiles.
- Complete mono PCM16 WAV and low-latency PCM16 streaming share one runtime.
- One-command V2ProPlus inspection, ONNX export, FP16 conversion, engine build, enqueue validation, and package publication in v1.
- Models and engines are prepared before service startup, keeping the `serve` hot path fully offline.

## Roxy Migurdia from *Mushoku Tensei: Jobless Reincarnation*: V2ProPlus Cantonese Demo

<div align="center">

<video src="assets/roxy-v2proplus-cantonese-demo.mp4" controls width="960"></video>

<p>The WebUI is still in testing and is not available in this release. Stay tuned for a future release.</p>

</div>

## Performance Benchmarks

> [!NOTE]
> **Environment:** RTX 5070 Ti 16 GB / driver 596.36 / CUDA runtime 12.8 /
> PyTorch 2.7.0+cu128 / TensorRT 11.2.1.2 / FP16

The workload uses the external Roxy V2ProPlus voice package with a fixed short text, seed, and sampling parameters. It runs 10 sessions; every session performs 10 warmups, 100 complete-WAV requests, 100 streaming requests over new connections, and 100 streaming requests over one persistent connection. The headline is the median across the 10 session-level statistics. The range shows session-to-session variation.

Formal requests run at concurrency 1. New-connection rows open a local HTTP/1.1 connection per request; keep-alive rows reuse one separately warmed connection per session. First-packet latency ends when the client reads the first server-emitted PCM chunk. Audible TTFA uses the first PCM sample above -45 dBFS within the earliest active 10 ms RMS frame and is constrained by that chunk's arrival time. Device output latency is not included.

| Metric | Median across 10 session-level statistics | Session range |
|---|---:|---:|
| Complete REST WAV wall P50 | **224.868 ms** | 223.440–233.007 ms |
| Complete REST WAV wall P95 | **270.228 ms** | 254.869–298.801 ms |
| Server inference P50 | **204.186 ms** | 199.756–208.611 ms |
| RTF P50 | **0.128717** | 0.127900–0.133376 |
| Streaming first-packet latency P50 | **95.096 ms** | 92.529–101.434 ms |
| Streaming first-packet latency P95 | **121.429 ms** | 110.241–143.452 ms |
| Keep-alive streaming first-packet latency P50 | **82.671 ms** | 81.487–84.616 ms |
| Keep-alive streaming first-packet latency P95 | **116.565 ms** | 100.100–132.897 ms |
| Audible streaming TTFA P50 | **101.721 ms** | 99.154–108.059 ms |
| Audible streaming TTFA P95 | **128.054 ms** | 116.866–150.077 ms |
| Keep-alive audible streaming TTFA P50 | **89.296 ms** | 88.112–91.241 ms |
| Keep-alive audible streaming TTFA P95 | **123.190 ms** | 106.725–139.522 ms |
| GPU busy-time P50 | **44.0%** | 43–45% |
| GPU busy-time P95 | **48.0%** | 48–49% |

All 1,000 complete-WAV, 1,000 new-connection streaming, and 1,000 keep-alive streaming requests reported `TensorRT-11` with `X-PyTorch-Fallback: false`. See the machine-readable [`benchmarks/README_BENCHMARK_SUMMARY.json`](benchmarks/README_BENCHMARK_SUMMARY.json).

`nvidia-smi` reports interval GPU busy-time rather than SM occupancy. The serialized GPT autoregressive chain remains the primary reason utilization does not approach 100% at concurrency 1.

### Reproduce the benchmark table

[`scripts/benchmark_readme.py`](scripts/benchmark_readme.py) is the canonical public benchmark. Its Markdown output contains exactly the same 14 metrics as the table above, using the same default workload and aggregation method.

Run it against an existing local API:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_readme.py `
  --host 127.0.0.1 --port 9881 --locale en `
  --model roxy-v2proplus `
  --report .\reports\benchmark.json `
  --markdown .\reports\benchmark.md
```

Run it inside the existing Docker container without rebuilding or recreating it:

```powershell
docker exec aniflive-tts /app/scripts/entrypoint.sh benchmark `
  --host 127.0.0.1 --port 9880 --locale en `
  --report /data/reports/benchmark.json `
  --markdown /data/reports/benchmark.md
```

Defaults are 10 sessions per model, 10 warm-up requests per session, 100 complete-WAV requests, 100 new-connection streaming requests, and 100 keep-alive streaming requests. Repeat `--model` to aggregate multiple voice packages. Every workload runs at concurrency 1.

### GPT-SoVITS Performance Comparison

#### RTF (lower is faster)

| Repository / system | RTF | Backend | Test conditions | Source |
|---|---:|---|---|---|
| GPT-SoVITS V2ProPlus | 0.014 | PyTorch parallel inference | RTX 4090; about four minutes of output | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| GPT-SoVITS V2ProPlus | 0.028 | PyTorch parallel inference | RTX 4060 Ti | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| GPT-SoVITS C++ TRT | 0.1020 | TensorRT | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| **AnifLive-TTS v1.1** | **0.128717** | **TensorRT 11 FP16** | **RTX 5070 Ti; 10 Roxy sessions** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS Minimal Inference TRT fitted | 0.2096 | TensorRT fitted | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

#### First-output latency (lower is faster)

| Repository / system | Metric | Latency | Test conditions | Source |
|---|---|---:|---|---|
| **AnifLive-TTS v1.1** | **Audible TTFA P50** | **89.296 ms** 🚀 | **RTX 5070 Ti; persistent HTTP/1.1; 10 Roxy sessions** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS C++ TRT Stream | First packet | 460 ms | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference ONNX Stream | First token | 1,000 ms | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT fitted | First token | 2,022 ms | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

### Public Performance Data from Other Open-Source TTS Systems

This is not a controlled benchmark. Except for AnifLive-TTS, every value is reported by its cited source. GPU, model capabilities, input, first-packet size, concurrency, and measurement protocol differ. The tables index public values for matching metrics and do not represent a same-condition ranking.

#### RTF (lower is faster)

| System | RTF | Runtime / model | Test conditions | Source |
|---|---:|---|---|---|
| Chatterbox-Flash (D=32, α=0.75) | 0.076 | Block diffusion | H100; concurrency 1; 50 utterances | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| Chatterbox-Flash (default D=16, α=0.5) | 0.107 | Block diffusion | H100; concurrency 1; 50 utterances | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| CosyVoice3 | 0.1091 | TRT-LLM; offline batch 1 | L20 | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice3.md#benchmark-with-offline-inference-mode) |
| CosyVoice2 | 0.1228 | TRT-LLM | L20; concurrency 1; client-server | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |
| **AnifLive-TTS v1.1** | **0.128717** | **TensorRT 11 FP16** | **RTX 5070 Ti; 10 Roxy sessions** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| VoxCPM2 | About 0.13 | Nano-vLLM / vLLM-Omni | RTX 4090 | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM#-highlights) |
| Fish Audio S2 | 0.195 | SGLang-based inference engine | H200; single GPU | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| IndexTTS 2.5 | 0.2065 | 2.5 BF16; KV cache | RTX 4090; overall | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |
| Qwen3-TTS-12Hz-0.6B | 0.288 | vLLM V0; concurrency 1 | Single accelerator; CUDA Graph | [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) |
| IndexTTS 2.0 | 0.3257 | 2.0 FP16; KV cache | RTX 4090; overall | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |

#### First-audio latency (lower is faster)

| System | Metric | Latency | Statistic | Test conditions | Source |
|---|---|---:|---|---|---|
| **AnifLive-TTS v1.1** | **Audible TTFA** | **89.296 ms** 🚀 | **P50** | **RTX 5070 Ti; persistent HTTP/1.1; 10 Roxy sessions** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| Qwen3-TTS-12Hz-0.6B | First-packet latency | 97 ms | Concurrency 1 | Single accelerator; 320 ms speech packet | [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) |
| Fish Audio S2 | TTFA | About 100 ms | Project-published value | H200; single GPU | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| Chatterbox-Flash (D=32, α=0.75) | TTFP | 103 ms | Concurrency 1; 50 utterances | H100 | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| Chatterbox-Flash (default D=16, α=0.5) | TTFP | 118 ms | Concurrency 1; 50 utterances | H100 | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| CosyVoice2 | First chunk | 196.13 ms | P50 | L20; concurrency 1; client-server | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |

IndexTTS 2.0/2.5 and VoxCPM2 do not publish first-audio latency under a comparable protocol.

AnifLive-TTS pre-packages voice profiles and reference conditioning for persistent local voice-cloning services. Users can rebuild TensorRT engines on the target GPU and verify quality through reproducible tests.

## Deterministic Quality Preservation

> [!NOTE]
> **Quality gate**　Each final streaming path is compared with the complete-WAV path at the same seed and sampling settings. These objective regression checks do not replace a subjective MOS evaluation.

| Voice package | Waveform correlation | Log-mel cosine | Speaker cosine | SI-SDR | Duration difference | Result |
|---|---:|---:|---:|---:|---:|---:|
| Miku V2ProPlus | 0.775697 | 0.995100 | 0.987893 | 1.792 dB | 0.000% | Pass |
| Roxy V2ProPlus | 0.995861 | 0.993346 | 0.983652 | 20.794 dB | 0.629% | Pass |

The hard gates are log-mel cosine `>=0.99`, speaker cosine `>=0.98`, and duration difference `<=3%`.

Under the fixed deterministic regression reference, the complete output remains
identical before and after TensorRT optimization. Performance optimizations do
not change generation steps or sampling settings.

## Optimizations And Tested Boundary

- All nine neural stages execute through TensorRT 11 `execute_async_v3()`.
- GPT KV ping-pong buffers, bindings, indices, and reference conditioning persist.
- A sampling CUDA Graph captures softmax, multinomial, and gather while preserving RNG semantics.
- Single-segment requests batch EOS checks every two steps.
- Startup warmup and HTTP/1.1 keep-alive avoid request-time setup.

Full GPT-step CUDA Graph capture is currently limited by TensorRT capture
compatibility. See the [performance engineering record](docs/PERFORMANCE_ENGINEERING.md).

## Architecture And API

AnifLive-TTS is the first-party FP16 TensorRT 11 speech inference platform for
AnifEngine-Voice. Its first validated v1 acoustic backend is `gsv-v2proplus`,
with the same contract reserved for future GPT-SoVITS model generations. Python owns the API, five-language frontend, model packages,
converter, and GPT AR scheduling. CUDA/TensorRT owns all nine model stages,
GPU sampling, and persistent buffers. One process preloads one active model.

- Languages: Putonghua/Mandarin (`zh`), Cantonese (`yue`), English (`en`), Japanese (`ja`), and Korean (`ko`); legacy adapters also accept `auto` and `auto_yue`.
- Canonical endpoint: `POST /v1/audio/speech`.
- Discovery: `GET /health`, `/v1/capabilities`, `/v1/models`, `/v1/voices`.
- Model selection: `POST /v1/models/activate` unloads the active package before loading one compatible local replacement.
- `stream=false` returns mono PCM16 WAV; `stream=true` returns PCM16 chunks.

## Quick Start

Users with an existing GPT-SoVITS voice-cloning model can convert its GPT/SoVITS checkpoints and reference audio directly into an AnifLive-TTS model package.

### 1. Install the local toolchain

Create the project environment before installing anything. The API never installs
dependencies or downloads shared assets at startup.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements\torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\base.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\tensorrt11-cu12.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\setup_shared_assets.py `
  --output D:\models\shared --accept-third-party-licenses
```

For the CUDA 12.6 profile, replace `requirements\torch-cu128.txt` with
`requirements\torch-cu126.txt`. Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
before accepting the shared-asset licenses.

### 2. Convert a model

```powershell
.\.venv\Scripts\aniflive-tts.exe model convert `
  --gpt D:\models\voice.ckpt --sovits D:\models\voice.pth `
  --reference-audio D:\models\reference.wav `
  --reference-text-file D:\models\reference.txt --reference-language ja `
  --model-id my-v2proplus --voice-profile default `
  --shared-dir D:\models\shared --output D:\models\my-v2proplus
```

The converter inspects checkpoint headers and tensor shapes, exports ONNX,
converts FP16, applies TensorRT 11 graph patches, builds fitted engines,
validates deserialize/enqueue, and atomically publishes the package.

> [!IMPORTANT]
> Safe mode uses `torch.load(weights_only=True)`. Use `--allow-unsafe-pickle` only with a trusted local checkpoint.

Engine fingerprints include TensorRT,
CUDA runtime, compute capability, ONNX, profiles, and build configuration.

### 3. Validate real engine enqueue

> [!TIP]
> Validate every newly converted model package with `--enqueue` before starting the API.

```powershell
.\.venv\Scripts\aniflive-tts.exe validate `
  --model-package D:\models\my-v2proplus `
  --shared-dir D:\models\shared `
  --enqueue
```

### 4. Start the Docker API

```powershell
Copy-Item .env.example .env
.\scripts\run_docker.ps1 -CudaProfile cu128 -Build
```

Use `-Build` for the first image build; later starts reuse that image. Model and
cache directories persist through host bind mounts. See the [deployment guide](docs/DEPLOYMENT_EN.md).

The default Compose configuration exposes the API on loopback only. AnifLive-TTS
does not provide public-edge authentication; place any external deployment behind
an authenticated reverse proxy and appropriate request limits.

## API

```powershell
curl.exe -X POST "http://127.0.0.1:9882/v1/audio/speech" `
  -H "Content-Type: application/json" `
  --output output.wav `
  --data '{"model":"my-v2proplus","voice_profile":"default","text":"今日はいい天気ですね。","language":"ja","stream":false,"generation":{"top_k":15,"top_p":1.0,"temperature":1.0,"seed":1234}}'
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/speech` | Canonical TTS and OpenAI adapter |
| `GET/POST /` | GPT-SoVITS flat compatibility adapter |
| `GET /health` | Readiness, GPU, CUDA, TensorRT, and engine proof |
| `GET /v1/capabilities` | Languages, streaming, and expression capability |
| `GET /v1/models` | Active model |
| `POST /v1/models/activate` | Switch to a compatible local model package |
| `GET /v1/voices` | Startup-cached voice profiles |

## Roadmap

**v1**

- [x] V2ProPlus model conversion and nine-stage TensorRT 11 inference
- [x] Five-language API, complete WAV, and low-latency PCM streaming
- [x] Docker release, deterministic quality gates, and an offline hot path

**Next: Neural Emotion Adapter**

- [ ] Implement controllable emotion, intensity, and style with timbre-preservation and latency acceptance

**Later: More GPT-SoVITS generations**

- [ ] Support V2 / V2Pro, V3, and V4 under the same API and model-package contract

## Compatibility

> [!WARNING]
> **Before deployment**　`cu128` is the configuration with complete local GPU E2E validation. The `cu126` compatibility profile has source and build-policy validation only; GPU E2E support is not claimed until its image passes the release workflow and validation on a compatible host.

v1 currently supports V2ProPlus; other GPT-SoVITS generations remain on the roadmap. Use `cu128` for RTX 50-series/Blackwell GPUs. `cu126` has not passed end-to-end GPU validation on the RTX 5070 Ti; see the [deployment guide](docs/DEPLOYMENT_EN.md).

## Documentation And License

- [API contract](docs/API_EN.md)
- [Deployment guide](docs/DEPLOYMENT_EN.md)
- [Performance engineering record](docs/PERFORMANCE_ENGINEERING.md)
- [Acceptance report](docs/ACCEPTANCE.md)
- [Rollback guide](docs/ROLLBACK.md)
- [Licensing](LICENSING.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Third-party media notice](assets/THIRD_PARTY_MEDIA.md)

AnifLive-TTS is the first-party TTS for AnifEngine-Voice. Its current acoustic implementation builds on the research and engineering of [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS), [GPT-SoVITS Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference), and [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp). Special thanks to GPT-SoVITS creator **花儿不哭** and all other GPT-SoVITS contributors.

Original AnifLive-TTS code is licensed under [PolyForm Noncommercial 1.0.0](LICENSE); commercial use requires a separate written Commercial License from Hiruynk. GPT-SoVITS-derived portions retain MIT, Minimal Inference-derived and applicable GPT-SoVITS C++ reference portions retain Apache-2.0, and dependencies retain their own terms. See [LICENSING.md](LICENSING.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and the cu126/cu128 image-derived SPDX SBOMs attached to the v1.1.0 release.
