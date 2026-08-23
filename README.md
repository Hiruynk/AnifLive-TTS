<div align="center">

<img src="assets/everynight_dance.gif" alt="Evernight dance" width="260">

# AnifLive-TTS

**A low-latency, high-quality multilingual voice-cloning TTS runtime with first-class Cantonese support**

[![Release](https://img.shields.io/badge/release-v1.0.0-2563eb?style=flat-square)](RELEASE_NOTES_v1.0.0.md)
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

- All eight neural stages use TensorRT 11 `execute_async_v3()`.
- Putonghua/Mandarin (`zh`), Cantonese (`yue`), English (`en`), Japanese (`ja`), and Korean (`ko`), plus GPT-SoVITS flat and OpenAI-compatible adapters.
- Supports GPT-SoVITS V2ProPlus voice-cloning models, packaging custom GPT/SoVITS checkpoints and reference audio as independent voice profiles.
- Complete mono PCM16 WAV and low-latency PCM16 streaming share one runtime.
- One-command V2ProPlus inspection, ONNX export, FP16 conversion, engine build, enqueue validation, and package publication in v1.
- Models and engines are prepared before service startup, keeping the `serve` hot path fully offline.

## Performance Benchmarks

> [!NOTE]
> **Environment:** RTX 5070 Ti 16 GB / driver 596.36 / CUDA runtime 12.8 /
> PyTorch 2.7.0+cu128 / TensorRT 11.2.1.2 / FP16

The workload fixes an external V2ProPlus test asset, a short text, seed, and sampling parameters. Each session performs 10 warmups, 100 complete-WAV requests, and 100 streaming requests. P50/P95 values are first computed independently within each 100-request session; the headline is the median across the ten session-level statistics. The range shows session-to-session variation.

Formal requests run at concurrency 1, with a new local HTTP/1.1 connection per request. TTFA is measured from the client sending the request until it reads the first PCM audio chunk.

| Metric | Median across 10 session-level statistics | Session range |
|---|---:|---:|
| Complete REST WAV wall P50 | **252.862 ms** | 249.456–263.883 ms |
| Complete REST WAV wall P95 | **311.778 ms** | 287.871–346.559 ms |
| Server inference P50 | **230.533 ms** | 227.369–244.072 ms |
| RTF P50 | **0.085426** | 0.084276–0.089150 |
| Streaming TTFA P50 | **78.964 ms** | 77.106–81.385 ms |
| Streaming TTFA P95 | **104.440 ms** | 90.404–127.497 ms |
| GPU busy-time P50 | **47.0%** | 46–48% |
| GPU busy-time P95 | **51.5%** | 51–53% |
| VRAM P50 | **4,554 MiB** | 4,543–4,776 MiB |

The workload produced 74 semantic tokens and 2.96 seconds of 32 kHz mono PCM16 audio. Every session reported `TensorRT-11` and `X-PyTorch-Fallback: false`. A separate 100-request payload audit found that the first HTTP audio chunk represents eight semantic tokens, while low-latency preview decoding uses another eight tokens as lookahead context. The first payload measured 4,093 samples / 127.906 ms at P50, with a 3,617–4,093 sample / 113.031–127.906 ms range. See the machine-readable [`benchmarks/README_BENCHMARK_SUMMARY.json`](benchmarks/README_BENCHMARK_SUMMARY.json).

`nvidia-smi` reports interval busy-time, not SM occupancy. The serial GPT AR dependency chain is the bottleneck at 218.319 ms session-median P50; SoVITS is only 7.416 ms.

### Reproduce the benchmark table

[`scripts/benchmark_readme.py`](scripts/benchmark_readme.py) is the canonical public benchmark. Its Markdown output contains exactly the same nine metrics as the table above, using the same default workload and aggregation method.

Run it against an existing local API:

```powershell
.\.venv\Scripts\python.exe .\scripts\benchmark_readme.py `
  --host 127.0.0.1 --port 9882 --locale en `
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

Defaults are 10 sessions, 10 warm-up requests per session, 100 complete-WAV requests and 100 streaming requests. Every formal request opens a new HTTP/1.1 connection at concurrency 1. Use `--sessions`, `--warmup` and `--runs` only for shorter diagnostics; results produced with non-default counts are not directly comparable with the published table.

### GPT-SoVITS Performance Comparison

#### RTF (lower is faster)

| Repository / system | RTF | Backend | Test conditions | Source |
|---|---:|---|---|---|
| GPT-SoVITS V2ProPlus | 0.014 | PyTorch parallel inference | RTX 4090; about four minutes of output | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| GPT-SoVITS V2ProPlus | 0.028 | PyTorch parallel inference | RTX 4060 Ti | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| **AnifLive-TTS v1** | **0.085426** | **TensorRT 11 FP16** | **RTX 5070 Ti; warm P50** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS C++ TRT | 0.1020 | TensorRT | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT fitted | 0.2096 | TensorRT fitted | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

#### First-output latency (lower is faster)

| Repository / system | Metric | Latency | Test conditions | Source |
|---|---|---:|---|---|
| **AnifLive-TTS v1** | **TTFA P50** | **78.964 ms** 🚀 | **RTX 5070 Ti; warm; 113.031–127.906 ms PCM payload** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS C++ TRT Stream | First packet | 460 ms | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference ONNX Stream | First token | 1,000 ms | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT fitted | First token | 2,022 ms | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

### Public Performance Data from Other Open-Source TTS Systems

This is not a controlled benchmark. Except for AnifLive-TTS, every value is reported by its cited source. GPU, model capabilities, input, first-packet size, concurrency, and measurement protocol differ. The tables index public values for matching metrics and do not represent a same-condition ranking.

#### RTF (lower is faster)

| System | RTF | Runtime / model | Test conditions | Source |
|---|---:|---|---|---|
| Chatterbox-Flash (D=32, α=0.75) | 0.076 | Block diffusion | H100; concurrency 1; 50 utterances | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| **AnifLive-TTS v1** | **0.085426** | **TensorRT 11 FP16** | **RTX 5070 Ti; warm P50** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| Chatterbox-Flash (default D=16, α=0.5) | 0.107 | Block diffusion | H100; concurrency 1; 50 utterances | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| CosyVoice3 | 0.1091 | TRT-LLM; offline batch 1 | L20 | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice3.md#benchmark-with-offline-inference-mode) |
| CosyVoice2 | 0.1228 | TRT-LLM | L20; concurrency 1; client-server | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |
| VoxCPM2 | About 0.13 | Nano-vLLM / vLLM-Omni | RTX 4090 | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM#-highlights) |
| Fish Audio S2 | 0.195 | SGLang-based inference engine | H200; single GPU | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| IndexTTS 2.5 | 0.2065 | 2.5 BF16; KV cache | RTX 4090; overall | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |
| Qwen3-TTS-12Hz-0.6B | 0.288 | vLLM V0; concurrency 1 | Single accelerator; CUDA Graph | [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) |
| IndexTTS 2.0 | 0.3257 | 2.0 FP16; KV cache | RTX 4090; overall | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |

#### First-audio latency (lower is faster)

| System | Metric | Latency | Statistic | Test conditions | Source |
|---|---|---:|---|---|---|
| **AnifLive-TTS v1** | **TTFA** | **78.964 ms** 🚀 | **P50** | **RTX 5070 Ti; warm; 113.031–127.906 ms PCM payload** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| Qwen3-TTS-12Hz-0.6B | First-packet latency | 97 ms | Concurrency 1 | Single accelerator; 320 ms speech packet | [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) |
| Fish Audio S2 | TTFA | About 100 ms | Project-published value | H200; single GPU | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| Chatterbox-Flash (D=32, α=0.75) | TTFP | 103 ms | Concurrency 1; 50 utterances | H100 | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| Chatterbox-Flash (default D=16, α=0.5) | TTFP | 118 ms | Concurrency 1; 50 utterances | H100 | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| CosyVoice2 | First chunk | 196.13 ms | P50 | L20; concurrency 1; client-server | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |

IndexTTS 2.0/2.5 and VoxCPM2 do not publish first-audio latency under a comparable protocol.

AnifLive-TTS pre-packages voice profiles and reference conditioning for persistent local voice-cloning services. Users can rebuild TensorRT engines on the target GPU and verify quality through reproducible tests.

## Deterministic Quality Preservation

> [!NOTE]
> **Quality gate**　The results below prove complete-output equality against a fixed deterministic regression reference. They are not a subjective MOS evaluation; every performance change must pass this gate first.

| Check | Measured | Gate | Result |
|---|---:|---:|---:|
| Complete WAV SHA-256 | Identical | Identical | Pass |
| Waveform correlation | 1.000000 | >=0.999 | Pass |
| Log-mel cosine | 1.000000 | >=0.99 | Pass |
| Log-mel MAE | 0.000000 | Lower is better | Pass |
| Speaker cosine | 1.000000 | >=0.98 | Pass |
| SI-SDR | 120 dB (cap) | >=30 dB | Pass |
| Duration difference | 0.000% | <=3% | Pass |

Under the fixed deterministic regression reference, the complete output remains
identical before and after TensorRT optimization. Performance optimizations do
not change generation steps or sampling settings.

## Optimizations And Tested Boundary

- All eight neural stages execute through TensorRT 11 `execute_async_v3()`.
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
converter, and GPT AR scheduling. CUDA/TensorRT owns all eight model stages,
GPU sampling, and persistent buffers. One process preloads one active model.

- Languages: Putonghua/Mandarin (`zh`), Cantonese (`yue`), English (`en`), Japanese (`ja`), and Korean (`ko`); legacy adapters also accept `auto` and `auto_yue`.
- Canonical endpoint: `POST /v1/audio/speech`.
- Discovery: `GET /health`, `/v1/capabilities`, `/v1/models`, `/v1/voices`.
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

For the CUDA 12.1 profile, replace `requirements\torch-cu128.txt` with
`requirements\torch-cu121.txt`. Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
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
| `GET /v1/voices` | Startup-cached voice profiles |

## Roadmap

**v1**

- [x] V2ProPlus model conversion and eight-stage TensorRT 11 inference
- [x] Five-language API, complete WAV, and low-latency PCM streaming
- [x] Docker release, deterministic quality gates, and an offline hot path

**Next: Neural Emotion Adapter**

- [ ] Implement controllable emotion, intensity, and style with timbre-preservation and latency acceptance

**Later: More GPT-SoVITS generations**

- [ ] Support V2 / V2Pro, V3, and V4 under the same API and model-package contract

## Compatibility

> [!WARNING]
> **Before deployment**　`cu128` is the configuration with complete local GPU E2E validation; `cu121` has build/load validation only and is not claimed for GPU E2E on the local RTX 5070 Ti.

v1 currently supports V2ProPlus; other GPT-SoVITS generations remain on the roadmap. Use `cu128` for RTX 50-series/Blackwell GPUs. `cu121` has not passed end-to-end GPU validation on the RTX 5070 Ti; see the [deployment guide](docs/DEPLOYMENT_EN.md).

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

Original AnifLive-TTS code is licensed under [PolyForm Noncommercial 1.0.0](LICENSE); commercial use requires a separate written Commercial License from AnifEngine. GPT-SoVITS-derived portions retain MIT, Minimal Inference-derived and applicable GPT-SoVITS C++ reference portions retain Apache-2.0, and dependencies retain their own terms. See [LICENSING.md](LICENSING.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and the cu121/cu128 image-derived SPDX SBOMs attached to the v1.0.0 release.
