<div align="center">

<img src="assets/everynight_dance.gif" alt="Evernight dance" width="260">

# AnifLive-TTS

**A low-latency, high-quality multilingual voice-cloning TTS runtime with first-class Cantonese support**

[![Release](https://img.shields.io/badge/release-v1.3.0-2563eb?style=flat-square)](RELEASE_NOTES_v1.3.0.md)
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
- Package-curated expression profiles support global and per-segment delivery without changing the V2ProPlus neural architecture.
- A local five-language WebUI provides model switching, natural-language expression selection, streaming playback, and live latency metrics.
- One-command V2ProPlus inspection, ONNX export, FP16 conversion, engine build, enqueue validation, and package publication in v1.
- Models and engines are prepared before service startup, keeping the `serve` hot path fully offline.

## Roxy Migurdia from *Mushoku Tensei: Jobless Reincarnation*: V2ProPlus Cantonese Demo

https://github.com/user-attachments/assets/0be5a03d-94a9-4d33-b05f-6bae6fb3cc40

The local WebUI shown in the demo is included in v1.3 and starts with `run_webui.bat` after the API is ready.

## Performance Benchmarks

> [!NOTE]
> **Environment:** RTX 5070 Ti 16 GB / driver 596.36 / CUDA runtime 12.8 /
> PyTorch 2.7.0+cu128 / TensorRT 11.2.1.2 / FP16

The canonical v1.3 workload uses one external Roxy V2ProPlus voice package with fixed text, seed, and sampling parameters. It runs 10 sessions; every session performs 10 warmups, 100 complete-WAV requests, 100 streaming requests over new connections, and 100 streaming requests over one persistent connection. The headline is the median across the 10 session-level statistics. The range shows session variation. Miku is excluded from the performance headline because its known model-specific streaming behavior is being investigated separately.

Formal requests run at concurrency 1. New-connection rows open a local HTTP/1.1 connection per request; keep-alive rows reuse one separately warmed connection per session. First-packet latency ends when the client reads the first server-emitted PCM chunk. Audible TTFA uses the first PCM sample above -45 dBFS within the earliest active 10 ms RMS frame and is constrained by that chunk's arrival time. Device output latency is not included.

| Metric | Median across 10 session-level statistics | Session range |
|---|---:|---:|
| Complete REST WAV wall P50 | **153.177 ms** | 146.508–166.100 ms |
| Complete REST WAV wall P95 | **189.548 ms** | 158.981–201.439 ms |
| Server inference P50 | **148.591 ms** | 142.121–161.275 ms |
| RTF P50 | **0.087680** | 0.083863–0.095077 |
| Streaming first-packet latency P50 | **67.077 ms** | 64.866–74.544 ms |
| Streaming first-packet latency P95 | **90.853 ms** | 71.753–120.376 ms |
| Keep-alive streaming first-packet latency P50 | **67.953 ms** | 65.456–76.345 ms |
| Keep-alive streaming first-packet latency P95 | **92.019 ms** | 81.640–136.052 ms |
| Audible streaming TTFA P50 | **73.702 ms** | 71.491–81.169 ms |
| Audible streaming TTFA P95 | **97.478 ms** | 78.378–127.001 ms |
| Keep-alive audible streaming TTFA P50 | **74.578 ms** | 72.081–82.970 ms |
| Keep-alive audible streaming TTFA P95 | **98.644 ms** | 88.265–142.677 ms |
| GPU busy-time P50 | **53.0%** | 50.5–54% |
| GPU busy-time P95 | **56.0%** | 56–57% |

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
  --model roxy-v2proplus `
  --report /data/reports/benchmark.json `
  --markdown /data/reports/benchmark.md
```

Defaults are 10 sessions per model, 10 warm-up requests per session, 100 complete-WAV requests, 100 new-connection streaming requests, and 100 keep-alive streaming requests. The release table uses only `roxy-v2proplus`; every workload runs at concurrency 1.

### GPT-SoVITS Performance Comparison

#### First-output latency (lower is faster)

| Repository / system | Metric | Latency | Test conditions | Source |
|---|---|---:|---|---|
| **AnifLive-TTS v1.3** | **Audible TTFA P50** | **74.578 ms** 🚀 | **RTX 5070 Ti; persistent HTTP/1.1; 10 Roxy sessions** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS C++ TRT Stream | First packet | 460 ms | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference ONNX Stream | First token | 1,000 ms | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT fitted | First token | 2,022 ms | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

#### RTF (lower is faster)

| Repository / system | RTF | Backend | Test conditions | Source |
|---|---:|---|---|---|
| GPT-SoVITS V2ProPlus | 0.014 | PyTorch parallel inference | RTX 4090; about four minutes of output | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| GPT-SoVITS V2ProPlus | 0.028 | PyTorch parallel inference | RTX 4060 Ti | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS#features) |
| **AnifLive-TTS v1.3** | **0.087680** | **TensorRT 11 FP16** | **RTX 5070 Ti; 10 Roxy sessions** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| GPT-SoVITS C++ TRT | 0.1020 | TensorRT | RTX 2080 Ti 22 GB | [GPT-SoVITS C++](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS-cpp#-performance-benchmarks) |
| GPT-SoVITS Minimal Inference TRT fitted | 0.2096 | TensorRT fitted | RTX 2080 Ti 22 GB; FP16 | [Minimal Inference](https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference#-performance-benchmarks) |

### Public Performance Data from Other Open-Source TTS Systems

This is not a controlled benchmark. Except for AnifLive-TTS, every value is reported by its cited source. GPU, model capabilities, input, first-packet size, concurrency, and measurement protocol differ. The tables index public values for matching metrics and do not represent a same-condition ranking.

#### First-audio latency (lower is faster)

| System | Metric | Latency | Statistic | Test conditions | Source |
|---|---|---:|---|---|---|
| **AnifLive-TTS v1.3** | **Audible TTFA** | **74.578 ms** 🚀 | **P50** | **RTX 5070 Ti; persistent HTTP/1.1; 10 Roxy sessions** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| Qwen3-TTS-12Hz-0.6B | First-packet latency | 97 ms | Concurrency 1 | Single accelerator; 320 ms speech packet | [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) |
| Fish Audio S2 | TTFA | About 100 ms | Project-published value | H200; single GPU | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| Chatterbox-Flash (D=32, α=0.75) | TTFP | 103 ms | Concurrency 1; 50 utterances | H100 | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| Chatterbox-Flash (default D=16, α=0.5) | TTFP | 118 ms | Concurrency 1; 50 utterances | H100 | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| CosyVoice2 | First chunk | 196.13 ms | P50 | L20; concurrency 1; client-server | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |

IndexTTS 2.0/2.5 and VoxCPM2 do not publish first-audio latency under a comparable protocol.

#### RTF (lower is faster)

| System | RTF | Runtime / model | Test conditions | Source |
|---|---:|---|---|---|
| Chatterbox-Flash (D=32, α=0.75) | 0.076 | Block diffusion | H100; concurrency 1; 50 utterances | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| **AnifLive-TTS v1.3** | **0.087680** | **TensorRT 11 FP16** | **RTX 5070 Ti; 10 Roxy sessions** | **[Local measurement](benchmarks/README_BENCHMARK_SUMMARY.json)** |
| Chatterbox-Flash (default D=16, α=0.5) | 0.107 | Block diffusion | H100; concurrency 1; 50 utterances | [Chatterbox-Flash paper](https://arxiv.org/abs/2605.30748) |
| CosyVoice3 | 0.1091 | TRT-LLM; offline batch 1 | L20 | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice3.md#benchmark-with-offline-inference-mode) |
| CosyVoice2 | 0.1228 | TRT-LLM | L20; concurrency 1; client-server | [QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice/blob/main/runtime/triton_trtllm/README.Cosyvoice2.DiT.md#benchmark-with-client-server-mode) |
| VoxCPM2 | About 0.13 | Nano-vLLM / vLLM-Omni | RTX 4090 | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM#-highlights) |
| Fish Audio S2 | 0.195 | SGLang-based inference engine | H200; single GPU | [Fish Audio S2](https://github.com/fishaudio/fish-speech#performance) |
| IndexTTS 2.5 | 0.2065 | 2.5 BF16; KV cache | RTX 4090; overall | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |
| Qwen3-TTS-12Hz-0.6B | 0.288 | vLLM V0; concurrency 1 | Single accelerator; CUDA Graph | [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621) |
| IndexTTS 2.0 | 0.3257 | 2.0 FP16; KV cache | RTX 4090; overall | [index-tts/index-tts](https://github.com/index-tts/index-tts#-inference-speed) |

AnifLive-TTS pre-packages voice profiles and reference conditioning for persistent local voice-cloning services. Users can rebuild TensorRT engines on the target GPU and verify quality through reproducible tests.

## Deterministic Quality Preservation

> [!NOTE]
> **Quality gate**　Each controlled-expression stream is compared with its complete-WAV output at the same seed and settings. Neutral complete WAV, streaming PCM, and semantic outputs are also checked against the immutable v1.2 baseline. These objective checks do not replace subjective listening.

| Voice package | Japanese expression rows | Minimum log-mel | Minimum speaker cosine | Maximum duration difference | Underruns |
|---|---:|---:|---:|---:|---:|
| Miku V2ProPlus | 18/18 | 0.999454 | 0.989519 | 0.302% | 0 |
| Roxy V2ProPlus | 18/18 | 0.994325 | 0.987721 | 0.300% | 0 |

These rows describe private local validation overlays only. Their expression
references, transcripts, and character media are not included in the public
source, image, or release bundle.

The post-policy five-language long/short matrix passed 20/20 rows per model. Its controlled-expression minima were `0.999660` log-mel and `0.988638` speaker cosine for Miku, and `0.998827` and `0.993495` for Roxy. Neutral complete WAV, streaming PCM, complete semantics, and streaming semantics remain exactly equal to v1.2 for both packages across `zh`, `yue`, `en`, `ja`, and `ko`.

Six independent short, long, and mixed-expression blind pairs produced five no-difference decisions, one preference for the Roxy complete output, and no reported artifact. The hard gates remain log-mel cosine `>=0.99`, speaker cosine `>=0.98`, and duration difference `<=3%`.

## Optimizations And Tested Boundary

- All nine neural stages execute through TensorRT 11 `execute_async_v3()`.
- Per-model fitted GPT engines use persistent TensorRT contexts, fixed KV buffers, and zero auxiliary streams.
- A sampling CUDA Graph captures softmax, multinomial, and gather while preserving RNG semantics.
- Only the first text segment uses the established 9+8 semantic-token preview; later segments keep the native full-context refill path.
- EOS checks remain batched every two steps, and the runtime retains its warm state for 25 seconds.
- Expression references are prepared at model activation and kept GPU-resident; neutral requests preserve the v1.2 output exactly.
- Expression selection and transition settings are package-driven, with no Miku/Roxy runtime branch.
- Startup warmup and HTTP/1.1 keep-alive avoid request-time setup.

Full GPT-step CUDA Graph capture is currently limited by TensorRT capture
compatibility. See the [performance engineering record](docs/PERFORMANCE_ENGINEERING.md).

## Architectures Evaluated for v1.2

| Candidate | Result | Decision |
|---|---|---|
| Transformer + TensorRT runtime optimization | Passed end-to-end latency and quality gates | Adopted |
| MTP-4 | Future-token accuracy did not meet the semantic quality gate | Not adopted |
| Mamba-2 hybrid | End-to-end benefit did not justify the quality and complexity tradeoff | Not adopted |
| Mamba-2 hybrid + MTP | Combined quality/performance gate was not met | Not adopted |

AnifLive-TTS does not adopt architectural changes solely for theoretical
efficiency. Experimental semantic backends are promoted only when they
outperform the production baseline without compromising speech quality. See
the [v1.2 semantic experiment record](docs/research/v1.2-semantic-experiments.md).

## Architectures Evaluated for v1.3

| Candidate | Result | Decision |
|---|---|---|
| Reference-directed expression control | Passed objective, compatibility, and blind-listening gates | Adopted |
| Hard-token acoustic lookahead with MTP future context | Cross-model quality and end-to-end latency gates were not met | Not adopted |
| Early-exit Transformer self-draft | Even one draft token made the first-17 path 50-61% slower because drafting cost exceeded the verification savings | Not adopted |
| Recurrent speculative decoding | Reduced target NFE, but drafter and verifier overhead limited wall-time benefit | Not adopted |
| SemanticPiece / EABPE | Sequence compression did not concentrate in the first 17 TTFA-critical tokens | Not adopted |
| Retrieval-based speculative decoding | The offline oracle found no candidate that met all first-17 coverage gates | Not adopted |
| FastStart distilled/pruned Transformer | Continuation and refill reliability gates were not met | Not adopted |
| D16 block diffusion | The evaluated low-rank adaptation did not pass the semantic-quality gate | Not adopted |

These decisions apply to the evaluated AnifLive-TTS V2ProPlus workload, not to every possible implementation of the candidate architectures. See the [v1.3 latency experiment record](docs/research/v1.3-latency-experiments.md) and [expression design record](docs/research/v1.3-reference-expression-design.md).

## Architecture And API

AnifLive-TTS is the first-party FP16 TensorRT 11 speech inference platform for
AnifEngine-Voice. Its first validated v1 acoustic backend is `gsv-v2proplus`,
with the same contract reserved for future GPT-SoVITS model generations. Python owns the API, five-language frontend, model packages,
converter, and GPT AR scheduling. CUDA/TensorRT owns all nine model stages,
GPU sampling, and persistent buffers. One process preloads one active model.

- Languages: Putonghua/Mandarin (`zh`), Cantonese (`yue`), English (`en`), Japanese (`ja`), and Korean (`ko`); legacy adapters also accept `auto` and `auto_yue`.
- Canonical endpoint: `POST /v1/audio/speech`.
- Discovery: `GET /health`, `/v1/capabilities`, `/v1/models`, `/v1/voices`.
- Expression discovery: `GET /v1/expressions`; global and per-segment expression requests use the canonical speech endpoint.
- Model selection: `POST /v1/models/activate` unloads the active package before loading one compatible local replacement.
- Cancellation: `POST /v1/audio/cancel` releases an abandoned stream before the next request.
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

Expression control is optional. The public source, image, and release bundle ship
without predefined expression profiles or reference media. Create a specification
for references that you have the right to use:

```json
{
  "schema": 1,
  "default_profile": "neutral",
  "preferred_policy": "semantic-style",
  "profiles": [
    {
      "id": "gentle-1",
      "emotion": "gentle",
      "intensity": 0.7,
      "reference_audio": "gentle.wav",
      "reference_text": "This transcript must exactly match the reference audio.",
      "reference_language": "en",
      "manual_verified": true
    }
  ]
}
```

Then add the verified references without modifying the source package:

```powershell
.\.venv\Scripts\aniflive-tts.exe model import-expressions `
  --model-package D:\models\my-v2proplus `
  --voice-profile default `
  --spec-file D:\models\expressions.json `
  --asset-root D:\models\expression-audio `
  --output D:\models\my-v2proplus-expression
```

The importer is model-ID agnostic and uses package metadata, so any validated V2ProPlus package follows the same contract.
Use clean, single-speaker WAV files and exact transcripts. The importer copies the
assets into the new output package, updates checksums, and leaves the original
neutral package unchanged.

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

### 5. Start the local WebUI

```powershell
.\run_webui.bat
```

After activating a package with expression profiles, highlight a complete clause
including its comma, period, semicolon, question mark, or exclamation mark, then
choose an expression. Unmarked text uses the voice's native neutral delivery.
Packages without imported profiles keep the expression UI unavailable.

## API

```powershell
curl.exe -X POST "http://127.0.0.1:9882/v1/audio/speech" `
  -H "Content-Type: application/json" `
  --output output.wav `
  --data '{"model":"my-v2proplus","voice_profile":"default","text":"今日はいい天気ですね。","language":"ja","stream":false,"generation":{"top_k":15,"top_p":1.0,"temperature":1.0,"seed":1234}}'
```

For one expression, add a symbolic profile returned by `GET /v1/expressions`:

```json
{
  "model": "my-v2proplus",
  "text": "The weather is nice today.",
  "language": "en",
  "stream": true,
  "expression": {
    "enabled": true,
    "profile": "gentle",
    "intensity": 0.7,
    "policy": "semantic-style"
  }
}
```

For multiple expressions, send complete, punctuation-terminated clauses. The
server rejects mid-phrase switches rather than risk skipped or unclear words:

```json
{
  "model": "my-v2proplus",
  "language": "en",
  "stream": true,
  "segments": [
    {
      "text": "I was worried, ",
      "expression": {"enabled": true, "profile": "shy", "intensity": 0.6}
    },
    {
      "text": "but now I am ready.",
      "expression": {"enabled": true, "profile": "confident", "intensity": 0.8}
    }
  ]
}
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
| `GET /v1/expressions` | Package-curated expression profiles and policies |
| `POST /v1/audio/cancel` | Cancel the active stream |

## Roadmap

**v1.3**

- [x] V2ProPlus model conversion and nine-stage TensorRT 11 inference
- [x] Five-language API, complete WAV, low-latency PCM streaming, and Docker delivery
- [x] Package-curated expression profiles, per-segment delivery, and the local WebUI

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

Original AnifLive-TTS code is licensed under [PolyForm Noncommercial 1.0.0](LICENSE); commercial use requires a separate written Commercial License from Hiruynk. GPT-SoVITS-derived portions retain MIT, Minimal Inference-derived and applicable GPT-SoVITS C++ reference portions retain Apache-2.0, and dependencies retain their own terms. See [LICENSING.md](LICENSING.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
