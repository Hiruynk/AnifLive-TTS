# v1.5 Platform and Model Capability Matrix

This matrix summarizes the service backends, sample rates, and platform behavior. Benchmark protocols and results are in the [performance engineering record](PERFORMANCE_ENGINEERING.md).

## Platform status

| Platform / target | Runtime status | Backend and precision | Model rate and configuration | Platform-specific claims |
|---|---|---|---|---|
| Windows, NVIDIA | Available service backends use CUDA／TensorRT（Docker）. | v2ProPlus uses TensorRT 11. v5Turbo uses FP16, native 4-step CFM and CFG 0. | v2ProPlus: 32 kHz. v5Turbo: 48 kHz. | See the [Windows deployment guide](DEPLOYMENT_EN.md). |
| macOS, Apple Silicon | Apple Silicon（macOS／MLX／Metal）. | FP16 recommended; frozen FP32 rollback worker available. | v5Turbo: 48 kHz mono PCM16, native 4-step CFM and CFG 0. | Tested on Mac Studio M5 Max, 40-core GPU, 128 GB, macOS 27.0. See the [macOS installation guide](MACOS_INSTALLATION_EN.md). |

The bundled MLX libraries have a macOS 26.2+ deployment floor; only macOS 27.0 has been verified. Other Apple Silicon models are untested; Intel Macs are unsupported.

## Capability status

| Capability | Windows / NVIDIA | macOS / Apple Silicon |
|---|---|---|
| v2ProPlus backend | CUDA／TensorRT（Docker）, 32 kHz. | Not available on macOS. |
| v5Turbo backend | CUDA／TensorRT（Docker）, FP16, native 4-step CFM, CFG 0, 48 kHz. | Apple Silicon（macOS／MLX／Metal）, 48 kHz. The current FP16 worker supports all five policies: `full-switch`, `identity-lock`, `semantic-style`, `acoustic-style`, and `sv-only`. They map to v5's semantic, style and CFM voice-anchor components; v5 has no separate v2ProPlus SV encoder. The frozen FP32 rollback worker retains its original policy set; check `expression.policies` and `expression.complete_policies`. |
| API, Python SDK, Voice / Reference, PCM streaming, complete WAV, Speech Session and cancellation | Shared API routes and SDK; model-specific features are returned by capabilities. | Dynamic Voice/Expression catalogs, on-demand Reference preparation/cache, `voice_profile=...`, per-segment expression, and complete WAV/PCM are supported. Session A and complete-WAV cancellation are supported; cancellation returns HTTP 409 with no partial WAV. Only Session A is a product contract; B–F remain research. |
| WebUI and Studio | WebUI and Studio remain available for the Windows deployment. | The shared WebUI can target the Mac service; browser playback on Mac is unverified. Studio workstation workflows remain Windows-only. |

Reference and expression behavior depends on the active model and worker; use `GET /v1/capabilities` and `GET /v1/expressions` to discover supported features.

## macOS behavior

The current FP16 worker uses Rust/MLX/Metal for neural inference and Python for HTTP and text processing. PyTorch 2.9.1 MPS is used only for Reference DSP; there is no CPU, PyTorch, or MPS neural fallback.

PCM streaming is incremental at the vocoder stage. AR generation reaches EOS and the full four-step CFM completes before the first PCM; the vocoder then emits windows. There is no pre-EOS semantic or acoustic stream. Complete WAV uses one full-length vocoder call.

Only Session A is part of the product contract. Session strategies B–F remain research. v2ProPlus and Studio workstation workflows are Windows-only; WebUI browser playback on Mac is unverified. For benchmarks and methods, see the [performance engineering record](PERFORMANCE_ENGINEERING.md).
