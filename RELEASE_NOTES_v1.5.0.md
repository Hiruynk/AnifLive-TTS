# AnifLive-TTS v1.5.0

## Highlights

- Added the GPT-SoVITS v5Turbo backend with CUDA/TensorRT/Triton, FP16, native four-step CFM, CFG 0 and 48 kHz PCM16. It works with the existing Voice/Reference, Expression, API and SDK workflows.
- Added Native Apple Silicon support with Rust/MLX/Metal on macOS, using FP16 or FP32 and native deployment. The macOS backend supports complete WAV, PCM streaming, Voice/Reference management and Speech Sessions. It has been verified on Mac Studio M5 Max.

## Runtime and installation

v2ProPlus and v5Turbo use the same service interface. The existing WebUI and Studio remain available. See the [Docker deployment guide](docs/DEPLOYMENT_EN.md), [v5Turbo package guide](docs/V1_5_V5_BUILD.md) and [macOS installation guide](docs/MACOS_INSTALLATION_EN.md).

## SDK

Install the wheel for your platform and connect to a running service. The [SDK manual](docs/SDK_EN.md) covers sync/async clients, Voice and Reference management, streaming and Speech Sessions.

## Performance

Keep-alive audible TTFA P50 is 75.70 ms for v2ProPlus and 80.93 ms for v5Turbo on RTX 5070 Ti, and 404.69 ms for v5Turbo on Mac Studio M5 Max. Measurement methods and RTF are listed in [Benchmarks](docs/BENCHMARKS.md).

---

# AnifLive-TTS v1.5.0（繁體中文）

## 新增內容

- 新增 GPT-SoVITS v5Turbo 後端，使用 CUDA／TensorRT／Triton、FP16、原生四步 CFM、CFG 0 及 48 kHz PCM16，並適配既有 Voice／Reference、情感控制、API 和 SDK。
- 新增 Apple Silicon 原生支援，以 Rust／MLX／Metal 在 macOS 使用 FP16 或 FP32 推理，無須 Docker。macOS 後端支援完整 WAV、PCM 串流、Voice／Reference 管理及 Speech Session，已在 Mac Studio M5 Max 驗證。

## Runtime 與安裝

v2ProPlus 與 v5Turbo 使用同一服務介面，既有 WebUI 及 Studio 繼續提供。請參閱 [Docker 部署指南](docs/DEPLOYMENT_ZH.md)、[v5Turbo 套件指南](docs/V1_5_V5_BUILD.md)及 [macOS 安裝指南](docs/MACOS_INSTALLATION_ZH.md)。

## SDK

安裝符合平台的 wheel，連接已啟動的服務。[SDK 使用手冊](docs/SDK_ZH.md)介紹同步／非同步用戶端、Voice／Reference 管理、串流及 Speech Session。

## 性能

持續連線可聽 TTFA P50：RTX 5070 Ti 上 v2ProPlus 為 75.70 ms、v5Turbo 為 80.93 ms；Mac Studio M5 Max 上 v5Turbo 為 404.69 ms。測量方法及 RTF 見 [Benchmarks](docs/BENCHMARKS.md)。
