# AnifLive-TTS v1.5.0 Beta 1

## Highlights

- Official Python SDK with synchronous `Client` and asynchronous `AsyncClient`, backed by a shared Rust transport core.
- Typed service discovery, model activation, complete WAV synthesis, incremental PCM streaming, cancellation, and speech sessions.
- Session lookup, ordered segments, idempotency, flush, and cancellation follow the existing server API.
- Lightweight client installation with no local CUDA, TensorRT, or model requirement. Server commands use the `runtime` extra.

## SDK

Install the supplied wheel matching your platform, then connect to a running AnifLive-TTS service. See the [SDK manual](docs/SDK_EN.md) for installation, sync/async examples, expressions, sessions, and typed errors.

The package version is `1.5.0b1`. Linux x64 wheels target CPython 3.10–3.12 with a manylinux_2_28 baseline. Windows x64 and macOS arm64 wheel workflows are prepared; their platform qualification is pending.

## Runtime, WebUI And Studio

The existing TensorRT runtime, WebUI, and Studio remain available. This beta adds a client SDK; it does not embed inference in client applications. Docker builds compile the client extension in a separate builder stage.

---

# AnifLive-TTS v1.5.0 Beta 1（繁體中文）

## 新增內容

- 官方 Python SDK 提供同步 `Client` 與非同步 `AsyncClient`，共用 Rust 傳輸核心。
- 型別化服務探索、模型啟用、完整 WAV 合成、增量 PCM 串流、取消及 Speech Session。
- Session 查詢、順序片段、冪等追加、flush 與取消沿用現有伺服器 API。
- 輕量用戶端不需要本機 CUDA、TensorRT 或模型；伺服器指令使用 `runtime` extra。

## SDK

安裝符合平台的隨附 wheel，再連接已啟動的 AnifLive-TTS 服務。安裝、同步／非同步範例、情感設定、session 與型別化錯誤請參閱 [SDK 使用手冊](docs/SDK_ZH.md)。

套件版本為 `1.5.0b1`。Linux x64 wheel 以 manylinux_2_28 為基準，目標為 CPython 3.10–3.12。Windows x64 與 macOS arm64 的 wheel 工作流程已備妥，平台驗證尚待執行。

## Runtime、WebUI 與 Studio

既有 TensorRT runtime、WebUI 與 Studio 繼續並存。本次 Beta 新增用戶端 SDK，不會把推理嵌入用戶端應用程式。Docker 透過獨立建置階段編譯 client extension。
