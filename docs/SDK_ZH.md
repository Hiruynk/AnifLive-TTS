# Python SDK（Beta）

AnifLive-TTS Python SDK 是連接 AnifLive-TTS 服務的型別化用戶端。Python facade 透過內附的 Rust 傳輸核心呼叫 canonical HTTP API；既有伺服器、WebUI 與 Studio 繼續並存。

## 下載與安裝

v1.5.0-beta 發布後，請開啟 [GitHub Releases](https://github.com/Hiruynk/AnifLive-TTS/releases)，選擇 `v1.5.0-beta`，並從 **Assets** 下載符合作業系統及架構的 `.whl`。本機預覽期間使用隨附的 wheel。

安裝下載的 wheel（請將範例路徑換成實際檔案位置）：

```bash
python -m pip install /path/to/aniflive_tts-1.5.0b1-cp310-abi3-manylinux_2_28_x86_64.whl
```

Linux x64 wheel 已於 CPython 3.10–3.12 驗證。Windows x64 與 macOS arm64 工作流程已備妥，尚待原生平台驗證。用戶端 wheel 不需要本機 CUDA、TensorRT、模型套件或 Docker。

此版本發布至套件索引後，可使用 `python -m pip install aniflive-tts==1.5.0b1`。伺服器指令請安裝 `runtime` extra，例如 `python -m pip install 'aniflive-tts[runtime]==1.5.0b1'`；它提供 Python 伺服器相依套件，推理仍沿用部署環境鎖定的 CUDA／TensorRT 相依套件。

從原始碼 checkout 開發：

```bash
python -m pip install -e '.[runtime]'
```

原始碼及 editable 建置需要 Rust 1.98.1、CMake 與 C 編譯器（Windows 使用 MSVC Build Tools）。一般 pip build isolation 會安裝已宣告的 Python 建置工具；使用 `--no-build-isolation` 時須先安裝所有 `[build-system].requires`，包括 `setuptools-rust`。Docker 建置會在 builder stage 提供這些工具。

## 選擇用戶端

同步程式使用 `Client`；asyncio 應用程式使用 `AsyncClient`。兩者均提供 `close()`、`is_closed` 並支援 context manager。`AsyncClient.close()` 會同步釋放本機資源；若需要可 await 的明確清理，請使用 `await client.aclose()`。建立 client 不會連線；`probe()`、`synthesize()` 等方法才會送出網路請求。

```python
from aniflive_tts import AsyncClient, Client

with Client("http://127.0.0.1:9880") as client:
    health = client.health()

async def check_async_health() -> None:
    async with AsyncClient("http://127.0.0.1:9880") as client:
        health = await client.health()
```

預設連線逾時為 10 秒、完整請求逾時為 120 秒、串流讀取閒置逾時為 60 秒。PCM 呼叫的 request timeout 只涵蓋請求建立至收到回應標頭；idle timeout 則限制等待串流資料的時間，串流本身沒有整體期限。`request_timeout=None` 會停用 request limit；`stream_idle_timeout=None` 會停用串流閒置逾時。AsyncClient 會綁定首次呼叫所在的 event loop。

## 查詢服務並檢查相容性

Client 提供 `health()`、`capabilities()`、`models()`、`voices()`、`expressions()`、`activate_model(model)` 與 `probe()`。`probe()` 會明確查詢 health 和 capabilities，並檢查服務 contract；建立 client 時不會自動探測。若應用程式依賴特定功能，請傳入 required features：

```python
from aniflive_tts import AsyncClient, RequiredFeature

async def check_required_features() -> None:
    async with AsyncClient("http://127.0.0.1:9880") as client:
        service = await client.probe(
            required=(RequiredFeature.PCM_STREAMING, RequiredFeature.SPEECH_SESSIONS)
        )
```

可要求 `PCM_STREAMING`、`EXPRESSIONS` 和 `SPEECH_SESSIONS`。若服務不符合要求，會引發 `CompatibilityError`。`capabilities.streaming.wav` 描述增量 WAV 串流功能；值為 `false` 表示不支援串流 WAV。完整 WAV 合成仍可使用 `stream=false`。

## 儲存完整 WAV

`Client.synthesize()` 和 `AsyncClient.synthesize()` 會回傳 `SynthesisResult`。`audio` 欄位包含完整 WAV 檔案；`metadata` 則包含解析後且允許公開的回應中繼資料。

```python
from pathlib import Path
from aniflive_tts import Client, Language, SynthesisRequest

request = SynthesisRequest(language=Language.EN, text="Hello from AnifLive-TTS.")
with Client("http://127.0.0.1:9880") as client:
    client.probe()
    result = client.synthesize(request)

Path("speech.wav").write_bytes(result.audio)
```

可參考 [sdk_sync_wav.py](../examples/sdk_sync_wav.py) 的命令列範例。每個 request 必須在 `text` 和 `segments` 之間擇一。請透過服務 discovery 查詢目前的 model、voice 和 expression profiles；不要假設某個 model package 一定提供特定 profile。

## 讀取增量 PCM

`stream()` 會回傳 `AudioStream`（同步）或 `AsyncAudioStream`（非同步）。串流內容是原始有號 16 位元 little-endian PCM 位元組，不是 WAV container。請從 stream 讀取 `sample_rate`、`channels` 和 `sample_format`；傳輸區塊邊界不固定，也不保證與 sample frame 對齊。SDK 不會播放或重新取樣音訊。

```python
import asyncio
from pathlib import Path
from aniflive_tts import AsyncClient, Language, RequiredFeature, SynthesisRequest

async def main() -> None:
    request = SynthesisRequest(language=Language.EN, text="Streaming PCM.")
    async with AsyncClient("http://127.0.0.1:9880") as client:
        await client.probe(required=(RequiredFeature.PCM_STREAMING,))
        stream = await client.stream(request)
        try:
            with Path("speech.pcm").open("wb") as output:
                async for chunk in stream:
                    output.write(chunk)
        finally:
            stream.close()

asyncio.run(main())
```

同步版本使用一般 iterator：

```python
from pathlib import Path
from aniflive_tts import Client, Language, RequiredFeature, SynthesisRequest

request = SynthesisRequest(language=Language.EN, text="Streaming PCM.")
with Client("http://127.0.0.1:9880") as client:
    client.probe(required=(RequiredFeature.PCM_STREAMING,))
    stream = client.stream(request)
    try:
        with Path("speech.pcm").open("wb") as output:
            for chunk in stream:
                output.write(chunk)
    finally:
        stream.close()
```

`stream.close()` 只會關閉本機的 response body，不保證取消伺服器端的合成。`client.cancel()` 會呼叫服務的 process-wide current-stream cancellation endpoint，也可能影響由 session 持有的串流；它不只作用於該 client 物件。要取消指定的 speech session，請使用 owner-scoped 的 `session.cancel()`。可參考 [sdk_async_stream.py](../examples/sdk_async_stream.py)。

## 使用 speech session

Speech session 會將有序 segments 附加至遠端 session，並提供單一 raw PCM consumer。依賴 session 時，請先要求 `PCM_STREAMING` 和 `SPEECH_SESSIONS`。Session handle 提供 `id` 和不可變的 `initial_info`；呼叫 `get()` 取得最新伺服器狀態。

有限批次合成的順序是：附加所有 segments、flush，然後開啟 audio stream 並讀取至 EOF：

```python
from pathlib import Path
from aniflive_tts import AsyncClient, Language, SegmentRequest, SessionCreate

async def finite_session(client: AsyncClient) -> None:
    session = await client.create_session(SessionCreate())
    try:
        await session.append(SegmentRequest(
            segment_id="part-1",
            text="A complete sentence.",
            language=Language.EN,
        ))
        await session.flush()  # 不再接受新 segment
        audio = await session.audio()  # flush 後讀取至 EOF
        try:
            with Path("speech.pcm").open("wb") as output:
                async for chunk in audio:
                    output.write(chunk)
        finally:
            audio.close()
    finally:
        await session.close()
```

若 session 仍為 open 且尚未 flush，先將 audio 讀至 EOF 可能會無限等待：服務會保持 consumer 開啟，以接收後續 segments。即時合成時，先啟動單一 audio reader task，再附加 segments；最後一段附加後呼叫 `flush()`，並繼續讀取至 EOF。不要開啟第二個 consumer；服務會回傳 HTTP 409。

`session.cancel()` 是遠端且限定 owner 的取消操作。`session.close()` 會送出 `DELETE /v1/sessions/{id}`；目前伺服器將 DELETE 視為 cancel alias，並保留 terminal record 直到清理。關閉 client 只會釋放本機資源，不會刪除遠端 sessions。[sdk_session.py](../examples/sdk_session.py) 同時示範有限批次與即時並行流程。

## 錯誤與傳輸行為

SDK 的基底例外是 `AnifLiveTTSError`。Client 端錯誤包含 `TransportError`、`TimeoutError`、`ProtocolError`、`CompatibilityError`、`ClosedError` 和 `ConcurrentReadError`。HTTP 錯誤會依序映射為 `BadRequestError`（400）、`NotFoundError`（404）、`ConflictError`（409）、`PayloadTooLargeError`（413）、`BusyError`（429）及 `ServerError`（5xx）；其他狀態使用 `APIError`。

`APIError` 提供 `status_code`、`server_code`、`retry_after`、`diagnostic` 和 `response_body`。`retry_after` 保留回應標頭的原始值。例外的 `str()` 和 `repr()` 會隱藏細節；服務訊息可能包含 request 資料，請避免記錄 diagnostic 或 response-body 欄位。

SDK 不會自動重試，包括 HTTP 429。請自行判斷操作是否適合重試，並在自訂政策中參考 `retry_after`。逾時或遺失回應不代表遠端操作已回復原狀。HTTPS 憑證驗證維持啟用、redirect 預設停用，且預設不讀取環境 proxy 變數。請依部署需求明確設定傳輸選項。

Canonical client surface 不提供舊式 flat routes、本機 model 路徑、參考音訊上傳或 reference override。SDK 是用戶端程式庫；請另行安裝並執行 AnifLive-TTS 服務。
