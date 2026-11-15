# Python SDK

The AnifLive-TTS Python SDK, introduced in v1.5-beta, is a typed client for an AnifLive-TTS service. Its Python facade uses the bundled Rust transport core and the canonical HTTP API. The existing server, WebUI, and Studio remain available.

## Download and install

Download the v1.5 `.whl` matching your operating system and architecture from [GitHub Releases](https://github.com/Hiruynk/AnifLive-TTS/releases) → **Assets**, or use the wheel supplied in the release bundle.

Install the downloaded wheel (replace the example path with its actual location):

```bash
python -m pip install /path/to/aniflive_tts-1.5.0-cp310-abi3-linux_x86_64.whl
```

The client requires CPython 3.10–3.12. Wheels cover Linux x64, Windows x64 and macOS arm64. The SDK connects to a running service over HTTP. Native Mac service installation is covered in the [macOS installation guide](MACOS_INSTALLATION_EN.md).

For development or server commands from a source checkout, install the `runtime` extra and follow the backend's installation guide for inference dependencies:

```bash
python -m pip install -e '.[runtime]'
```

Source and editable builds require Rust 1.98.1, CMake, and a C compiler (MSVC Build Tools on Windows). Normal pip build isolation installs the declared Python build tools. If using `--no-build-isolation`, install all `[build-system].requires` first, including `setuptools-rust`. Docker builds provide these tools in a builder stage.

## Choose a client

Use `Client` for synchronous code and `AsyncClient` inside asyncio applications. Both expose `close()` and `is_closed` and support context managers. `AsyncClient.close()` performs synchronous local cleanup; use `await client.aclose()` when explicit awaitable cleanup is needed. Construction does not contact the service; calls such as `probe()` and `synthesize()` perform network requests.

```python
from aniflive_tts import AsyncClient, Client

with Client("http://127.0.0.1:9880") as client:
    health = client.health()

async def check_async_health() -> None:
    async with AsyncClient("http://127.0.0.1:9880") as client:
        health = await client.health()
```

The default timeouts are 10 seconds to connect, 120 seconds for a complete request, and 60 seconds idle while reading a stream. For PCM calls, the request timeout covers setup through response headers; the idle timeout applies while waiting for stream data, and there is no whole-stream deadline. `request_timeout=None` disables the request limit; `stream_idle_timeout=None` disables the stream idle limit. Async client use is bound to the event loop of its first call.

## API key authentication

Pass `api_key` to either client for an endpoint that accepts Bearer authentication. The SDK sends `Authorization: Bearer <api_key>` on each request. Read the key from an environment variable or a secret store.

```python
import os

from aniflive_tts import AsyncClient, Client

api_key = os.environ["ANIFLIVE_TTS_API_KEY"]

with Client("https://tts.example.com", api_key=api_key) as client:
    health = client.health()

async def check_authenticated_health() -> None:
    async with AsyncClient("https://tts.example.com", api_key=api_key) as client:
        health = await client.health()
```

`api_key` defaults to `None`. A supplied key must be non-empty and contain only visible ASCII characters. Custom `headers` are still supported. To supply `Authorization` yourself, omit `api_key`. Supplying both raises `ValueError`, regardless of the header name's casing. The SDK hides Bearer keys in its debug output and suppresses error details that echo the key.

## Discover and check compatibility

The client exposes `health()`, `capabilities()`, `models()`, `voices()`, `expressions()`, `activate_model(model)`, and `probe()`. Use `expressions(voice_profile="<voice-id>")` to discover a registered voice's available emotion references and intensities; omitting the argument retains the model's default catalog. Both synchronous and asynchronous clients support this query. `probe()` explicitly requests health and capabilities and checks the service contract. Construction alone does not probe. Pass required features when your application depends on them:

```python
from aniflive_tts import AsyncClient, RequiredFeature

async def check_required_features() -> None:
    async with AsyncClient("http://127.0.0.1:9880") as client:
        service = await client.probe(
            required=(RequiredFeature.PCM_STREAMING, RequiredFeature.SPEECH_SESSIONS)
        )
```

Available requirements are `PCM_STREAMING`, `EXPRESSIONS`, `SPEECH_SESSIONS`, and `VOICE_MANAGEMENT`. A failed requirement raises `CompatibilityError`.

`capabilities.streaming.wav` describes incremental WAV support: `false` means streamed WAV is unavailable. Complete WAV synthesis remains available with `stream=false`.

## Register a zero-shot voice

A shared v2ProPlus package can start with an empty voice library. Probe for
`RequiredFeature.VOICE_MANAGEMENT`, then use `create_voice(VoiceCreate(...))`,
`upload_reference(voice_id, audio_bytes, ReferenceCreate(...))`, and
`prepare_reference(voice_id, reference_id)`. Poll `reference(...)` until
`prepare_state == "ready"`; preparation is asynchronous. Pass the returned Voice
ID as `voice_profile` to `synthesize()`, `stream()`, or `create_session()`.
`AsyncClient` exposes the same operations with `await`.

[sdk_voice_reference.py](../examples/sdk_voice_reference.py) provides a complete
upload → prepare → WAV example. References need 3–10 seconds of speech and a
matching transcript. English sentence transcripts should use normal sentence
casing; the upstream frontend spells all-capital words as letters. ASR output
uses `transcript_source="asr"` with its provider; human confirmation is a separate
`transcript_confirmed` field.

Use `voice()`, `update_voice()`, `delete_voice()`, `references()`,
`update_reference()`, `replace_reference_audio()` and `delete_reference()` to
manage the library. Audio and metadata survive restart; GPU conditioning is
prepared again. Label-only edits keep conditioning. Requests and sessions pin
their selected voice while in use. Emotion selection uses
`Expression(profile=reference_id_or_emotion_id, intensity=...)` and chooses the
nearest available Reference rather than continuous interpolation. See the
[Voice / Reference API contract](V1_5_VOICE_API.md) for states and metadata.

## v2ProPlus and v5Turbo behavior

**Backends:** v2ProPlus and Windows v5 use CUDA／TensorRT（Docker）; the native Mac service uses Apple Silicon（macOS／MLX／Metal）. v2ProPlus outputs 32 kHz. v5Turbo uses `gsv-v5turbo`, FP16, four-step CFM and CFG 0, and outputs 48 kHz. Read the response or stream metadata for the active sample rate.

The v5 backend accepts all five Reference-conditioning policies: `full-switch`, `identity-lock`, `semantic-style`, `acoustic-style`, and `sv-only`. `full-switch` is the default and selects a Reference by ID; `emotion_id` plus a supported discrete intensity selects a matching Reference. Other policies map References onto v5's semantic, style, and CFM voice-anchor components; v5 has no separate v2ProPlus SV encoder.

The current Mac FP16 worker supports all five policies and per-segment expression. The frozen FP32 rollback worker retains its original policy set; check `expression.policies` and `expression.complete_policies` in service capabilities. For Mac PCM, generation reaches AR EOS and completes CFM before vocoder output begins. Session A supports flush, cancellation, and recovery; Session policies B–F remain research features. Cancelling a complete-WAV request returns HTTP 409 without partial audio. See the [macOS API notes](API_EN.md#macos-apple-silicon-behavior) and [platform matrix](V1_5_PLATFORM_MATRIX.md).

Example request with an explicit policy:

```python
from aniflive_tts import (
    Client, ConditioningPolicy, Expression, Language, SynthesisRequest,
)

request = SynthesisRequest(
    language=Language.EN,
    text="A complete sentence.",
    voice_profile="my-voice",
    expression=Expression(
        enabled=True,
        profile="happy-reference",
        intensity=0.7,
        policy=ConditioningPolicy.SEMANTIC_STYLE,
    ),
)
with Client("http://127.0.0.1:9880") as client:
    result = client.synthesize(request)
```

## Save a complete WAV

`Client.synthesize()` and `AsyncClient.synthesize()` return a `SynthesisResult`. Its `audio` field contains the complete WAV file; `metadata` contains the parsed safe response metadata.

```python
from pathlib import Path
from aniflive_tts import Client, Language, SynthesisRequest

request = SynthesisRequest(language=Language.EN, text="Hello from AnifLive-TTS.")
with Client("http://127.0.0.1:9880") as client:
    client.probe()
    result = client.synthesize(request)

Path("speech.wav").write_bytes(result.audio)
```

See [sdk_sync_wav.py](../examples/sdk_sync_wav.py) for a runnable command-line example. A request uses exactly one of `text` or `segments`. Use the service's discovery routes for active models, voices, and expression profiles; do not assume a package-specific profile exists.

## Read incremental PCM

`stream()` returns an `AudioStream` (sync) or `AsyncAudioStream` (async). The stream yields raw signed 16-bit little-endian PCM bytes, not a WAV container. Read `sample_rate`, `channels`, and `sample_format` from the stream; transport chunk boundaries are arbitrary and need not align with sample frames. The SDK does not play audio or resample it.

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

The synchronous form uses a normal iterator:

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

`stream.close()` closes this local response body. It does not promise to cancel remote synthesis. `client.cancel()` calls the service's process-wide current-stream cancellation endpoint and can affect a session-owned stream; it is not scoped to that client object. Use `session.cancel()` for owner-scoped session cancellation. See [sdk_async_stream.py](../examples/sdk_async_stream.py).

## Use speech sessions

Speech sessions append ordered segments to a remote session and expose one raw PCM consumer. Probe for both `PCM_STREAMING` and `SPEECH_SESSIONS` before depending on them. A session handle has an `id` and immutable `initial_info`; call `get()` for a fresh server snapshot. The macOS runtime supports Session A with flush, cancellation, and recovery; Session B neural context continuity is not supported.

For finite batch audio, append all segments, flush, then open and drain the audio stream:

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
        await session.flush()  # prevents further appends
        audio = await session.audio()  # read to EOF after flush
        try:
            with Path("speech.pcm").open("wb") as output:
                async for chunk in audio:
                    output.write(chunk)
        finally:
            audio.close()
    finally:
        await session.close()
```

Reading an open, unflushed session to EOF can wait indefinitely: the service keeps the consumer open for later segments. For live generation, start a single audio reader task before appending, append segments while that task consumes, then call `flush()` after the final append and keep draining until EOF. Do not open a second consumer; the service returns HTTP 409.

`session.cancel()` is a remote, owner-scoped cancellation. `session.close()` sends `DELETE /v1/sessions/{id}`; the server currently treats DELETE as a cancel alias and retains the terminal record until pruning. Closing a client only releases local resources and does not delete its remote sessions. See [sdk_session.py](../examples/sdk_session.py) for finite and concurrent live flows.

## Errors and transport behavior

Catch `AnifLiveTTSError` for the SDK base class. Client-side failures include `TransportError`, `TimeoutError`, `ProtocolError`, `CompatibilityError`, `ClosedError`, and `ConcurrentReadError`. HTTP failures map to `BadRequestError` (400), `NotFoundError` (404), `ConflictError` (409), `PayloadTooLargeError` (413), `BusyError` (429), and `ServerError` (5xx); other statuses use `APIError`.

`APIError` exposes `status_code`, `server_code`, `retry_after`, `diagnostic`, and `response_body`. `retry_after` preserves the response header value. Error `str()` and `repr()` redact details; avoid logging diagnostic or response-body attributes because service messages can contain request data.

The SDK does not retry automatically, including on HTTP 429. Decide whether an operation is safe to retry and use `retry_after` when applying your own policy. A timeout or lost response does not prove that a remote operation was rolled back. HTTPS certificate verification stays enabled, redirects are disabled, and environment proxy variables are not used by default. Set transport options explicitly for your deployment.

The canonical client surface does not expose legacy flat routes or local server model paths. The SDK remains a client library; install and run the AnifLive-TTS service separately.
