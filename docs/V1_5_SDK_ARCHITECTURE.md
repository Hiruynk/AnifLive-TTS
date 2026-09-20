# v1.5 SDK architecture (PR00)

Status: design baseline, 2026-09-20. Audited server baseline:
`2bb0b48d2d4821eec6f0d19208c27538812be2fc`. This PR adds no SDK
implementation and makes no wheel, performance, or platform qualification claim.

## Authority and boundary

The [API audit](V1_5_SDK_API_AUDIT.md) records the current wire contract.
The existing API implementation and contract tests, API_EN/API_ZH,
V1_4_ARCHITECTURE, V1_4_ACCEPTANCE, V1_4_QUALIFICATION,
RELEASE_VERIFICATION, SECURITY and LICENSING remain authoritative.
The [packaging ADR](adr/ADR-V1_5-RUST-SDK-PACKAGING.md) governs installation.

```text
application -> aniflive_tts Python facade -> PyO3 private _native
            -> Rust client protocol core -> canonical HTTP API
            -> existing Python/TensorRT service
```

This is a client SDK, not embedded inference. Server/model loading, CUDA,
TensorRT, GPT-SoVITS, workstation, workers and training remain unchanged.
No playback, resampling, WebSocket protocol, CPU inference implementation or
downstream repository changes belong to this track. Downstream adapters use the
official SDK; missing client primitives must be added here, never implemented
as a private Python HTTP wrapper downstream.

## Code ownership

| Layer | Responsibilities |
| --- | --- |
| `rust/aniflive-tts-client` | async Rust core; URL/routes, serde wire models, pool, status/header/JSON decoding, timeout/cancel, body ownership, compatibility and all session calls |
| `rust/aniflive-tts-python` | PyO3 conversion, native exceptions, Python futures, sync waiting and native resource handles |
| `src/aniflive_tts` facade | stable public exports, context managers, iterator ergonomics, type stubs, convenience construction; no HTTP or duplicate serialization |
| server | validation, model/voice/expression availability, scheduling, session state/idempotency and actual synthesis |

Public surface includes Client/AsyncClient, typed requests/results, Language,
Expression, Generation, ContinuityPolicy A-F, ConditioningPolicy and session
types. Expose discovery/probe, activation, complete WAV, direct PCM/cancel and
every session operation in the audit with sync/async parity. Server state is
authoritative; a session handle retains its ID and client reference, while
refresh obtains current state. Preserve extra response fields and raw safe
metadata. Backend identity is opaque; TensorRT timing fields are optional.

## Runtime, pool and Python event loop

Use one async reqwest core. Each client owns a persistent reqwest client/pool,
shared by its stream/session handles; do not construct a transport per call.
Use the maintained pyo3-async-runtimes Tokio bridge and a lazily initialized
process-scoped Tokio multithread runtime. Import and client construction make
no network request or model/server import. Defer runtime work until needed.
Do not create a Python event loop or call asyncio.run from SDK code.

Async methods capture the calling Python event loop/task context via the bridge.
Bind AsyncClient to the loop of first use; reject use from another loop.
Rust tasks own Rust data, never borrowed Python references across await.
Sync operations submit the same core future to Tokio and wait with the GIL
released; no nested Tokio block_on on a Tokio worker. Reject blocking facade
calls made on a running Python event-loop thread with guidance to AsyncClient.
The bridge documents Python-future cancellation support; SDK cleanup still
needs its own cancellation guards and race tests.
See [bridge API](https://docs.rs/pyo3-async-runtimes/0.29.0/pyo3_async_runtimes/tokio/fn.future_into_py.html).

Runtime lifetime is process-scoped, not client-scoped. Client close cancels and
awaits its own tasks/bodies, rejects new calls, and drops pool references; it
does not shut down Tokio used by another client. Streams keep resource handles
alive until closed. No interpreter callbacks from destructors. Explicit close
is deterministic; finalizers only signal cancellation/drop resources without
blocking or remote requests. Forked-child reuse and subinterpreters are outside
Beta support; create clients in the worker process.

## Streaming and cancellation

Return stream handles after successful status and mandatory PCM metadata
validation, before collecting the body. Yield nonempty bytes in order with no
full-response buffering or sample conversion. HTTP chunk boundaries are not
PCM frame boundaries; callers must not infer frames from individual chunks.
A Rust stream has one active reader. Concurrent next calls fail clearly.
Use demand-driven next_chunk with no unbounded producer queue; retain only
the active transport chunk plus a bounded delivery slice. Choose and test the
concrete byte bound in PR05, including a malicious oversized HTTP chunk.
Library/socket buffering is separate and must be measured.

States are open, exhausted, failed and closed. EOF closes the body. Read failure,
idle timeout or cancellation terminates that stream; iteration cannot silently
resume or retry. Repeated local stream close is harmless. Close must wake an
in-flight next call without waiting for a mutex held across network I/O.
Async task cancellation stays asyncio.CancelledError and drops/cancels the
associated Rust operation; a cancelled stream read closes that stream.

Local stream close only releases the HTTP consumer. It does not promise remote
synthesis cancellation and does not issue the global direct-cancel endpoint.
client.cancel targets the runtime current stream, which may belong to a speech
session; it is not scoped to the calling SDK client. session.cancel is an
explicit owner-scoped remote operation; session.close maps to DELETE, currently
a cancel alias that retains the terminal record until pruning.
Closing a client closes local resources and does not delete all remote sessions.
For finite sessions append, flush, then drain audio; live applications consume
audio concurrently with append/flush. Draining an open session before calling
flush can wait indefinitely. Disconnect effects remain server-defined (see audit). Remote operations that
lose their response can have an unknown outcome; do not claim rollback.
A session's one-consumer conflict, stable segment-ID duplicate success/conflict,
flush, expiry and continuity metadata pass through unchanged. Policies B-F
remain experimental/unqualified where the server reports that status; synthesis
continuity does not establish what a downstream user actually heard.

## Timeouts, errors and transport policy

Configuration exposes base_url, connect_timeout, request_timeout,
stream_idle_timeout, optional headers and User-Agent. Proposed initial defaults
for fixture validation: 10 seconds connect, 120 seconds request, 60 seconds
stream read idle; explicit None disables the corresponding request/idle limit.
For complete/JSON calls, request_timeout bounds the entire operation including
body. For streams it bounds request establishment through response headers;
there is no implicit whole-stream deadline. Idle time bounds an outstanding
network body read, not the application's deliberate pause between next calls.
Both facades use the same values and semantics. No client timeout implies
successful remote cancellation.

Use TLS verification by default; support custom trust roots through explicit
configuration. Disable redirects by default so a synthesis POST or credentials
cannot silently move to another origin. Accept http/https URLs only, reject URL
credentials/query/fragment, normalize a path prefix once, and percent-encode
dynamic IDs as one path segment. Additional proxy headers do not introduce an
AnifLive-TTS authentication protocol. Use an SDK-version User-Agent.

Default retries are zero, including synthesis, activation, append and 429.
Expose Retry-After rather than sleep secretly. Any later opt-in retry applies
only to bounded discovery GET failures; do not retry partially consumed audio.
Stable segment IDs permit caller-directed retry with an explicitly documented
unknown-outcome case.

Expose typed transport, timeout, protocol, compatibility and API errors under
AnifLiveTTSError. Map 400/404/409/413/429/5xx per audit; retain generic APIError
for other statuses. Preserve status, server code, Retry-After and bounded safe
diagnostics even for malformed error JSON. Validate required PCM headers and
content type; tolerate absent optional inference metrics and unknown JSON fields.
Bound JSON/error body sizes before decoding (concrete limits tested in PR02).
Do not log text, arbitrary headers, audio, request bodies or session IDs by
default. Error repr must redact credentials and sensitive payloads.

## Compatibility and verification

Target the audited v1.4+ canonical contract, not an exact patch version.
probe explicitly reads health and capabilities, records identity/version and
checks requested features. capabilities.streaming.wav=false means streamed WAV
is unsupported; it does not disable the complete WAV synthesis method. Preserve
the current selective request unknown-field behavior rather than inventing a
server-wide rejection policy. A reachable unready service is distinguishable from
an incompatible service. No construction-time probe; backend strings and
version strings alone cannot reject a compatible CPU implementation.

GPU-free fixture tests must prove sync/async parity, malformed/unknown fields,
status mapping, pool reuse, timeout phases, backpressure, slow consumer,
cancel-before/after-headers, pending-read close, double-close, EOF, event-loop
affinity and session duplicate/conflict/lifecycle behavior. Keep existing server
tests; compare serialization with their canonical fixtures.
Clean installed-wheel tests must assert no torch/tensorrt/fastapi/uvicorn import.
Typing tests check public exports and examples.

Real-server synthesis/session/cancellation qualification and matched raw-HTTP
versus SDK p50/p95 latency/first-packet/TTFA measurements are later release gates.
Investigate any consistent >10% regression; this design makes no speed claim.
