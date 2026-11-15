# API Contract

Python applications can use the [official SDK](SDK_EN.md). The server API is the wire contract; the Rust client core handles transport and protocol, while the Python facade provides typed sync/async interfaces.

The canonical endpoint is `POST /v1/audio/speech` with `model`,
`voice_profile`, `text`, `language`, `stream`, `expression`, and `generation`.
Supported core language identifiers are `zh`, `yue`, `en`, `ja`, and `ko`.

`stream=false` returns a complete mono PCM16 WAV. `stream=true` returns raw
little-endian signed 16-bit PCM (S16LE) chunks with Content-Type
`application/octet-stream` and the `X-TTS-Sample-Format: s16le` response header.

Example complete-WAV request:

```json
{
  "model": "my-v2proplus",
  "voice_profile": "default",
  "text": "こんにちは。",
  "language": "ja",
  "stream": false,
  "generation": {"top_k": 15, "top_p": 1.0, "temperature": 1.0, "seed": 1234}
}
```

Use `stream=true` for raw PCM. Response metadata describes the sample rate; do not assume every model uses the same rate.

The expression schema is stable in v1. Disabled expression uses native neutral
delivery. If the active package provides curated profiles, enabled expression
accepts `profile`, `intensity`, and optional `policy`; otherwise the request is
rejected with HTTP 400. Available profiles and the package-preferred policy are
reported by `GET /v1/expressions`.

A request may provide either `text` or a `segments` array, never both. Each
segment contains `text` plus an optional symbolic expression object. Segment
settings inherit omitted intensity and policy values from the top-level
expression object. The speech endpoint does not accept local paths, URLs, raw
audio uploads, or per-request Reference overrides. Manage stored Voice and
Reference resources through the existing Voice/Reference routes.
Expression changes must occur after a comma, period, semicolon, question mark,
exclamation mark, or paragraph boundary. Select complete clauses and include the
terminating punctuation; mid-phrase switches are rejected to prevent skipped or
unclear words.

Discovery endpoints are `/health`, `/v1/capabilities`, `/v1/models`,
`/v1/voices`, and `/v1/expressions`. `POST /v1/audio/cancel` cancels the active
stream. Legacy GPT-SoVITS flat requests and OpenAI `input`/`voice`
requests are compatibility adapters over the same runtime.

`POST /v1/models/activate` accepts a model ID for a compatible package
already present in the local model registry. The runtime rejects activation
while synthesis is active, unloads the current package, and only then loads the
replacement. Existing synthesis endpoints and URLs do not change after
activation.
Activation changes only the active local model; the request contract remains the same.

## Windows v5Turbo behavior

**Backend:** CUDA／TensorRT（Docker）. v2ProPlus outputs 32 kHz. v5Turbo uses
`gsv-v5turbo`, FP16, native four-step CFM and CFG 0, and outputs 48 kHz. It
requires `speed=1.0`, `noise_scale=0.5`, four sample steps, and no `cut_punc`.

The v5Turbo backend accepts all five Reference-conditioning policies:
`full-switch`, `identity-lock`, `semantic-style`, `acoustic-style`, and
`sv-only`. `full-switch` is the default and selects a stored Reference by ID;
a stored `emotion_id` plus a supported discrete intensity can select a matching
Reference. The other policies map References onto v5's semantic, style, and
CFM voice-anchor components; v5 has no separate v2ProPlus SV encoder.

Voice, Reference, and Speech Session metadata use the existing API contract.
See [Voice / Reference behavior](V1_5_VOICE_API.md) and the [platform matrix](V1_5_PLATFORM_MATRIX.md).

## macOS Apple Silicon behavior

**Backend:** Apple Silicon（macOS／MLX／Metal）. Native v5 outputs 48 kHz mono PCM16 with four-step CFM and CFG 0. `stream=false` returns a complete WAV; `stream=true` returns raw PCM. Streaming starts after autoregressive generation reaches EOS and the full CFM completes, then emits incrementally from the vocoder; it is not pre-EOS semantic or acoustic streaming.

The current FP16 worker supports all five Reference-conditioning policies and per-segment expression: `full-switch`, `identity-lock`, `semantic-style`, `acoustic-style`, and `sv-only`. They map to v5's semantic, style, and CFM voice-anchor components; v5 has no separate v2ProPlus SV encoder. The frozen FP32 rollback worker retains its original policy set. Check `expression.policies` and `expression.complete_policies` in capabilities before selecting a policy.

Registered Voices and expression profiles are listed by `/v1/voices` and `/v1/expressions`; query profiles for a Voice with `GET /v1/expressions?voice_profile=<id>`. Requests select that Voice with `voice_profile`. References are prepared on demand and cached. Generation accepts `top_k` 1–50, `top_p` in (0, 1], `temperature` greater than 0, and `seed` greater than or equal to -1. v5 uses `speed=1.0`, `noise_scale=0.5`, four sample steps, and no `cut_punc`. Compatibility adapters accept `auto`, `auto_yue`, and supported aliases; canonical `ja` uses mixed Japanese/English text processing.

Session A supports append, flush, cancellation, and recovery; strategies B–F remain research features. Cancelling an active complete-WAV request ends synthesis with HTTP 409 and no partial WAV, after which the service can synthesize again. The shared WebUI can connect to the Mac service; v2ProPlus and Studio workstation workflows remain Windows-only. Hardware requirements and Reference DSP dependencies are in [macOS installation](MACOS_INSTALLATION_EN.md). See also [Voice / Reference behavior](V1_5_VOICE_API.md) and the [platform matrix](V1_5_PLATFORM_MATRIX.md).
