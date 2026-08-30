# API Contract

The canonical endpoint is `POST /v1/audio/speech` with `model`,
`voice_profile`, `text`, `language`, `stream`, `expression`, and `generation`.
Supported core language identifiers are `zh`, `yue`, `en`, `ja`, and `ko`.

`stream=false` returns a complete mono PCM16 WAV. `stream=true` returns raw
little-endian signed 16-bit PCM (S16LE) chunks with Content-Type
`application/octet-stream` and the `X-TTS-Sample-Format: s16le` response header.

The expression schema is stable in v1. Disabled expression uses native neutral
delivery. If the active package provides curated profiles, enabled expression
accepts `profile`, `intensity`, and optional `policy`; otherwise the request is
rejected with HTTP 400. Available profiles and the package-preferred policy are
reported by `GET /v1/expressions`.

A request may provide either `text` or a `segments` array, never both. Each
segment contains `text` plus an optional symbolic expression object. Segment
settings inherit omitted intensity and policy values from the top-level
expression object. Local paths, URLs, uploads, and reference overrides are not
accepted by the public API.
Expression changes must occur after a comma, period, semicolon, question mark,
exclamation mark, or paragraph boundary. Select complete clauses and include the
terminating punctuation; mid-phrase switches are rejected to prevent skipped or
unclear words.

Discovery endpoints are `/health`, `/v1/capabilities`, `/v1/models`,
`/v1/voices`, and `/v1/expressions`. `POST /v1/audio/cancel` cancels the active
stream. Legacy GPT-SoVITS flat requests and OpenAI `input`/`voice`
requests are compatibility adapters over the same runtime.

`POST /v1/models/activate` accepts `{"model":"my-v2proplus"}` for a
compatible package already present in the local model registry. The runtime
rejects activation while synthesis is active, unloads the current package, and
only then loads the replacement. Existing synthesis endpoints and URLs do not
change after activation.
