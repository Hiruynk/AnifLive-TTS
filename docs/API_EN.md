# API Contract

The canonical endpoint is `POST /v1/audio/speech` with `model`,
`voice_profile`, `text`, `language`, `stream`, `expression`, and `generation`.
Supported core language identifiers are `zh`, `yue`, `en`, `ja`, and `ko`.

`stream=false` returns a complete mono PCM16 WAV. `stream=true` returns raw
little-endian signed 16-bit PCM (S16LE) chunks with Content-Type
`application/octet-stream` and the `X-TTS-Sample-Format: s16le` response header.

The expression schema is stable in v1. Disabled expression uses the model's
native delivery. Enabled expression returns HTTP 501 with
`expression_not_implemented`; it is never silently ignored.

Discovery endpoints are `/health`, `/v1/capabilities`, `/v1/models`, and
`/v1/voices`. Legacy GPT-SoVITS flat requests and OpenAI `input`/`voice`
requests are compatibility adapters over the same runtime.

`POST /v1/models/activate` accepts `{"model":"my-v2proplus"}` for a
compatible package already present in the local model registry. The runtime
rejects activation while synthesis is active, unloads the current package, and
only then loads the replacement. Existing synthesis endpoints and URLs do not
change after activation.
