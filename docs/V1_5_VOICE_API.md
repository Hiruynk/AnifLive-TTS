# v1.5 Voice / Reference contract

Voice and Reference resources belong to the TTS service and are available to
client applications through its API and SDK. For v2ProPlus, one shared base
engine serves all registered voices; creating a Voice or Reference does not
train or rebuild that engine. See [API behavior](API_EN.md), the [Python SDK](SDK_EN.md),
and [performance engineering details](PERFORMANCE_ENGINEERING.md).

## Shared v2ProPlus engine package

Prepare the official v2ProPlus `s1v3.ckpt`, `v2Pro/s2Gv2ProPlus.pth` and shared
assets using the existing [Quick Start](../README.md#quick-start). A shared
package does not need a character Reference during conversion:

```powershell
docker compose build aniflive-tts
docker compose run --rm aniflive-tts convert --shared-base --gpt /data/shared/s1v3.ckpt --sovits /data/shared/v2Pro/s2Gv2ProPlus.pth --shared-dir /data/shared --model-id shared-v2proplus --output /data/models/shared-v2proplus
$env:ANIFLIVE_TTS_MODEL_PACKAGE="/data/models/shared-v2proplus"
docker compose up -d aniflive-tts
```

The API starts with an empty Voice library. Add Voice/Reference resources through
the SDK or the routes below; new characters reuse these engines. The standard
Docker Compose and Studio launcher persist runtime voices in `data/voices`.
Existing fixed-voice packages remain supported.

## Resources

| Method | Resource | Behavior |
|---|---|---|
| GET / POST | `/v1/voices` | List / create (`id` optional, `name`) |
| GET / PATCH / DELETE | `/v1/voices/{id}` | Read / rename / delete |
| GET / POST | `/v1/voices/{id}/references` | List / upload audio + metadata |
| GET / PATCH / DELETE | `/v1/voices/{id}/references/{ref}` | Read / metadata or audio replacement / delete |
| POST | `/v1/voices/{id}/references/{ref}/prepare` | Prepare conditioning; poll reference state |
| GET | `/v1/voices/{id}/references/{ref}/audio` | Read normalized WAV |

Caller-supplied safe stable IDs support portable character folders; conflicts
return 409. `default` remains reserved for legacy fixed voice packages.
Uploads use multipart fields `audio` (file) and `metadata` (JSON). No server
paths or fetch URLs are accepted. For v2ProPlus, supported Reference decoding is WAV/FLAC/Ogg/MP3/M4A/WebM,
normalized to mono PCM16 at 32 kHz; references must be 3–10 seconds and at most 25 MiB.

Reference metadata: `id` (create only), `emotion_id`, Unicode `emotion_label`,
`intensity` (0–1), `reference_language`, `transcript`, `transcript_source`
(`manual`/`asr`/`imported`), optional `transcript_provider`,
`transcript_confirmed` (independent human confirmation), `is_default`.
A missing transcript is storable but cannot be prepared. For v5Turbo, the
metadata intensity field is used with supported discrete selection values; it
does not define continuous emotion control. English transcripts should use
normal sentence casing: the upstream text frontend spells all-capital
words as individual letters, which can misalign a sentence transcript with its audio.

States: needs_transcript, unprepared, queued, preparing, ready, failed.
Unprepared means no work is queued (including restart/cache eviction).
Audio/transcript/language updates invalidate conditioning; label changes do not.
Active requests/sessions pin a Voice; conflicting asset changes return 409.

Use canonical `voice_profile=Voice.id` for speech and sessions. On v2ProPlus,
manual reference selection reuses `Expression.profile=Reference.id`; emotion ID
plus intensity selects the nearest available Reference in that Voice. There is
no cross-voice fallback. The same Voice and Reference routes are used by
v5Turbo, with its conditioning policies described below. Model engine readiness
and individual Voice readiness are separate. Session metadata and
session-scoped cancellation use the existing API contract.

## Model-specific Reference behavior

### v2ProPlus

The existing v2ProPlus Reference and expression behavior remains available.
Its output rate is 32 kHz. The shared-engine setup, 32 kHz Reference normalization,
and nearest-reference selection above describe this backend.

### Windows v5Turbo

**Backend:** CUDA／TensorRT（Docker）. The v5Turbo model family is
`gsv-v5turbo`; it uses FP16, native four-step CFM and CFG 0, and outputs 48 kHz.
It accepts `full-switch`, `identity-lock`, `semantic-style`, `acoustic-style`,
and `sv-only`. `full-switch` is the default and selects a stored Reference ID;
`emotion_id` with a supported discrete intensity selects a matching Reference
in the same Voice. Other policies map to v5's semantic, GE/style, and CFM
voice-anchor components; v5 has no separate v2ProPlus SV encoder. Top-level
and per-segment policies use the same selection contract.

### Native macOS v5Turbo

**Backend:** Apple Silicon（macOS／MLX／Metal）. The current FP16 worker supports
the same five policies, per-segment expressions, and mixed policies. A frozen
FP32 rollback worker retains its original policy set; read
`expression.policies` and `expression.complete_policies` from capabilities
before selecting a policy. Registered Voices and References use the same API;
References are prepared on demand and cached. v2ProPlus remains Windows-only.
