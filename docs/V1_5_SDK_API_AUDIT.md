# v1.5 SDK API Audit

Baseline: AnifLive-TTS source revision 2bb0b48d2d4821eec6f0d19208c27538812be2fc. This audits the server contract present at that revision; it does not change server behavior. The route implementation is in src/aniflive_tts/service.py. src/aniflive_tts/api.py configures the runtime and imports the FastAPI app; repository-root api.py is only the Uvicorn launcher. The API reports service version 1.4.0 at this baseline.

## Canonical surface

The canonical HTTP surface is the routes below. API_EN.md and API_ZH.md document synthesis and discovery, while V1_4_ARCHITECTURE.md lists the speech-session routes. The two API pages do not provide full session schemas, status codes, or the response-header contract.

| Method and route | Request | Successful response | Principal error statuses |
|---|---|---|---|
| GET /health | None | 200 JSON health snapshot | No route-specific error mapping |
| GET /v1/capabilities | None | 200 JSON capability snapshot | No route-specific error mapping |
| GET /v1/models | None | 200 JSON model list | No route-specific error mapping |
| GET /v1/voices | None | 200 JSON active voice list | No route-specific error mapping |
| GET /v1/expressions | None | 200 JSON expression catalog or unavailable marker | No route-specific error mapping |
| POST /v1/models/activate | JSON object with required model | 200 JSON activation result | 400 invalid/unknown model; 409 active synthesis or nonterminal session; 500 activation failure |
| POST /v1/audio/speech | JSON object; canonical structured form or compatibility form described below | 200 audio/wav or streamed application/octet-stream | 400 invalid request; 413 oversized body; 429 busy; 500 sanitized synthesis failure |
| POST /v1/audio/cancel | No body required | 200 JSON cancellation result | No route-specific error mapping |
| POST /v1/sessions | JSON object; fields below are optional | 201 speech.session JSON | 400 invalid request; 413 oversized body; 429 session limit |
| GET /v1/sessions/{session_id} | None | 200 speech.session JSON | 404 unknown session |
| POST /v1/sessions/{session_id}/segments | JSON segment object | 201 new segment; 200 exact duplicate | 400 invalid request; 404 unknown session; 409 conflict/closed session/model mismatch; 413 limits |
| GET /v1/sessions/{session_id}/audio | None | 200 raw PCM S16LE stream | 404 unknown session; 409 consumer already attached or terminal session |
| POST /v1/sessions/{session_id}/flush | None | 200 speech.session JSON | 404 unknown session |
| POST /v1/sessions/{session_id}/cancel | None | 200 speech.session plus active_audio_cancelled | 404 unknown session |
| DELETE /v1/sessions/{session_id} | None | 200 same result as session cancel | 404 unknown session |

For errors created by RequestError or SpeechSessionError, the JSON body is {"code": <HTTP status>, "message": <safe message>}; HTTP status matches code. _json_response adds Retry-After: 1 to every 429. Malformed JSON, non-object JSON, and most field validation errors return 400. The JSON parser enforces a 65,536-byte body limit and returns 413. Framework-level responses outside these handlers are not separately specified by API_EN.md/API_ZH.md. POST /v1/audio/cancel returns {"cancelled": boolean, "model": active_model_id}; it targets the process's current stream owner and can cancel that owner's speech session as well as a direct synthesis stream. With no active stream, cancelled is false.

## Discovery response schemas

### GET /health

TensorRTService.health returns these top-level fields: service, version, status, ready, backend, model, voice, cuda, tensorrt, gpu, engine_fingerprint, engine_count, engines, reference, expression, request_segment_char_limit, streaming, startup_warmup, active_requests, gpu_resource_lease, and warm_retention. RuntimeServiceManager.health adds switching and available_model_count and overrides status to switching during activation.

status is ok, not_ready, or switching; ready is boolean. backend is TensorRT-11. cuda, tensorrt, gpu, engine_fingerprint, startup_warmup, warm_retention, and GPU lease contents can be null or runtime-dependent. reference exposes only configured and language. streaming reports enabled, transport=pcm_s16le, and complete_wav=request stream=false. This is a runtime health payload, not a minimal stable client schema.

### GET /v1/capabilities

The response has service, version, model_family, backend, precision, languages, streaming, expression, pytorch_fallback, adaptive_punctuation_segments, warm_retention_seconds, and speech_sessions.

At this revision languages is [zh, yue, en, ja, ko]; backend is TensorRT-11; pytorch_fallback is false. streaming contains pcm16 and wav booleans. expression contains native, controlled_profiles, continuous_vector, and segmented booleans. speech_sessions contains committed_segments, pcm_stream, cancel, flush, context_mode, neural_state_continuity, acoustic_latent_continuity, default_policy, policies, qualification, and limits. Policies A-F are described under Session lifecycle.

streaming is scoped to stream=true encodings: pcm16=true and wav=false means incremental WAV streaming is unsupported. Complete WAV is available separately with stream=false; health.streaming.complete_wav also says request stream=false, and _tts_response rejects stream=true with a WAV response_format because the RIFF data length is not known in advance. The API pages could make this distinction explicit, but the capability value is consistent with the implementation.

### GET /v1/models

Returns {"object": "list", "data": [...]}. Each model item has id, object="model", owned_by="aniflive-tts-local", description, model_family, voice_profiles (array of identifiers), and active (boolean). It lists compatible packages discovered in the local registry; it does not return package paths.

### GET /v1/voices

Returns {"object": "list", "data": [...]}. The current implementation emits one active voice item with id, reference_language, and reference_configured. It does not expose reference audio, reference text, or filesystem paths.

### GET /v1/expressions

Returns {"object": "list", enabled, default, profiles, policies}, plus catalog metadata when enabled. With no catalog, enabled=false, default=null, profiles=[], and policies=[]. With a catalog, fields also include preferred_policy, playback_policy, and runtime_policy. Each public profile item has id, intensity_levels, languages, reference_languages, and preferred_policies. The profiles are package-dependent; callers must use this route/capabilities rather than assume a profile exists.

## Synthesis request and response

### Canonical structured request

POST /v1/audio/speech accepts the canonical fields:

| Field | Rules at this revision |
|---|---|
| model | Optional; defaults to the active model. A different model returns 400. |
| voice_profile | Optional; defaults to the active profile. A different profile returns 400. |
| text | Non-empty string, at most 1,000 characters. Must be supplied without segments. |
| segments | Non-empty array of at most 64 objects, used instead of text. Sum of segment text is at most 1,000 characters. Each object accepts text and optional expression only. |
| language | Required canonical code: zh, yue, en, ja, or ko. Canonical structured requests reject auto and auto_yue. |
| stream | Optional boolean; defaults false. The parser also accepts 0/1 and common true/false strings. |
| response_format | Optional. Defaults to wav when stream=false and pcm when stream=true. Non-stream accepts only wav; stream accepts pcm, pcm_s16le, or s16le. |
| expression | Optional object: enabled defaults false; profile optional; intensity defaults 0.5 and is in [0,1]; policy optional. |
| generation | Optional object. Recognized keys are top_k, top_p, temperature, seed, noise_scale, and speed. Bounds/defaults are below. |

Generation defaults and validation: top_k defaults 15 and must be an integer from 1 through 50; top_p defaults 1.0 and must be finite in [0.000001, 1.0]; temperature defaults 1.0 and must be finite and at least 0.000001; seed defaults -1 and must be an integer >= -1; noise_scale defaults 0.5 and must be finite in [0,10]; speed defaults 1.0 and, despite accepting positive numeric input during parsing, is rejected unless effectively 1.0 by this engine. The implementation also recognizes selected legacy top-level generation fields. pause_length defaults 0.440 and is bounded to [0,10]; cut_punc is limited to 64 characters; sample_steps must equal 32; if_sr must be false. These are runtime constraints, not a reason for the SDK to invent additional canonical fields.

Structured requests must contain exactly one of text or segments. A segment expression object accepts enabled, profile, intensity, and policy; omitted controls inherit the top-level expression values. The allowed policies are full-switch, identity-lock, semantic-style, acoustic-style, and sv-only. When omitted, the package preferred policy is used for an enabled profile, or full-switch when expression is disabled. Enabled expression on a package without expression profiles returns 400; an unknown profile also returns 400.

An expression switch between segments must follow a safe punctuation or paragraph boundary. The implementation accepts ASCII and CJK comma, period, semicolon, question, and exclamation marks, plus a blank-line boundary. Segment switches inside a phrase return 400. Text segments are coalesced when adjacent controls are identical, while their joined text is preserved.

The public voice reference is package-fixed. Alternate reference paths/text/languages, URLs, uploads, and extra inp_refs are rejected. _assert_fixed_reference checks that supplied legacy aliases match the configured identity reference; an exact match is accepted, but a different value is rejected. Canonical SDK requests should omit all reference override fields.

### Compatibility form on the same route

openai_speech selects the structured branch if the JSON object contains text, segments, voice_profile, or generation. Otherwise it treats input as the text field and accepts voice, model, text_lang, or text_language. voice is restricted to default or the active voice identifier. model must be active. If neither text_lang nor text_language is supplied, this branch defaults text_lang to auto; auto and auto_yue are accepted here. Supplying language alone does not select the structured branch, so an input request should use text_lang/text_language rather than relying on language.

Legacy GPT-SoVITS flat fields are compatibility inputs, not the canonical SDK schema. See exclusions.

### Synthesis responses and headers

stream=false returns a complete mono PCM16 WAV body with Content-Type audio/wav. There is no Content-Disposition header. Its current response headers are:

- X-TTS-Service, X-TTS-Version, X-TensorRT-Backend, X-TensorRT-Engine-Count, X-PyTorch-Fallback, X-TTS-Model
- X-TTS-Sample-Rate, X-TTS-Output-Samples, X-TTS-Segments, X-TTS-Inference-Seconds, X-TTS-Seed, X-TTS-Queue-Ms, X-TTS-Pause-Mode
- X-TTS-Expression, X-TTS-Expression-Policy, X-TTS-First-Context-Tokens
- X-TTS-Stage-Text-Seconds, X-TTS-Stage-GPT-Encoder-Seconds, X-TTS-Stage-GPT-Decode-Seconds, X-TTS-Stage-SoVITS-Seconds, X-TTS-Stage-Postprocess-Seconds, X-TTS-Semantic-Tokens, X-TTS-GPT-Steps, X-TTS-Sample-CUDA-Graph, X-TTS-SoVITS-Invocations

stream=true returns incremental mono signed 16-bit little-endian PCM chunks with Content-Type application/octet-stream. The current headers are X-TTS-Service, X-TTS-Version, X-TensorRT-Backend, X-TensorRT-Engine-Count, X-PyTorch-Fallback, X-TTS-Model, X-TTS-Stream, X-TTS-Sample-Format, X-TTS-Sample-Rate, X-TTS-Channels, X-TTS-Recommended-Prebuffer-Ms, X-TTS-Queue-Ms, X-TTS-Pause-Mode, X-TTS-Expression, X-TTS-Expression-Policy, X-TTS-First-Context-Tokens, and Cache-Control=no-store. X-TTS-Stream is pcm_s16le, X-TTS-Sample-Format is s16le, and channels is 1.

These additional TensorRT and profiling headers are emitted by service.py but are not enumerated by API_EN.md/API_ZH.md as stable wire fields. A client should require the media type/PCM format metadata it needs and tolerate absent or unknown diagnostic headers. TensorRT-specific metadata should remain optional in SDK types.

## Speech-session schema and lifecycle

### Creation

POST /v1/sessions accepts an optional model, voice_profile, and continuity_policy. model and voice_profile default to the active identities and must match them; mismatch returns 400. continuity_policy defaults to A and accepts A-F case-insensitively. A new session returns 201 with:

- id: generated sess_ identifier; object: speech.session
- state: open; model; voice_profile; sample_rate; format=pcm_s16le
- continuity_policy; context; audio_consumer_attached=false
- created_at and updated_at UTC timestamps
- segments: initially empty array

Defaults from SpeechSessionManager are maximum_open=8 sessions, ttl_seconds=600, maximum_segments=64, maximum_characters=16,000, and maximum_retained=64. GET /v1/capabilities exposes the first four limits. Reaching maximum_open returns 429 with Retry-After: 1.

### Appending segments

POST /v1/sessions/{session_id}/segments accepts segment_id (required), text (required), language (required core language), and optional paragraph_id, pause_after_ms, expression, and generation.

segment_id and paragraph_id, when present, must be 1-128 characters from letters, digits, dot, underscore, colon, and hyphen. text is subject to the synthesis 1,000-character limit; total session text is limited to 16,000 characters. pause_after_ms defaults to 0, must be an integer in [0,10,000]. The server stores segments in insertion order and returns a public segment object: segment_id, sequence (zero-based), characters, language, paragraph_id, pause_after_ms, state, created_at, updated_at, and error.

A new segment is queued and returns 201. Reusing a segment_id with the same request fingerprint returns the existing segment and 200. Reuse with different content returns 409. The fingerprint is SHA-256 over sorted compact JSON containing text, language, expression, generation, paragraph_id, and pause_after_ms. The raw text/language/expression/generation inputs are fingerprinted before synthesis normalization; retries should therefore resend the same JSON values and field presence.

Segments transition queued → streaming → completed. Cancellation/expiry marks queued or streaming segments cancelled. A synthesis failure marks the current segment failed, sets the session cancelled, and ends the raw PCM stream; the current audio response does not carry a structured error event.

### Audio consumer, flush, cancel, and delete

GET /v1/sessions/{session_id}/audio attaches the single allowed consumer and returns raw mono PCM S16LE with Content-Type application/octet-stream. A second consumer returns 409. It may be opened while state is open or flushing, but not after closed, cancelled, or expired. It waits for committed segments and continues waiting while the session is open and no segment is ready. If the HTTP consumer disconnects while a segment is actively being generated, the cancelled generator marks the session cancelled and stops that segment. If it disconnects while the generator is waiting for a future segment, the consumer detaches and the session remains open for a later consumer.

POST /v1/sessions/{session_id}/flush changes open to flushing and prevents new appends. The consumer drains queued segments in order; when no queued segment remains, the audio iterator changes flushing to closed and ends the stream. Calling flush does not itself return the audio. Repeated flush calls return the current session state; only open transitions to flushing. POST /cancel sets cancelled, clears retained context, and stops queued/active work. Repeated cancel calls on terminal states return that existing state. DELETE calls the same cancel handler; it is a cancel alias, not removal of the session record. GET remains available for terminal state until the record is pruned.

Avoid a lifecycle deadlock in SDK examples: fully reading the audio body before calling flush can wait indefinitely because the open session keeps the stream alive. For finite batch synthesis, append all segments, call flush, then GET and drain audio to EOF. For live synthesis, consume GET audio in a task while appending segments, then call flush concurrently after the final append and continue draining.

The audio stream currently sends X-TTS-Service, X-TTS-Version, X-TensorRT-Backend, X-TensorRT-Engine-Count, X-PyTorch-Fallback, X-TTS-Model, X-TTS-Stream=pcm_s16le, X-TTS-Sample-Format=s16le, X-TTS-Sample-Rate, X-TTS-Channels=1, X-TTS-Session-ID, X-TTS-Session-Context=committed-neural-v1, X-TTS-Session-Context-Policy, X-TTS-Neural-State-Continuity, X-TTS-Acoustic-Latent-Continuity=false, X-TTS-Continuity-Qualification=experimental-unqualified, Cache-Control=no-store, and X-Accel-Buffering=no.

The session JSON context object has mode, policy, neural_state_continuity, acoustic_latent_continuity, committed_segments, previous_language, previous_expression, paragraph_id, previous_pause_ms, previous_output_samples, total_output_samples, hard_boundary, retained_phonemes, retained_semantic_tokens, retained_neural_bytes, and last_plan. last_plan is null or contains configured_policy, effective_policy, reason, phones, semantic_tokens, and acoustic_latent=false. Tensor shapes/retained byte counts are diagnostics; no acoustic latent is retained.

A-F continuity policy mapping from ContinuityPolicy and CONTINUITY_POLICY_CONTRACTS:
- A: no neural history; compatibility default.
- B: previous phones/BERT.
- C: B plus at most 32 semantic tokens.
- D: B plus at most 64 semantic tokens.
- E: expression-aware; caps expression-switch context to 12 phones and 48 tokens.
- F: boundary-adaptive choice based on paragraph, pause, language, punctuation, and expression changes.

The implementation caps retained state at 75 phones and 64 semantic tokens. docs/V1_4_ARCHITECTURE.md and docs/V1_4_IMPLEMENTATION_STATUS.md mark B-F experimental/unqualified; docs/V1_4_CONTINUITY_DIAGNOSTICS.md says its reports are diagnostic only and not release qualification. Capabilities and audio headers also label qualification experimental-unqualified. Keep A as the safe default; do not promise B-F quality.

## Model activation

POST /v1/models/activate requires {"model": <safe local model identifier>}. It resolves only a compatible package in the local registry. Success returns HTTP 200 with code=0, message=Success, changed, model, voice, backend=TensorRT-11, and pytorch_fallback=false. Activating the already active ready model returns changed=false. A new activation is serialized against session creation. It returns 409 while synthesis is active or any speech session remains nonterminal; it clears retained session neural context before a model transition. Unknown/unsafe model IDs return 400. Runtime loading failure returns a sanitized 500 message. The existing synthesis URLs remain unchanged.

## Compatibility and exclusions for the SDK surface

Canonical routes are the /health, /v1/capabilities, /v1/models, /v1/voices, /v1/expressions, /v1/models/activate, /v1/audio/speech, /v1/audio/cancel, and /v1/sessions routes listed above.

The implementation also registers these compatibility/legacy routes; they are outside the canonical SDK surface:

- GET / and POST /: GPT-SoVITS flat synthesis schema. GET uses query parameters; POST uses JSON. These call the same synthesis runtime.
- GET /model/config: older model metadata payload.
- GET/POST /change_refer: compatibility stub validates the configured fixed reference and returns success; it does not install a caller-selected reference.
- GET/POST /set_model: compatibility stub always reports that runtime switching is disabled. Use POST /v1/models/activate for registry-based activation.
- The input/voice branch on POST /v1/audio/speech is a compatibility adapter on a canonical URL, not the canonical typed request shape. It alone defaults the language to auto and accepts auto_yue.
- Local path, URL, file upload, extra reference audio, custom reference text/language, direct model paths, and selectable reference assets are not supported as canonical SDK operations. Reference inputs that exactly equal the configured fixed reference may pass the legacy validation guard but do not select a new reference.

Other workstation, training, evaluation, worker, or internal routes are not part of this public inference API. Do not derive client SDK methods from unrelated /api/workstation routes.

## Unknown-field and forward-compatibility behavior

There is no single server-wide additional-properties policy:

- JSON request bodies must be objects, but the parser has no schema-level unknown-field rejection.
- Canonical top-level synthesis fields not consumed by normalization are generally ignored. Unknown keys in generation and top-level expression objects are also ignored. Some known legacy top-level fields are still interpreted.
- A segment object rejects fields other than text and expression; its nested expression rejects fields outside enabled, profile, intensity, and policy.
- Session creation, session append, and model activation consume selected keys and ignore unrelated body keys.
- Response objects are emitted as ordinary dictionaries without strict response models. This source alone does not guarantee a stable additive-field policy, and tests do not verify unknown response-field tolerance. The Rust SDK should ignore unknown response keys for forward compatibility, while validating required fields and value types.

## Evidence and unresolved contract items

Relevant implementation symbols in src/aniflive_tts/service.py: openai_speech, _structured_speech_values, _options_from_values, _tts_response, _read_json_body, create_speech_session, append_speech_session_segment, stream_speech_session_audio, flush_speech_session, cancel_speech_session, delete_speech_session, capabilities, health, list_expressions, cancel_audio, list_models, activate_model, and list_voices. Session state and serialization live in speech_session.py: SpeechSession.public_dict, CommittedSpeechSegment.public_dict, SpeechSessionManager.append, attach_consumer, next_segment, finish_segment, flush, and cancel. Policy values live in continuity.py: ContinuityPolicy and CONTINUITY_POLICY_CONTRACTS. Expression schema values live in expression.py: ConditioningPolicy and ExpressionCatalog.public_metadata.

The API pages are docs/API_EN.md and docs/API_ZH.md. Supporting authorities are docs/V1_4_ARCHITECTURE.md (session paths and experimental status), docs/V1_4_ACCEPTANCE.md (session acceptance behavior), docs/V1_4_IMPLEMENTATION_STATUS.md (B-F remain experimental), and docs/V1_4_CONTINUITY_DIAGNOSTICS.md (diagnostics do not qualify release behavior).

Contract coverage in tests/test_api_contract.py includes test_capabilities_reserve_expression_interface, test_speech_session_streams_committed_segments_and_is_idempotent, test_speech_session_policy_is_explicit_and_invalid_policy_fails_closed, test_speech_session_rejects_conflicting_duplicate_and_model_switch, test_session_cancel_uses_owner_scoped_stream_cancellation, test_runtime_stream_owner_prevents_cross_session_cancel, test_speech_session_response_detaches_consumer_before_body_iteration, test_speech_session_response_consumer_detach_survives_caller_cancellation, test_expression_enabled_rejects_package_without_profiles, test_segmented_expression_request_is_structured_and_inherits_defaults, test_expression_switch_rejects_mid_phrase_segments, test_canonical_api_rejects_text_and_segments_together, test_stream_transport_declares_little_endian_pcm, the JSON size-limit tests, test_api_hides_reference_and_runtime_paths, and test_canonical_api_requires_one_of_five_core_language_codes. The file does not exercise mid-segment disconnect cancellation or the open-session wait/flush sequence, assert every response header, or define a universal unknown-field policy.

Documentation and test follow-ups (no server behavior change is implied):
1. Clarify in API_EN.md/API_ZH.md that capabilities.streaming.wav describes incremental streaming support; complete WAV is requested with stream=false.
2. Document the current DELETE-to-cancel alias and terminal-state idempotency for sessions.
3. Document and test the current selective request unknown-field behavior; preserve the observed behavior unless a separate compatibility decision changes it.
4. Expand public API documentation and tests for discovery response shapes, complete WAV metadata, Retry-After, active-segment disconnect cancellation, and the open-session wait/flush sequence.
