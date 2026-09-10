# AnifLive-TTS v1.4 Implementation Status

Date: 2026-09-10

## Current Status

The v1.4.0 local release candidate has completed the approved functional-smoke
acceptance, human audio review, real training continuation, GPU handoff and
local installation checks. The cu128 runtime was exercised on the acceptance
GPU; cu126 remains build/static-audit only. Each new voice still needs its own
qualification. Release assets and checksums remain frozen; subsequent README
preview and documentation updates apply to the repository only.

## Historical Implementation Snapshot (2026-09-08)

Production qualification was incomplete at the time of this snapshot.

- Training input v2 separates train, validation and sealed test. Transcript and
  speaker verification are digest-bound; audio acceptance alone is insufficient.
- The production chain now includes validation checkpoint selection, a new
  blinded human reference lock, one-time holdout evaluation and conversion
  parity before packaging. See `V1_4_TRAINING_PIPELINE.md`.
- The calibrated speaker retention limits are `0.84/0.82`, with generated
  absolute floors `0.72/0.64`. Content and audio hard gates remain required.
- The candidate run passed checkpoint selection under the calibrated contract.
  Its 1,996 retained artifacts exceeded the old transport budget; verified
  handoff recovery preserves the original result and creates a distinct retry.
  The human reference lock and one-time sealed holdout subsequently passed.
  Conversion qualification remains required.
- Broker and registry share a 4,096-file checkpoint-selection budget; other
  job types retain 1,024. Job results have a bounded 256 KiB JSON budget;
  other JSON fields retain 64 KiB. Recovery accepts only verified transport
  failures and matching retry ancestry; it cannot bypass quality or sealed-test gates.
- New lineage leaves use an indexed cycle-check fast path. Self-links and
  cycles involving existing nodes still fail in both Python and SQLite.
  A deep-DAG regression reproduces the old repeated traversal, and migration
  tests preserve existing lineage.
- Normalizer loading now preserves the declared source-root priority even when
  those roots are already in PYTHONPATH or another text module was imported.
  Two regression cases and real five-language Docker checks cover the fix.
- The complete Linux installed-worker suite, including Node-based frontend
  tests, passes **966 tests, with 2 skipped**. The two skipped cases require
  Windows-specific rejection/junction semantics. CPU tests do not prove
  TensorRT quality or GPU handoff acceptance.
- The current local worker includes the normalizer, evidence-envelope and
  ONNX validation fixes. Its immutable image passed the installed-worker suite.
  Switching workers must not interrupt an active job.
- The existing candidate passed all 89 PyTorch-to-ONNX output checks, including
  eight fixed cases for each GPT stage, and exported all nine deployment ONNX
  files. The same candidate then built and imported all nine TensorRT 11
  engines successfully. ONNX/TensorRT greedy semantic sequences agree exactly,
  but complete-audio parity currently fails spectral, speaker and some duration
  gates. The comparator uses different sampling/decoding contracts; passive
  stage-input diagnostics are being collected before any gate decision changes.
  Packaging and latency qualification remain blocked by conversion parity.
- Passive decoder-input evidence isolated a speaker frontend mismatch: the old
  export omitted native Kaldi DC removal and pre-emphasis and used different
  frame, window and mel-bank definitions. The repaired frontend matches native
  E-reference embeddings at cosine > 0.999999999999, with unchanged weights.
  Native-oracle and dynamic-length ONNX regressions pass. A new conversion of
  the same locked candidate is in progress; old engine evidence is retained.
- Conversion's PyTorch semantic reference now uses the exported-stage sampling
  and EOS contract. The previous native training loop used a different sampler
  and repetition policy, so equal seeds did not define equal inference inputs.
  This affects conversion QA only; selection and consumed holdout are unchanged.
- The worker image now removes the CPU ONNX Runtime distribution introduced by
  ASR dependencies and restores the existing GPU 1.22.0 pin. The build checks
  CUDA-provider availability; actual ONNX CUDA and TensorRT semantic execution
  succeeded, and faster-whisper VAD still loads with the same distribution.
- Broker progress forwarding now includes checkpoint, reference, holdout and
  conversion jobs, including the first labelled stdout/stderr event. Twelve
  Docker broker cases cover the six progress-producing job types.
- English, Traditional Chinese and Simplified Chinese guides now document
  verification, checkpoint selection, blind references, sealed holdout and
  parity gates.
- Final in-app-browser interaction and visual verification remains outstanding.
  A browser-tool startup failure prevented a fresh inspection this session.

The implementation and measurements below are retained as the **2026-09-01
historical snapshot**. They must not override the current status or be
presented as final validation of subsequent source changes.


## Release Decision

**IMPLEMENTATION IN PROGRESS; RELEASE QUALIFICATION PENDING.** This branch
remains an isolated v1.4 workspace and must not be published until Dataset
Factory acquisition, audio quality and repeated GPU-handoff gates are recorded.
The v1.3 repository and inference baseline remain unchanged.

## Implemented And Tested

- **AnifLive-TTS Studio** at port 9891 with Overview, Synthesis, Expressions,
  Datasets, Training, Evaluation, Models, Engines, Jobs and Settings. Advanced
  TSE diagnostics remain internal to Dataset Factory.
- The preserved **AnifLive-TTS WebUI** remains a separate lightweight synthesis
  surface at port 9890. Studio embeds the same v1.3 synthesis behavior instead
  of replacing it.
- Installable Studio PWA assets, offline shell, visible install action and
  foreground `run_studio.bat` / `run_webui.bat` launchers.
- Full-viewport Mixkit grayscale motion background, restrained glass depth,
  responsive module navigation and inline page transitions without an overlay
  card. The Overview product title can render only as one line or as
  `AnifLive-TTS` followed by `Studio`; no line break may occur inside the
  `AnifLive-TTS` product name.
- Preserved v1.3 segmented expressions, expression tags and captions,
  PCM streaming, cancellation, WAV download, audio-aligned gold playback text,
  five languages, eight request metrics and request history.
- Transactional SQLite control plane with projects, jobs, dependency graphs,
  events, priorities, pause/resume/cancel/retry, resource leases, expression
  drafts, qualification evidence and immutable artifact lineage.
- Dataset Factory supports standard media, target-speaker acquisition and
  existing GPT-SoVITS `.list` projects. Target acquisition now uses managed
  FSMN-VAD, a multi-region reference prototype, a generic workstation TensorRT
  speaker verifier, two-pass Sortformer evidence, conservative SNR routing,
  MossFormer2 salvage, per-clip source lineage, SenseVoice suggestions,
  mandatory review, expression candidates and immutable freeze before Training
  handoff. No voice package or voice name participates in acquisition routing.
- Long regions prefer silence-validated cuts. A non-silence forced split is
  quarantined for review. Rejected clips and both original/processed salvage
  candidates remain available with stable IDs and checksums instead of being
  deleted or renamed from transcripts.
- V2ProPlus training from validated dataset inputs through deterministic GPT
  and SoVITS checkpoints, immutable deployment checkpoint manifests and safe
  loading of the pinned upstream `HParams` type.
- Generic V2ProPlus checkpoint conversion to FP16 ONNX, all nine TensorRT 11
  engines, package checksum/fingerprint validation and five-language evaluation
  job wiring. No voice-name branch exists in conversion or worker execution.
- Evaluation Lab evidence composition, blind A/B evidence import, security
  evidence import and promotion gates that fail closed when evidence is absent,
  changed or mismatched.
- Speech Session API for committed, ordered segments with idempotency,
  cancellation, flush, one audio consumer, bounded state and model binding.
- Continuity policies A-F with bounded phones/BERT/exact-semantic state.
  Policy A remains the compatibility default; B-F stay opt-in experiments until
  their listening qualification is complete.
- Digest-pinned Linux Docker worker broker with source-owned commands,
  read-only inputs, isolated outputs, `--network none`, read-only root,
  capability drop, no-new-privileges and strict result/artifact verification.
- Process-owned `gpu:0` lease shared by resident inference and GPU workers.
  Workers wait with a persisted reason instead of overlapping inference VRAM;
  uncertain container cleanup quarantines the lease until reconciliation proves
  that the container is absent.

## Deliberate Boundaries

- Dereverb remains disabled because no redistribution-safe backend has passed
  the v1.4 quality gate. Selecting a non-`none` backend fails clearly.
- Offline ASR output is integrity-checked but remains mandatory-review data; it
  is not silently accepted as a training transcript.
- The adaptive Sortformer policy passed the labelled 31-minute acquisition
  fixture, including stationary overlap examples. This qualifies the recorded
  workload, not every voice or acoustic condition; uncertain evidence still
  routes to human review and operator-confirmed ranges remain available.
- Continuity policies B-F are executable research policies, not production
  defaults. Acoustic latent continuity is neither implemented nor claimed.
- GPU workers safely wait for inference to release its resident lease. Automatic
  inference drain/unload/worker/reload orchestration is outside this snapshot.
- A one-epoch training smoke model proves workflow execution, not speech
  quality. Formal multilingual evaluation requires a release-candidate model
  and the configured offline ASR assets.

## Qualification Checklist Recorded on 2026-09-08

This earlier checklist is historical. Final acceptance used the subsequently
approved functional-smoke scope, without exhaustive continuity matrices or
a two-hour soak.

1. Complete human audio/transcript review of the 31-minute acquisition result,
   freeze the dataset, then execute Training -> TensorRT build -> package ->
   evaluation. The automatic acquisition pass is not a package qualification.
2. Run fixed-seed v1.3/v1.4 synthesis parity and the canonical five-language
   evaluation on a release-candidate V2ProPlus package in Linux Docker.
3. Complete long- and short-form blind listening for expression quality and the
   A-F continuity matrix; record immutable evidence artifacts.
4. Repeat inference-to-worker-to-inference GPU handoffs while checking Docker
   cleanup, NVIDIA process ownership and SQLite WAL integrity.
5. Re-run the security/privacy scan and generate release checksums after the
   candidate file set and qualification evidence are frozen.

## Historical Validation Snapshot (2026-09-01)

- Complete repository test suite in the final offline Linux worker environment:
  **776 passed, 19 skipped**.
- Focused Dataset Factory, acquisition, TSE, workstation and worker regression
  suite: **307 passed, 15 skipped**; the sandbox-blocked FFmpeg video decode
  case passed when rerun with process execution enabled.
- Focused Dataset Factory frontend/PWA regression suite: **27 passed, 3
  skipped**.
- Release security scan passes across **468 tracked and candidate untracked
  files** at the exact `1.4.0.dev0` version.
- Fatal Ruff checks pass for the v1.4 application and newly added workstation
  tests. JavaScript syntax passes for all shipped Studio and WebUI modules.
- Desktop, tablet and mobile layouts were inspected with the Codex in-app
  Browser. All eleven Studio modules fit without page-level horizontal overflow;
  the Overview title and mobile module sheet pass their responsive contracts.
- Live HTTP smoke checks pass for both Studio and the preserved WebUI, PWA
  assets, offline shell and every read-only workstation control-plane endpoint.
  `/api/status` correctly remains unavailable while no inference runtime is
  attached; this is an explicit disconnected state rather than a Studio fault.
- The worker image used for that historical snapshot was pinned as
  `aniflive-tts-workstation-worker@sha256:06976aa57b82419fbaee9cf76d778306b91623b39eb50443733edb53f0213427`.
  On the RTX 5070 Ti it reports PyTorch 2.10.0+cu128, TensorRT 11.2.1.2 and a
  working CUDA device.
- A real Studio API job traversed scheduler, GPU lease, final digest-pinned
  Docker broker and artifact handoff. The offline container completed 32 kHz
  dataset processing in 4.563 seconds, returned four checksum-verified artifacts
  and left no managed container behind.
- A separate offline Linux Docker target-speaker qualification processed
  `1862.58 s` of labelled source audio into `363` stable records. The recorded
  policy achieved `1.0` overlap-containment recall, zero contamination routed
  directly to clean, zero non-target clips routed directly to clean and zero
  target clips auto-rejected. The final status is deliberately
  `waiting_for_review`; see
  `benchmarks/V1_4_VOICE_ACQUISITION_QUALIFICATION.json`.
- The Mixkit background video and derived poster have recorded provenance,
  checksums and license text. No background asset is fetched at runtime.
