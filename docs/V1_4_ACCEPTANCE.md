# AnifLive-TTS v1.4 Acceptance

## Baseline

- v1.3 inference output and public API behavior are the golden baseline.
- TensorRT 11 remains the only production neural runtime.
- No training, TSE or dataset worker may import into the inference process.
- Existing v1.0–v1.3 model packages remain valid.

## Synthesis

- All v1.3 Synthesis controls and metrics remain present.
- Audio-aligned playback highlighting must use measured PCM activity rather
  than elapsed-time interpolation.
- Segmented expression text must round-trip without missing, duplicated or
  reordered characters.
- Cancellation releases the active TensorRT request.
- Desktop and mobile layouts have no horizontal overflow or incoherent overlap.

## Sessions

- Stable segment IDs are idempotent.
- Reusing a segment ID with different content returns a conflict.
- A session binds one active model and voice profile.
- Only one audio consumer may attach.
- Flush drains queued segments in order and closes the PCM stream.
- Cancel stops active synthesis and clears retained context.
- Model activation is rejected while a session remains open.
- Session TTL and maximum-open limits are enforced.
- Neural continuity may be enabled only after fixed-seed semantic and PCM parity
  pass on all five languages and long-form listening passes.

## Workstation

- The product surface is named `AnifLive-TTS Studio`; the lightweight preserved
  surface is named `AnifLive-TTS WebUI`.
- At every viewport width the Overview title is either a single line or breaks
  exactly between `AnifLive-TTS` and `Studio`. It may not break inside the
  `AnifLive-TTS` product name.
- Studio is installable as a PWA and its offline shell contains no runtime model,
  reference recording, account state or credentials.
- Projects and jobs survive process restart.
- Job transitions are validated.
- Project status, progress and job-count metrics are derived transactionally
  from their jobs; they may not remain at a cosmetic `draft` or `0%` after work
  is queued, running, cancelled, failed or completed.
- Dataset paths cannot escape configured import roots.
- Dataset, TSE, training, evaluation, engine and package jobs use explicit
  adapters and resource requirements.
- GPU-exclusive jobs cannot overlap with resident inference; both must use the
  same fail-closed GPU lease.
- Artifact lineage records dataset, checkpoint, expression bank, engine
  fingerprint, evaluation and package checksum.
- A TSE overlap extraction is successful only when blind separated candidates
  are selected by the workstation-owned TensorRT speaker verifier, every output
  window passes the target and ambiguity thresholds, output is finite and
  exact-length, and the matched mixture improves SI-SDR by at least 3 dB.
- TSE source revision, model revision and checkpoint checksum must be present in
  the worker report. The separator checkpoint remains external and read-only.
- Operator-confirmed separation ranges must be sorted, non-overlapping, finite,
  bounded by the decoded source, converted to exact sample bounds, and recorded
  with per-segment trigger provenance. Invalid ranges fail the worker before
  neural execution.
- Automatic overlap coverage may be claimed only after stationary as well as
  identity-changing overlaps pass a labelled detector evaluation; successful
  source separation alone does not satisfy the detector gate.
- Dataset, training, engine, package and evaluation jobs must execute through
  the digest-pinned Linux Docker broker; a preparation manifest by itself does
  not satisfy functional acceptance.
- A successful worker result must be imported only after every regular output
  file is reopened on the host and its size and SHA-256 are reverified.

## Security And Privacy

- Public assets contain no credentials, private reference audio, model weights,
  checkpoints, transcripts or local absolute paths.
- The WebUI binds to loopback by default. A non-loopback bind requires an
  explicit operator override and a matching trusted host configuration.
- Every WebUI request rejects untrusted `Host` authorities. State-changing
  requests reject cross-origin and `Sec-Fetch-Site: cross-site` callers.
- Routes that parse a JSON mutation require `Content-Type: application/json`;
  no-body cancellation and worker-control routes remain available.
- User text is rendered with text nodes or form values.
- Workstation file operations use validated local paths.
- Model packages retain checksum and contained-path validation.

## Release Gate

- Existing test suite passes.
- New workstation, session, path-containment and compatibility tests pass.
- Synthesis A/B output matches the v1.3 baseline at fixed seeds.
- Long and short multilingual listening passes.
- Training/TSE workers survive cancellation and restart.
- Inference and the host worker resolve the same host-mounted workstation
  directory. A GPU worker stays queued with an explicit wait reason while
  inference is resident, starts only after inference releases `gpu:0`, and
  repeated handoffs leave no overlapping NVIDIA process or SQLite corruption.
- Closing the foreground workstation lifecycle gives Docker cleanup its full
  grace window and leaves no managed GPU worker container behind.
- Desktop and mobile visual verification passes.
- The v1.3 repository remains unchanged.
