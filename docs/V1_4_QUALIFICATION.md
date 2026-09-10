# v1.4 production qualification

AnifLive-TTS v1.4 composes production qualification from four immutable JSON
artifacts. A manually written `aniflive-tts-qualification-v1` report is not
eligible for import or promotion.

## Dataset acquisition is an upstream gate

Target-speaker acquisition has its own machine-readable qualification before a
dataset can be frozen. The current offline Linux Docker run processed 1862.58
seconds and passed its recorded labelled-fixture safety gates. Its public,
asset-free summary is
`benchmarks/V1_4_VOICE_ACQUISITION_QUALIFICATION.json`.

An acquisition `passed` result means that source decoding, VAD, generic speaker
verification, diarization routing, separation, transcription and lineage gates
completed without neural fallback. It does **not** accept transcripts, approve
review candidates, freeze a dataset, train a model, build TensorRT engines or
qualify a package. The report must remain `waiting_for_review` until a human has
reviewed every required audio and transcript decision. Production qualification
below begins only after that reviewed dataset is frozen and its model artifacts
exist.

## Required sources

1. `aniflive-tts-workstation-evaluation-v1`
   - Produced by the Linux TensorRT evaluation worker.
   - Supplies multilingual content, speaker identity, streaming parity,
     TensorRT runtime, TTFA, and RTF evidence.
   - Must directly reference the artifact being qualified.
2. Two separate `aniflive-tts-blind-ab-evidence-v1` artifacts
   - One for `long-form-continuity`.
   - One for `expression-quality`.
   - Each records an explicit human `passed` or `failed` decision, blinded
     protocol, trial count, listener count, and the sample and randomization
     manifest hashes.
3. `aniflive-tts-security-verification-v1`
   - Produced only after the release security command passes every required
     check.

The composer reopens every registered file, verifies its registry SHA-256,
validates its schema and subject, derives the eight production gates, writes a
new `aniflive-tts-qualification-v1` artifact, and immediately re-verifies the
composition before importing it. Promotion repeats these checks. A changed or
missing nested source invalidates promotion.

## Recording blind A/B decisions

The blind test harness must first preserve the sample manifest and its hidden
randomization manifest. After listening is complete, record the human decision:

```powershell
python scripts/record_blind_ab_evidence.py `
  --output D:\evidence\long-form.json `
  --subject-kind artifact `
  --subject-id artifact_00000000-0000-4000-8000-000000000000 `
  --gate long-form-continuity `
  --decision passed `
  --completed-trials 20 `
  --listener-count 2 `
  --sample-manifest-sha256 <sha256> `
  --randomization-sha256 <sha256> `
  --operator listener-panel-1 `
  --summary "No candidate regression detected in the blinded long-form set"
```

Run it separately for `expression-quality`. The command refuses to overwrite an
existing artifact. Automated evaluation never creates either decision.

## Producing security evidence

```powershell
python scripts/check_release_security.py `
  --include-untracked `
  --evidence-output D:\evidence\security.json `
  --subject-kind artifact `
  --subject-id artifact_00000000-0000-4000-8000-000000000000
```

No security artifact is written when a check fails.

## Register and compose

Register each source as a ready `evaluation` artifact. Give the three explicit
evidence files these metadata values so the Evaluation Lab can place them in the
correct selectors:

- `long-form-blind-ab`
- `expression-blind-ab`
- `security-verification`

Use `automated-evaluation` for an imported worker report when it does not already
have worker metadata. The metadata key is `qualification_evidence_kind`; it is a
UI classification only. The backend trusts the file schema, subject, registry
identity, and checksum instead.

The Evaluation Lab's **Compose verified gates** action calls
`POST /api/workstation/qualifications/compose`. It stays disabled until a ready
subject and all four distinct sources are selected. A failed blind decision or
security check produces a failed qualification, never an automatic pass.
