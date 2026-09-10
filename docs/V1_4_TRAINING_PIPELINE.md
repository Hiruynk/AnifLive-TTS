# AnifLive-TTS v1.4 V2ProPlus Training Pipeline

The v1.4 workstation prepares and trains GPT-SoVITS V2ProPlus models only in
the pinned Linux CUDA worker. Windows is the control plane; neural
preprocessing and training do not run on the host.

## Input Contract

The preprocessing list contains one UTF-8 record per clip:

```text
clip.wav|speaker-id|ja|transcript
```

Supported language identifiers are `yue`, `zh`, `ja`, `en`, and `ko` (`JP` is
accepted as the upstream-compatible alias for `ja`). Audio basenames must be
unique because GPT-SoVITS keys all derived assets by basename.

Dataset Factory freezes reviewed train/validation/test splits and materializes
an immutable `aniflive-v2proplus-training-input-v2` bundle:

```text
training-bundle/
  train/voice.list + wav/
  validation/manifest.json + wav/
  test/manifest.json + wav/
  frozen-manifest.json
  training-input.json
```

The default split is 85/10/5, deterministic and group-aware. With at least
40 accepted clips, validation and test each contain at least five clips.
Audio acceptance, verified transcript and speaker digests, and a split are
required; expression verification is required only when the project requires
expressions. Editing a verified value invalidates its verification. Older
frozen datasets without this evidence remain readable lineage records but
cannot start new production training.

Each manifest records stable item IDs, file SHA-256, text, language, speaker,
quality and verification evidence. Training subprocesses see only the train
list. Checkpoint selection reads validation only. Test stays sealed until both
the validation checkpoint winner and the human reference decision are locked.
`Continue to Training` creates the project from this bundle without a manual
path handoff.

Studio's AI Components manager supplies checksum-pinned, read-only paths for:

- the BERT and HuBERT model directories;
- the speaker-verification checkpoint;
- the GPT and SoVITS V2ProPlus starting checkpoints; and
- the V2ProPlus discriminator checkpoint.

The locked component is installed separately from the repository and worker
image. The worker remains network-disabled and verifies the immutable input
bundle and every component before neural preprocessing starts. Manual projects
may still use equivalent local, read-only assets through the Advanced UI.

## Strict Preprocessing

The training worker invokes the installed preprocessing module automatically
when the selected dataset is a frozen Dataset Factory bundle. The equivalent
low-level command is:

```bash
python -m aniflive_tts.workstation_preprocessing \
  --input-list /input/train/voice.list \
  --wav-dir /input/train/wav \
  --output /output \
  --pretrained-sovits-g /models/voice.pth \
  --bert-dir /shared/chinese-roberta-wwm-ext-large \
  --hubert-dir /shared/chinese-hubert-base \
  --speaker-model /shared/sv/pretrained_eres2netv2w24s4ep4.ckpt
```

The command invokes the four scripts from the checksum-pinned GPT-SoVITS
source revision. It validates exact per-clip outputs after every stage. In
particular, speaker-vector generation must create one `.pt` file for every
input row. The upstream script catches per-file exceptions and can otherwise
exit with status zero after producing nothing; AnifLive-TTS treats that state
as a hard failure.

Torchaudio 2.10 uses the pinned TorchCodec 0.10 loader in this worker. The
image build decodes a real 32 kHz probe WAV through `torchaudio.load()` so an
unusable codec backend cannot pass the image gate.

V2ProPlus exports use the standard `06` two-byte model header. For the
upstream semantic and training scripts, the worker creates a mode-0600 private
working copy with the regular torch archive `PK` header. The source checkpoint
is never modified, the source digest is rechecked, and the working copy is
removed after the job.

## Training Gate

The training adapter validates that text and semantic rows are aligned and
that the speaker-vector directory exactly matches them before it launches a
subprocess. Both pinned GPT-SoVITS import roots are always present in
`PYTHONPATH`:

```text
/opt/aniflive-tts/gpt-sovits
/opt/aniflive-tts/gpt-sovits/GPT_SoVITS
```

A successful job must produce the requested deployable GPT and/or SoVITS
checkpoint. Neural subprocesses execute with a private writable job directory;
the pinned GPT-SoVITS source and all input mounts remain read-only. The worker
result records checksums for configurations, logs, deployable weights, and
resume checkpoints; missing outputs fail the job.

Checkpoint retention separates resumable optimizer state from deployable
candidates. The latest resume state is retained; GPT and SoVITS candidates
saved at the configured epoch cadence remain available for validation
selection. A successful two-stage run publishes a checksummed candidate
manifest. The final epoch is not automatically the deployment winner.
Partial GPT-only or SoVITS-only runs are not production-ready pairs.

## Pause And Resume

With the checkpoint-aware Docker controller and worker, Studio's Pause action
waits for the next complete epoch checkpoint. The job remains running while
the checkpoint is written and verified; the job log explains the wait.
Pausing during preprocessing therefore waits until the first training
checkpoint. Cancel remains available to stop the worker immediately.

Resume creates a new continuation job linked to the paused attempt. The
original attempt and its evidence remain visible. The continuation restores
the saved dataset features, framework state, candidate weights and resolved
training budget; it does not recalculate epoch totals or start from pretrained
weights. Repeating Resume returns the same continuation instead of scheduling
another copy. Missing, corrupt or ambiguous checkpoints block continuation
with an error.

GPT recovery includes optimizer, scheduler, RNG and pending accumulated
gradients. SoVITS recovery includes both generator and discriminator states,
optimizers, schedulers, AMP and RNG. A checkpoint is an epoch boundary, not a
save of an arbitrary incomplete batch. Complete published snapshots retain
the latest two recovery points. These files currently live under the local
workstation job output; preserve that directory while an attempt is paused.
Recovery does not promise bit-identical SoVITS results across processes.

## Validation Checkpoint Selection

Selection first sweeps GPT candidates using a temporary latest SoVITS
checkpoint, then SoVITS candidates using the provisional GPT winner. It
evaluates the top three from each sweep as at most nine joint pairs on full
validation, with seeds `1234`, `2026`, and `7`.

Each train-only probe reference is evaluated independently. A pair must have
at least one reference passing every hard gate; failed references are retained
in the report rather than averaged into a successful reference. Evidence
filenames include the reference index. Ranking considers identity, content
error, stability and earlier epochs only after hard gates pass.

The source-calibrated speaker contract uses median/P10 retention limits of
`0.84/0.82`, with generated absolute safety floors of `0.72/0.64`.
Qualified production source/control measurements motivated these limits;
there are no model-name branches. The report imports runtime constants
directly. Content, generation-success and audio-integrity gates still apply.
This validation result does not replace holdout evaluation or human listening.

## Reference Lock And Sealed Holdout

Reference candidates come only from verified train clips and must meet
duration, quality, silence, clipping and route requirements. Target-speaker
projects exclude separated salvage from deployment-reference selection by
default. The robust speaker centroid contributes to candidate ranking.

The reference worker prepares five candidates using a fixed sentence set.
Studio presents blinded audio choices without the automatic ranking or item
mapping. A human decision is required for the current blind manifest; a choice
from an earlier run cannot approve a new run.

Only after checkpoint and reference locks may `holdout.evaluate` first open
test. The test is consumed once. Failure records NO-GO and blocks engine
building; it must not be reused to select another checkpoint.

## Deployment Chain

```text
training.prepare
  -> checkpoint.select       (validation only)
  -> reference.select
  -> human reference lock
  -> holdout.evaluate        (sealed test, once)
  -> engine.prepare          (9 TensorRT 11 engines)
  -> conversion.parity
  -> model.package
  -> evaluation.prepare
  -> qualification
  -> promotion
```

Successful training queues checkpoint selection. Successful reference
selection waits for human evidence. Later jobs are queued only after their
required gates pass. Retrying a failed selection creates a new attempt that
reuses the existing training dependency and preserves the original evidence.
It does not require another training run.

Checkpoint and reference manifests are immutable and checksummed. The control
plane materializes only their declared artifacts, rechecking registry
metadata and bytes before Docker dispatch. Missing, mixed or modified
artifacts fail closed.

Conversion runs in Linux Docker. The isolation ladder is PyTorch baseline,
ONNX neural-stage parity, TensorRT complete output, then TensorRT streaming.
GPT parity includes teacher-forced logits and greedy semantic sequences;
stochastic waveform comparison alone is insufficient. A parity failure
blocks packaging.

A built package remains qualification-pending until formal evaluation and
required human evidence pass. Evaluation includes the five supported
languages, TensorRT-only execution, complete/streaming quality, identity,
content accuracy, TTFA, RTF and the canonical benchmark. Packaging success
does not permit production promotion or release.

Production neural inference remains TensorRT 11 only. PyTorch is permitted in
the isolated preparation, training and QA workers, not as an inference
fallback. All corrections and selection rules are model- and dataset-agnostic.

## Recovering A Completed Checkpoint Handoff

A successful checkpoint sweep can produce thousands of individual validation
artifacts. The broker and registry share a bounded limit of 4,096 files for
checkpoint selection; other worker jobs retain their 1,024-file limit.
Contained paths, regular-file checks, byte sizes and every SHA-256 remain
mandatory. Job-result JSON is bounded at 256 KiB so the artifact IDs fit;
other JSON fields retain the 64 KiB bound. This changes transport capacity,
not any model-quality gate.

For a retained completed result rejected by the former artifact-count limit,
the Docker-only recovery command first performs a dry-run:

```bash
python -m aniflive_tts.workstation_handoff_recovery \
  --workstation-dir /workstation \
  --source-job-id <failed-checkpoint-job-id>
```

After the retained result passes verification, repeat with `--apply` to
import it. The command accepts only that specific transport failure, verifies
the preparation/result identities and pinned image, checks every artifact,
and requires a passed checkpoint report with `test_split_accessed=false`.

Recovery creates a paused retry and claims it atomically, preventing another
scheduler from starting a neural rerun. It imports the original bytes through
the normal artifact registry and completion gates, recording original job,
run, result, manifest and image digests. If a previous recovery failed only
while recording an oversized result or lost its metadata-import lease, the
command follows bounded, matching retry ancestry to the original retained output. The old failed jobs and
retained reports remain unchanged. Normal reference selection follows only when the
project has automatic production building enabled. Registry insertion avoids
repeated ancestor traversal for new leaves while retaining self-link, parent
existence and non-leaf cycle checks in the application and database. Recovery does not approve
a reference, consume test, change a quality decision or promote a package.

Official training outputs that serialize the pinned `utils.HParams` record are
loaded with PyTorch `weights_only=True` plus an allowlist containing only that
exact bundled class. They do not require the unrestricted pickle option. Any
other unsupported global remains a hard failure unless the operator explicitly
opts into unsafe loading for a trusted local checkpoint.


## Conversion Validation And Diagnostics

The worker installs only the pinned GPU ONNX Runtime implementation after
resolving ASR dependencies. The CPU distribution pulled by faster-whisper
shares the same module path and must be removed before reinstalling the GPU
wheel. Image validation requires the CUDA provider; actual GPU execution is
verified by conversion parity. Faster-whisper VAD still uses the available CPU
provider within that same GPU distribution.


Conversion requires finite outputs and per-element absolute-plus-relative
agreement between PyTorch and ONNX Runtime. Discrete outputs must match exactly.
GPT encoder and step each run all eight fixed probes; a passing probe cannot
replace a failed probe. Validation uses bounded CPU thread pools and retains
the case identifier, tolerances and execution settings.

Stochastic SoVITS validation replays the actual nonzero PyTorch normal draws
in an in-memory ONNX copy. Draw count, shape and dtype must agree. Production
ONNX retains its random operators. The report records graph and draw hashes;
saved validation inputs allow failures to be reproduced without retraining.

The spectrogram export expresses the same Hann-windowed transform as a portable
FP32 radix-2 FFT. It preserves reflection padding, hop, magnitude epsilon and
output layout, and retains the original strict spectrogram tolerance. Native
transform agreement and dynamic-length ONNX tests cover this representation;
the resulting TensorRT engine still requires runtime parity qualification.

Conversion retains subprocess logs and validation reports when an export fails,
including exact inputs for failed numerical comparisons. These diagnostics do
not authorize changing checkpoints, reference locks or consuming holdout again.
FP16 export, TensorRT construction and audio conversion parity are separate
gates; passing PyTorch-to-ONNX checks does not imply that later gates passed.

The speaker export must also agree with the independent native Kaldi frontend,
not just its own export wrapper. Its fixed 16 kHz contract uses 400-sample
snipped frames at a 160-sample hop, per-frame DC removal, 0.97 pre-emphasis,
a symmetric Povey window, right zero-padding to 512 samples and the native
80-band mel filters. The speaker network weights remain unchanged. Export
validation compares ONNX speaker outputs to the native feature implementation.

Complete-audio conversion QA uses original PyTorch checkpoint weights with the
same exported semantic-stage sampling and EOS contract as TensorRT. Training's
native repetition policy and sampler are a different workload even with an
identical seed. Each audio case records the comparison contract and decoder
input hashes, shapes and numerical differences. Unaligned acoustic randomness
is explicitly recorded; diagnostics must not silently approve failed quality
gates or replace the original reports.
## Human Content Adjudication

An ASR-only conversion failure can be reviewed by a human using the exact
generated WAV and expected sentence. The optional `content_review` file input
on `conversion.parity` uses schema
`aniflive-conversion-content-review-v1` and a `reviews` list. Each entry
contains `audio_sha256`, `text`, `decision`, `user_response` and `origin`.
Only an unambiguous `content-complete` decision with a recorded response and
origin can adjudicate the matching WAV hash and exact sentence.

The worker retains the review file and its digest alongside the conversion
report. Reviewed cases preserve their ASR hypotheses, error rates and
`automated_content_regression`, and identify the applied
`human_content_review`. Changed audio, changed text, duplicate entries or an
unconfirmed decision cannot reuse the approval. This review addresses content
completeness only; semantic, spectral, speaker, duration and artifact gates
remain mandatory. The original failed job remains part of the evidence.
This mechanism does not approve voice preference, naturalness, runtime latency
or release qualification.

## Runtime Engine Identity

Engine loading still requires matching TensorRT, CUDA, compute capability,
GPU model, SM count and operating-system architecture. Reported total GPU
memory can vary with driver reservations: positive integer capacities may
differ by at most both 64 MiB and 0.5 percent of the smaller capacity.
Missing, malformed or substantially different capacity records require a
rebuild. Selection preserves the original manifest, fingerprint and engine
bytes; actual TensorRT deserialization and runtime qualification remain
necessary.

## Streaming Overlap Default

New conversions use 12 latent overlap frames in the CLI, Docker training
worker and standalone exporter. An explicit positive even value remains
supported, and existing packages keep the overlap shape recorded in their
engine configuration. This setting changes the exported streaming decoder
contract and therefore requires conversion and runtime qualification.

Run 10's controlled overlap comparison passed all five conversion cases
(with one hash-bound human ASR adjudication) and the five-language real
stream/complete gates. In the same 3-session, 30-request Japanese workload,
keep-alive audible TTFA P50/P95 changed from 93.96/131.43 ms at 32 frames
to 66.67/83.97 ms at 12 frames. These diagnostic results support the new
default; they are not the formal 10-session release benchmark or long-form
human listening qualification. Minimum first-preview audio protections
remain unchanged.

## Chinese ASR Orthography

Runtime evaluation retains the original ASR hypothesis and raw CER. For
Mandarin and Cantonese it additionally records OpenCC `t2s` scoring, including
the normalizer version and both normalized strings. Baseline comparison
rescores both original hypotheses with the same installed normalizer, retains
both raw scores and records the derived comparison. Legacy reports without
their original hypotheses cannot pass this comparison.

Only script variants are normalized. Word substitutions, omissions and
Cantonese-to-Mandarin wording changes still count as errors. Content,
speaker and performance regression limits remain unchanged. Original
baseline reports are never rewritten to obtain a passing result.

## Dataset-Aware Initial Epoch Budget

New Quick, Balanced and High Quality worker runs default to
`epoch_policy=adaptive`. Their preset epoch counts are ceilings. Advanced
defaults to `epoch_policy=fixed`, preserving explicit user choices; it can
explicitly opt into adaptive ceilings. Fixed policy is also available for
reproducible preset runs. Existing projects and checkpoints are not rewritten.

After preprocessing and alignment validation, the worker measures only the
audio named in the training text inventory. Validation/test audio and unrelated
files do not contribute. Unreadable, missing or zero-duration training audio
fails before training. The versioned initial heuristic uses a 600-second,
200-clip reference budget:

- Duration multiplier: `min(1, sqrt(600 / training_seconds))`.
- Estimated steps per epoch: `ceil(training_clips / batch_size)`.
- Nominal update budget: `ceil(200 / batch_size) * preset_epoch_ceiling`.
- Choose the smaller duration-scaled ceiling and update-limited epoch count,
  retaining at least two candidate epochs when the caller's ceiling permits it.
  Report when even this minimum exceeds the nominal update budget.
- Small datasets never receive extra repetitions just to hit an update target.
  Fewer than 120 seconds or 40 clips is recorded as limited data.
- Adaptive runs save every epoch so validation can choose an earlier candidate.

These are auditable engineering defaults, not empirically established optimal
epoch counts or an overfitting detector. Updates are estimates because training
sampler padding and gradient accumulation can change actual step counts.
Duration does not capture speaker consistency, phonetic diversity or recording
quality. Validation checkpoint selection and human checks of content, identity
and timbre remain necessary. A model that underfits needs a reviewed extension
of its budget, not automatic qualification or unlimited repetition.

`training-budget.json` records requested ceilings, resolved epochs, audio
duration, counts, batch sizes, estimated updates, policy version and limitations.
`training-audio-inventory.json` is a separate hashed artifact, keeping large
inventories out of the job result. The resolved plan drives the actual GPT and
SoVITS config files, and the training report retains the budget.

Resuming requires `epoch_policy=fixed` and the previously resolved epoch
totals (use Advanced with those values when they differ from preset defaults).
The worker rejects adaptive resume rather than silently recalculate or extend
an existing training run.
