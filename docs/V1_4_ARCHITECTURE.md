# AnifLive-TTS v1.4 Architecture

## Product

AnifLive-TTS v1.4 extends the TensorRT inference runtime into a local voice
production workstation. The v1.3 synthesis path remains the golden inference
baseline.

The workstation follows three execution lanes:

1. **Linux inference runtime** — the existing TensorRT 11 service and compatible
   `/v1/audio/speech` API inside the production Docker container.
2. **Local control plane** — projects, expression metadata, model lineage,
   evaluation records and job coordination served by the WebUI process. The
   Windows host may run this lane.
3. **Workers** — dataset, TSE, training, evaluation, engine-build and package
   adapters that run outside the inference request process.

This separation keeps long-running jobs from importing training stacks or
claiming GPU memory inside the latency-sensitive service.

## Execution Boundary

| Environment | Permitted work |
|---|---|
| Windows host | WebUI, SQLite control plane, job coordination, local file inventory and preparation manifests |
| Linux Docker | TensorRT inference and every neural TSE, training, evaluation or engine-build operation |

No neural work executes on Windows. A Windows worker may inventory files and
prepare a validated job manifest; it does not load a TSE, training, evaluation
or TensorRT build stack. Linux neural workers must use the CUDA 12.8 container
toolchain and claim the same GPU-exclusive resource lease as inference.

## Modules

| Area | Module | Responsibility |
|---|---|---|
| Create | Synthesis | Segmented expression editing, streaming playback, metrics and downloads |
| Create | Expressions | Package-qualified expression profiles and future draft banks |
| Data | Dataset Factory | Standard, target-speaker and GPT-SoVITS-list acquisition; segmentation, ASR, review, expression annotation and freeze |
| Train | Training | Experiment configuration, checkpoints and resource scheduling |
| Train | Evaluation | Quality, identity, multilingual, streaming and performance gates |
| Deploy | Models | Dataset-to-checkpoint-to-engine-to-package lineage |
| Deploy | Engines | TensorRT engine inventory and runtime compatibility |
| System | Jobs | Queue, state, progress, cancellation and logs |
| System | Settings | Local runtime and session capabilities |

## Dataset Target-Speaker Stage

Target-speaker acquisition is a Dataset Factory stage, not a second normal-user
module. Its advanced inspector and standalone internal route remain available
for diagnosis. The stage combines three independent responsibilities:

- pinned FSMN-VAD discovers candidate speech regions inside the Linux worker;
- pinned `MossFormer2_SS_16K` performs blind two-speaker separation inside the
  Linux CUDA worker;
- a workstation-owned TensorRT 11 ERes2NetV2 engine verifies the target speaker
  against the operator-supplied reference without requiring a pre-existing
  voice package.

The source revision, model revision and checkpoint checksum are mandatory. The
checkpoint remains external and is mounted read-only; no model weight is kept
in the repository or worker image. Per-window ambiguity fails closed to review.
The Dataset path preserves every source sample range and emits independent,
stable clip IDs plus raw and processed audio. Ambiguous or contaminated clips
are routed to separation or human review; rejected clips are retained. The
legacy standalone TSE route may still emit a concatenated artifact, but Dataset
Factory never feeds that artifact back through VAD for a second segmentation.
There is no Roxy/Miku name branch and no Windows neural execution. The measured
RTX 5070 Ti qualification and the current overlap-detector boundary are in
`V1_4_TSE_OVERLAP_SEPARATION.md`.

## Synthesis Compatibility

The v1.4 Synthesis module directly preserves the validated v1.3 interaction
contract:

- model-exclusive activation;
- Japanese, Putonghua/Mandarin, Cantonese, English and Korean;
- expression underlines, captions and tags;
- PCM16 streaming, cancellation and WAV download;
- audio-aligned gold playback text;
- eight request metrics and the five most recent requests.

The workstation shell changes navigation and visual hierarchy without replacing
the annotation or acoustic playback-alignment implementations.

The Overview Voice Pulse reflects live workstation activity. It becomes active
while a job is running or while Synthesis playback is active, and returns to
idle after both sources stop.

## Speech Sessions

The development API adds:

    POST   /v1/sessions
    GET    /v1/sessions/{id}
    POST   /v1/sessions/{id}/segments
    GET    /v1/sessions/{id}/audio
    POST   /v1/sessions/{id}/flush
    POST   /v1/sessions/{id}/cancel
    DELETE /v1/sessions/{id}

The caller appends committed speech segments with stable IDs. AnifLive-TTS
enforces ordering, idempotency, one audio consumer, cancellation, flush, model
binding and session expiry.

The current development contract reports `context.mode=committed-neural-v1`.
Policy `A` is the backward-compatible default and retains metadata only.
Callers may explicitly select one of six experimental policies when creating a
session:

| Policy | Context used for the next committed segment |
| --- | --- |
| A | None; fixed-seed baseline |
| B | Previous phones and BERT features |
| C | B plus at most 32 exact semantic tokens |
| D | B plus at most 64 exact semantic tokens |
| E | Expression-aware limits; 12 phones and 48 tokens at a style switch |
| F | Boundary-adaptive selection of A, C, D or E |

Retained state is capped at 75 phones and 64 exact semantic tokens, is bound to
one model and voice profile, and is committed only after the segment stream
finishes successfully. Cancellation, expiry, terminal close, model activation
and process shutdown release retained GPU tensors. No acoustic latent is
retained or claimed. The API and stream headers mark B-F as experimental and
unqualified until the fixed-seed A-F report and long-form quality gates run on
the production RTX 5070 Ti TensorRT environment.

AnifEngine-Voice remains responsible for turn control, partial LLM output,
speech planning, interruption policy and expression planning.

## Local Data

Workstation state is stored outside the repository under the local application
data directory, or under ANIFLIVE_TTS_WORKSTATION_DIR.

Datasets, checkpoints, references and model packages remain in explicit local
artifact locations. Local inventory reads are restricted to roots configured by
ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS.

The artifact registry records datasets, checkpoints, expression banks,
evaluations, TensorRT engines and packages. Artifact identity, checksum and
parent links are immutable. Parent links form an acyclic lineage graph, and
artifact files remain inside the workstation artifact store.

## Workstation Persistence And Scheduling

Projects, jobs and job events use transactional SQLite WAL storage. Fresh
database bootstrap and schema migration are concurrency-safe. Workers claim
jobs with expiring owner tokens, heartbeat while running and release trusted
resource leases on terminal state. A second workstation process never marks a
live job failed; only expired leases are recovered.

Job resource classes are derived from the adapter type rather than supplied by
the caller. GPU-exclusive workers and the inference service share the same
resource lease contract. Dependencies, cancellation, monotonic progress, logs
and terminal immutability are enforced by the store.

The default local launcher and Docker Compose deployment both use the host
directory `./data/workstation`. Compose bind-mounts it at `/data/workstation`
and sets `ANIFLIVE_TTS_WORKSTATION_DIR` in the inference container. The
TensorRT service acquires `gpu:0` before its first CUDA probe and heartbeats the
lease for the full lifetime of its resident engines, reference conditioning and
CUDA context. It releases the lease only after those assets are unloaded.
Docker-backed GPU jobs claim the same SQLite lease, so they cannot start merely
because inference is between requests while its VRAM is still allocated.

The foreground inference launcher rejects differing host and Compose
workstation paths. Queued GPU work exposes a persisted wait reason instead of
appearing idle, and successful claim clears that reason in the same transaction
that acquires the job-owned lease. On workstation shutdown the supervisor stops
the worker first, giving Docker cleanup 30 seconds before the WebUI is stopped.

This development snapshot does not yet implement a cross-process
drain/unload/run-worker/reload handoff. A GPU-exclusive workstation job remains
blocked while the inference service is ready; the operator must stop or unload
inference before such a job can run. Conversely, inference startup and model
activation fail clearly while a GPU worker owns the lease. Request, stream,
warm-retention and model cleanup paths validate the process-owned residency
lease and never create a second cross-process reservation.

Production residency leases use a 120-second minimum TTL and a heartbeat no
slower than one quarter of that TTL, leaving margin beyond SQLite's bounded
busy timeout. A lost heartbeat is fail-stop: the inference process exits with
code 70 instead of serving or retaining CUDA allocations past an unrenewed
lease. Docker may restart it, but startup cannot touch CUDA until `gpu:0` is
claimable again. The expired token is reclaimed by the ordinary store recovery
path; it is never replaced while the failed process remains alive.

Dataset inventory rejects symlinks, Windows junctions and other reparse points,
then rechecks every resolved media path against configured import roots.

Heavy model stacks remain outside the WebUI and inference processes. Explicit
worker adapters receive validated project records, cancellation callbacks and
progress sinks.

## Worker Runner

The bounded CLI worker supports one-shot execution and a long-running poller:

```powershell
aniflive-tts worker --once `
  --job-type dataset.inventory `
  --workstation-dir D:\AnifLive-TTS\workstation `
  --import-root D:\VoiceData
```

```bash
aniflive-tts worker --poll-seconds 2 --heartbeat-seconds 15 \
  --workstation-dir /workstation --import-root /data
```

The worker selects only registered adapter types. It accepts no executable,
script or shell argument. Each job is claimed with an owner token, heartbeats
before lease expiry, observes cooperative cancellation and writes progress,
events and terminal state through the same claim token. `SIGINT` and `SIGTERM`
request a clean stop; an active adapter transitions to `cancelled` before the
poller exits.

`dataset.inventory` performs validated local inventory on the control plane.
`dataset.process`, TSE, training, evaluation, engine and package adapters write
immutable preparation manifests and then execute them through the configured
Linux Docker broker. Dataset processing covers decode, canonical resampling,
optional conservative denoise, VAD, segmentation, optional offline ASR,
five-language normalization and quality metadata. Training emits a verified GPT
and SoVITS checkpoint pair; the dependent engine job converts that pair to FP16
ONNX and nine TensorRT 11 engines; packaging validates checksums, fingerprints,
deserialization and I/O contracts; evaluation runs the resulting package. When
the Docker backend or a required input is absent, the adapter is `blocked`; it
is never reported as completed neural work.

An optional, admin-configured Linux Docker broker can execute those validated
manifests. Its image must be pinned by digest and already exist locally. Adapter
source code owns every container entrypoint and argument; jobs cannot supply
runtime flags. Inputs are mounted read-only, output is isolated per run, the
container requests the NVIDIA GPU, defaults to no network and runs with a
read-only root filesystem, no new privileges and no Linux capabilities. A
strict versioned result manifest must match the job, project, run, Linux
platform and image digest before the worker records success. See
`V1_4_DOCKER_WORKER_BROKER.md` for the complete contract.

The shipped worker image is built from the CUDA 12.8 Linux development image
and is pinned by immutable digest in the local broker configuration. Image
construction itself runs TorchCodec decoding, GPT-SoVITS imports and all five
text-normalization frontends. Production execution keeps `--pull=never` and
`--network none`, so a job cannot fetch code, weights or models at runtime.
