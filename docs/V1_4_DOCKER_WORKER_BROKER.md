# AnifLive-TTS v1.4 Linux Docker Worker Broker

## Boundary

The workstation control plane may run on Windows, but neural TSE, training,
evaluation and TensorRT engine-build work remains inside a Linux NVIDIA Docker
container. The broker does not build or pull an image. It starts only an image
that is already present locally and pinned by an administrator to a SHA-256
digest.

Without an explicit broker config, neural preparation jobs remain `blocked`.
They are never reported as completed.

## Admin Configuration

The worker accepts one admin-owned config path:

```powershell
aniflive-tts worker --docker-broker-config D:\AnifLive-TTS\broker.json `
  --workstation-dir D:\AnifLive-TTS\workstation `
  --import-root D:\VoiceData
```

The versioned config contains no executable, entrypoint, argument or mount
fields:

```json
{
  "schema": "aniflive-tts-docker-broker-config-v1",
  "image": "ghcr.io/example/aniflive-tts-worker@sha256:<64 lowercase hex characters>",
  "network": "none",
  "timeout_seconds": 86400,
  "poll_seconds": 0.25
}
```

`network` defaults to `none`. `host`, `default` and container namespace sharing
are rejected. A named Docker network can be selected only in this admin config;
it cannot be selected by a project or job.

## Fixed Execution Contract

Adapter-owned source code selects the container subcommand for each supported
job type. Project and job JSON cannot provide an image, command, executable,
entrypoint, flags, network, mount or shell.

Every invocation uses:

- `--pull=never` and a digest-pinned image;
- `--platform linux/amd64`;
- an NVIDIA `device=0` GPU request;
- a read-only container root filesystem;
- `no-new-privileges`, `cap-drop=ALL`, a private IPC namespace and a PID limit;
- a bounded, non-executable `/tmp` tmpfs;
- a read-only preparation manifest;
- one read-only bind mount for each validated input;
- one new, per-run read-write output directory;
- AnifLive-TTS job, project, run, type and manifest-digest labels.

The broker also supplies a fixed set of validated identity environment values
for the result schema, run ID, image digest, manifest digest and Linux platform.
Jobs cannot add or replace container environment variables.

The SQLite workstation directory is never mounted into a worker container.
Only the per-run output directory is writable.

The host worker and production inference container still coordinate through
that database: Docker Compose bind-mounts the host's
`ANIFLIVE_TTS_WORKSTATION_HOST_DIR` at `/data/workstation`, and the inference
service holds and heartbeats the same `gpu:0` lease for the entire period that
TensorRT engines, reference tensors and the CUDA context remain resident. The
host worker must claim that lease before it starts a GPU-exclusive container,
so it cannot run beside idle-but-resident inference VRAM. The neural worker
sees neither the database nor its lease token.

`ANIFLIVE_TTS_WORKSTATION_HOST_DIR` and the host control plane's
`ANIFLIVE_TTS_WORKSTATION_DIR` must resolve to the same directory. The v1.4
`run_tts.bat` launcher enforces that invariant before Compose starts. It owns
`aniflive-tts-v14-workstation-api` in the foreground, streams Compose logs and,
on Ctrl+C, allows up to 60 seconds for Docker to stop inference and release its
CUDA context. It refuses to take ownership of an already-running container.

While `gpu:0` is leased, a GPU-exclusive job remains `queued` with the stable
reason `Waiting for gpu:0; TensorRT inference or another GPU job is resident`.
The worker records the reason once, exposes it through the job API and WebUI,
and clears it atomically when the job is claimed or leaves the queue. This is a
scheduling state, not a failed or completed job.

Automatic inference quiesce, unload and reload handoff is not implemented in
this development snapshot. GPU jobs remain blocked while inference is loaded;
stopping or explicitly unloading inference is currently required before the
host worker can claim `gpu:0`.

If the inference heartbeat cannot renew its residency lease, inference exits
fail-stop with code 70. Process termination is intentional: it guarantees that
the operating system releases the CUDA context and VRAM before the old lease
can expire and a worker becomes eligible to claim the resource. A restarting
container must reacquire `gpu:0` before its first CUDA probe.

## Result Contract

The container writes `/aniflive/output/result.json` using
`aniflive-tts-docker-worker-result-v1`. The broker rejects unknown or duplicate
fields, malformed JSON, a mismatched job/project/type/run/image/platform,
manifest digest, non-completed outcomes and execution-control fields in the
payload.

Declared artifacts must use portable relative POSIX paths inside the output
directory. The broker rejects links, traversal, Windows alternate-data-stream
syntax, reserved device names and components ending in a dot or space, then
verifies each file's exact size and SHA-256 checksum before a job can succeed.

Successful outputs do not remain trusted merely because they passed the result
manifest parser. While the job lease is still live, the host reopens each
regular file, rejects linked/reparse path components, copies it into a unique
workstation-owned location below `artifacts/worker/<job-id>/`, and computes
SHA-256 again while copying. A declared digest or size mismatch fails the job.
Verified files and parent links are first registered as `building`. Their
promotion to `ready` and the owning job's transition to `succeeded` occur in
one SQLite transaction. Cancellation, failure, lease expiry or a failed final
transition removes the staged rows and leaves no published artifact.

The output kind is also bound to the trusted job type: training may publish a
checkpoint, evaluation an evaluation report, engine build an engine, packaging
a package, and TSE a dataset. The referenced job must exist and its type and
project must match the handoff, so a container cannot relabel or re-home an
output. Optional `parent_artifact_ids` in trusted job parameters are validated
against existing immutable records before the transaction commits.

Artifact IDs derive from the job ID, relative path and host-verified content
digest. Retries are therefore deterministic even if the manifest order changes.
An existing staged result is reused only when its full immutable identity,
including type, building status, path, digest, metadata and parent lineage,
matches. Every
destination directory component is checked while it is created; a symlink,
junction or other reparse point is rejected before any bytes are written.

Direct artifact registration follows the same storage invariant. A `ready`
record must identify an existing regular file inside the workstation artifact
store. The server computes its SHA-256 itself; a caller may provide a digest as
an assertion, but a mismatch is rejected. Planned metadata-only records remain
valid without a path or digest.

## Cancellation And Restart

Cancellation and timeout stop and force-remove the labelled container, then
verify its absence through Docker before the GPU job may become terminal.
Non-zero stop/remove results are acceptable only when absence is independently
confirmed. Otherwise the job and GPU reservation remain quarantined.

A worker restart lists only containers with the AnifLive-TTS managed label,
retains containers for unexpired live jobs and removes stale or duplicate
containers. Only after every stale removal is verified does the store recover
expired GPU jobs and release their resource leases. Expired job-owned GPU
leases are not reusable through generic lease expiry, so Docker control errors
cannot create concurrent inference and worker CUDA ownership. Live jobs owned
by another worker are not interrupted. Long-running supervisors invoke the
same bounded reconciliation hook after a lease window when recovering from a
host-process crash.

Broker unit tests use a fake command runner. They do not start, build or pull a
Docker image.

## Coordinated Lease Validation

Run this qualification only after the v1.4 inference and worker images exist;
do not reuse a v1.3 container for either side.

1. Set both host variables to the same absolute directory, start
   `run_tts.bat`, and confirm its log prints that directory as the workstation
   lease path.
2. Start `run_webui.bat` from the same checkout and queue one GPU-exclusive
   job. Confirm the job remains `queued`, reports `Waiting for GPU`, and no
   labelled worker container starts while inference is ready.
3. Stop the inference terminal with Ctrl+C. Confirm
   `aniflive-tts-v14-workstation-api` is stopped before accepting worker
   execution.
4. Confirm the queued job changes to `running`, its wait reason disappears and
   exactly one labelled Linux worker container owns GPU 0.
5. Stop the workstation terminal during a disposable worker job. The
   supervisor gives the worker 30 seconds to stop, remove and verify its Docker
   container before terminating the WebUI. Confirm no managed worker container
   remains and the job is cancelled, paused, or quarantined rather than falsely
   completed.
6. Repeat the sequence for at least 100 inference/worker handoffs while
   checking SQLite integrity, WAL recovery, lease uniqueness and NVIDIA process
   ownership. This step qualifies Docker Desktop's host-bind locking behavior;
   static unit tests do not claim that cross-boundary result.
