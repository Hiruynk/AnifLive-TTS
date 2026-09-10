# AnifLive-TTS v1.4.0.dev0

## Development Status

**NO-GO pending release qualification.** This file identifies the isolated
v1.4 development source for CI version-integrity checks. It is not a public
release note and must not be used to create a GitHub release, container tag, or
production package.

The feature implementation now includes AnifLive-TTS Studio and the preserved
AnifLive-TTS WebUI, Dataset Factory, TSE, V2ProPlus training, TensorRT engine
build, packaging, evaluation, artifact lineage, the Linux Docker worker broker,
GPU resource leasing and the Speech Session API. The final worker image and a
real scheduler-to-Docker artifact handoff have passed on the RTX 5070 Ti.

The v1.3 inference path remains the immutable production baseline. The
approved performance comparison and real checkpoint pause/resume checks have
passed. New packages record their semantic sampling contract; legacy packages
retain their existing behavior. Final qualification, public Docker handoff,
brief long-form/expression listening and local release-package acceptance
remain required before v1.4 can be declared releasable.

The release blockers and current validation evidence are maintained in
`docs/V1_4_IMPLEMENTATION_STATUS.md`. A final `RELEASE_NOTES_v1.4.0.md` may be
created only after every v1.4 acceptance gate passes.
