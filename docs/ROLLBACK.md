# Container Rollback

For v1.5, retain the complete previous image, model-package directory,
`source-lock.json`, engine fingerprint and checksums together. v2ProPlus and
v5Turbo have different acoustic engines and sample rates; do not copy an
engine or prepared conditioning cache from one backend into the other. Voice
records and their audio files are persistent data and should be backed up
before an upgrade. Reprepare References for the selected runtime.

Keep the last validated image tag and model package fingerprint. Start a new
candidate on a spare localhost port, run health, TensorRT proof, five-language,
stream, performance, and quality checks, then change the local port binding.

If validation fails, stop the candidate and restart the previous image with
the unchanged host-mounted model/shared/cache directories. Do not rebuild
engines, alter checkpoints, or delete persistent data during rollback.

Restore external routing through the deployment configuration. For Apple Silicon, follow the [macOS rollback instructions](MACOS_INSTALLATION_EN.md).
