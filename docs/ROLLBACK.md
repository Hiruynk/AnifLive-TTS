# Container Rollback

Keep the last validated image tag and model package fingerprint. Start a new
candidate on a spare localhost port, run health, TensorRT proof, five-language,
stream, performance, and quality checks, then change the local port binding.

If validation fails, stop the candidate and restart the previous image with
the unchanged host-mounted model/shared/cache directories. Do not rebuild
engines, alter checkpoints, or delete persistent data during rollback.

Public routing rollback is deployment-specific and must be implemented only in
the corresponding external overlay, never in the generic AnifLive-TTS image.
