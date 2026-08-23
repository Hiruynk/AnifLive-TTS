# Docker Deployment

The generic image supports `cu128` and a provisional `cu121` profile. Models,
shared resources, cache, and reports are host bind mounts. The serving
entrypoint only validates and serves; it never downloads dependencies or
builds engines.

```powershell
Copy-Item .env.example .env
```

Configure the host-to-container mapping:

```dotenv
ANIFLIVE_TTS_MODELS_DIR=D:/models/packages
ANIFLIVE_TTS_MODEL_PACKAGE=/data/models/my-v2proplus
ANIFLIVE_TTS_SHARED_HOST_DIR=D:/models/shared
ANIFLIVE_TTS_CACHE_HOST_DIR=D:/models/cache
ANIFLIVE_TTS_REPORTS_DIR=D:/models/reports
```

Build the image, then rebuild and validate the hardware-specific engines inside
the target Linux container:

```powershell
docker compose build
docker compose run --rm --entrypoint aniflive-tts aniflive-tts `
  model rebuild-engines --model-package /data/models/my-v2proplus
docker compose run --rm --entrypoint aniflive-tts aniflive-tts `
  validate --model-package /data/models/my-v2proplus `
  --shared-dir /data/shared --source-dir /app/minimal_inference --enqueue
.\scripts\run_docker.ps1 -CudaProfile cu128
```

For `cu121`, add `-f docker-compose.yml -f docker-compose.cu121.yml` to both
`docker compose` commands and start with `-CudaProfile cu121`.

Subsequent starts enforce `--pull never --no-build`: deleting only the
container while retaining the image and host bind mounts causes no downloads,
installs, or builds. Deleting the image requires another explicit build.
Missing model packages or mismatched engine fingerprints fail with
`ENGINE_REBUILD_REQUIRED`.

Release images include OCI source-revision metadata. The release workflow
exports an immutable digest, image-derived SPDX SBOM, and BuildKit provenance
record for each CUDA profile. Follow [release image verification](RELEASE_VERIFICATION.md)
before changing package visibility or deploying a published tag.

CUDA 12.8 passed full GPU E2E on RTX 5070 Ti. CUDA 12.1 engines build and load,
but cu121 cannot execute the PyTorch CUDA support operations on this Blackwell
GPU. Validate the cu121 tag separately on compatible Ampere/Ada hardware.

Character assets and external routing configuration belong in private
deployment overlays outside the generic image and source release.
