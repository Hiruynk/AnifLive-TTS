# Docker Deployment

**CUDA／TensorRT（Docker）** serves v2ProPlus (32 kHz) and v5Turbo (48 kHz, FP16, 4-step CFM, CFG 0). Models, shared resources and caches use host bind mounts. Prepare the model package before starting the API.

The measured host is RTX 5070 Ti with CUDA 12.8, TensorRT 11 and Triton. For **Apple Silicon（macOS／MLX／Metal）**, see the [macOS installation guide](MACOS_INSTALLATION_EN.md).

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

For v2ProPlus packages only, build the image, then rebuild and validate the
hardware-specific engines inside the target Linux container:

```powershell
docker compose build
docker compose run --rm --entrypoint aniflive-tts aniflive-tts `
  model rebuild-engines --model-package /data/models/my-v2proplus
docker compose run --rm --entrypoint aniflive-tts aniflive-tts `
  validate --model-package /data/models/my-v2proplus `
  --shared-dir /data/shared --source-dir /app/minimal_inference --enqueue
.\scripts\run_docker.ps1 -CudaProfile cu128
```

## v5Turbo packages

Use a `gsv-v5turbo` package prepared with the [v5Turbo build guide](V1_5_V5_BUILD.md). The v2ProPlus `model rebuild-engines` command above builds v2ProPlus engines; v5Turbo uses its own builders. Validate newly built engines on the target GPU before deployment.

For `cu126`, add `-f docker-compose.yml -f docker-compose.cu126.yml` to both
`docker compose` commands and start with `-CudaProfile cu126`.

Subsequent starts enforce `--pull never --no-build`: deleting only the
container while retaining the image and host bind mounts causes no downloads,
installs, or builds. Deleting the image requires another explicit build.
Missing model packages or mismatched engine fingerprints fail with
`ENGINE_REBUILD_REQUIRED`.

Release images include OCI source-revision metadata. The release workflow
exports an immutable digest, image-derived SPDX SBOM, and BuildKit provenance
record for each CUDA profile. Follow [release image verification](RELEASE_VERIFICATION.md)
before changing package visibility or deploying a published tag.

CUDA 12.8 is tested on RTX 5070 Ti. CUDA 12.6 uses the same PyTorch 2.10 baseline and has source/build checks; GPU operation still needs validation on a compatible host.

Keep model/Reference data and routing credentials in persistent host directories and deployment configuration.
