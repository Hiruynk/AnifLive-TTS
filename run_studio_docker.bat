@echo off
setlocal
cd /d "%~dp0"

set "ANIFLIVE_TTS_CUDA_FLAVOR=cu128"
if not "%~1"=="" set "ANIFLIVE_TTS_CUDA_FLAVOR=%~1"
if "%ANIFLIVE_TTS_CUDA_FLAVOR%"=="cu128" (
    set "ANIFLIVE_TTS_CUDA_BASE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04@sha256:9175fa92f96de35a8cfb9493f0dfcf9435c7a597e9d95ad41d2cae382a95e3f9"
) else if "%ANIFLIVE_TTS_CUDA_FLAVOR%"=="cu126" (
    set "ANIFLIVE_TTS_CUDA_BASE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:8aef630a54bc5c5146ae5ce68e6af5caa3df0fb690bb91544175c91f307e4356"
) else (
    echo [AnifLive-TTS Studio] Choose cu128 or cu126.
    exit /b 1
)

docker image inspect aniflive-tts-studio:1.4.0-%ANIFLIVE_TTS_CUDA_FLAVOR% >nul 2>&1
if errorlevel 1 (
    docker build --build-arg "CUDA_BASE_IMAGE=%ANIFLIVE_TTS_CUDA_BASE%" --build-arg "TORCH_REQUIREMENTS=requirements/torch-%ANIFLIVE_TTS_CUDA_FLAVOR%.txt" -t aniflive-tts:1.4.0-%ANIFLIVE_TTS_CUDA_FLAVOR% .
    if errorlevel 1 exit /b 1
    docker build -f Dockerfile.workstation-worker --build-arg BASE_IMAGE=aniflive-tts:1.4.0-%ANIFLIVE_TTS_CUDA_FLAVOR% -t aniflive-tts-workstation-worker:1.4.0-%ANIFLIVE_TTS_CUDA_FLAVOR% .
    if errorlevel 1 exit /b 1
    docker build -f Dockerfile.studio --build-arg WORKER_IMAGE=aniflive-tts-workstation-worker:1.4.0-%ANIFLIVE_TTS_CUDA_FLAVOR% -t aniflive-tts-studio:1.4.0-%ANIFLIVE_TTS_CUDA_FLAVOR% .
    if errorlevel 1 exit /b 1
)

docker run --rm --mount "type=bind,source=%CD%,target=/workspace" --mount type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock aniflive-tts-studio:1.4.0-%ANIFLIVE_TTS_CUDA_FLAVOR% -m aniflive_tts.studio_docker --host-project-root "%CD%" --studio-image aniflive-tts-studio:1.4.0-%ANIFLIVE_TTS_CUDA_FLAVOR% --worker-image aniflive-tts-workstation-worker:1.4.0-%ANIFLIVE_TTS_CUDA_FLAVOR% --runtime-image aniflive-tts:1.4.0-%ANIFLIVE_TTS_CUDA_FLAVOR%
if errorlevel 1 exit /b %ERRORLEVEL%
start "" "http://127.0.0.1:9891/"
exit /b 0
