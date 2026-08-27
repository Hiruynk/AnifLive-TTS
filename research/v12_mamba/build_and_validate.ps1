param(
    [string]$Image = "aniflive-tts-mamba2-dev:trt11-cu128",
    [string]$OutputDirectory = (Join-Path $PSScriptRoot "out"),
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker CLI is not installed or not available on PATH."
}

$serverVersion = & docker info --format '{{.ServerVersion}}' 2>$null
if ($LASTEXITCODE -ne 0 -or -not $serverVersion) {
    throw "Docker daemon is unavailable. This script does not launch Docker Desktop or any GUI."
}

if (-not $SkipBuild) {
    & docker build --pull=false --file (Join-Path $PSScriptRoot "Dockerfile.devel") --tag $Image $PSScriptRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Mamba-2 development image build failed."
    }
}

New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
$resolvedOutput = (Resolve-Path $OutputDirectory).Path

& docker run --rm --pull never --gpus all --network none `
    --mount "type=bind,source=$resolvedOutput,target=/out" `
    $Image
if ($LASTEXITCODE -ne 0) {
    throw "Mamba-2 TensorRT 11 validation failed."
}

Write-Host "Validation report: $resolvedOutput\mamba2-update-feasibility.json"
