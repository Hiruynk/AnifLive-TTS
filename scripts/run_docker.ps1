param(
    [ValidateSet("cu128", "cu126")][string]$CudaProfile = "cu128",
    [switch]$Pull,
    [switch]$Build
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$compose = @("compose", "-f", "docker-compose.yml")
if ($CudaProfile -eq "cu126") { $compose += @("-f", "docker-compose.cu126.yml") }
$compose += @("up", "-d")
if (-not $Pull) { $compose += @("--pull", "never") }
if ($Build) {
    $compose += "--build"
} else {
    $compose += "--no-build"
}

& docker @compose
if ($LASTEXITCODE -ne 0) { throw "docker compose failed with exit code $LASTEXITCODE" }
