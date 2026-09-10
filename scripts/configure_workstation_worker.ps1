[CmdletBinding()]
param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$Image = "aniflive-tts-workstation-worker:dev",

    [Parameter()]
    [string]$WorkstationDir = "",

    [Parameter()]
    [ValidateRange(1, 604800)]
    [int]$TimeoutSeconds = 86400,

    [Parameter()]
    [ValidateRange(0.05, 5.0)]
    [double]$PollSeconds = 0.25
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($WorkstationDir)) {
    $WorkstationDir = Join-Path $PSScriptRoot "..\data\workstation"
}

if (-not (Get-Command docker.exe -ErrorAction SilentlyContinue)) {
    throw "Docker CLI was not found. Install/start Docker Desktop before configuring the worker."
}

$inspectText = & docker.exe image inspect $Image 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "The local worker image '$Image' was not found. Build it explicitly before running this script. $inspectText"
}

$images = @($inspectText | ConvertFrom-Json)
if ($images.Count -ne 1) {
    throw "Docker returned an unexpected image inspection result for '$Image'."
}
$imageInfo = $images[0]
if ($imageInfo.Os -ne "linux" -or $imageInfo.Architecture -ne "amd64") {
    throw "The worker image must be linux/amd64, got $($imageInfo.Os)/$($imageInfo.Architecture)."
}

$repository = ($Image -split "@", 2)[0]
$lastSlash = $repository.LastIndexOf("/")
$lastColon = $repository.LastIndexOf(":")
if ($lastColon -gt $lastSlash) {
    $repository = $repository.Substring(0, $lastColon)
}
$digestReference = @($imageInfo.RepoDigests) |
    Where-Object { $_ -like "$repository@sha256:*" } |
    Select-Object -First 1
if (-not $digestReference) {
    throw "The local image has no resolvable repository digest. Rebuild it with Docker BuildKit, then retry."
}
if ($digestReference -cnotmatch "^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$") {
    throw "Docker returned a worker digest that does not satisfy the broker contract: $digestReference"
}

& docker.exe image inspect $digestReference *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker cannot resolve the worker by immutable digest: $digestReference"
}

$resolvedWorkstation = [System.IO.Path]::GetFullPath($WorkstationDir)
[System.IO.Directory]::CreateDirectory($resolvedWorkstation) | Out-Null
$configPath = Join-Path $resolvedWorkstation "docker-broker.json"
$document = [ordered]@{
    schema = "aniflive-tts-docker-broker-config-v1"
    image = $digestReference
    network = "none"
    timeout_seconds = $TimeoutSeconds
    poll_seconds = $PollSeconds
}
$json = $document | ConvertTo-Json -Depth 3
[System.IO.File]::WriteAllText(
    $configPath,
    $json + [Environment]::NewLine,
    [System.Text.UTF8Encoding]::new($false)
)

Write-Host "[AnifLive-TTS] Linux worker image: $digestReference"
Write-Host "[AnifLive-TTS] Broker config: $configPath"
Write-Host "[AnifLive-TTS] No image was built or pulled by this command."
