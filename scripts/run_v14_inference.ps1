[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $projectRoot

function Resolve-WorkstationPath([string] $Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return [IO.Path]::GetFullPath((Join-Path $projectRoot "data\workstation"))
    }
    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
}

$composeWorkstation = Resolve-WorkstationPath $env:ANIFLIVE_TTS_WORKSTATION_HOST_DIR
$hostWorkstation = Resolve-WorkstationPath $env:ANIFLIVE_TTS_WORKSTATION_DIR
if ($composeWorkstation -ne $hostWorkstation) {
    throw "ANIFLIVE_TTS_WORKSTATION_HOST_DIR and ANIFLIVE_TTS_WORKSTATION_DIR must identify the same directory."
}
New-Item -ItemType Directory -Force -Path $composeWorkstation | Out-Null
$env:ANIFLIVE_TTS_WORKSTATION_HOST_DIR = $composeWorkstation
$env:ANIFLIVE_TTS_WORKSTATION_DIR = $hostWorkstation

$docker = Get-Command docker.exe -ErrorAction Stop
& $docker.Source info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is not ready. No inference container was started."
}
& $docker.Source compose config --quiet
if ($LASTEXITCODE -ne 0) {
    throw "The AnifLive-TTS Docker Compose configuration is invalid."
}

$containerName = "aniflive-tts-v14-workstation-api"
$running = & $docker.Source inspect --format '{{.State.Running}}' $containerName 2>$null
if ($LASTEXITCODE -eq 0 -and $running -eq "true") {
    throw "$containerName is already running. This terminal will not take ownership of another process's container."
}

$script:stopRequested = $false
$cancelHandler = [ConsoleCancelEventHandler] {
    param($Sender, $EventArgs)
    $script:stopRequested = $true
    $EventArgs.Cancel = $true
}
[Console]::add_CancelKeyPress($cancelHandler)

$compose = $null
$owned = $false
$exitCode = 0
try {
    Write-Host "[AnifLive-TTS] Workstation lease: $composeWorkstation"
    Write-Host "[AnifLive-TTS] Linux Docker logs follow. Press Ctrl+C to stop inference."
    $arguments = @(
        "compose", "up", "--no-build", "--pull", "never", "aniflive-tts"
    )
    $compose = Start-Process -FilePath $docker.Source -ArgumentList $arguments -NoNewWindow -PassThru
    $owned = $true
    while (-not $compose.HasExited -and -not $script:stopRequested) {
        Start-Sleep -Milliseconds 200
        $compose.Refresh()
    }
    if (-not $script:stopRequested) {
        $compose.WaitForExit()
        $exitCode = $compose.ExitCode
    }
}
finally {
    [Console]::remove_CancelKeyPress($cancelHandler)
    if ($owned) {
        Write-Host "[AnifLive-TTS] Stopping inference and releasing resident CUDA state..."
        & $docker.Source compose stop --timeout 60 aniflive-tts
        $stopExitCode = $LASTEXITCODE
        if ($null -ne $compose -and -not $compose.HasExited) {
            if (-not $compose.WaitForExit(65000)) {
                $compose.Kill()
                $compose.WaitForExit()
            }
        }
        if ($stopExitCode -ne 0 -and $exitCode -eq 0) {
            $exitCode = $stopExitCode
        }
    }
}

exit $exitCode
