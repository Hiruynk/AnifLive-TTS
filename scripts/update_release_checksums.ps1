param(
    [string]$Version = "1.3.0"
)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Dist = Join-Path $Project "dist"
$Artifacts = @(
    (Join-Path $Dist "AnifLive-TTS-v$Version-docker-source-bundle.zip"),
    (Join-Path $Dist "SBOM-AnifLive-TTS-v$Version-cu128.spdx.json"),
    (Join-Path $Dist "SBOM-AnifLive-TTS-v$Version-cu126.spdx.json"),
    (Join-Path $Dist "RELEASE-METADATA-AnifLive-TTS-v$Version-cu128.json"),
    (Join-Path $Dist "RELEASE-METADATA-AnifLive-TTS-v$Version-cu126.json"),
    (Join-Path $Dist "TRIVY-AnifLive-TTS-v$Version-cu128.json"),
    (Join-Path $Dist "TRIVY-AnifLive-TTS-v$Version-cu126.json"),
    (Join-Path $Dist "RELEASE_NOTES_v$Version.md")
)

foreach ($Artifact in $Artifacts) {
    if (-not (Test-Path -LiteralPath $Artifact -PathType Leaf)) {
        throw "Missing release artifact: $Artifact"
    }
}

$Lines = foreach ($Artifact in $Artifacts) {
    $Hash = (Get-FileHash -LiteralPath $Artifact -Algorithm SHA256).Hash.ToLowerInvariant()
    "$Hash  $([IO.Path]::GetFileName($Artifact))"
}
Set-Content -LiteralPath (Join-Path $Dist "SHA256SUMS") -Value $Lines -Encoding ascii
Get-Item -LiteralPath $Artifacts | Select-Object Name, Length
