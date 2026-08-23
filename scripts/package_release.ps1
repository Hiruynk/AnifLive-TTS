param(
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Dist = Join-Path $Project "dist"
New-Item -ItemType Directory -Path $Dist -Force | Out-Null
$Stage = Join-Path $Dist (".release-stage-" + [guid]::NewGuid().ToString("N"))
$Bundle = Join-Path $Stage "AnifLive-TTS"
New-Item -ItemType Directory -Path $Bundle -Force | Out-Null

try {
    $Status = & git -C $Project status --porcelain
    if ($LASTEXITCODE -ne 0) { throw "git status failed with exit code $LASTEXITCODE" }
    if ($Status) {
        throw "Release packaging requires a completely clean Git working tree."
    }

    & (Join-Path $Project ".venv\Scripts\python.exe") `
        (Join-Path $Project "scripts\check_release_security.py") `
        --expected-version $Version
    if ($LASTEXITCODE -ne 0) {
        throw "Release security gate failed with exit code $LASTEXITCODE"
    }

    $SourceArchive = Join-Path $Stage "source.zip"
    & git -C $Project archive --format=zip --output=$SourceArchive HEAD
    if ($LASTEXITCODE -ne 0) { throw "git archive failed with exit code $LASTEXITCODE" }
    Expand-Archive -LiteralPath $SourceArchive -DestinationPath $Bundle

    $Zip = Join-Path $Dist "AnifLive-TTS-v$Version-docker-source-bundle.zip"
    if (Test-Path -LiteralPath $Zip) {
        $Backup = "$Zip.previous-$(Get-Date -Format yyyyMMddTHHmmss)"
        Move-Item -LiteralPath $Zip -Destination $Backup
    }
    Compress-Archive -LiteralPath $Bundle -DestinationPath $Zip -CompressionLevel Optimal
    $ReleaseNotesName = "RELEASE_NOTES_v$Version.md"
    Copy-Item -LiteralPath (Join-Path $Bundle $ReleaseNotesName) `
        -Destination (Join-Path $Dist $ReleaseNotesName) -Force

    Get-Item -LiteralPath $Zip, (Join-Path $Dist $ReleaseNotesName) |
        Select-Object Name, Length
}
finally {
    $ResolvedDist = [IO.Path]::GetFullPath($Dist).TrimEnd('\') + '\'
    $ResolvedStage = [IO.Path]::GetFullPath($Stage)
    if ($ResolvedStage.StartsWith($ResolvedDist, [StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $ResolvedStage)) {
        Remove-Item -LiteralPath $ResolvedStage -Recurse -Force
    }
}
