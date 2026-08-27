param(
    [string]$Version = "1.2.0",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = if ([string]::IsNullOrWhiteSpace($Python)) {
    Join-Path $Project ".venv\Scripts\python.exe"
} else {
    (Resolve-Path -LiteralPath $Python).Path
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}
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

    & $Python `
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
    $ReleaseNotesPath = Join-Path $Dist $ReleaseNotesName
    Copy-Item -LiteralPath (Join-Path $Bundle $ReleaseNotesName) `
        -Destination $ReleaseNotesPath -Force

    $ChecksumPath = Join-Path $Dist "SHA256SUMS-v$Version-source"
    $ChecksumLines = Get-Item -LiteralPath $Zip, $ReleaseNotesPath | ForEach-Object {
        $Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$Hash  $($_.Name)"
    }
    Set-Content -LiteralPath $ChecksumPath -Value $ChecksumLines -Encoding ascii

    Get-Item -LiteralPath $Zip, $ReleaseNotesPath, $ChecksumPath |
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
