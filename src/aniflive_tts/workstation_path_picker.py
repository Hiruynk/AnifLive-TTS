from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal


class WorkstationPathPickerError(RuntimeError):
    pass


PickerKind = Literal["file", "files", "directory", "save-file"]


def _windows_picker_script() -> str:
    return r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
$owner = New-Object System.Windows.Forms.Form
$owner.ShowInTaskbar = $false
$owner.TopMost = $true
$owner.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$owner.Size = New-Object System.Drawing.Size(1, 1)
$owner.Opacity = 0
$owner.Show()
$owner.Activate()
$kind = $env:ANIFLIVE_TTS_PICKER_KIND
$title = $env:ANIFLIVE_TTS_PICKER_TITLE
$initial = $env:ANIFLIVE_TTS_PICKER_INITIAL
$filter = $env:ANIFLIVE_TTS_PICKER_FILTER

if ($kind -eq 'directory') {
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = $title
    if ($dialog.PSObject.Properties.Name -contains 'UseDescriptionForTitle') {
        $dialog.UseDescriptionForTitle = $true
    }
    if ($initial) { $dialog.SelectedPath = $initial }
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
        @($dialog.SelectedPath) | ConvertTo-Json -Compress
    } else {
        @() | ConvertTo-Json -Compress
    }
    exit 0
}

if ($kind -eq 'save-file') {
    $dialog = New-Object System.Windows.Forms.SaveFileDialog
} else {
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Multiselect = ($kind -eq 'files')
    $dialog.CheckFileExists = $true
}
$dialog.Title = $title
$dialog.RestoreDirectory = $true
if ($initial) { $dialog.InitialDirectory = $initial }
if ($filter) { $dialog.Filter = $filter }
if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
    if ($kind -eq 'files') { @($dialog.FileNames) | ConvertTo-Json -Compress }
    else { @($dialog.FileName) | ConvertTo-Json -Compress }
} else {
    @() | ConvertTo-Json -Compress
}
"""


def pick_workstation_paths(
    *,
    kind: PickerKind,
    title: str,
    initial_directory: Path | None = None,
    file_filter: str = "All files (*.*)|*.*",
) -> list[str]:
    if kind not in {"file", "files", "directory", "save-file"}:
        raise WorkstationPathPickerError("Unsupported path picker kind")
    if not isinstance(title, str) or not title.strip() or len(title) > 120:
        raise WorkstationPathPickerError("Path picker title is malformed")
    if not isinstance(file_filter, str) or not file_filter or len(file_filter) > 500:
        raise WorkstationPathPickerError("Path picker filter is malformed")
    if sys.platform != "win32":
        raise WorkstationPathPickerError(
            "The native path picker is available from the local Windows Studio control plane"
        )

    initial = ""
    if initial_directory is not None:
        candidate = Path(initial_directory).expanduser().absolute()
        if candidate.is_dir():
            initial = str(candidate)

    environment = os.environ.copy()
    environment.update(
        {
            "ANIFLIVE_TTS_PICKER_KIND": kind,
            "ANIFLIVE_TTS_PICKER_TITLE": title.strip(),
            "ANIFLIVE_TTS_PICKER_INITIAL": initial,
            "ANIFLIVE_TTS_PICKER_FILTER": file_filter,
        }
    )
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-Sta",
                "-Command",
                _windows_picker_script(),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
            env=environment,
            creationflags=creation_flags,
        )
        payload = json.loads(result.stdout.strip() or "[]")
    except (json.JSONDecodeError, OSError, subprocess.SubprocessError) as error:
        raise WorkstationPathPickerError("The native path picker could not be opened") from error
    if isinstance(payload, str):
        payload = [payload]
    if not isinstance(payload, list) or any(not isinstance(value, str) for value in payload):
        raise WorkstationPathPickerError("The native path picker returned malformed data")
    return [value for value in payload if value]


__all__ = ["WorkstationPathPickerError", "pick_workstation_paths"]
