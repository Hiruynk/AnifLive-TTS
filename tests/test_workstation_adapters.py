from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aniflive_tts.workstation import (
    JOB_PROJECT_KINDS,
    JOB_RESOURCE_CLASSES,
    JOB_TYPES,
    WorkstationError,
    WorkstationStore,
)
from aniflive_tts.workstation_adapters import (
    ADAPTER_REGISTRY,
    AdapterCancelled,
    AdapterEvent,
    AllowedCommand,
    CommandAllowlist,
    run_adapter,
)


def _records(
    tmp_path: Path,
    *,
    project_kind: str,
    job_type: str,
    config: dict | None = None,
    parameters: dict | None = None,
):
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(
        kind=project_kind, name=f"{project_kind} project", config=config or {}
    )
    job = store.create_job(
        job_type=job_type, project_id=project["id"], parameters=parameters or {}
    )
    return project, job


def test_registry_is_explicit_and_covers_every_job_type() -> None:
    assert frozenset(ADAPTER_REGISTRY) == JOB_TYPES
    for job_type, adapter in ADAPTER_REGISTRY.items():
        assert adapter.job_type == job_type
        assert adapter.project_kinds == JOB_PROJECT_KINDS[job_type]
        assert adapter.resource_class == JOB_RESOURCE_CLASSES[job_type]


def test_adapter_rejects_mismatched_project_and_resource_contract(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    project, job = _records(
        tmp_path,
        project_kind="dataset",
        job_type="dataset.inventory",
        config={"source": str(source)},
    )
    wrong_project = dict(project, id="dataset_00000000-0000-0000-0000-000000000001")
    with pytest.raises(WorkstationError, match="IDs do not match"):
        run_adapter(
            job,
            wrong_project,
            manifest_root=tmp_path / "manifests",
            allowed_path_roots=(tmp_path,),
        )
    wrong_resource = dict(job, resource_class="gpu-exclusive")
    with pytest.raises(WorkstationError, match="requires resource class"):
        run_adapter(
            wrong_resource,
            project,
            manifest_root=tmp_path / "manifests",
            allowed_path_roots=(tmp_path,),
        )
    with pytest.raises(WorkstationError, match="job_id is malformed"):
        run_adapter(
            dict(job, id="../../outside"),
            project,
            manifest_root=tmp_path / "manifests",
            allowed_path_roots=(tmp_path,),
        )


def test_dataset_inventory_runs_locally_and_publishes_progress(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "voice.wav").write_bytes(b"RIFF")
    (source / "notes.txt").write_text("not media", encoding="utf-8")
    project, job = _records(
        tmp_path,
        project_kind="dataset",
        job_type="dataset.inventory",
        config={"source": str(source)},
    )
    events: list[AdapterEvent] = []
    result = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        emit_event=events.append,
    )
    assert result.outcome == "completed"
    assert result.payload["execution_performed"] is True
    assert result.payload["media_files"] == 1
    assert result.payload["examined_files"] == 2
    assert events[0].progress == 0.02
    assert events[-1].progress == 1.0


def test_dataset_inventory_observes_cooperative_cancellation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(4):
        (source / f"{index}.wav").write_bytes(b"RIFF")
    project, job = _records(
        tmp_path,
        project_kind="dataset",
        job_type="dataset.inventory",
        config={"source": str(source)},
    )
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(AdapterCancelled, match="cancellation"):
        run_adapter(
            job,
            project,
            manifest_root=tmp_path / "manifests",
            allowed_path_roots=(tmp_path,),
            cancel_requested=cancelled,
        )


def test_dataset_process_prepares_fixed_linux_worker_manifest_and_handoff(
    tmp_path: Path,
) -> None:
    source = tmp_path / "voice.mp4"
    source.write_bytes(b"fixed-media")
    asr_model = tmp_path / "asr-model"
    asr_model.mkdir()
    project, job = _records(
        tmp_path,
        project_kind="dataset",
        job_type="dataset.process",
        config={
            "source": str(source),
            "asr_model": str(asr_model),
            "enable_afftdn": True,
        },
    )
    output = tmp_path / "docker-output"
    output.mkdir()
    canonical = output / "dataset-pipeline" / "canonical.wav"
    canonical.parent.mkdir()
    canonical.write_bytes(b"RIFFfixed")
    digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
    observed: dict[str, object] = {}

    class Execution:
        output_root = output
        image_digest = "sha256:" + "a" * 64
        artifacts = (
            {
                "kind": "dataset",
                "relative_path": "dataset-pipeline/canonical.wav",
                "sha256": digest,
                "size_bytes": canonical.stat().st_size,
            },
        )

        def as_dict(self):
            return {
                "backend": "linux-docker",
                "payload": {"schema": "aniflive-dataset-pipeline-v1"},
                "artifacts": [dict(self.artifacts[0])],
            }

    class Broker:
        def describe(self, job_type: str):
            assert job_type == "dataset.process"
            return {"available": True, "kind": "linux-docker"}

        def execute(self, job_type: str, **kwargs):
            observed.update(kwargs)
            assert job_type == "dataset.process"
            manifest = json.loads(Path(kwargs["manifest_path"]).read_text(encoding="utf-8"))
            assert manifest["container_input_paths"] == {
                "asr_model": "/aniflive/input/asr_model",
                "source": "/aniflive/input/source.mp4",
            }
            assert manifest["settings"] == {"enable_afftdn": True}
            return Execution()

    result = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        worker_broker=Broker(),
    )
    assert result.outcome == "completed"
    assert result.payload["execution_performed"] is True
    assert observed["input_paths"] == {
        "asr_model": str(asr_model.resolve()),
        "source": str(source.resolve()),
    }
    assert result.artifact_handoff is not None
    assert result.artifact_handoff.artifacts[0]["kind"] == "dataset"


def test_target_speaker_adapter_requires_generic_speaker_and_fsmn_assets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    reference = tmp_path / "reference.wav"
    speaker = tmp_path / "speaker-component"
    vad = tmp_path / "vad-component"
    diarization = tmp_path / "diarization-component"
    source.write_bytes(b"RIFFsource")
    reference.write_bytes(b"RIFFreference")
    speaker.mkdir()
    vad.mkdir()
    diarization.mkdir()
    project, job = _records(
        tmp_path,
        project_kind="dataset",
        job_type="dataset.target-speaker",
        config={
            "acquisition_mode": "target-speaker",
            "source": str(source),
            "reference": str(reference),
            "speaker_engine": str(speaker),
            "vad_model": str(vad),
            "diarization_model": str(diarization),
        },
    )

    result = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )

    assert result.outcome == "prepared-only"
    assert result.payload["missing_inputs"] == ()
    manifest = json.loads(
        Path(result.payload["manifest_path"]).read_text(encoding="utf-8")
    )
    assert set(manifest["input_paths"]) >= {
        "source",
        "reference",
        "speaker_engine",
        "vad_model",
        "diarization_model",
    }

    missing_project, missing_job = _records(
        tmp_path / "missing",
        project_kind="dataset",
        job_type="dataset.target-speaker",
        config={
            "acquisition_mode": "target-speaker",
            "source": str(source),
            "reference": str(reference),
            "speaker_engine": str(speaker),
        },
    )
    missing = run_adapter(
        missing_job,
        missing_project,
        manifest_root=tmp_path / "missing-manifests",
        allowed_path_roots=(tmp_path,),
    )
    assert missing.payload["readiness"] == "blocked"
    assert "vad_model" in missing.payload["missing_inputs"]


def test_tse_manifest_preserves_manual_separation_ranges_as_settings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    reference = tmp_path / "reference.wav"
    source.write_bytes(b"RIFFsource")
    reference.write_bytes(b"RIFFreference")
    package = tmp_path / "package"
    separator = tmp_path / "separator"
    package.mkdir()
    separator.mkdir()
    ranges = [
        {"start_seconds": 0.25, "end_seconds": 1.5},
        {"start_seconds": 2.0, "end_seconds": 3.0},
    ]
    project, job = _records(
        tmp_path,
        project_kind="tse",
        job_type="tse.prepare",
        config={
            "source": str(source),
            "reference": str(reference),
            "model_package": str(package),
            "separation_model": str(separator),
            "separation_segments": ranges,
        },
    )
    result = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )
    manifest = json.loads(
        Path(result.payload["manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["settings"]["separation_segments"] == ranges
    assert "separation_segments" not in manifest["container_input_paths"]


def test_preparation_adapter_is_honest_and_manifest_is_immutable(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    project, job = _records(
        tmp_path,
        project_kind="training",
        job_type="training.prepare",
        config={
            "dataset": str(dataset),
            "preset": "balanced",
            "auto_build_production": True,
            "model_id": "voice-v2",
            "reference_selection_policy": "reviewed-quality-speaker-centroid-v2",
            "reference_status": "pending-checkpoint-selection",
            "speaker_component": str(tmp_path / "speaker-component"),
            "voice_profile": "default",
        },
        parameters={"parent_artifact_ids": []},
    )
    events: list[AdapterEvent] = []
    first = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        emit_event=events.append,
    )
    second = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )
    assert first.outcome == "prepared-only"
    assert first.payload["execution_performed"] is False
    assert first.payload["readiness"] == "blocked"
    assert first.payload["backend_available"] is False
    assert first.payload["manifest_sha256"] == second.payload["manifest_sha256"]
    manifest = json.loads(Path(first.payload["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["disposition"] == "prepared-only"
    assert manifest["backend"]["execution_performed"] is False
    assert manifest["backend"]["available"] is False
    assert manifest["settings"] == {"preset": "balanced"}
    assert events[-1].level == "warning"


def test_checkpoint_selection_materializes_all_verified_dependency_paths(
    tmp_path: Path,
) -> None:
    paths = {
        key: tmp_path / key
        for key in (
            "checkpoint_candidates",
            "dataset",
            "shared_dir",
            "asr_model",
            "speaker_component",
        )
    }
    for path in paths.values():
        path.mkdir()
    project, job = _records(
        tmp_path,
        project_kind="training",
        job_type="checkpoint.select",
        config={key: str(path) for key, path in paths.items()},
    )

    result = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )

    manifest = json.loads(Path(result.payload["manifest_path"]).read_text(encoding="utf-8"))
    assert result.payload["missing_inputs"] == ()
    assert manifest["input_paths"] == {
        key: str(path.resolve()) for key, path in paths.items()
    }
    assert set(manifest["container_input_paths"]) == set(paths)


def test_checkpoint_engine_preparation_requires_conversion_inputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    project, job = _records(
        tmp_path,
        project_kind="evaluation",
        job_type="engine.prepare",
        config={"checkpoint": str(checkpoint)},
    )

    result = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )

    assert set(result.payload["missing_inputs"]) == {
        "shared_dir",
        "model_package | reference",
    }


def test_checkpoint_engine_preparation_accepts_generic_package_template(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    shared = tmp_path / "shared"
    package = tmp_path / "package"
    for path in (checkpoint, shared, package):
        path.mkdir()
    project, job = _records(
        tmp_path,
        project_kind="evaluation",
        job_type="engine.prepare",
        config={
            "checkpoint": str(checkpoint),
            "shared_dir": str(shared),
            "model_package": str(package),
        },
    )

    result = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )

    assert result.payload["missing_inputs"] == ()


def test_adapter_does_not_mount_irrelevant_project_paths(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    shared = tmp_path / "shared"
    package = tmp_path / "package"
    sealed_dataset = tmp_path / "sealed-dataset"
    for path in (checkpoint, shared, package, sealed_dataset):
        path.mkdir()
    project, job = _records(
        tmp_path,
        project_kind="evaluation",
        job_type="engine.prepare",
        config={
            "checkpoint": str(checkpoint),
            "shared_dir": str(shared),
            "model_package": str(package),
            "dataset": str(sealed_dataset),
        },
    )

    result = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )

    manifest = json.loads(Path(result.payload["manifest_path"]).read_text(encoding="utf-8"))
    assert set(manifest["input_paths"]) == {
        "checkpoint",
        "shared_dir",
        "model_package",
    }
    assert "dataset" not in manifest["container_input_paths"]


def test_model_packaging_rejects_an_engine_directory_without_package_contract(
    tmp_path: Path,
) -> None:
    engine = tmp_path / "engine"
    engine.mkdir()
    project, job = _records(
        tmp_path,
        project_kind="evaluation",
        job_type="model.package",
        config={"engine_dir": str(engine)},
    )

    result = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )

    assert result.payload["missing_inputs"] == ("model_package",)


def test_preparation_manifest_creation_is_concurrent_and_idempotent(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pretrained_gpt = tmp_path / "pretrained-gpt.ckpt"
    pretrained_sovits_g = tmp_path / "pretrained-sovits-g.pth"
    pretrained_sovits_d = tmp_path / "pretrained-sovits-d.pth"
    for path in (pretrained_gpt, pretrained_sovits_g, pretrained_sovits_d):
        path.write_bytes(b"fixed-test-weight")
    project, job = _records(
        tmp_path,
        project_kind="training",
        job_type="training.prepare",
        config={
            "dataset": str(dataset),
            "pretrained_gpt": str(pretrained_gpt),
            "pretrained_sovits_g": str(pretrained_sovits_g),
            "pretrained_sovits_d": str(pretrained_sovits_d),
            "shared_dir": str(dataset),
        },
    )

    def prepare(_index: int):
        return run_adapter(
            job,
            project,
            manifest_root=tmp_path / "manifests",
            allowed_path_roots=(tmp_path,),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(prepare, range(16)))
    assert len({result.payload["manifest_sha256"] for result in results}) == 1
    assert len({result.payload["manifest_path"] for result in results}) == 1


def test_preparation_records_trusted_backend_availability_without_running_it(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pretrained_gpt = tmp_path / "pretrained-gpt.ckpt"
    pretrained_sovits_g = tmp_path / "pretrained-sovits-g.pth"
    pretrained_sovits_d = tmp_path / "pretrained-sovits-d.pth"
    for path in (pretrained_gpt, pretrained_sovits_g, pretrained_sovits_d):
        path.write_bytes(b"fixed-test-weight")
    project, job = _records(
        tmp_path,
        project_kind="training",
        job_type="training.prepare",
        config={
            "dataset": str(dataset),
            "pretrained_gpt": str(pretrained_gpt),
            "pretrained_sovits_g": str(pretrained_sovits_g),
            "pretrained_sovits_d": str(pretrained_sovits_d),
            "shared_dir": str(dataset),
        },
    )
    executable = Path(sys.executable).resolve()
    commands = CommandAllowlist(
        {
            "training.prepare": AllowedCommand(
                "training-worker-v1", executable, ("-c", "print('fixed command')")
            )
        },
        executable_roots=(executable.parent,),
    )
    result = run_adapter(
        job,
        project,
        manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
        command_allowlist=commands,
    )
    assert result.payload["readiness"] == "ready"
    assert result.payload["backend_available"] is True
    assert result.payload["command_id"] == "training-worker-v1"
    assert result.payload["execution_performed"] is False


def test_user_parameters_cannot_supply_an_executable(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    project, job = _records(
        tmp_path,
        project_kind="training",
        job_type="training.prepare",
        config={"dataset": str(dataset)},
        parameters={"executable": str(Path(sys.executable))},
    )
    with pytest.raises(WorkstationError, match="trusted command allowlist"):
        run_adapter(
            job,
            project,
            manifest_root=tmp_path / "manifests",
            allowed_path_roots=(tmp_path,),
        )


def test_input_paths_must_be_local_existing_and_allowlisted(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    project, job = _records(
        tmp_path,
        project_kind="tse",
        job_type="tse.prepare",
        config={"source": str(outside), "reference": str(outside)},
    )
    with pytest.raises(WorkstationError, match="outside the configured local roots"):
        run_adapter(
            job,
            project,
            manifest_root=tmp_path / "manifests",
            allowed_path_roots=(allowed,),
        )


def test_command_allowlist_uses_fixed_argv_and_never_a_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = Path(sys.executable).resolve()
    commands = CommandAllowlist(
        {
            "training.prepare": AllowedCommand(
                "fixed-test-command", executable, ("-c", "print('allowlisted')")
            )
        },
        executable_roots=(executable.parent,),
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    events: list[AdapterEvent] = []
    invoked: dict[str, object] = {}
    original_popen = subprocess.Popen

    def observed_popen(*args, **kwargs):
        invoked["argv"] = args[0]
        invoked["shell"] = kwargs.get("shell")
        return original_popen(*args, **kwargs)

    monkeypatch.setattr("aniflive_tts.workstation_adapters.subprocess.Popen", observed_popen)
    result = commands.execute(
        "training.prepare",
        manifest_path=manifest,
        cancel_requested=lambda: False,
        emit_event=events.append,
    )
    assert result["exit_code"] == 0
    assert result["output"].strip() == "allowlisted"
    assert events[-1].progress == 1.0
    assert invoked["shell"] is False
    assert invoked["argv"] == [str(executable), "-c", "print('allowlisted')"]


def test_command_allowlist_rejects_relative_or_untrusted_executables(tmp_path: Path) -> None:
    executable = Path(sys.executable).resolve()
    with pytest.raises(WorkstationError, match="absolute"):
        CommandAllowlist(
            {"training.prepare": AllowedCommand("relative", Path("python"))},
            executable_roots=(executable.parent,),
        )
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    with pytest.raises(WorkstationError, match="outside"):
        CommandAllowlist(
            {"training.prepare": AllowedCommand("outside", executable)},
            executable_roots=(trusted,),
        )

def test_evaluation_adapter_passes_baseline_and_filters_project_metadata(tmp_path: Path):
    from aniflive_tts.workstation_evaluation import _ALLOWED_SETTINGS

    package, shared, asr = (tmp_path / name for name in ("package", "shared", "asr"))
    for directory in (package, shared, asr):
        directory.mkdir()
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}")
    project, job = _records(
        tmp_path, project_kind="evaluation", job_type="evaluation.prepare",
        config={
            "model_package": str(package), "shared_dir": str(shared),
            "asr_model": str(asr), "baseline_report": str(baseline),
            "source_training_project_id": "training-source",
            "source_package_job_id": "package-source",
            "qualification_status": "blocked", "blocked_reason": "pending baseline",
            "benchmark_language": "ja", "benchmark_sessions": 1,
        },
        parameters={"benchmark_sessions": 2},
    )
    result = run_adapter(
        job, project, manifest_root=tmp_path / "manifests",
        allowed_path_roots=(tmp_path,),
    )
    manifest = json.loads(Path(result.payload["manifest_path"]).read_text())
    assert manifest["input_paths"]["baseline_report"] == str(baseline)
    assert manifest["container_input_paths"]["baseline_report"] == "/aniflive/input/baseline_report.json"
    assert manifest["settings"] == {"benchmark_language": "ja", "benchmark_sessions": 2}
    assert set(ADAPTER_REGISTRY["evaluation.prepare"].spec.setting_keys) == _ALLOWED_SETTINGS


def test_evaluation_baseline_cannot_escape_allowed_roots(tmp_path: Path):
    allowed = tmp_path / "inputs"
    allowed.mkdir()
    baseline = tmp_path / "outside.json"
    baseline.write_text("{}")
    project, job = _records(
        tmp_path, project_kind="evaluation", job_type="evaluation.prepare",
        config={
            "model_package": str(allowed), "shared_dir": str(allowed),
            "asr_model": str(allowed), "baseline_report": str(baseline),
        },
    )
    with pytest.raises(WorkstationError, match="outside|allowed"):
        run_adapter(job, project, manifest_root=tmp_path / "manifests", allowed_path_roots=(allowed,))
