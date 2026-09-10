from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml

GPT_SOVITS_TRAINING_REVISION = "48b1a0169a28582a8984402f82cf438d3bfa6aca"
_TRAINING_ROOT = Path("/opt/aniflive-tts/gpt-sovits")
_REVISION_FILE = _TRAINING_ROOT / "ANIFLIVE_TTS_GPT_SOVITS_REVISION"
_DATASET_ENTRIES = (
    "2-name2text.txt",
    "3-bert",
    "4-cnhubert",
    "5-wav32k",
    "6-name2semantic.tsv",
    "7-sv_cn",
)
_ALLOWED_SETTINGS = frozenset(
    {
        "epoch_policy",
        "experiment_name",
        "gpt_batch_size",
        "gpt_epochs",
        "gpt_learning_rate",
        "gradient_checkpointing",
        "preset",
        "save_every_epoch",
        "seed",
        "sovits_batch_size",
        "sovits_epochs",
        "sovits_learning_rate",
        "stage",
        "source_dataset_id",
        "source_dataset_artifact_id",
        "source_dataset_manifest_sha256",
        "training_bundle_sha256",
    }
)
_LINEAGE_SETTINGS = frozenset(
    {
        "source_dataset_id",
        "source_dataset_artifact_id",
        "source_dataset_manifest_sha256",
        "training_bundle_sha256",
    }
)
_ADVANCED_FIELDS = frozenset(
    {
        "gpt_batch_size",
        "gpt_epochs",
        "gpt_learning_rate",
        "gradient_checkpointing",
        "save_every_epoch",
        "seed",
        "sovits_batch_size",
        "sovits_epochs",
        "sovits_learning_rate",
    }
)
_EPOCH_PATTERN = re.compile(r"(?:Train\s+)?Epoch:?\s*(\d+)", re.IGNORECASE)
_CHECKPOINT_EPOCH_PATTERN = re.compile(
    r"(?:^|[-_])e(?:poch)?[-_]?(\d+)(?=[-_.]|$)", re.IGNORECASE
)


class TrainingWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingPlan:
    preset: str
    stage: str
    experiment_name: str
    gpt_epochs: int
    sovits_epochs: int
    gpt_batch_size: int
    sovits_batch_size: int
    gpt_learning_rate: float
    sovits_learning_rate: float
    save_every_epoch: int
    seed: int
    gradient_checkpointing: bool
    epoch_policy: str = "fixed"


_PRESET_VALUES: dict[str, dict[str, Any]] = {
    "quick": {
        "gpt_epochs": 4,
        "sovits_epochs": 4,
        "gpt_batch_size": 4,
        "sovits_batch_size": 4,
        "gpt_learning_rate": 0.01,
        "sovits_learning_rate": 0.0001,
        "save_every_epoch": 1,
        "seed": 1234,
        "gradient_checkpointing": False,
    },
    "balanced": {
        "gpt_epochs": 12,
        "sovits_epochs": 8,
        "gpt_batch_size": 6,
        "sovits_batch_size": 4,
        "gpt_learning_rate": 0.01,
        "sovits_learning_rate": 0.0001,
        "save_every_epoch": 1,
        "seed": 1234,
        "gradient_checkpointing": False,
    },
    "high-quality": {
        "gpt_epochs": 20,
        "sovits_epochs": 12,
        "gpt_batch_size": 6,
        "sovits_batch_size": 4,
        "gpt_learning_rate": 0.01,
        "sovits_learning_rate": 0.0001,
        "save_every_epoch": 1,
        "seed": 1234,
        "gradient_checkpointing": True,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_epoch(path: Path, label: str) -> int:
    match = _CHECKPOINT_EPOCH_PATTERN.search(path.name)
    if match is None:
        raise TrainingWorkerError(f"{label} checkpoint has no parseable epoch: {path.name}")
    return int(match.group(1))


def _write_checkpoint_candidates_manifest(
    output: Path,
    *,
    gpt_weights: Sequence[Path],
    sovits_weights: Sequence[Path],
) -> tuple[Path, dict[str, Any]]:
    if not gpt_weights:
        raise TrainingWorkerError("training produced no GPT checkpoint candidates")
    if not sovits_weights:
        raise TrainingWorkerError("training produced no SoVITS checkpoint candidates")

    def record(path: Path, label: str) -> dict[str, Any]:
        resolved = path.resolve(strict=True)
        try:
            relative = resolved.relative_to(output.resolve(strict=True)).as_posix()
        except ValueError as error:
            raise TrainingWorkerError("deployment checkpoint escaped the worker output") from error
        return {
            "epoch": _checkpoint_epoch(path, label),
            "relative_path": relative,
            "sha256": _sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }

    payload = {
        "schema": "aniflive-tts-v2proplus-checkpoint-candidates-v1",
        "model_family": "gsv-v2proplus",
        "selection_required": True,
        "gpt": sorted(
            (record(path, "GPT") for path in gpt_weights),
            key=lambda row: (row["epoch"], row["relative_path"]),
        ),
        "sovits": sorted(
            (record(path, "SoVITS") for path in sovits_weights),
            key=lambda row: (row["epoch"], row["relative_path"]),
        ),
    }
    path = output / "checkpoint-candidates.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, payload


def _bounded_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise TrainingWorkerError(f"{field} must be an integer between {minimum} and {maximum}")
    return value


def _bounded_float(value: Any, field: str, minimum: float, maximum: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise TrainingWorkerError(f"{field} must be finite and between {minimum} and {maximum}")
    return float(value)


def _experiment_name(value: Any, fallback: str) -> str:
    if value is None:
        value = fallback
    if not isinstance(value, str):
        raise TrainingWorkerError("experiment_name must be text")
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-_")
    if not normalized or len(normalized) > 64:
        raise TrainingWorkerError("experiment_name must form a 1-64 character safe identifier")
    return normalized


def resolve_training_plan(settings: Mapping[str, Any], *, project_id: str) -> TrainingPlan:
    if not isinstance(settings, Mapping):
        raise TrainingWorkerError("training settings must be a JSON object")
    unknown = set(settings) - _ALLOWED_SETTINGS
    if unknown:
        raise TrainingWorkerError("unsupported training settings: " + ", ".join(sorted(unknown)))
    settings = {key: value for key, value in settings.items() if key not in _LINEAGE_SETTINGS}
    raw_preset = settings.get("preset", "balanced")
    if not isinstance(raw_preset, str):
        raise TrainingWorkerError("preset must be text")
    preset = raw_preset.strip().lower().replace("_", "-").replace(" ", "-")
    if preset not in {*_PRESET_VALUES, "advanced"}:
        raise TrainingWorkerError("preset must be Quick, Balanced, High Quality or Advanced")
    stage = settings.get("stage", "both")
    if stage not in {"both", "gpt", "sovits"}:
        raise TrainingWorkerError("stage must be both, gpt or sovits")
    if preset != "advanced":
        overridden = _ADVANCED_FIELDS.intersection(settings)
        if overridden:
            raise TrainingWorkerError(
                f"{preset} is a fixed preset; custom fields require Advanced: "
                + ", ".join(sorted(overridden))
            )
        values = dict(_PRESET_VALUES[preset])
    else:
        values = {
            "gpt_epochs": _bounded_int(settings.get("gpt_epochs", 20), "gpt_epochs", 1, 100),
            "sovits_epochs": _bounded_int(
                settings.get("sovits_epochs", 12), "sovits_epochs", 1, 100
            ),
            "gpt_batch_size": _bounded_int(
                settings.get("gpt_batch_size", 6), "gpt_batch_size", 1, 32
            ),
            "sovits_batch_size": _bounded_int(
                settings.get("sovits_batch_size", 4), "sovits_batch_size", 1, 32
            ),
            "gpt_learning_rate": _bounded_float(
                settings.get("gpt_learning_rate", 0.01),
                "gpt_learning_rate",
                1e-6,
                0.05,
            ),
            "sovits_learning_rate": _bounded_float(
                settings.get("sovits_learning_rate", 0.0001),
                "sovits_learning_rate",
                1e-7,
                0.001,
            ),
            "save_every_epoch": _bounded_int(
                settings.get("save_every_epoch", 1), "save_every_epoch", 1, 10
            ),
            "seed": _bounded_int(settings.get("seed", 1234), "seed", 0, 2_147_483_647),
            "gradient_checkpointing": settings.get("gradient_checkpointing", False),
        }
        if not isinstance(values["gradient_checkpointing"], bool):
            raise TrainingWorkerError("gradient_checkpointing must be boolean")
    policy = settings.get("epoch_policy", "fixed" if preset == "advanced" else "adaptive")
    if not isinstance(policy, str) or policy not in {"adaptive", "fixed"}:
        raise TrainingWorkerError("epoch_policy must be adaptive or fixed")
    return TrainingPlan(
        epoch_policy=policy,
        preset=preset,
        stage=str(stage),
        experiment_name=_experiment_name(settings.get("experiment_name"), project_id),
        **values,
    )


def plan_training_budget(
    plan: TrainingPlan, *, duration_seconds: float, clips: int, resumed: bool = False,
) -> tuple[TrainingPlan, dict[str, Any]]:
    """A bounded initial search budget, not a detector of overfit or convergence."""
    if (
        isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(duration_seconds) or duration_seconds <= 0
        or isinstance(clips, bool) or not isinstance(clips, int) or clips < 2
    ):
        raise TrainingWorkerError("training budget requires positive audio duration and two clips")
    if resumed and plan.epoch_policy != "fixed":
        raise TrainingWorkerError(
            "Resume requires epoch_policy=fixed and the previously resolved epoch totals"
        )
    scale = min(1.0, math.sqrt(600.0 / duration_seconds))
    stages = {}
    resolved = {}
    for stage in ("gpt", "sovits"):
        ceiling = getattr(plan, f"{stage}_epochs")
        batch = getattr(plan, f"{stage}_batch_size")
        steps = math.ceil(clips / batch)
        nominal_update_budget = math.ceil(200 / batch) * ceiling
        # Keep two candidates where the caller's ceiling permits it. Small
        # datasets never gain extra repetitions merely to meet an update target.
        floor = min(2, ceiling)
        recommended = max(floor, min(
            ceiling, math.ceil(ceiling * scale),
            max(1, nominal_update_budget // steps),
        ))
        epochs = recommended if plan.epoch_policy == "adaptive" else ceiling
        resolved[f"{stage}_epochs"] = epochs
        stages[stage] = {
            "requested_epoch_ceiling": ceiling, "recommended_epochs": recommended,
            "resolved_epochs": epochs, "batch_size": batch,
            "estimated_steps_per_epoch": steps,
            "estimated_updates": steps * epochs,
            "nominal_update_budget": nominal_update_budget,
            "epoch_floor_exceeds_update_budget": steps * epochs > nominal_update_budget,
            "estimated_audio_exposure_seconds": duration_seconds * epochs,
        }
    budget = {
        "schema": "aniflive-training-budget-v1",
        "policy_version": "duration-update-capped-v1",
        "mode": plan.epoch_policy, "resumed": resumed,
        "train_audio_seconds": duration_seconds, "train_clips": clips,
        "reference_audio_seconds": 600.0, "reference_clips": 200,
        "duration_scale": scale, "stages": stages,
        "limited_data": duration_seconds < 120 or clips < 40,
        "validation_checkpoint_selection_required": True,
        "human_audio_quality_review_required": True,
        "convergence_or_overfit_established": False,
        "limitations": (
            "Initial heuristic budget; duration is not speech diversity. Estimated updates "
            "exclude sampler padding and gradient accumulation. Validation and human listening "
            "must assess underfit, overfit, timbre and identity before qualification."
        ),
    }
    return replace(
        plan, **resolved,
        save_every_epoch=1 if plan.epoch_policy == "adaptive" else plan.save_every_epoch,
    ), budget


def measure_training_audio(dataset: Path) -> dict[str, Any]:
    """Read only the audio named by the verified training text inventory."""
    import soundfile as sf

    names = [
        row.split("\t", 1)[0].strip()
        for row in (dataset / "2-name2text.txt").read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    audio_root = (dataset / "5-wav32k").resolve(strict=True)
    total = 0.0
    inventory = []
    for name in names:
        if not name or Path(name).name != name:
            raise TrainingWorkerError("training audio name is unsafe")
        path = audio_root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(audio_root):
            raise TrainingWorkerError("training audio is missing or outside its directory")
        try:
            info = sf.info(str(path))
        except (RuntimeError, OSError) as error:
            raise TrainingWorkerError("training audio header is unreadable") from error
        duration = info.frames / info.samplerate if info.samplerate > 0 else 0
        if not math.isfinite(duration) or duration <= 0:
            raise TrainingWorkerError("training audio has no finite positive duration")
        total += duration
        inventory.append({"name": name, "frames": info.frames, "sample_rate": info.samplerate})
    return {"duration_seconds": total, "clips": len(names), "inventory": inventory}


def _validate_training_input_bundle(dataset: Path) -> tuple[Path, Path, dict[str, Any]]:
    descriptor_path = dataset / "training-input.json"
    if descriptor_path.is_symlink() or not descriptor_path.is_file():
        raise TrainingWorkerError("dataset is neither preprocessed nor a training input bundle")
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TrainingWorkerError("training input descriptor could not be read") from error
    if not isinstance(descriptor, dict) or descriptor.get("schema") not in {
        "aniflive-v2proplus-training-input-v1",
        "aniflive-v2proplus-training-input-v2",
    }:
        raise TrainingWorkerError("training input bundle schema is unsupported")
    is_v2 = descriptor["schema"] == "aniflive-v2proplus-training-input-v2"
    training_root = dataset / "train" if is_v2 else dataset
    training_list = training_root / "voice.list"
    wav_dir = training_root / "wav"
    if training_list.is_symlink() or not training_list.is_file() or wav_dir.is_symlink() or not wav_dir.is_dir():
        raise TrainingWorkerError("training input bundle is incomplete")
    if _sha256_file(training_list) != descriptor.get("training_list_sha256"):
        raise TrainingWorkerError("training input list failed integrity validation")
    files = descriptor.get("audio_files")
    if not isinstance(files, list):
        raise TrainingWorkerError("training input audio inventory is malformed")
    train_files = [
        record
        for record in files
        if isinstance(record, Mapping) and (not is_v2 or record.get("split") == "train")
    ]
    if len(train_files) < 2:
        raise TrainingWorkerError("training input bundle needs at least two train audio files")
    for record in files:
        if not isinstance(record, Mapping):
            raise TrainingWorkerError("training input audio inventory is malformed")
        if is_v2 and record.get("split") != "train":
            continue
        relative = record.get("path")
        sha256 = record.get("sha256")
        expected_prefix = "train/wav/" if is_v2 else "wav/"
        if (
            not isinstance(relative, str)
            or not relative.startswith(expected_prefix)
            or ".." in Path(relative).parts
            or not isinstance(sha256, str)
        ):
            raise TrainingWorkerError("training input audio inventory is unsafe")
        candidate = dataset / relative
        if candidate.is_symlink() or not candidate.is_file() or _sha256_file(candidate) != sha256:
            raise TrainingWorkerError("training input audio failed integrity validation")
    return training_list, wav_dir, descriptor


def _require_regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise TrainingWorkerError(f"missing regular {label}: {path}")
    return path.resolve(strict=True)


def _require_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise TrainingWorkerError(f"missing {label} directory: {path}")
    return path.resolve(strict=True)


def validate_preprocessed_dataset(dataset: Path) -> dict[str, Any]:
    dataset = _require_directory(dataset, "preprocessed dataset")
    resolved: dict[str, Path] = {}
    for name in _DATASET_ENTRIES:
        candidate = dataset / name
        resolved[name] = (
            _require_regular(candidate, f"dataset asset {name}")
            if candidate.suffix in {".txt", ".tsv"}
            else _require_directory(candidate, f"dataset asset {name}")
        )
    text_rows = [
        line
        for line in resolved["2-name2text.txt"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    semantic_rows = [
        line
        for line in resolved["6-name2semantic.tsv"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(text_rows) < 2 or len(semantic_rows) < 3:
        raise TrainingWorkerError("preprocessed dataset needs at least two aligned examples")
    text_names = [line.split("\t", 1)[0].strip() for line in text_rows]
    semantic_names = [line.split("\t", 1)[0].strip() for line in semantic_rows[1:]]
    if (
        any(not name for name in text_names)
        or len(text_names) != len(set(text_names))
        or semantic_names != text_names
    ):
        raise TrainingWorkerError("preprocessed text and semantic rows are not exactly aligned")
    expected_speaker_embeddings = {f"{Path(name).name}.pt" for name in text_names}
    actual_speaker_embeddings = {
        path.name
        for path in resolved["7-sv_cn"].iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_speaker_embeddings != expected_speaker_embeddings:
        missing = sorted(expected_speaker_embeddings - actual_speaker_embeddings)
        raise TrainingWorkerError(
            "speaker-vector preprocessing is incomplete: "
            f"expected {len(expected_speaker_embeddings)}, got {len(actual_speaker_embeddings)}"
            + (f"; missing={', '.join(missing[:8])}" if missing else "")
        )
    return {
        "text_rows": len(text_rows),
        "semantic_rows": max(0, len(semantic_rows) - 1),
        "speaker_embedding_rows": len(actual_speaker_embeddings),
        "assets": {
            name: (
                _sha256_file(path)
                if path.is_file()
                else sum(1 for child in path.iterdir() if child.is_file() and not child.is_symlink())
            )
            for name, path in resolved.items()
        },
    }


def _verify_training_source() -> Path:
    revision = _require_regular(_REVISION_FILE, "training source revision").read_text(
        encoding="ascii"
    ).strip()
    if revision != GPT_SOVITS_TRAINING_REVISION:
        raise TrainingWorkerError("GPT-SoVITS training source revision is not trusted")
    for relative in (
        "GPT_SoVITS/s1_train.py",
        "GPT_SoVITS/s2_train.py",
        "GPT_SoVITS/configs/s1longer-v2.yaml",
        "GPT_SoVITS/configs/s2v2ProPlus.json",
    ):
        _require_regular(_TRAINING_ROOT / relative, f"training source {relative}")
    return _TRAINING_ROOT


def _stage_dataset(dataset: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name in _DATASET_ENTRIES:
        os.symlink(dataset / name, destination / name, target_is_directory=(dataset / name).is_dir())
    (destination / "logs_s2_v2ProPlus").mkdir()


def _copy_resume(resume: Path, *, gpt_root: Path, sovits_root: Path) -> dict[str, int]:
    resume = _require_directory(resume, "resume checkpoint")
    gpt_source = resume / "gpt"
    sovits_source = resume / "sovits"
    gpt_files = sorted(gpt_source.glob("*.ckpt")) if gpt_source.is_dir() else []
    generator_files = sorted(sovits_source.glob("G_*.pth")) if sovits_source.is_dir() else []
    discriminator_files = sorted(sovits_source.glob("D_*.pth")) if sovits_source.is_dir() else []
    if bool(generator_files) != bool(discriminator_files):
        raise TrainingWorkerError("SoVITS resume requires both generator and discriminator checkpoints")
    if {path.name[2:] for path in generator_files} != {path.name[2:] for path in discriminator_files}:
        raise TrainingWorkerError("SoVITS resume checkpoint filenames must form matching G/D pairs")
    if not gpt_files and not (generator_files and discriminator_files):
        raise TrainingWorkerError(
            "resume checkpoint must contain gpt/*.ckpt or paired sovits/G_*.pth and D_*.pth"
        )
    for source in gpt_files:
        _require_regular(source, "GPT resume checkpoint")
        shutil.copy2(source, gpt_root / source.name)
    for source in [*generator_files, *discriminator_files]:
        _require_regular(source, "SoVITS resume checkpoint")
        shutil.copy2(source, sovits_root / source.name)
    return {
        "gpt_checkpoints": len(gpt_files),
        "sovits_generator_checkpoints": len(generator_files),
        "sovits_discriminator_checkpoints": len(discriminator_files),
    }


def _resume_entrypoint(source: Path, output: Path, stage: str) -> Path:
    """Stage a fail-closed resume runner without editing the pinned upstream tree."""
    filename = {"gpt": "s1_train.py", "sovits": "s2_train.py"}.get(stage)
    if filename is None:
        raise TrainingWorkerError("Unknown resume stage")
    original = _require_regular(source / "GPT_SoVITS" / filename, "training entrypoint")
    text = original.read_text(encoding="utf-8")
    original_sha256 = _sha256_file(original)
    if stage == "gpt":
        marker = '    print("ckpt_path:", ckpt_path)\n'
        if text.count(marker) != 1:
            raise TrainingWorkerError("Pinned GPT resume entrypoint contract changed")
        text = text.replace(marker,
            '    if ckpt_path is None:\n'
            '        raise RuntimeError("Required GPT resume checkpoint was not found; refusing fresh training")\n'
            + marker, 1)
    else:
        begin = "    except:  # 如果首次不能加载，加载pretrain\n"
        end = "    # scheduler_g"
        load = "        _, _, _, epoch_str = utils.load_checkpoint("
        increment = "        epoch_str += 1\n"
        if text.count(begin) != 1 or text.count(load) != 2 or text.count(increment) != 1:
            raise TrainingWorkerError("Pinned SoVITS resume entrypoint contract changed")
        start = text.index(begin)
        finish = text.find(end, start)
        if finish < 0:
            raise TrainingWorkerError("Pinned SoVITS scheduler boundary changed")
        text = text[:start] + (
            '    except Exception as error:\n'
            '        raise RuntimeError("Required SoVITS resume failed; refusing pretrained fallback") from error\n\n'
        ) + text[finish:]
        text = text.replace(load, "        _, _, _, discriminator_epoch = utils.load_checkpoint(", 1)
        text = text.replace(increment,
            '        if discriminator_epoch != epoch_str:\n'
            '            raise RuntimeError("SoVITS G/D resume epochs do not match")\n'
            + increment, 1)
    compile(text, filename, "exec")
    root = output / "resume-runners"
    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    target.write_text(text, encoding="utf-8")
    (root / (filename + ".source.json")).write_text(json.dumps({
        "schema": "aniflive-training-resume-runner-v1",
        "source_revision": GPT_SOVITS_TRAINING_REVISION,
        "source_sha256": original_sha256,
        "runner_sha256": _sha256_file(target),
        "stage": stage, "fallback_to_fresh_training": False,
    }, indent=2) + "\n", encoding="utf-8")
    return target


def restore_snapshot_budget(
    root: Path, requested_plan: TrainingPlan, *, source_fingerprint: str,
) -> tuple[TrainingPlan, dict[str, Any], Path]:
    """Restore the already resolved budget; never rerun the adaptive heuristic."""
    from .training_snapshot import TrainingSnapshotError, read_training_snapshot

    try:
        directory, manifest = read_training_snapshot(
            root, expected_source_fingerprint=source_fingerprint,
        )
        document = json.loads((directory / "training-plan.json").read_text(encoding="utf-8"))
        budget = json.loads((directory / "training-budget.json").read_text(encoding="utf-8"))
        if document.get("schema") != "aniflive-training-resume-plan-v1":
            raise TrainingWorkerError("Saved training plan schema is unsupported")
        if document.get("requested_plan") != asdict(requested_plan):
            raise TrainingWorkerError("Training settings changed since the recovery checkpoint")
        raw_plan = document["resolved_plan"]
        checked = resolve_training_plan(
            {**raw_plan, "preset": "advanced", "epoch_policy": "fixed"},
            project_id="resume",
        )
        restored = replace(
            checked, preset=requested_plan.preset, epoch_policy=requested_plan.epoch_policy,
        )
        if asdict(restored) != raw_plan:
            raise TrainingWorkerError("Saved resolved training plan is inconsistent")
        if budget.get("schema") != "aniflive-training-budget-v1":
            raise TrainingWorkerError("Saved training budget schema is unsupported")
        for stage in ("gpt", "sovits"):
            if budget["stages"][stage]["resolved_epochs"] != getattr(restored, stage + "_epochs"):
                raise TrainingWorkerError("Saved training budget does not match checkpoint plan")
            if budget["stages"][stage]["batch_size"] != getattr(restored, stage + "_batch_size"):
                raise TrainingWorkerError("Saved training batch size does not match checkpoint plan")
        stage = manifest["metadata"]["stage"]
        if restored.stage not in {"both", stage}:
            raise TrainingWorkerError("Recovery stage does not belong to this training plan")
        if manifest["metadata"]["epoch"] > getattr(restored, stage + "_epochs"):
            raise TrainingWorkerError("Recovery checkpoint exceeds its saved epoch budget")
        budget["resumed"] = True
        budget["resume_source"] = {
            "snapshot_id": directory.name,
            "original_budget_sha256": manifest["files"]["training-budget.json"]["sha256"],
            "budget_recomputed": False,
        }
        return restored, budget, directory
    except (TrainingSnapshotError, OSError, ValueError, KeyError, TypeError) as error:
        raise TrainingWorkerError(f"Training recovery metadata is invalid: {error}") from error


def _write_configs(
    source: Path,
    output: Path,
    dataset_work: Path,
    plan: TrainingPlan,
    inputs: Mapping[str, Path],
) -> tuple[Path, Path]:
    config_root = output / "configs"
    config_root.mkdir()
    gpt_output = output / "work" / "gpt"
    gpt_output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints" / "gpt").mkdir(parents=True)
    (output / "checkpoints" / "sovits").mkdir(parents=True)
    with (source / "GPT_SoVITS/configs/s1longer-v2.yaml").open(encoding="utf-8") as stream:
        gpt = yaml.safe_load(stream)
    gpt["train"].update(
        {
            "seed": plan.seed,
            "epochs": plan.gpt_epochs,
            "batch_size": plan.gpt_batch_size,
            "save_every_n_epoch": plan.save_every_epoch,
            "precision": "16-mixed",
            "if_save_latest": True,
            "if_save_every_weights": True,
            "half_weights_save_dir": str(output / "checkpoints" / "gpt"),
            "exp_name": plan.experiment_name,
        }
    )
    gpt["optimizer"]["lr"] = plan.gpt_learning_rate
    gpt.update(
        {
            "output_dir": str(gpt_output),
            "pretrained_s1": str(inputs["pretrained_gpt"]),
            "train_semantic_path": str(dataset_work / "6-name2semantic.tsv"),
            "train_phoneme_path": str(dataset_work / "2-name2text.txt"),
        }
    )
    gpt_path = config_root / "gpt-v2proplus.yaml"
    gpt_path.write_text(yaml.safe_dump(gpt, sort_keys=True), encoding="utf-8")

    sovits = json.loads(
        (source / "GPT_SoVITS/configs/s2v2ProPlus.json").read_text(encoding="utf-8")
    )
    sovits["train"].update(
        {
            "seed": plan.seed,
            "epochs": plan.sovits_epochs,
            "batch_size": plan.sovits_batch_size,
            "learning_rate": plan.sovits_learning_rate,
            "gpu_numbers": "0",
            "pretrained_s2G": str(inputs["pretrained_sovits_g"]),
            "pretrained_s2D": str(inputs["pretrained_sovits_d"]),
            "if_save_latest": True,
            "if_save_every_weights": True,
            "save_every_epoch": plan.save_every_epoch,
            "grad_ckpt": plan.gradient_checkpointing,
            "lora_rank": 0,
        }
    )
    sovits["model"]["version"] = "v2ProPlus"
    sovits["data"]["exp_dir"] = str(dataset_work)
    sovits["s2_ckpt_dir"] = str(dataset_work)
    sovits["save_weight_dir"] = str(output / "checkpoints" / "sovits")
    sovits["name"] = plan.experiment_name
    sovits["version"] = "v2ProPlus"
    sovits_path = config_root / "sovits-v2proplus.json"
    sovits_path.write_text(
        json.dumps(sovits, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return gpt_path, sovits_path


def _emit_progress(progress: float, stage: str, message: str) -> None:
    print(
        "ANIFLIVE_TTS_PROGRESS "
        + json.dumps(
            {"progress": round(progress, 6), "stage": stage, "message": message},
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _training_pythonpath(source: Path, inherited: str | None) -> str:
    """Return the import roots required by the pinned GPT-SoVITS scripts."""

    candidates = [source, source / "GPT_SoVITS"]
    if inherited:
        candidates.extend(Path(value) for value in inherited.split(os.pathsep) if value)
    result: list[str] = []
    for candidate in candidates:
        value = str(candidate)
        if value not in result:
            result.append(value)
    return os.pathsep.join(result)


def _run_stage(
    *,
    name: str,
    argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    start_progress: float,
    end_progress: float,
    epochs: int,
) -> None:
    tail: deque[str] = deque(maxlen=80)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )
        assert process.stdout is not None
        last_epoch = 0
        for line in process.stdout:
            log.write(line)
            log.flush()
            tail.append(line.rstrip())
            match = _EPOCH_PATTERN.search(line)
            if match:
                raw_epoch = int(match.group(1))
                epoch = min(epochs, max(1, raw_epoch + (1 if name == "gpt" else 0)))
                if epoch > last_epoch:
                    last_epoch = epoch
                    progress = start_progress + (end_progress - start_progress) * epoch / epochs
                    _emit_progress(progress, name, f"{name} epoch {epoch}/{epochs}")
        returncode = process.wait()
    if returncode != 0:
        raise TrainingWorkerError(
            f"{name} training exited with code {returncode}: " + "\n".join(tail)[-6000:]
        )


def _copy_resume_outputs(output: Path, dataset_work: Path) -> list[Path]:
    resume_root = output / "resume"
    gpt_resume = resume_root / "gpt"
    sovits_resume = resume_root / "sovits"
    gpt_resume.mkdir(parents=True)
    sovits_resume.mkdir(parents=True)
    retained: list[Path] = []

    def move_latest(paths: Sequence[Path], destination: Path) -> None:
        candidates = [path for path in paths if path.is_file() and not path.is_symlink()]
        if not candidates:
            return
        source = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
        target = destination / source.name
        shutil.move(str(source), target)
        retained.append(target)

    move_latest(list((output / "work" / "gpt" / "ckpt").glob("*.ckpt")), gpt_resume)
    resume_root = dataset_work / "logs_s2_v2ProPlus"
    move_latest(list(resume_root.glob("G_*.pth")), sovits_resume)
    move_latest(list(resume_root.glob("D_*.pth")), sovits_resume)
    return retained


def _copy_selection_assets(output: Path, dataset_source: Path) -> list[Path]:
    source = dataset_source / "7-sv_cn"
    if source.is_symlink() or not source.is_dir():
        raise TrainingWorkerError("speaker-vector preprocessing assets are missing")
    destination = output / "selection-assets" / "7-sv_cn"
    destination.mkdir(parents=True)
    copied: list[Path] = []
    for path in sorted(source.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        target = destination / path.name
        shutil.copy2(path, target)
        if _sha256_file(target) != _sha256_file(path):
            raise TrainingWorkerError("speaker-vector selection asset copy failed")
        copied.append(target)
    if len(copied) < 2:
        raise TrainingWorkerError("checkpoint selection requires at least two speaker vectors")
    return copied


def validate_training_resume_request(manifest: Mapping[str, Any]) -> None:
    if manifest.get("resume_requested") is True:
        inputs = manifest.get("container_input_paths")
        if (not isinstance(inputs, Mapping)
                or not isinstance(inputs.get("resume_checkpoint"), str)
                or not inputs["resume_checkpoint"].strip()):
            raise TrainingWorkerError(
                "Resume was requested but its checkpoint input is missing; refusing fresh training"
            )
    inputs = manifest.get("container_input_paths")
    if isinstance(inputs, Mapping) and isinstance(inputs.get("resume_checkpoint"), str):
        root = Path(inputs["resume_checkpoint"])
        if root.is_symlink() or not (root / "latest.json").is_file():
            raise TrainingWorkerError(
                "Resume requires a verified recovery snapshot; legacy weights lack complete training state"
            )


def run_training(
    manifest: Mapping[str, Any], output: Path
) -> tuple[dict[str, Any], list[Path]]:
    validate_training_resume_request(manifest)
    if sys.platform != "linux":
        raise TrainingWorkerError("V2ProPlus training only runs in the Linux worker")
    try:
        import torch
    except ImportError as error:
        raise TrainingWorkerError("PyTorch is missing from the Linux training image") from error
    if not torch.cuda.is_available():
        raise TrainingWorkerError("CUDA is unavailable inside the Linux training worker")
    inputs_value = manifest.get("container_input_paths")
    if not isinstance(inputs_value, Mapping):
        raise TrainingWorkerError("worker manifest has no container input paths")
    required = (
        "dataset",
        "pretrained_gpt",
        "pretrained_sovits_g",
        "pretrained_sovits_d",
        "shared_dir",
    )
    if any(not isinstance(inputs_value.get(key), str) for key in required):
        raise TrainingWorkerError("training inputs are incomplete")
    inputs = {
        "dataset": _require_directory(Path(str(inputs_value["dataset"])), "dataset"),
        "pretrained_gpt": _require_regular(
            Path(str(inputs_value["pretrained_gpt"])), "pretrained GPT"
        ),
        "pretrained_sovits_g": _require_regular(
            Path(str(inputs_value["pretrained_sovits_g"])), "pretrained SoVITS generator"
        ),
        "pretrained_sovits_d": _require_regular(
            Path(str(inputs_value["pretrained_sovits_d"])), "pretrained SoVITS discriminator"
        ),
        "shared_dir": _require_directory(
            Path(str(inputs_value["shared_dir"])), "V2ProPlus training component"
        ),
    }
    settings = manifest.get("settings", {})
    plan = resolve_training_plan(
        settings if isinstance(settings, Mapping) else {}, project_id=str(manifest["project_id"])
    )
    requested_plan = plan
    modern_directory = None
    modern_manifest = None
    resume_root_value = inputs_value.get("resume_checkpoint")
    if isinstance(resume_root_value, str) and (Path(resume_root_value) / "latest.json").exists():
        from .training_snapshot import read_training_snapshot
        modern_directory, modern_manifest = read_training_snapshot(Path(resume_root_value))
    if inputs_value.get("resume_checkpoint") and modern_directory is None and plan.epoch_policy != "fixed":
        raise TrainingWorkerError(
            "Resume requires epoch_policy=fixed and the previously resolved epoch totals"
        )
    source = _verify_training_source()
    output.mkdir(parents=True, exist_ok=True)
    work_root = output / "work"
    work_root.mkdir()
    preprocessing_report: dict[str, Any] | None = None
    preprocessing_artifacts: list[Path] = []
    dataset_source = inputs["dataset"]
    if modern_directory is not None:
        from .training_recovery_driver import restore_recovery_dataset
        dataset_source = restore_recovery_dataset(modern_directory, work_root / "preprocessed")
    if not all((dataset_source / name).exists() for name in _DATASET_ENTRIES):
        training_list, wav_dir, bundle = _validate_training_input_bundle(dataset_source)
        shared = inputs["shared_dir"]
        from .workstation_preprocessing import run_v2proplus_preprocessing

        preprocessed = work_root / "preprocessed"
        preprocessing_report = run_v2proplus_preprocessing(
            input_list=training_list,
            wav_dir=wav_dir,
            output=preprocessed,
            pretrained_sovits_g=inputs["pretrained_sovits_g"],
            bert_dir=shared / "chinese-roberta-wwm-ext-large",
            hubert_dir=shared / "chinese-hubert-base",
            speaker_model=shared / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt",
        )
        preprocessing_report["training_input"] = {
            "dataset_id": bundle.get("dataset_id"),
            "frozen_manifest_sha256": bundle.get("frozen_manifest_sha256"),
            "training_list_sha256": bundle.get("training_list_sha256"),
        }
        preprocessing_report_path = output / "preprocessing-report.json"
        preprocessing_report_path.write_text(
            json.dumps(preprocessing_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        preprocessing_artifacts.append(preprocessing_report_path)
        preprocessing_log_root = output / "logs" / "preprocessing"
        for log in sorted((preprocessed / "preprocessing-logs").glob("*.log")):
            preprocessing_log_root.mkdir(parents=True, exist_ok=True)
            destination = preprocessing_log_root / log.name
            shutil.copy2(log, destination)
            preprocessing_artifacts.append(destination)
        dataset_source = preprocessed
    dataset_report = validate_preprocessed_dataset(dataset_source)
    audio_inventory = measure_training_audio(dataset_source)
    recovery_identity = None
    if modern_directory is not None:
        from .training_recovery_driver import identify_training_inputs, identity_fingerprint
        recovery_identity = identify_training_inputs(
            dataset=dataset_source, dataset_entries=_DATASET_ENTRIES,
            requested_plan=asdict(requested_plan),
            pretrained_paths={key: inputs[key] for key in
                              ("pretrained_gpt", "pretrained_sovits_g", "pretrained_sovits_d")},
            source_revision=GPT_SOVITS_TRAINING_REVISION,
        )
        plan, budget, modern_directory = restore_snapshot_budget(
            Path(resume_root_value), requested_plan,
            source_fingerprint=identity_fingerprint(recovery_identity),
        )
    else:
        plan, budget = plan_training_budget(
            plan, duration_seconds=audio_inventory["duration_seconds"],
            clips=audio_inventory["clips"], resumed=bool(inputs_value.get("resume_checkpoint")),
        )
    inventory_path = output / "training-audio-inventory.json"
    inventory_path.write_text(
        json.dumps(audio_inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    budget["audio_inventory_sha256"] = _sha256_file(inventory_path)
    budget["audio_inventory_path"] = inventory_path.name
    budget["training_text_sha256"] = _sha256_file(dataset_source / "2-name2text.txt")
    budget_path = output / "training-budget.json"
    budget_path.write_text(
        json.dumps(budget, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _emit_progress(
        0.04, "training-budget",
        f"Resolved {plan.epoch_policy} budget: GPT {plan.gpt_epochs}, SoVITS {plan.sovits_epochs} epochs",
    )
    dataset_work = work_root / "dataset"
    _stage_dataset(dataset_source, dataset_work)
    resume_report: dict[str, int] | None = None
    resume_value = inputs_value.get("resume_checkpoint")
    if isinstance(resume_value, str) and modern_directory is None:
        (work_root / "gpt" / "ckpt").mkdir(parents=True, exist_ok=True)
        resume_report = _copy_resume(
            Path(resume_value),
            gpt_root=work_root / "gpt" / "ckpt",
            sovits_root=dataset_work / "logs_s2_v2ProPlus",
        )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "CUDA_HOME",
            "LD_LIBRARY_PATH",
            "PATH",
            "PYTHONPATH",
            "VIRTUAL_ENV",
        }
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "_CUDA_VISIBLE_DEVICES": "0",
            "HOME": "/tmp/aniflive-training-home",
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": "/tmp/aniflive-training-cache/huggingface",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "MPLCONFIGDIR": "/tmp/aniflive-training-cache/matplotlib",
            "NLTK_DATA": os.environ.get("NLTK_DATA", "/opt/aniflive-tts/nltk_data"),
            "TORCH_HOME": "/tmp/aniflive-training-cache/torch",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "XDG_CACHE_HOME": "/tmp/aniflive-training-cache/xdg",
            "PYTHONUNBUFFERED": "1",
            "version": "v2ProPlus",
            "hz": "25hz",
        }
    )
    environment["PYTHONPATH"] = _training_pythonpath(source, environment.get("PYTHONPATH"))
    for path in (
        environment["HOME"],
        environment["HF_HOME"],
        environment["MPLCONFIGDIR"],
        environment["TORCH_HOME"],
        environment["XDG_CACHE_HOME"],
    ):
        Path(path).mkdir(parents=True, exist_ok=True)
    from .workstation_preprocessing import stage_v2proplus_checkpoint

    private_checkpoint_root = work_root / ".private-checkpoint"
    staged_generator = stage_v2proplus_checkpoint(
        inputs["pretrained_sovits_g"], private_checkpoint_root
    )
    config_inputs = dict(inputs)
    config_inputs["pretrained_sovits_g"] = staged_generator.path
    logs: list[Path] = []
    recovery_context = None
    recovery_artifacts: list[Path] = []
    stages = [stage for stage in ("gpt", "sovits") if plan.stage in {"both", stage}]
    completed_stages = []
    if modern_manifest is not None:
        saved = modern_manifest["metadata"]
        completed_stages = list(saved.get("completed_stages", []))
        if saved["epoch"] >= getattr(plan, saved["stage"] + "_epochs"):
            completed_stages.append(saved["stage"])
        stages = [stage for stage in stages if stage not in completed_stages]
        if not stages:
            raise TrainingWorkerError("Recovery checkpoint already completed its saved training budget")
        resume_report = {"snapshot_id": modern_directory.name, "stage": saved["stage"],
                         "epoch": saved["epoch"], "global_step": saved["global_step"],
                         "gpt_checkpoints": int(saved["stage"] == "gpt"),
                         "sovits_generator_checkpoints": int(saved["stage"] == "sovits"),
                         "budget_recomputed": False}
    try:
        gpt_config, sovits_config = _write_configs(
            source, output, dataset_work, plan, config_inputs
        )
        if not resume_value or modern_directory is not None:
            from .training_recovery_driver import prepare_training_recovery
            recovery_context = prepare_training_recovery(
                manifest=manifest, output=output, dataset=dataset_source,
                dataset_entries=_DATASET_ENTRIES, requested_plan=asdict(requested_plan),
                resolved_plan=asdict(plan), budget=budget,
                pretrained_paths={key: inputs[key] for key in
                                  ("pretrained_gpt", "pretrained_sovits_g", "pretrained_sovits_d")},
                source_revision=GPT_SOVITS_TRAINING_REVISION, identity=recovery_identity,
            )
        if modern_directory is not None:
            from .training_recovery_driver import materialize_recovery_checkpoint
            materialize_recovery_checkpoint(
                modern_directory, modern_manifest["metadata"], output=output, dataset_work=dataset_work,
            )
            recovery_context.update(
                resume_stage=modern_manifest["metadata"]["stage"],
                resume_directory=str(modern_directory),
            )
        _emit_progress(0.05, "validation", "Validated V2ProPlus training inputs")
        for index, stage in enumerate(stages):
            start = 0.1 + index * 0.8 / len(stages)
            end = 0.1 + (index + 1) * 0.8 / len(stages)
            log_path = output / "logs" / f"{stage}.log"
            logs.append(log_path)
            resume_required = bool(resume_report and resume_report[
                "gpt_checkpoints" if stage == "gpt" else "sovits_generator_checkpoints"
            ])
            entrypoint = (
                _resume_entrypoint(source, output, stage) if resume_required
                else source / "GPT_SoVITS" / ("s1_train.py" if stage == "gpt" else "s2_train.py")
            )
            if recovery_context is not None:
                from .training_entrypoints import stage_recovery_entrypoint
                recovery_context["completed_stages"] = completed_stages + stages[:index]
                context_path = output / "training-recovery-context.json"
                context_path.write_text(json.dumps(recovery_context, sort_keys=True), encoding="utf-8")
                environment["ANIFLIVE_TTS_TRAINING_RECOVERY_CONTEXT"] = str(context_path)
                entrypoint = stage_recovery_entrypoint(source, output, stage)
            if stage == "gpt":
                argv = (
                    sys.executable,
                    "-s",
                    str(entrypoint),
                    "--config_file",
                    str(gpt_config),
                )
                epochs = plan.gpt_epochs
            else:
                argv = (
                    sys.executable,
                    "-s",
                    str(entrypoint),
                    "--config",
                    str(sovits_config),
                )
                epochs = plan.sovits_epochs
            stage_cwd = work_root / "stage-cwd" / stage
            stage_cwd.mkdir(parents=True, exist_ok=True)
            _run_stage(
                name=stage,
                argv=argv,
                cwd=stage_cwd,
                environment=environment,
                log_path=log_path,
                start_progress=start,
                end_progress=end,
                epochs=epochs,
            )
            if recovery_context is not None and Path(recovery_context["pause_ack_path"]).exists():
                from .training_snapshot import read_training_snapshot
                directory, snapshot = read_training_snapshot(
                    Path(recovery_context["snapshot_root"]),
                    expected_source_fingerprint=recovery_context["source_fingerprint"],
                )
                ack_path = Path(recovery_context["pause_ack_path"])
                ack = json.loads(ack_path.read_text(encoding="utf-8"))
                if ack.get("snapshot_id") != directory.name:
                    raise TrainingWorkerError("Pause acknowledgement does not match the saved checkpoint")
                report = {
                    "schema": "aniflive-tts-v2proplus-training-report-v2",
                    "status": "paused", "pause_ack": ack,
                    "plan": asdict(plan), "training_budget": budget,
                    "resume": resume_report,
                    "recovery": {"snapshots_available": True,
                                 "checkpoint": snapshot["metadata"]},
                }
                report_path = output / "training-report.json"
                report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                return report, [report_path, ack_path]
        if recovery_context is not None:
            from .training_snapshot import read_training_snapshot
            read_training_snapshot(
                Path(recovery_context["snapshot_root"]),
                expected_source_fingerprint=recovery_context["source_fingerprint"],
            )
            recovery_artifacts = [
                path for path in Path(recovery_context["snapshot_root"]).rglob("*") if path.is_file()
            ]
        gpt_weights = sorted((output / "checkpoints" / "gpt").glob("*.ckpt"))
        sovits_weights = sorted((output / "checkpoints" / "sovits").glob("*.pth"))
        if plan.stage in {"both", "gpt"} and not gpt_weights:
            raise TrainingWorkerError("GPT training completed without a deployable checkpoint")
        if plan.stage in {"both", "sovits"} and not sovits_weights:
            raise TrainingWorkerError("SoVITS training completed without a deployable checkpoint")
        resume_artifacts = _copy_resume_outputs(output, dataset_work)
        selection_assets = _copy_selection_assets(output, dataset_source)
    finally:
        shutil.rmtree(private_checkpoint_root, ignore_errors=True)
    candidates_manifest: Path | None = None
    checkpoint_candidates: dict[str, Any] | None = None
    if gpt_weights and sovits_weights:
        candidates_manifest, checkpoint_candidates = _write_checkpoint_candidates_manifest(
            output,
            gpt_weights=gpt_weights,
            sovits_weights=sovits_weights,
        )
    report = {
        "schema": "aniflive-tts-v2proplus-training-report-v2",
        "status": "passed",
        "backend": "PyTorch CUDA training in isolated Linux worker",
        "model_family": "gsv-v2proplus",
        "training_source": {
            "repository": "https://github.com/RVC-Boss/GPT-SoVITS",
            "revision": GPT_SOVITS_TRAINING_REVISION,
        },
        "plan": asdict(plan),
        "training_budget": budget,
        "dataset": dataset_report,
        "preprocessing": preprocessing_report,
        "generator_checkpoint_staging": {
            "format_header": staged_generator.original_header,
            "private_working_copy_repaired": staged_generator.repaired,
            "source_sha256": staged_generator.source_sha256,
        },
        "resume": resume_report,
        "recovery": {"snapshots_available": bool(recovery_artifacts),
                     "controller_pause_resume_protocol": "epoch-boundary-v1"},
        "configs": {
            "gpt_sha256": _sha256_file(gpt_config),
            "sovits_sha256": _sha256_file(sovits_config),
        },
        "outputs": {
            "gpt_weights": len(gpt_weights),
            "sovits_weights": len(sovits_weights),
            "resume_checkpoints": len(resume_artifacts),
            "selection_speaker_vectors": len(selection_assets),
            "deployment_ready": False,
            "checkpoint_selection_required": checkpoint_candidates is not None,
        },
        "checkpoint_candidates": checkpoint_candidates,
    }
    report_path = output / "training-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _emit_progress(1.0, "complete", "V2ProPlus training completed")
    shutil.rmtree(work_root, ignore_errors=True)
    return report, [
        report_path,
        budget_path,
        inventory_path,
        *([candidates_manifest] if candidates_manifest is not None else []),
        gpt_config,
        sovits_config,
        *preprocessing_artifacts,
        *logs,
        *gpt_weights,
        *sovits_weights,
        *resume_artifacts,
        *recovery_artifacts,
        *selection_assets,
    ]


__all__ = [
    "GPT_SOVITS_TRAINING_REVISION",
    "TrainingPlan",
    "TrainingWorkerError",
    "resolve_training_plan",
    "run_training",
    "validate_preprocessed_dataset",
]
