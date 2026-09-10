from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .workstation_training import GPT_SOVITS_TRAINING_REVISION

_TRAINING_ROOT = Path("/opt/aniflive-tts/gpt-sovits")
_REVISION_FILE = _TRAINING_ROOT / "ANIFLIVE_TTS_GPT_SOVITS_REVISION"
_LANGUAGES = frozenset({"zh", "ja", "en", "ko", "yue"})


class PreprocessingWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingExample:
    wav_name: str
    speaker: str
    language: str
    text: str


@dataclass(frozen=True)
class StagedCheckpoint:
    path: Path
    source_sha256: str
    staged_sha256: str
    original_header: str
    repaired: bool


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PreprocessingWorkerError(f"missing regular {label}: {path}")
    return path.resolve(strict=True)


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise PreprocessingWorkerError(f"missing {label} directory: {path}")
    return path.resolve(strict=True)


def _safe_wav_name(value: str, *, line_number: int) -> str:
    normalized = value.strip().replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    if not name or name in {".", ".."} or "\x00" in name:
        raise PreprocessingWorkerError(
            f"training list line {line_number} has an invalid audio filename"
        )
    return name


def parse_training_list(path: Path, *, maximum_examples: int = 100_000) -> tuple[TrainingExample, ...]:
    path = _regular(path, "GPT-SoVITS training list")
    if not 1 <= maximum_examples <= 1_000_000:
        raise ValueError("maximum_examples must be between 1 and 1000000")
    examples: list[TrainingExample] = []
    names: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        fields = raw_line.split("|")
        if len(fields) != 4:
            raise PreprocessingWorkerError(
                f"training list line {line_number} must have wav|speaker|language|text"
            )
        wav_value, speaker, language, text = (field.strip() for field in fields)
        wav_name = _safe_wav_name(wav_value, line_number=line_number)
        normalized_language = language.casefold()
        aliases = {"jp": "ja", "ja": "ja", "zh": "zh", "en": "en", "ko": "ko", "yue": "yue"}
        normalized_language = aliases.get(normalized_language, normalized_language)
        if normalized_language not in _LANGUAGES:
            raise PreprocessingWorkerError(
                f"training list line {line_number} has unsupported language {language!r}"
            )
        if not speaker or not text:
            raise PreprocessingWorkerError(
                f"training list line {line_number} has an empty speaker or transcript"
            )
        if wav_name in names:
            raise PreprocessingWorkerError(
                f"training list contains a duplicate audio basename: {wav_name}"
            )
        names.add(wav_name)
        examples.append(
            TrainingExample(
                wav_name=wav_name,
                speaker=speaker,
                language=normalized_language,
                text=text,
            )
        )
        if len(examples) > maximum_examples:
            raise PreprocessingWorkerError(
                f"training list exceeds the {maximum_examples} example limit"
            )
    if len(examples) < 2:
        raise PreprocessingWorkerError("training list needs at least two examples")
    return tuple(examples)


def stage_v2proplus_checkpoint(source: Path, scratch: Path) -> StagedCheckpoint:
    """Create a private, torch-loadable working copy without changing the source."""

    source = _regular(source, "V2ProPlus SoVITS generator")
    scratch.mkdir(parents=True, exist_ok=True)
    if scratch.is_symlink() or not scratch.is_dir():
        raise PreprocessingWorkerError("checkpoint scratch path is not a regular directory")
    target = scratch / "pretrained-s2g-v2proplus.working.pth"
    if target.exists() or target.is_symlink():
        raise PreprocessingWorkerError("private checkpoint working copy already exists")
    source_sha256 = _sha256_file(source)
    with source.open("rb") as input_stream:
        header = input_stream.read(2)
        if header not in {b"06", b"PK"}:
            raise PreprocessingWorkerError(
                "SoVITS preprocessing accepts only a standard V2ProPlus 06 header "
                "or a regular PK torch archive"
            )
        with target.open("xb") as output_stream:
            output_stream.write(b"PK")
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    try:
        target.chmod(0o600)
        if target.stat().st_size != source.stat().st_size:
            raise PreprocessingWorkerError("private checkpoint working copy is truncated")
        if _sha256_file(source) != source_sha256:
            raise PreprocessingWorkerError("source checkpoint changed while it was staged")
        return StagedCheckpoint(
            path=target.resolve(strict=True),
            source_sha256=source_sha256,
            staged_sha256=_sha256_file(target),
            original_header=header.decode("ascii"),
            repaired=header == b"06",
        )
    except Exception:
        target.unlink(missing_ok=True)
        raise


def _validate_named_outputs(
    directory: Path,
    expected_names: set[str],
    *,
    label: str,
) -> int:
    if directory.is_symlink() or not directory.is_dir():
        raise PreprocessingWorkerError(f"{label} produced no output directory")
    actual = {
        item.name
        for item in directory.iterdir()
        if item.is_file() and not item.is_symlink()
    }
    missing = sorted(expected_names - actual)
    unexpected = sorted(actual - expected_names)
    if missing or unexpected or len(actual) != len(expected_names):
        detail = []
        if missing:
            detail.append("missing=" + ", ".join(missing[:8]))
        if unexpected:
            detail.append("unexpected=" + ", ".join(unexpected[:8]))
        raise PreprocessingWorkerError(
            f"{label} output count mismatch: expected {len(expected_names)}, "
            f"got {len(actual)}" + ("; " + "; ".join(detail) if detail else "")
        )
    return len(actual)


def validate_speaker_embedding_outputs(
    output: Path, examples: tuple[TrainingExample, ...]
) -> int:
    expected = {f"{example.wav_name}.pt" for example in examples}
    return _validate_named_outputs(
        output / "7-sv_cn", expected, label="speaker-vector preprocessing"
    )


def _validate_rows(path: Path, expected_names: set[str], *, delimiter: str, label: str) -> int:
    path = _regular(path, label)
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        names.append(_safe_wav_name(raw.split(delimiter, 1)[0], line_number=len(names) + 1))
    actual = set(names)
    if len(names) != len(actual) or actual != expected_names:
        raise PreprocessingWorkerError(
            f"{label} row mismatch: expected {len(expected_names)}, got {len(names)}"
        )
    return len(names)


def _verify_source() -> Path:
    source = _directory(_TRAINING_ROOT, "pinned GPT-SoVITS source")
    revision = _regular(_REVISION_FILE, "GPT-SoVITS source revision").read_text(
        encoding="ascii"
    ).strip()
    if revision != GPT_SOVITS_TRAINING_REVISION:
        raise PreprocessingWorkerError("GPT-SoVITS preprocessing source revision is not trusted")
    for name in (
        "1-get-text.py",
        "2-get-hubert-wav32k.py",
        "2-get-sv.py",
        "3-get-semantic.py",
    ):
        _regular(source / "GPT_SoVITS" / "prepare_datasets" / name, f"preprocessor {name}")
    return source


def _pythonpath(source: Path, inherited: str | None) -> str:
    values = [source, source / "GPT_SoVITS", source / "GPT_SoVITS" / "eres2net"]
    if inherited:
        values.extend(Path(item) for item in inherited.split(os.pathsep) if item)
    unique: list[str] = []
    for value in values:
        text = str(value)
        if text not in unique:
            unique.append(text)
    return os.pathsep.join(unique)


def _run_upstream(
    source: Path,
    scratch: Path,
    environment: dict[str, str],
    script_name: str,
    log_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-s",
            str(source / "GPT_SoVITS" / "prepare_datasets" / script_name),
        ],
        cwd=str(scratch),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise PreprocessingWorkerError(
            f"{script_name} exited with code {completed.returncode}: "
            + completed.stdout[-4000:]
        )


def run_v2proplus_preprocessing(
    *,
    input_list: Path,
    wav_dir: Path,
    output: Path,
    pretrained_sovits_g: Path,
    bert_dir: Path,
    hubert_dir: Path,
    speaker_model: Path,
    s2_config: Path | None = None,
) -> dict[str, Any]:
    if sys.platform != "linux":
        raise PreprocessingWorkerError("V2ProPlus preprocessing only runs in the Linux worker")
    try:
        import torch
        import torchcodec
    except ImportError as error:
        raise PreprocessingWorkerError(
            "Linux preprocessing requires PyTorch and the pinned TorchCodec loader"
        ) from error
    if not torch.cuda.is_available():
        raise PreprocessingWorkerError("CUDA is unavailable inside the preprocessing worker")

    source = _verify_source()
    wav_dir = _directory(wav_dir, "training audio")
    bert_dir = _directory(bert_dir, "BERT model")
    hubert_dir = _directory(hubert_dir, "HuBERT model")
    speaker_model = _regular(speaker_model, "speaker embedding model")
    examples = parse_training_list(input_list)
    for example in examples:
        _regular(wav_dir / example.wav_name, f"training audio {example.wav_name}")

    output = output.resolve()
    if output.exists():
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise PreprocessingWorkerError("preprocessing output must be a new or empty directory")
    else:
        output.mkdir(parents=True)
    scratch = output / ".private-preprocessing-scratch"
    scratch.mkdir(mode=0o700)
    checkpoint: StagedCheckpoint | None = None
    expected_names = {example.wav_name for example in examples}
    try:
        checkpoint = stage_v2proplus_checkpoint(
            pretrained_sovits_g, scratch / "checkpoint"
        )
        config = (
            _regular(s2_config, "V2ProPlus SoVITS config")
            if s2_config is not None
            else _regular(
                source / "GPT_SoVITS" / "configs" / "s2v2ProPlus.json",
                "V2ProPlus SoVITS config",
            )
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"CUDA_HOME", "LD_LIBRARY_PATH", "PATH", "VIRTUAL_ENV"}
        }
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "_CUDA_VISIBLE_DEVICES": "0",
                "HOME": str(scratch / "home"),
                "HF_HUB_OFFLINE": "1",
                "HF_HOME": str(scratch / "cache" / "huggingface"),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONPATH": _pythonpath(source, os.environ.get("PYTHONPATH")),
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
                "XDG_CACHE_HOME": str(scratch / "cache" / "xdg"),
                "all_parts": "1",
                "bert_pretrained_dir": str(bert_dir),
                "cnhubert_base_dir": str(hubert_dir),
                "exp_name": "aniflive-v2proplus-preprocessing",
                "i_part": "0",
                "inp_text": str(_regular(input_list, "GPT-SoVITS training list")),
                "inp_wav_dir": str(wav_dir),
                "is_half": "True",
                "opt_dir": str(output),
                "pretrained_s2G": str(checkpoint.path),
                "s2config_path": str(config),
                "sv_path": str(speaker_model),
                "version": "v2ProPlus",
            }
        )
        Path(environment["HOME"]).mkdir(parents=True)
        Path(environment["HF_HOME"]).mkdir(parents=True)
        Path(environment["XDG_CACHE_HOME"]).mkdir(parents=True)
        logs = output / "preprocessing-logs"

        _run_upstream(source, scratch, environment, "1-get-text.py", logs / "text.log")
        text_part = output / "2-name2text-0.txt"
        _validate_rows(text_part, expected_names, delimiter="\t", label="text preprocessing")
        chinese = {f"{item.wav_name}.pt" for item in examples if item.language == "zh"}
        _validate_named_outputs(output / "3-bert", chinese, label="BERT preprocessing")

        _run_upstream(
            source,
            scratch,
            environment,
            "2-get-hubert-wav32k.py",
            logs / "hubert.log",
        )
        _validate_named_outputs(
            output / "4-cnhubert",
            {f"{name}.pt" for name in expected_names},
            label="HuBERT preprocessing",
        )
        _validate_named_outputs(
            output / "5-wav32k", expected_names, label="32 kHz audio preprocessing"
        )

        _run_upstream(source, scratch, environment, "2-get-sv.py", logs / "speaker.log")
        speaker_rows = validate_speaker_embedding_outputs(output, examples)

        _run_upstream(
            source,
            scratch,
            environment,
            "3-get-semantic.py",
            logs / "semantic.log",
        )
        semantic_part = output / "6-name2semantic-0.tsv"
        semantic_rows = _validate_rows(
            semantic_part,
            expected_names,
            delimiter="\t",
            label="semantic preprocessing",
        )
        os.replace(text_part, output / "2-name2text.txt")
        semantic_text = semantic_part.read_text(encoding="utf-8").strip()
        (output / "6-name2semantic.tsv").write_text(
            "item_name\tsemantic_audio\n" + semantic_text + "\n",
            encoding="utf-8",
        )
        semantic_part.unlink()
        report = {
            "schema": "aniflive-tts-v2proplus-preprocessing-report-v1",
            "status": "passed",
            "model_family": "gsv-v2proplus",
            "examples": len(examples),
            "languages": sorted({example.language for example in examples}),
            "speaker_embedding_rows": speaker_rows,
            "semantic_rows": semantic_rows,
            "checkpoint_staging": {
                "format_header": checkpoint.original_header,
                "private_working_copy_repaired": checkpoint.repaired,
                "source_sha256": checkpoint.source_sha256,
            },
            "dependencies": {
                "torch": str(torch.__version__),
                "torchcodec": str(torchcodec.__version__),
                "gpt_sovits_revision": GPT_SOVITS_TRAINING_REVISION,
            },
        }
        (output / "preprocessing-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a strict V2ProPlus training dataset in the Linux worker"
    )
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained-sovits-g", type=Path, required=True)
    parser.add_argument("--bert-dir", type=Path, required=True)
    parser.add_argument("--hubert-dir", type=Path, required=True)
    parser.add_argument("--speaker-model", type=Path, required=True)
    parser.add_argument("--s2-config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_v2proplus_preprocessing(
            input_list=args.input_list,
            wav_dir=args.wav_dir,
            output=args.output,
            pretrained_sovits_g=args.pretrained_sovits_g,
            bert_dir=args.bert_dir,
            hubert_dir=args.hubert_dir,
            speaker_model=args.speaker_model,
            s2_config=args.s2_config,
        )
    except (OSError, PreprocessingWorkerError) as error:
        print(f"AnifLive-TTS preprocessing failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PreprocessingWorkerError",
    "StagedCheckpoint",
    "TrainingExample",
    "parse_training_list",
    "run_v2proplus_preprocessing",
    "stage_v2proplus_checkpoint",
    "validate_speaker_embedding_outputs",
]
