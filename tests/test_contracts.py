from aniflive_tts.backend.contracts import (
    OPTIONAL_LEGACY_STAGE_INPUTS,
    STAGE_IO_CONTRACTS,
    STAGE_ORDER,
)
from aniflive_tts.cli import build_parser
from aniflive_tts.converter import _jsonable_profiles


def test_tensor_rt_stages_are_complete() -> None:
    assert len(STAGE_ORDER) == 9
    assert tuple(STAGE_IO_CONTRACTS) == STAGE_ORDER
    assert all(inputs and outputs for inputs, outputs in STAGE_IO_CONTRACTS.values())


def test_only_v1_stream_acoustic_noise_is_optional_for_package_migration() -> None:
    assert OPTIONAL_LEGACY_STAGE_INPUTS == {
        "sovits_stream": ("acoustic_noise",),
    }
    assert "acoustic_noise" in STAGE_IO_CONTRACTS["sovits_stream"][0]


def test_rebuild_engines_cli_is_implemented(tmp_path) -> None:
    args = build_parser().parse_args(
        ["model", "rebuild-engines", "--model-package", str(tmp_path)]
    )
    assert args.handler.__name__ == "_rebuild_engines"
    assert args.optimization_level == 5


def test_convert_cli_exposes_generic_stream_overlap() -> None:
    parser = build_parser()
    common = [
        "model",
        "convert",
        "--gpt",
        "model.ckpt",
        "--sovits",
        "model.pth",
        "--reference-audio",
        "reference.wav",
        "--reference-text-file",
        "reference.txt",
        "--reference-language",
        "ja",
        "--model-id",
        "generic-v2proplus",
        "--output",
        "package",
    ]
    assert parser.parse_args(common).stream_overlap_frames == 32
    assert (
        parser.parse_args([*common, "--stream-overlap-frames", "12"]).stream_overlap_frames
        == 12
    )


def test_expression_import_cli_is_admin_only_and_explicit(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "model",
            "import-expressions",
            "--model-package",
            str(tmp_path / "source"),
            "--spec-file",
            str(tmp_path / "expressions.json"),
            "--asset-root",
            str(tmp_path / "assets"),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert args.handler.__name__ == "_import_expressions"
    assert args.voice_profile == "default"


def test_enqueue_validation_accepts_runtime_paths(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "validate",
            "--model-package",
            str(tmp_path / "model"),
            "--enqueue",
            "--shared-dir",
            str(tmp_path / "shared"),
            "--source-dir",
            str(tmp_path / "source"),
        ]
    )
    assert args.enqueue is True
    assert args.language == "ja"


def test_engine_profiles_are_fingerprint_serializable() -> None:
    profiles = _jsonable_profiles()
    assert profiles["gpt_step"]["samples"] == {
        "min": [1, 1],
        "opt": [1, 1],
        "max": [1, 1],
    }
