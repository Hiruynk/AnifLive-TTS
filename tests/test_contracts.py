from aniflive_tts.backend.contracts import STAGE_IO_CONTRACTS, STAGE_ORDER
from aniflive_tts.cli import build_parser
from aniflive_tts.converter import _jsonable_profiles


def test_eight_tensor_rt_stages_are_complete() -> None:
    assert len(STAGE_ORDER) == 8
    assert tuple(STAGE_IO_CONTRACTS) == STAGE_ORDER
    assert all(inputs and outputs for inputs, outputs in STAGE_IO_CONTRACTS.values())


def test_rebuild_engines_cli_is_implemented(tmp_path) -> None:
    args = build_parser().parse_args(
        ["model", "rebuild-engines", "--model-package", str(tmp_path)]
    )
    assert args.handler.__name__ == "_rebuild_engines"
    assert args.optimization_level == 5


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
