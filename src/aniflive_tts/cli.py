from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _convert(args: argparse.Namespace) -> int:
    from .converter import convert_model

    result = convert_model(
        gpt=args.gpt,
        sovits=args.sovits,
        reference_audio=args.reference_audio,
        reference_text_file=args.reference_text_file,
        reference_language=args.reference_language,
        model_id=args.model_id,
        voice_profile=args.voice_profile,
        output=args.output,
        shared_dir=args.shared_dir,
        source_dir=args.source_dir,
        allow_unsafe_pickle=args.allow_unsafe_pickle,
        max_len=args.max_len,
        workspace_mib=args.workspace_mib,
        optimization_level=args.optimization_level,
    )
    print(result)
    return 0


def _validate(args: argparse.Namespace) -> int:
    from .validate import validate_model_package

    report = validate_model_package(
        args.model_package,
        enqueue=args.enqueue,
        shared_dir=args.shared_dir,
        source_dir=args.source_dir,
        text=args.text,
        language=args.language,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _rebuild_engines(args: argparse.Namespace) -> int:
    from .converter import rebuild_engines

    result = rebuild_engines(
        model_package=args.model_package,
        workspace_mib=args.workspace_mib,
        optimization_level=args.optimization_level,
        force=args.force,
    )
    print(result)
    return 0


def _serve(args: argparse.Namespace) -> int:
    os.environ["ANIFLIVE_TTS_MODEL_PACKAGE"] = str(args.model_package.resolve())
    os.environ["ANIFLIVE_TTS_SHARED_DIR"] = str(args.shared_dir.resolve())
    os.environ.setdefault(
        "ANIFLIVE_TTS_SOURCE_DIR",
        str(Path(__file__).resolve().parents[2] / "minimal_inference"),
    )
    from .api import create_app
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aniflive-tts")
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    convert = model_commands.add_parser("convert")
    convert.add_argument("--gpt", type=Path, required=True)
    convert.add_argument("--sovits", type=Path, required=True)
    convert.add_argument("--reference-audio", type=Path, required=True)
    convert.add_argument("--reference-text-file", type=Path, required=True)
    convert.add_argument("--reference-language", required=True)
    convert.add_argument("--model-id", required=True)
    convert.add_argument("--voice-profile", default="default")
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--shared-dir", type=Path, default=Path("data/shared"))
    convert.add_argument("--source-dir", type=Path, default=Path("minimal_inference"))
    convert.add_argument("--allow-unsafe-pickle", action="store_true")
    convert.add_argument("--max-len", type=int, default=1000)
    convert.add_argument("--workspace-mib", type=int, default=4096)
    convert.add_argument("--optimization-level", type=int, default=5)
    convert.set_defaults(handler=_convert)
    rebuild = model_commands.add_parser("rebuild-engines")
    rebuild.add_argument("--model-package", type=Path, required=True)
    rebuild.add_argument("--workspace-mib", type=int, default=4096)
    rebuild.add_argument("--optimization-level", type=int, choices=range(0, 6), default=5)
    rebuild.add_argument("--force", action="store_true")
    rebuild.set_defaults(handler=_rebuild_engines)
    validate = commands.add_parser("validate")
    validate.add_argument("--model-package", type=Path, required=True)
    validate.add_argument("--enqueue", action="store_true")
    validate.add_argument("--shared-dir", type=Path, default=Path("data/shared"))
    validate.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "minimal_inference",
    )
    validate.add_argument("--text", default="今日はいい天気ですね。")
    validate.add_argument("--language", choices=("zh", "yue", "en", "ja", "ko"), default="ja")
    validate.set_defaults(handler=_validate)
    serve = commands.add_parser("serve")
    serve.add_argument("--model-package", type=Path, required=True)
    serve.add_argument("--shared-dir", type=Path, default=Path("data/shared"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=9880)
    serve.set_defaults(handler=_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args) or 0)
