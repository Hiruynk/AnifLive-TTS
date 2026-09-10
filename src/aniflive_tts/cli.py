from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _default_source_dir() -> Path:
    configured = os.environ.get("ANIFLIVE_TTS_SOURCE_DIR")
    if configured:
        return Path(configured).expanduser()

    candidates = (
        Path.cwd() / "minimal_inference",
        Path(__file__).resolve().parents[2] / "minimal_inference",
    )
    for candidate in candidates:
        if (candidate / "run_trt_inference.py").is_file():
            return candidate
    return Path("minimal_inference")


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
        stream_overlap_frames=args.stream_overlap_frames,
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


def _import_expressions(args: argparse.Namespace) -> int:
    from .expression_import import import_expression_profiles

    result = import_expression_profiles(
        model_package=args.model_package,
        voice_profile=args.voice_profile,
        spec_file=args.spec_file,
        asset_root=args.asset_root,
        output=args.output,
    )
    print(result)
    return 0


def _migrate_engine_metadata(args: argparse.Namespace) -> int:
    from .model_package import migrate_engine_metadata

    result = migrate_engine_metadata(args.model_package)
    print(result)
    return 0


def _serve(args: argparse.Namespace) -> int:
    os.environ["ANIFLIVE_TTS_MODEL_PACKAGE"] = str(args.model_package.resolve())
    os.environ["ANIFLIVE_TTS_SHARED_DIR"] = str(args.shared_dir.resolve())
    os.environ.setdefault(
        "ANIFLIVE_TTS_SOURCE_DIR",
        str(_default_source_dir().resolve()),
    )
    from .api import create_app
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)
    return 0


def _webui(args: argparse.Namespace) -> int:
    from .webui import run_webui

    run_webui(
        host=args.host,
        port=args.port,
        upstream=args.upstream,
        allow_non_loopback=args.allow_non_loopback,
        surface=args.surface,
    )
    return 0


def _workstation(args: argparse.Namespace) -> int:
    from .workstation_supervisor import (
        WorkstationSupervisorConfig,
        default_broker_config,
        run_workstation,
    )

    workstation_dir = (
        args.workstation_dir or Path("data/workstation")
    ).expanduser().resolve()
    configured_roots = tuple(
        Path(value).expanduser().resolve()
        for value in os.environ.get(
            "ANIFLIVE_TTS_WORKSTATION_IMPORT_ROOTS", ""
        ).split(os.pathsep)
        if value.strip()
    )
    broker_config = args.docker_broker_config
    if broker_config is None:
        broker_config = default_broker_config(workstation_dir)
    return run_workstation(
        WorkstationSupervisorConfig(
            host=args.host,
            port=args.port,
            upstream=args.upstream,
            workstation_dir=workstation_dir,
            import_roots=tuple(
                args.import_root or configured_roots or (Path("data").resolve(),)
            ),
            docker_broker_config=broker_config,
            runtime_handoff_config=(
                args.runtime_handoff_config
                or (workstation_dir / "runtime-handoff.json"
                    if (workstation_dir / "runtime-handoff.json").is_file() else None)
            ),
            allow_non_loopback=args.allow_non_loopback,
        )
    )


def _worker(args: argparse.Namespace) -> int:
    from .workstation import WorkstationStore
    from .workstation_docker import DockerWorkerBroker
    from .workstation_worker import WorkstationWorker

    store = WorkstationStore(args.workstation_dir)
    component_root = (store.root / "components").resolve()
    component_root.mkdir(parents=True, exist_ok=True)
    import_roots = tuple(
        dict.fromkeys((*tuple(args.import_root or store.allowed_import_roots()), component_root))
    )
    worker_broker = None
    if args.docker_broker_config is not None:
        worker_broker = DockerWorkerBroker.from_config(
            args.docker_broker_config,
            output_root=store.root / "docker-worker-output",
            allowed_input_roots=import_roots,
        )

    def report(record) -> None:
        print(json.dumps(dict(record), ensure_ascii=False, sort_keys=True), flush=True)

    worker = WorkstationWorker(
        store,
        manifest_root=args.manifest_dir,
        allowed_path_roots=import_roots,
        enabled_job_types=args.job_type,
        worker_broker=worker_broker,
        poll_seconds=args.poll_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        maximum_inventory_files=args.max_inventory_files,
        reporter=report,
    )
    handoff_config = getattr(args, "runtime_handoff_config", None)
    if handoff_config is None:
        configured = os.environ.get("ANIFLIVE_TTS_RUNTIME_HANDOFF_CONFIG", "")
        handoff_config = Path(configured) if configured else store.root / "runtime-handoff.json"
    if handoff_config.is_file():
        if worker_broker is None:
            raise ValueError("Automatic runtime handoff requires the Linux Docker broker")
        from .workstation_handoff import RuntimeHandoff

        worker.runtime_handoff = RuntimeHandoff(
            store, handoff_config, stopping=lambda: worker.stopping
        )
    try:
        with worker.signal_handlers():
            summary = worker.run(once=args.once, max_jobs=args.max_jobs)
    finally:
        if worker.runtime_handoff is not None:
            worker.runtime_handoff.close()
    report({"kind": "worker-summary", **summary.as_dict()})
    return 1 if summary.failed or summary.blocked else 0


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
    convert.add_argument("--source-dir", type=Path, default=_default_source_dir())
    convert.add_argument("--allow-unsafe-pickle", action="store_true")
    convert.add_argument("--max-len", type=int, default=1000)
    convert.add_argument(
        "--stream-overlap-frames",
        type=int,
        default=12,
        help="V2ProPlus latent overlap frames encoded into sovits_stream.engine",
    )
    convert.add_argument("--workspace-mib", type=int, default=4096)
    convert.add_argument("--optimization-level", type=int, default=5)
    convert.set_defaults(handler=_convert)
    import_expressions = model_commands.add_parser("import-expressions")
    import_expressions.add_argument("--model-package", type=Path, required=True)
    import_expressions.add_argument("--voice-profile", default="default")
    import_expressions.add_argument("--spec-file", type=Path, required=True)
    import_expressions.add_argument("--asset-root", type=Path, required=True)
    import_expressions.add_argument("--output", type=Path, required=True)
    import_expressions.set_defaults(handler=_import_expressions)
    rebuild = model_commands.add_parser("rebuild-engines")
    rebuild.add_argument("--model-package", type=Path, required=True)
    rebuild.add_argument("--workspace-mib", type=int, default=4096)
    rebuild.add_argument("--optimization-level", type=int, choices=range(0, 6), default=5)
    rebuild.add_argument("--force", action="store_true")
    rebuild.set_defaults(handler=_rebuild_engines)
    migrate = model_commands.add_parser("migrate-engine-metadata")
    migrate.add_argument("--model-package", type=Path, required=True)
    migrate.set_defaults(handler=_migrate_engine_metadata)
    validate = commands.add_parser("validate")
    validate.add_argument("--model-package", type=Path, required=True)
    validate.add_argument("--enqueue", action="store_true")
    validate.add_argument("--shared-dir", type=Path, default=Path("data/shared"))
    validate.add_argument(
        "--source-dir",
        type=Path,
        default=_default_source_dir(),
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
    webui = commands.add_parser("webui")
    webui.add_argument("--host", default="127.0.0.1")
    webui.add_argument("--port", type=int, default=9890)
    webui.add_argument("--upstream", default="http://127.0.0.1:9880")
    webui.add_argument(
        "--surface",
        choices=("classic", "studio"),
        default="classic",
        help="Choose the lightweight WebUI or full Studio shell",
    )
    webui.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help="Allow an explicitly requested non-loopback WebUI bind",
    )
    webui.set_defaults(handler=_webui)
    workstation = commands.add_parser(
        "workstation",
        help="Run the WebUI and configured Linux Docker worker together",
    )
    workstation.add_argument("--host", default="127.0.0.1")
    workstation.add_argument("--port", type=int, default=9891)
    workstation.add_argument("--upstream", default="http://127.0.0.1:9880")
    workstation.add_argument("--workstation-dir", type=Path, default=None)
    workstation.add_argument(
        "--import-root", type=Path, action="append", default=None
    )
    workstation.add_argument("--docker-broker-config", type=Path, default=None)
    workstation.add_argument("--runtime-handoff-config", type=Path, default=None)
    workstation.add_argument("--allow-non-loopback", action="store_true")
    workstation.set_defaults(handler=_workstation)
    worker = commands.add_parser(
        "worker", help="Run fixed local workstation job adapters"
    )
    worker.add_argument(
        "--once", action="store_true", help="Process at most one claimable job"
    )
    worker.add_argument(
        "--workstation-dir", type=Path, default=None, help="SQLite workstation state directory"
    )
    worker.add_argument(
        "--manifest-dir", type=Path, default=None, help="Immutable worker preparation manifests"
    )
    worker.add_argument(
        "--import-root",
        type=Path,
        action="append",
        default=None,
        help="Allowed local artifact root; repeat to allow more than one",
    )
    worker.add_argument(
        "--job-type",
        action="append",
        choices=(
            "dataset.inventory",
            "dataset.process",
            "tse.prepare",
            "training.prepare",
            "evaluation.prepare",
            "engine.prepare",
            "model.package",
        ),
        default=None,
        help="Enable one fixed adapter type; repeat to enable several",
    )
    worker.add_argument(
        "--docker-broker-config",
        type=Path,
        default=None,
        help="Admin-owned, digest-pinned Linux Docker worker broker config",
    )
    worker.add_argument(
        "--runtime-handoff-config", type=Path, default=None,
        help="Administrator-owned managed inference GPU handoff config",
    )
    worker.add_argument("--poll-seconds", type=float, default=2.0)
    worker.add_argument("--heartbeat-seconds", type=float, default=15.0)
    worker.add_argument("--max-inventory-files", type=int, default=200_000)
    worker.add_argument(
        "--max-jobs", type=int, default=None, help="Stop after this many processed jobs"
    )
    worker.set_defaults(handler=_worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args) or 0)
