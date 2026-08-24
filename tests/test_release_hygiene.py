from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LICENSE_ID = "PolyForm-Noncommercial-1.0.0"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_license_metadata_is_consistent() -> None:
    workflow = _read(".github/workflows/container.yml")
    assert f'license = {{ text = "{LICENSE_ID}" }}' in _read("pyproject.toml")
    assert f'org.opencontainers.image.licenses="{LICENSE_ID}"' in _read("Dockerfile")
    assert "sbom: true" in workflow
    assert "docker buildx imagetools inspect" in workflow
    assert "{{ json .SBOM.SPDX }}" in workflow
    assert "spdx-tools==0.8.3" in workflow
    assert 'pyspdxtools -i "${output}"' in workflow
    assert "sbom_only:" in workflow
    assert "export-existing-sbom:" in workflow
    assert "offline_only:" in workflow
    assert "verify-existing-offline:" in workflow
    assert "ghcr.io/hiruynk/aniflive-tts" in workflow
    assert "fail-fast: false" in workflow
    assert "Reclaim runner disk space" in workflow
    assert "Reclaim post-build disk space" in workflow
    assert "docker buildx prune --all --force" in workflow
    assert "/usr/local/lib/android" in workflow
    assert "/usr/share/dotnet" in workflow
    for readme in ("README.md", "README_ZH_HK.md", "README_ZH_CN.md"):
        assert "PolyForm_Noncommercial_1.0.0" in _read(readme)


def test_original_code_author_is_hiruynk() -> None:
    assert "Required Notice: Copyright 2026 Hiruynk." in _read("LICENSE")
    assert 'authors = [{ name = "Hiruynk" }]' in _read("pyproject.toml")
    assert "owned or controlled by Hiruynk" in _read("LICENSING.md")


def test_image_data_licenses_and_attribution_are_present() -> None:
    notices = _read("THIRD_PARTY_NOTICES.md")
    required = (
        "licenses/FAST-LANGDETECT-MIT.txt",
        "licenses/FASTTEXT-LID-CC-BY-SA-3.0.txt",
        "licenses/NLTK-DATA-NOTICES.md",
        "licenses/CMUDICT-NOTICE.txt",
    )
    for relative in required:
        assert (ROOT / relative).is_file()
        assert relative in notices
    assert "CC BY-SA 3.0" in notices


def test_docker_context_excludes_private_and_generated_assets() -> None:
    dockerfile = _read("Dockerfile")
    ignored = _read(".dockerignore").splitlines()
    required = {
        ".git",
        ".venv",
        ".env",
        ".env.*",
        "Miku",
        "data",
        "dist",
        "reports",
        "*.ckpt",
        "*.pth",
        "*.engine",
        "*.onnx",
        "*.onnx.data",
        "*.wav",
        "*.mp4",
        ".cloudflared-token",
        "cloudflare-token*",
        "tunnel-token*",
    }
    assert required <= set(ignored)
    assert "!.env.example" in ignored
    assert "COPY . /app" not in dockerfile
    assert "COPY src /app/src" in dockerfile
    assert "COPY minimal_inference /app/minimal_inference" in dockerfile


def test_release_bundle_uses_only_committed_files() -> None:
    script = _read("scripts/package_release.ps1")
    assert "git archive" in script
    assert "status --porcelain" in script
    assert "check_release_security.py" in script
    assert "robocopy" not in script.lower()
    assert "generate_sbom" not in script
    assert "docker-source-bundle.zip" in script
    assert not (ROOT / "scripts" / "generate_sbom.py").exists()


def test_fast_langdetect_uses_an_immutable_offline_model() -> None:
    dockerfile = _read("Dockerfile")
    runtime = _read("src/aniflive_tts/api.py")
    segmenter = _read(
        "minimal_inference/GPT_SoVITS/text/LangSegmenter/langsegmenter.py"
    )
    workflow = _read(".github/workflows/container.yml")
    assert "ANIFLIVE_TTS_FAST_LANGDETECT_MODEL=" in dockerfile
    assert 'os.environ["ANIFLIVE_TTS_FAST_LANGDETECT_MODEL"]' in runtime
    assert "runtime downloads are disabled" in runtime
    assert "custom_model_path=" in segmenter
    assert "allow_fallback=False" in segmenter
    assert "--network none" in workflow
    assert "low_memory=False" in workflow


def test_benchmarks_name_wall_and_server_rtf_explicitly() -> None:
    api_benchmark = _read("scripts/benchmark_api.py")
    comparison = _read("scripts/benchmark_compare.py")
    for script in (api_benchmark, comparison):
        assert '"wall_rtf"' in script
        assert '"server_rtf"' in script
        assert '"rtf"' not in script


def test_windows_launcher_never_terminates_an_unrelated_process() -> None:
    launcher = _read("run_tts.bat").lower()
    assert "taskkill" not in launcher
    assert "already in use by pid" in launcher


def test_checkpoint_safety_policy_reaches_exporter() -> None:
    converter = _read("src/aniflive_tts/converter.py")
    exporter = _read("minimal_inference/export_onnx.py")
    checkpoint = _read("minimal_inference/GPT_SoVITS/process_ckpt.py")
    assert 'export_command.append("--allow_unsafe_pickle")' in converter
    assert 'torch.load(args.gpt_path, map_location="cpu", weights_only=True)' in exporter
    assert 'load_sovits_new(args.sovits_path, weights_only=True)' in exporter
    assert "if not args.allow_unsafe_pickle" in exporter
    assert "def load_sovits_new(sovits_path, *, weights_only=True):" in checkpoint


def test_environment_example_is_tracked_and_generic() -> None:
    example = _read(".env.example").lower()
    assert "tunnel" not in example
    assert "token" not in example
    assert "miku" not in example
