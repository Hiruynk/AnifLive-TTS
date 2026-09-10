from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from aniflive_tts.qualification import parse_security_evidence
from scripts import check_release_security

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    (
        ".pytest-artifact-registry/workstation/workstation.sqlite3",
        ".pytest-expression-bank/artifacts/engine.plan",
        "tmp85m0u687/artifacts/voice.pkg",
        "workstation/workstation.sqlite3",
    ),
)
def test_release_path_gate_rejects_generated_runtime_artifacts(path: str) -> None:
    with pytest.raises(SystemExit, match=re.escape(path)):
        check_release_security._check_tracked_paths([path])


def test_release_path_gate_allows_intentional_plan_and_package_fixtures() -> None:
    check_release_security._check_tracked_paths(
        ["docs/deployment.plan", "fixtures/example.pkg"]
    )


def test_gitignore_excludes_generated_runtime_state() -> None:
    entries = set((ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
    assert {"/.pytest-*/", "/tmp*/", "*.sqlite3"} <= entries


def test_security_policy_preserves_private_reporting_and_closed_contributions() -> None:
    policy = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "/security/advisories/new" in policy
    assert "does not currently accept external contributions or pull requests" in policy
    assert "weights_only=True" in policy
    assert "not a security sandbox" in policy


def test_external_actions_are_pinned_to_full_commit_shas() -> None:
    pattern = re.compile(r"^\s*-?\s*uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE)
    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        for action, revision in pattern.findall(workflow.read_text(encoding="utf-8")):
            if not action.startswith("./"):
                assert re.fullmatch(r"[0-9a-f]{40}", revision), (
                    f"{workflow.name}: {action}@{revision} is mutable"
                )


def test_container_release_records_source_and_evidence() -> None:
    workflow = (ROOT / ".github" / "workflows" / "container.yml").read_text(
        encoding="utf-8"
    )
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "VCS_REF=${{ github.sha }}" in workflow
    assert "BUILD_DATE=${{ steps.build_date.outputs.value }}" in workflow
    assert "index:org.opencontainers.image.revision=${{ github.sha }}" in workflow
    assert "RELEASE-METADATA-AnifLive-TTS-v1.4.0-" in workflow
    assert "provenance: mode=max" in workflow
    assert "pyspdxtools -i" in workflow
    assert workflow.count('python scripts/normalize_spdx_sbom.py "${output}"') == 2
    assert 'org.opencontainers.image.revision="${VCS_REF}"' in dockerfile


def test_release_checksums_cover_all_image_evidence() -> None:
    script = (ROOT / "scripts" / "update_release_checksums.ps1").read_text(
        encoding="utf-8"
    )
    assert "RELEASE-METADATA-AnifLive-TTS-v$Version-cu128.json" in script
    assert "RELEASE-METADATA-AnifLive-TTS-v$Version-cu126.json" in script
    assert "TRIVY-AnifLive-TTS-v$Version-cu128.json" in script
    assert "TRIVY-AnifLive-TTS-v$Version-cu126.json" in script


def test_public_webui_contains_no_login_or_credential_state() -> None:
    backend = (ROOT / "src" / "aniflive_tts" / "webui.py").read_text(
        encoding="utf-8"
    ).lower()
    frontend_files = [ROOT / "run_webui.bat", ROOT / "run_studio.bat"]
    frontend_files.extend(
        path
        for path in sorted((ROOT / "webui").rglob("*"))
        if path.is_file()
        and path.suffix.lower() in {".css", ".html", ".js", ".json", ".md", ".txt"}
    )
    frontend = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in frontend_files
        if path.is_file()
    ).lower()

    for forbidden in (
        "login.html",
        "username",
        "password",
        "password_hash",
        "credential",
        "session_cookie",
        "authorization",
    ):
        assert forbidden not in frontend
    assert "sessionstorage" not in frontend
    for forbidden in (
        "login.html",
        "password_hash",
        "session_cookie",
        "sessionstorage",
        '@app.get("/login',
        '@app.post("/login',
        '@app.get("/signin',
        '@app.post("/signin',
    ):
        assert forbidden not in backend
    assert frontend.count("localstorage") == 4
    assert 'const locale_key = "aniflive.uilocale"' in frontend


def test_readmes_avoid_defensive_webui_authentication_copy() -> None:
    forbidden = (
        "contains no login screen or stored account credentials",
        "authenticated overlay is not part of the public source",
        "不包含登入頁或已儲存的帳戶憑證",
        "本機登入版本不會進入公開原始碼",
        "不包含登录页或已保存的账户凭据",
        "本地登录版本不会进入公开源代码",
    )
    for name in ("README.md", "README_ZH_HK.md", "README_ZH_CN.md"):
        content = (ROOT / name).read_text(encoding="utf-8")
        for phrase in forbidden:
            assert phrase not in content


def test_expression_runtime_has_no_character_name_branches() -> None:
    runtime_files = (
        ROOT / "src" / "aniflive_tts" / "expression.py",
        ROOT / "src" / "aniflive_tts" / "service.py",
        ROOT / "src" / "aniflive_tts" / "streaming.py",
        ROOT / "src" / "aniflive_tts" / "webui.py",
        ROOT / "webui" / "annotation_editor.js",
        ROOT / "webui" / "index.html",
    )
    pattern = re.compile(r"\b(?:miku|roxy)\b", re.IGNORECASE)
    for path in runtime_files:
        matches = pattern.findall(path.read_text(encoding="utf-8"))
        assert not matches, f"{path.relative_to(ROOT)} contains a voice-name branch"


def test_security_gate_emits_parseable_explicit_evidence(tmp_path: Path) -> None:
    output = tmp_path / "security-evidence.json"
    subject_id = "artifact_12345678-1234-4123-8123-123456789abc"
    check_release_security._write_evidence(
        output,
        subject_kind="artifact",
        subject_id=subject_id,
        files=["LICENSE", "SECURITY.md"],
    )
    evidence = parse_security_evidence(json.loads(output.read_text(encoding="utf-8")))
    assert evidence["status"] == "passed"
    assert evidence["subject"] == {"kind": "artifact", "id": subject_id}
    assert evidence["release_tree_sha256"] == check_release_security._release_tree_sha256(
        ["LICENSE", "SECURITY.md"]
    )
