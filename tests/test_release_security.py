from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    assert "RELEASE-METADATA-AnifLive-TTS-v1.3.0-" in workflow
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
    public_files = [ROOT / "src" / "aniflive_tts" / "webui.py", ROOT / "run_webui.bat"]
    public_files.extend(sorted((ROOT / "webui").glob("*")))
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in public_files
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
        assert forbidden not in combined
    assert "sessionstorage" not in combined
    assert combined.count("localstorage") == 2
    assert 'const locale_key = "aniflive.uilocale"' in combined


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
