import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "package_candidate", Path(__file__).resolve().parents[1] / "scripts/package_candidate.py"
)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


@pytest.mark.parametrize("name", [
    "webui/login.html", "webui/login.js", "webui/signin.css",
    "credentials.json", "src/private/accounts.json",
    "docker-compose.local.yml", "reports/local-account.json", "../outside",
    "my-account.json", "webui/local-login.html", "scripts/accounts.json",
])
def test_local_auth_and_private_files_cannot_enter_candidate(name):
    with pytest.raises(ValueError):
        candidate.validate_candidate_path(name)


def test_public_studio_and_controller_files_remain_included():
    candidate.validate_candidate_path("webui/index.html")
    candidate.validate_candidate_path("src/aniflive_tts/runtime_handoff_control.py")
