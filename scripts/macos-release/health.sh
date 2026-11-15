#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/macos-release/common.sh
source "$script_dir/common.sh"
macos_release_require_python
macos_release_require_model_package
base_url="${ANIFLIVE_TTS_BASE_URL:-$(macos_release_url)}"

"$ANIFLIVE_TTS_PYTHON" - "$base_url" "$ANIFLIVE_TTS_MODEL_PACKAGE" <<'PY'
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

base_url = sys.argv[1].rstrip("/")
package = Path(sys.argv[2]).resolve()
manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))

def get_json(path: str) -> dict:
    with urlopen(base_url + path, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"{path} returned HTTP {response.status}")
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} returned a non-object response")
    return value

try:
    health = get_json("/health")
    capabilities = get_json("/v1/capabilities")
except (OSError, URLError, ValueError, RuntimeError) as error:
    print(json.dumps({"ready": False, "error": str(error)}, sort_keys=True))
    raise SystemExit(1)

checks = {
    "health_ready": health.get("ready") is True and health.get("status") == "ok",
    "mlx_metal_backend": health.get("backend") == "MLX-Metal",
    "expected_model": health.get("model") == manifest.get("model_id"),
    "sample_rate_48khz": health.get("sample_rate") == 48000,
    "no_neural_fallback": health.get("neural_fallback") is False,
    "capabilities_v5_family": capabilities.get("model_family") == "gsv-v5turbo",
    "capabilities_no_pytorch_fallback": capabilities.get("pytorch_fallback") is False,
}
summary = {
    "ready": all(checks.values()), "checks": checks,
    "backend": health.get("backend"), "model": health.get("model"),
    "sample_rate_hz": health.get("sample_rate"),
    "quality_qualified": health.get("quality_qualified"),
    "streaming_quality_qualified": health.get("streaming", {}).get("quality_qualified"),
    "streaming_enabled": health.get("streaming", {}).get("enabled"),
    "source": "loopback HTTP health/capabilities; no synthesis request was made",
}
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
raise SystemExit(0 if summary["ready"] else 1)
PY
