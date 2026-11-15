#!/usr/bin/env bash
# Shared path/config defaults for the portable macOS release scripts.

MACOS_RELEASE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACOS_RELEASE_REPO_ROOT="$(cd "$MACOS_RELEASE_SCRIPT_DIR/../.." && pwd)"

if [[ -n "${ANIFLIVE_TTS_MACOS_ENV_FILE:-}" ]]; then
    if [[ ! -r "$ANIFLIVE_TTS_MACOS_ENV_FILE" ]]; then
        printf 'macOS config is not readable: %s\n' "$ANIFLIVE_TTS_MACOS_ENV_FILE" >&2
        return 1 2>/dev/null || exit 1
    fi
    # The config is explicitly selected and user-owned; do not source a default file.
    # shellcheck source=/dev/null
    source "$ANIFLIVE_TTS_MACOS_ENV_FILE"
fi

: "${ANIFLIVE_TTS_MACOS_RUNTIME_ROOT:=${HOME}/Library/Application Support/AnifLive-TTS/macos-v1.5}"
: "${ANIFLIVE_TTS_MACOS_DATA_ROOT:=${ANIFLIVE_TTS_MACOS_RUNTIME_ROOT}/assets}"
: "${ANIFLIVE_TTS_SHARED_DIR:=${ANIFLIVE_TTS_MACOS_DATA_ROOT}/upstream}"
: "${ANIFLIVE_TTS_SOURCE_DIR:=${ANIFLIVE_TTS_SHARED_DIR}/official}"
: "${ANIFLIVE_TTS_NATIVE_FRONTEND_SCRIPT:=${MACOS_RELEASE_REPO_ROOT}/scripts/macos-v5turbo/prepare_frontend.py}"
: "${ANIFLIVE_TTS_CACHE_DIR:=${ANIFLIVE_TTS_MACOS_RUNTIME_ROOT}/runtime/cache}"
: "${ANIFLIVE_TTS_VOICE_DIR:=${ANIFLIVE_TTS_MACOS_RUNTIME_ROOT}/runtime/voices}"
: "${ANIFLIVE_TTS_PID_FILE:=${ANIFLIVE_TTS_MACOS_RUNTIME_ROOT}/run/server.pid}"
: "${ANIFLIVE_TTS_VENV:=${ANIFLIVE_TTS_MACOS_RUNTIME_ROOT}/venv}"
: "${ANIFLIVE_TTS_PYTHON:=${ANIFLIVE_TTS_VENV}/bin/python}"
: "${ANIFLIVE_TTS_HOST:=127.0.0.1}"
: "${ANIFLIVE_TTS_PORT:=29962}"
: "${ANIFLIVE_TTS_STOP_TIMEOUT_SECONDS:=45}"

macos_release_error() {
    printf 'macOS release: %s\n' "$*" >&2
}

macos_release_check_platform() {
    local system arch version major minor
    system="$(uname -s)"
    arch="$(uname -m)"
    if [[ "$system" != "Darwin" || "$arch" != "arm64" ]]; then
        macos_release_error "native MLX/Metal requires macOS on Apple Silicon (Darwin/arm64); got ${system}/${arch}"
        return 1
    fi
    version="$(sw_vers -productVersion)"
    IFS=. read -r major minor _ <<< "$version"
    major="${major:-0}"
    minor="${minor:-0}"
    if (( major < 26 || (major == 26 && minor < 2) )); then
        macos_release_error "the selected bundled MLX libraries require macOS 26.2 or later; got ${version}"
        return 1
    fi
}

macos_release_require_python() {
    if [[ ! -x "$ANIFLIVE_TTS_PYTHON" ]]; then
        macos_release_error "Python environment is missing; set ANIFLIVE_TTS_VENV or run install.sh"
        return 1
    fi
    if ! "$ANIFLIVE_TTS_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
        macos_release_error "the selected Python environment must use Python 3.12: $ANIFLIVE_TTS_PYTHON"
        return 1
    fi
}

macos_release_require_model_package() {
    if [[ -z "${ANIFLIVE_TTS_MODEL_PACKAGE:-}" ]]; then
        macos_release_error "set ANIFLIVE_TTS_MODEL_PACKAGE to a complete external model package"
        return 1
    fi
    if [[ ! -f "$ANIFLIVE_TTS_MODEL_PACKAGE/manifest.json" || ! -f "$ANIFLIVE_TTS_MODEL_PACKAGE/checksums.json" ]]; then
        macos_release_error "model package must include manifest.json and checksums.json: $ANIFLIVE_TTS_MODEL_PACKAGE"
        return 1
    fi
}

macos_release_openjtalk_dict() {
    if [[ -n "${ANIFLIVE_TTS_OPEN_JTALK_DICT_DIR:-}" ]]; then
        printf '%s\n' "$ANIFLIVE_TTS_OPEN_JTALK_DICT_DIR"
        return 0
    fi
    "$ANIFLIVE_TTS_PYTHON" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("pyopenjtalk")
locations = list(spec.submodule_search_locations or ()) if spec is not None else []
if not locations:
    raise SystemExit("pyopenjtalk is not installed in the configured Python environment")
print(Path(locations[0]).resolve() / "open_jtalk_dic_utf_8-1.11")
PY
}

macos_release_url() {
    printf 'http://%s:%s' "$ANIFLIVE_TTS_HOST" "$ANIFLIVE_TTS_PORT"
}
