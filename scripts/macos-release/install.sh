#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/macos-release/common.sh
source "$script_dir/common.sh"

macos_release_check_platform
build_worker="${ANIFLIVE_TTS_BUILD_WORKER:-1}"
install_sdk="${ANIFLIVE_TTS_INSTALL_SDK_WHEEL:-1}"
case "$build_worker" in 0|1) ;; *) macos_release_error "ANIFLIVE_TTS_BUILD_WORKER must be 0 or 1"; exit 2 ;; esac
case "$install_sdk" in 0|1) ;; *) macos_release_error "ANIFLIVE_TTS_INSTALL_SDK_WHEEL must be 0 or 1"; exit 2 ;; esac
if [[ "$build_worker" == "1" || ( "$install_sdk" == "1" && -z "${ANIFLIVE_TTS_SDK_WHEEL:-}" ) ]]; then
    command -v cargo >/dev/null 2>&1 || { macos_release_error "Cargo is required to build the Rust SDK or worker"; exit 1; }
    command -v rustc >/dev/null 2>&1 || { macos_release_error "rustc is required to build the Rust SDK or worker"; exit 1; }
fi
mkdir -p "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT" "$ANIFLIVE_TTS_MACOS_DATA_ROOT" \
         "$ANIFLIVE_TTS_CACHE_DIR" "$ANIFLIVE_TTS_VOICE_DIR" \
         "$(dirname "$ANIFLIVE_TTS_PID_FILE")" "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/build"

if [[ ! -x "$ANIFLIVE_TTS_PYTHON" ]]; then
    if [[ -e "$ANIFLIVE_TTS_VENV" ]]; then
        macos_release_error "configured environment path exists but has no executable Python; inspect it before replacement: $ANIFLIVE_TTS_VENV"
        exit 1
    fi
    base_python="${ANIFLIVE_TTS_BASE_PYTHON:-python3.12}"
    command -v "$base_python" >/dev/null 2>&1 || {
        macos_release_error "Python 3.12 is required (or set ANIFLIVE_TTS_BASE_PYTHON)"
        exit 1
    }
    "$base_python" -m venv "$ANIFLIVE_TTS_VENV"
fi
macos_release_require_python

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/cache/pip}"
mkdir -p "$PIP_CACHE_DIR"

case "${ANIFLIVE_TTS_PRECISION:-FP32}" in
    FP32) lock_name="requirements-api.lock" ;;
    FP16) lock_name="requirements-api-fp16.lock" ;;
    *) macos_release_error "ANIFLIVE_TTS_PRECISION must be FP32 or FP16"; exit 2 ;;
esac
if [[ "${ANIFLIVE_TTS_INSTALL_DEPENDENCIES:-1}" == "1" ]]; then
    "$ANIFLIVE_TTS_PYTHON" -m pip install \
        --requirement "$MACOS_RELEASE_REPO_ROOT/rust/macos-v5turbo/$lock_name"
elif [[ "${ANIFLIVE_TTS_INSTALL_DEPENDENCIES:-1}" != "0" ]]; then
    macos_release_error "ANIFLIVE_TTS_INSTALL_DEPENDENCIES must be 0 or 1"
    exit 2
fi

openjtalk_dict="$(macos_release_openjtalk_dict)"
if [[ ! -f "$openjtalk_dict/sys.dic" ]]; then
    macos_release_error "the PyOpenJTalk open_jtalk_dic_utf_8-1.11 dictionary is missing at $openjtalk_dict"
    macos_release_error "provision the pinned dictionary locally, set ANIFLIVE_TTS_OPEN_JTALK_DICT_DIR, and rerun; install/start will not trigger an implicit first-use download"
    exit 1
fi

if [[ "$install_sdk" == "1" && -n "${ANIFLIVE_TTS_SDK_WHEEL:-}" ]]; then
    [[ -f "$ANIFLIVE_TTS_SDK_WHEEL" ]] || { macos_release_error "SDK wheel not found: $ANIFLIVE_TTS_SDK_WHEEL"; exit 1; }
    "$ANIFLIVE_TTS_PYTHON" -m pip install --force-reinstall --no-deps "$ANIFLIVE_TTS_SDK_WHEEL"
    "$ANIFLIVE_TTS_PYTHON" -c 'import aniflive_tts; import aniflive_tts._native'
elif [[ "$install_sdk" == "1" ]]; then
    wheel_dir="$(mktemp -d "$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/build/sdk-wheel.XXXXXX")"
    cleanup_wheel() { [[ ! -d "$wheel_dir" ]] || rm -rf "$wheel_dir"; }
    trap cleanup_wheel EXIT
    export CARGO_TARGET_DIR="${ANIFLIVE_TTS_SDK_CARGO_TARGET_DIR:-$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/build/cargo-target-sdk}"
    mkdir -p "$CARGO_TARGET_DIR"
    export PATH="$(dirname "$ANIFLIVE_TTS_PYTHON"):$PATH"
    "$ANIFLIVE_TTS_PYTHON" -m pip wheel --no-deps --wheel-dir "$wheel_dir" "$MACOS_RELEASE_REPO_ROOT"
    wheels=("$wheel_dir"/*.whl)
    if ((${#wheels[@]} != 1)); then
        macos_release_error "expected one Rust-backed SDK wheel, found ${#wheels[@]}"
        exit 1
    fi
    "$ANIFLIVE_TTS_PYTHON" -m pip install --force-reinstall --no-deps "${wheels[0]}"
    "$ANIFLIVE_TTS_PYTHON" -c 'import aniflive_tts; import aniflive_tts._native'
fi

if [[ "$build_worker" == "1" ]]; then
    if [[ -z "${ANIFLIVE_TTS_NATIVE_OUTPUT:-}" ]]; then
        source_commit="$(git -C "$MACOS_RELEASE_REPO_ROOT" rev-parse --short HEAD 2>/dev/null || printf 'source')"
        export ANIFLIVE_TTS_NATIVE_OUTPUT="$ANIFLIVE_TTS_MACOS_RUNTIME_ROOT/build/native-worker-${source_commit}-$(date +%Y%m%d%H%M%S)"
    fi
    "$script_dir/build.sh" --output "$ANIFLIVE_TTS_NATIVE_OUTPUT"
fi

"$ANIFLIVE_TTS_PYTHON" -m pip check
cat <<EOF
macOS Python/SDK/worker installation completed.
Python environment: $ANIFLIVE_TTS_VENV
Native payload output: ${ANIFLIVE_TTS_NATIVE_OUTPUT:-not-built}
Complete model package: supply a separately provisioned ANIFLIVE_TTS_MODEL_PACKAGE before start.sh.
No model weights, voice audio, or reference assets were downloaded or copied by this installer.
EOF
