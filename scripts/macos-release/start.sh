#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/macos-release/common.sh
source "$script_dir/common.sh"
macos_release_check_platform
macos_release_require_python
macos_release_require_model_package
openjtalk_dict="$(macos_release_openjtalk_dict)"

for path in \
    "$ANIFLIVE_TTS_SOURCE_DIR/GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py" \
    "$ANIFLIVE_TTS_NATIVE_FRONTEND_SCRIPT" \
    "$openjtalk_dict/sys.dic" \
    "$ANIFLIVE_TTS_MACOS_DATA_ROOT/nltk_data" \
    "$ANIFLIVE_TTS_MACOS_DATA_ROOT/cache/nltk" \
    "$ANIFLIVE_TTS_MACOS_DATA_ROOT/cache/fast_langdetect/lid.176.bin"; do
    if [[ ! -e "$path" ]]; then
        macos_release_error "required pinned local frontend resource is missing: $path"
        exit 1
    fi
done

package_profile="$("$ANIFLIVE_TTS_PYTHON" - "$ANIFLIVE_TTS_MODEL_PACKAGE" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser().resolve(strict=True)
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("format") != "aniflive-tts-model-package":
    raise SystemExit("Unsupported model-package format")
if (manifest.get("model_family") != "gsv-v5turbo" or manifest.get("precision") not in {"FP32", "FP16"}
        or manifest.get("sample_rate") != 48000 or manifest.get("cfm_steps") != 4
        or float(manifest.get("cfg_rate", -1)) != 0.0):
    raise SystemExit("Model package does not match a supported FP32/FP16 v5-turbo profile")
# A shared-base package (reference_mode "dynamic") has no fixed voice and no
# packaged reference inputs; voices come from the Voice/Reference API.
required = ["worker_binary", "models_dir", "metal_library"]
if manifest.get("reference_mode") != "dynamic":
    required.append("reference_inputs")
for key in required:
    value = manifest.get("native_runtime", {}).get(key)
    if not isinstance(value, str):
        raise SystemExit(f"Model package has no native_runtime.{key}")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.exists():
        raise SystemExit(f"Model package native_runtime.{key} is missing or escapes the package")
print(manifest["precision"], manifest["model_id"])
PY
)"
package_precision="${package_profile%% *}"

if [[ -L "$ANIFLIVE_TTS_PID_FILE" ]]; then
    macos_release_error "refusing to use a symlink PID file: $ANIFLIVE_TTS_PID_FILE"
    exit 1
fi
if [[ -e "$ANIFLIVE_TTS_PID_FILE" ]]; then
    old_pid="$(sed -n '1p' "$ANIFLIVE_TTS_PID_FILE")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        macos_release_error "PID file points to a live process; run stop.sh and verify it is this service: $old_pid"
        exit 1
    fi
    macos_release_error "stale PID file exists; inspect and remove it explicitly before starting: $ANIFLIVE_TTS_PID_FILE"
    exit 1
fi

mkdir -p "$ANIFLIVE_TTS_CACHE_DIR" "$ANIFLIVE_TTS_VOICE_DIR" "$(dirname "$ANIFLIVE_TTS_PID_FILE")"
export ANIFLIVE_TTS_MODEL_PACKAGE
export ANIFLIVE_TTS_MACOS_DATA_ROOT ANIFLIVE_TTS_SHARED_DIR ANIFLIVE_TTS_SOURCE_DIR
export ANIFLIVE_TTS_NATIVE_FRONTEND_SCRIPT ANIFLIVE_TTS_CACHE_DIR ANIFLIVE_TTS_VOICE_DIR
export OPEN_JTALK_DICT_DIR="$openjtalk_dict"
export ANIFLIVE_TTS_HOST ANIFLIVE_TTS_PORT
export PYTHONDONTWRITEBYTECODE=1 MLX_ENABLE_TF32=0
# Product PCM streaming is the canonical vocoder stream and needs no research
# opt-in. Only a frozen pre-1.5-handoff worker (which still names that opt-in
# in its canonical-stream check) needs the flag to stream at all.
worker_binary="$ANIFLIVE_TTS_MODEL_PACKAGE/native/bin/aniflive-macos-v5turbo"
if LC_ALL=C grep -aq "Canonical stream requires unqualified research opt-in" "$worker_binary"; then
    export ANIFLIVE_TTS_ALLOW_UNQUALIFIED_STREAM=1
else
    export ANIFLIVE_TTS_ALLOW_UNQUALIFIED_STREAM=0
fi
export ANIFLIVE_TTS_NATIVE_CANONICAL_STREAM=1
export ANIFLIVE_TTS_BOOTSTRAP_STREAM_V2=0
export ANIFLIVE_TTS_NATIVE_CFM_COORDINATE_V2=0
export ANIFLIVE_TTS_NATIVE_CFM_STAGE=0
export ANIFLIVE_TTS_NATIVE_CFM_COMPILED=0
export ANIFLIVE_TTS_NATIVE_CFM_PREPARE=0
export ANIFLIVE_TTS_NATIVE_CFM_BATCHED_QKV=0
export ANIFLIVE_TTS_NATIVE_PRECISION="$package_precision"
export ANIFLIVE_TTS_NATIVE_AR_COMPILED=0
# The same AR accelerations run for FP32 and FP16 (Half GEMV accumulates in F32).
export ANIFLIVE_TTS_NATIVE_AR_COMBINED=1 ANIFLIVE_TTS_NATIVE_AR_MIXED_GEMV=1
export ANIFLIVE_TTS_NATIVE_AR_OPTIMIZED=1 ANIFLIVE_TTS_NATIVE_AR_STATIC=1
export ANIFLIVE_TTS_NATIVE_AR_RESIDUAL_NORM=1
# FP16 reference preparation runs MPS DSP only (no Torch speech model) and must
# never fall back to CPU. Set for both precisions: a later /v1/models/activate
# can switch this process to an FP16 package.
export PYTORCH_ENABLE_MPS_FALLBACK=0
if [[ "$package_precision" == "FP16" ]]; then
    # Bound the MLX freed-buffer cache (allocator policy, no numerical effect).
    export ANIFLIVE_TTS_MLX_CACHE_LIMIT_MB="${ANIFLIVE_TTS_MLX_CACHE_LIMIT_MB:-1024}"
fi
export ANIFLIVE_TTS_NATIVE_COMPLETE_MODE=cached
export ANIFLIVE_TTS_NATIVE_FRONTEND_MODE=persistent-python
export ANIFLIVE_TTS_NATIVE_FRONTEND_METADATA_CACHE=1
export ANIFLIVE_TTS_NATIVE_ONSET_TRIM=1
export ANIFLIVE_TTS_NATIVE_ONSET_TRIM_MAX_MS=60
unset PYTHONPATH ANIFLIVE_COMPONENT_CPU_REFERENCE ANIFLIVE_TTS_RESEARCH_AR_TOP_K
unset ANIFLIVE_HTTP_AUDIT_DIR ANIFLIVE_HTTP_AUDIT_TRACE ANIFLIVE_TTS_AUDIT_TRACE

cd "$MACOS_RELEASE_REPO_ROOT"
# macOS framework Python replaces argv[0]; keep an explicit process-owner marker.
owner_token="$(uuidgen)"
"$ANIFLIVE_TTS_PYTHON" -X "aniflive_macos_owner=$owner_token" -m uvicorn aniflive_tts.api:create_app --factory \
    --host "$ANIFLIVE_TTS_HOST" --port "$ANIFLIVE_TTS_PORT" --no-access-log &
server_pid=$!
pid_tmp="$ANIFLIVE_TTS_PID_FILE.tmp.$$"
cleanup() {
    if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ -f "$ANIFLIVE_TTS_PID_FILE" && "$(sed -n '1p' "$ANIFLIVE_TTS_PID_FILE")" == "${server_pid:-}" ]]; then
        rm -f "$ANIFLIVE_TTS_PID_FILE"
    fi
    rm -f "$pid_tmp"
}
handle_signal() {
    local exit_code="$1"
    cleanup
    trap - EXIT INT TERM
    exit "$exit_code"
}
trap cleanup EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
{
    printf '%s\n' "$server_pid"
    printf '%s\n' "$MACOS_RELEASE_REPO_ROOT"
    printf '%s\n' "$ANIFLIVE_TTS_MODEL_PACKAGE"
    printf '%s\n' "$ANIFLIVE_TTS_HOST"
    printf '%s\n' "$ANIFLIVE_TTS_PORT"
    printf '%s\n' "$ANIFLIVE_TTS_PYTHON"
    printf '%s\n' "$owner_token"
} > "$pid_tmp"
mv "$pid_tmp" "$ANIFLIVE_TTS_PID_FILE"

set +e
wait "$server_pid"
server_status=$?
set -e
cleanup
server_pid=""
trap - EXIT INT TERM
exit "$server_status"
