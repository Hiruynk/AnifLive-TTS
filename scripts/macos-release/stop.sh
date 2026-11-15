#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/macos-release/common.sh
source "$script_dir/common.sh"

pid_file="$ANIFLIVE_TTS_PID_FILE"
if [[ ! -e "$pid_file" ]]; then
    printf 'No managed macOS API PID file at %s\n' "$pid_file"
    exit 0
fi
if [[ -L "$pid_file" ]]; then
    macos_release_error "refusing to read a symlink PID file: $pid_file"
    exit 1
fi

pid="$(sed -n '1p' "$pid_file")"
owner_root="$(sed -n '2p' "$pid_file")"
owner_package="$(sed -n '3p' "$pid_file")"
owner_host="$(sed -n '4p' "$pid_file")"
owner_port="$(sed -n '5p' "$pid_file")"
owner_python="$(sed -n '6p' "$pid_file")"
owner_token="$(sed -n '7p' "$pid_file")"
if [[ ! "$pid" =~ ^[0-9]+$ || "$owner_root" != "$MACOS_RELEASE_REPO_ROOT" \
      || -z "$owner_package" || -z "$owner_host" || ! "$owner_port" =~ ^[0-9]+$ \
      || -z "$owner_python" || ! "$owner_token" =~ ^[A-Fa-f0-9-]{36}$ ]]; then
    macos_release_error "PID ownership record is incomplete or belongs to another source tree: $pid_file"
    exit 1
fi
if [[ -n "${ANIFLIVE_TTS_MODEL_PACKAGE:-}" && "$ANIFLIVE_TTS_MODEL_PACKAGE" != "$owner_package" ]]; then
    macos_release_error "PID record belongs to a different model package; refusing to signal it"
    exit 1
fi
if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pid_file"
    printf 'Removed stale PID record for exited process %s\n' "$pid"
    exit 0
fi

command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
case "$command_line" in
    *"-X aniflive_macos_owner=$owner_token "*"aniflive_tts.api:create_app"*"--factory"*"--host $owner_host"*"--port $owner_port"*) ;;
    *)
        macos_release_error "PID $pid is not the recorded AnifLive-TTS uvicorn process; refusing to signal"
        exit 1
        ;;
esac

kill -TERM "$pid"
deadline=$((SECONDS + ANIFLIVE_TTS_STOP_TIMEOUT_SECONDS))
while kill -0 "$pid" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
        macos_release_error "owned server did not exit after SIGTERM within ${ANIFLIVE_TTS_STOP_TIMEOUT_SECONDS}s; no force kill was sent"
        exit 1
    fi
    sleep 1
done
if [[ -f "$pid_file" && "$(sed -n '1p' "$pid_file")" == "$pid" ]]; then
    rm -f "$pid_file"
fi
printf 'Stopped owned AnifLive-TTS server PID %s\n' "$pid"
