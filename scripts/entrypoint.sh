#!/usr/bin/env bash
set -Eeuo pipefail

log() { printf '[aniflive-tts] %s\n' "$*" >&2; }
fail() { log "ERROR: $*"; exit 1; }

command="${1:-serve}"
shift || true

case "${command}" in
  serve)
    : "${ANIFLIVE_TTS_MODEL_PACKAGE:=/data/models/active}"
    : "${ANIFLIVE_TTS_SHARED_DIR:=/data/shared}"
    : "${ANIFLIVE_TTS_PORT:=9880}"
    [[ -f "${ANIFLIVE_TTS_MODEL_PACKAGE}/manifest.json" ]] || fail "Missing model package. Run 'aniflive-tts model convert' explicitly before serve."
    [[ -d "${ANIFLIVE_TTS_SHARED_DIR}" ]] || fail "Missing shared resources: ${ANIFLIVE_TTS_SHARED_DIR}"
    python -m aniflive_tts validate --model-package "${ANIFLIVE_TTS_MODEL_PACKAGE}" >/data/reports/startup-validation.json
    log "Starting strict TensorRT 11 API on 0.0.0.0:${ANIFLIVE_TTS_PORT}; no runtime download/build is permitted."
    exec python -m aniflive_tts serve --model-package "${ANIFLIVE_TTS_MODEL_PACKAGE}" --shared-dir "${ANIFLIVE_TTS_SHARED_DIR}" --host 0.0.0.0 --port "${ANIFLIVE_TTS_PORT}" "$@"
    ;;
  convert)
    exec python -m aniflive_tts model convert "$@"
    ;;
  validate)
    exec python -m aniflive_tts validate "$@"
    ;;
  benchmark)
    exec python /app/scripts/benchmark_readme.py "$@"
    ;;
  webui)
    exec python -m aniflive_tts webui "$@"
    ;;
  *) exec "${command}" "$@" ;;
esac
