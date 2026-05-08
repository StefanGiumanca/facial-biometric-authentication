#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${HOST_IP:-}" ]]; then
  HOST_IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
fi

if [[ -z "${HOST_IP:-}" ]]; then
  HOST_IP="$(ipconfig getifaddr en1 2>/dev/null || true)"
fi

if [[ -z "${HOST_IP:-}" ]]; then
  HOST_IP="$(ifconfig en0 2>/dev/null | awk '/inet / {print $2; exit}')"
fi

if [[ -z "${HOST_IP:-}" ]]; then
  echo "Could not detect your Wi-Fi IP automatically."
  echo "Run: HOST_IP=YOUR_WIFI_IP scripts/docker-dev.sh"
  exit 1
fi

export HOST_IP
export EXPO_MODE="${EXPO_MODE:-lan}"
export BACKEND_MODE="${BACKEND_MODE:-docker}"

echo "[docker-dev] HOST_IP=$HOST_IP"
echo "[docker-dev] Expo mode=$EXPO_MODE"
echo "[docker-dev] Backend:   http://$HOST_IP:8000/docs"
echo "[docker-dev] Web admin: http://$HOST_IP:5173"

if [[ "$BACKEND_MODE" == "native" ]]; then
  export DATABASE_URL="${DATABASE_URL:-postgresql+psycopg2://postgres:postgres@localhost:5433/visionauth}"
  export ADMIN_KEY="${ADMIN_KEY:-dev-admin-key}"
  export OCR_USE_GPU="${OCR_USE_GPU:-1}"
  export OMP_NUM_THREADS="${VISIONAUTH_THREADS:-4}"
  export OPENBLAS_NUM_THREADS="${VISIONAUTH_THREADS:-4}"
  export MKL_NUM_THREADS="${VISIONAUTH_THREADS:-4}"
  export NUMEXPR_NUM_THREADS="${VISIONAUTH_THREADS:-4}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/.cache}"

  echo "[docker-dev] Backend mode=native"
  docker compose up --build postgres web-admin mobile &
  COMPOSE_PID=$!

  cleanup() {
    kill "$COMPOSE_PID" 2>/dev/null || true
    docker compose stop postgres web-admin mobile >/dev/null 2>&1 || true
  }
  trap cleanup EXIT INT TERM

  venv/bin/python -m uvicorn backend.api.app:app --host 0.0.0.0 --port 8000 --reload
else
  echo "[docker-dev] Backend mode=docker"
  COMPOSE_PROFILES=docker-backend docker compose up --build
fi
