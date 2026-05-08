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

echo "[docker-dev] HOST_IP=$HOST_IP"
echo "[docker-dev] Expo mode=$EXPO_MODE"
echo "[docker-dev] Backend:   http://$HOST_IP:8000/docs"
echo "[docker-dev] Web admin: http://$HOST_IP:5173"

docker compose up --build
