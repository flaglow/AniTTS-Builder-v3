#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="${IMAGE_NAME:-anitts-builder-v3}"
INSTALL_CUML="${INSTALL_CUML:-1}"  # Optional: INSTALL_CUML=0 ./setup.sh
NO_CACHE="${NO_CACHE:-0}"          # Optional: NO_CACHE=1 ./setup.sh

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker command not found. Please install Docker first."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[ERROR] Docker daemon is not available. Please start Docker and retry."
  exit 1
fi

BUILD_ARGS=()
if [[ "$NO_CACHE" == "1" ]]; then
  BUILD_ARGS+=(--no-cache)
fi

echo "[INFO] Building Docker image: ${IMAGE_NAME}"
echo "[INFO] Build options: INSTALL_CUML=${INSTALL_CUML}, NO_CACHE=${NO_CACHE}"

docker build "${BUILD_ARGS[@]}" \
  --build-arg INSTALL_CUML="${INSTALL_CUML}" \
  -t "${IMAGE_NAME}" .

echo "[INFO] Build completed: ${IMAGE_NAME}"
