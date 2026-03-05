#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="${IMAGE_NAME:-anitts-builder-v3}"
APP_PORT="${APP_PORT:-7860}"
USE_GPU="${USE_GPU:-1}"                    # USE_GPU=0 ./start.sh mydata
ALLOW_CPU_FALLBACK="${ALLOW_CPU_FALLBACK:-1}"
DATA_PREFIX="${DATA_PREFIX:-data_}"

INPUT_NAME="${1:-}"
if [[ -z "${INPUT_NAME}" ]]; then
  read -rp "Enter folder name suffix (e.g., mydata): " INPUT_NAME
fi

if [[ -z "${INPUT_NAME}" ]]; then
  echo "[ERROR] No input provided. Exiting."
  exit 1
fi

if [[ ! "${INPUT_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "[ERROR] Invalid suffix: '${INPUT_NAME}'"
  echo "        Allowed chars: A-Z, a-z, 0-9, underscore(_), hyphen(-)"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker command not found. Please install Docker first."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[ERROR] Docker daemon is not available. Please start Docker and retry."
  exit 1
fi

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "[WARN] Docker image '${IMAGE_NAME}' not found. Running setup.sh to build it."
  bash "${SCRIPT_DIR}/setup.sh"
fi

DATA_DIR="${SCRIPT_DIR}/${DATA_PREFIX}${INPUT_NAME}"
CONTAINER_NAME="anitts-container-${INPUT_NAME}"

mkdir -p "${DATA_DIR}"
mkdir -p "${DATA_DIR}/audio_mp3"
mkdir -p "${DATA_DIR}/audio_wav"
mkdir -p "${DATA_DIR}/result"
mkdir -p "${DATA_DIR}/transcribe"
mkdir -p "${DATA_DIR}/video"

echo "[INFO] Data folder ready: ${DATA_DIR}"

# --rm인데 비정상 종료 시 남을 수 있어 방어적으로 정리
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "[WARN] Removing existing container: ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

run_container() {
  local use_gpu_flag="$1"
  local -a gpu_args=()
  if [[ "${use_gpu_flag}" == "1" ]]; then
    gpu_args+=(--gpus all)
  fi

  echo "[INFO] Starting AniTTS (GPU=${use_gpu_flag}) at http://localhost:${APP_PORT}"
  docker run -it --rm \
    -p "${APP_PORT}:7860" \
    "${gpu_args[@]}" \
    --name "${CONTAINER_NAME}" \
    -v "${SCRIPT_DIR}:/workspace/AniTTS-Builder-v3" \
    -v "${DATA_DIR}:/workspace/AniTTS-Builder-v3/data" \
    "${IMAGE_NAME}" \
    python3 main.py
}

if [[ "${USE_GPU}" == "1" ]]; then
  set +e
  run_container "1"
  rc=$?
  set -e
  if [[ $rc -ne 0 && "${ALLOW_CPU_FALLBACK}" == "1" ]]; then
    echo "[WARN] GPU run failed (exit=${rc}). Retrying with CPU mode..."
    run_container "0"
  elif [[ $rc -ne 0 ]]; then
    exit "$rc"
  fi
else
  run_container "0"
fi
