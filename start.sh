#!/usr/bin/env bash
set -euo pipefail

# 현재 스크립트가 있는 디렉터리
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# 폴더 이름 suffix 입력 받기 (예: mydata)
read -rp "Enter folder name suffix (e.g., mydata): " INPUT_NAME

# 빈 입력 방지
if [[ -z "${INPUT_NAME}" ]]; then
  echo "No input provided. Exiting..."
  read -rp "Press Enter to continue..." _
  exit 1
fi

# data_<INPUT_NAME> 폴더 및 하위 디렉터리 생성
DATA_DIR="${SCRIPT_DIR}/data_${INPUT_NAME}"

mkdir -p "${DATA_DIR}"
mkdir -p "${DATA_DIR}/audio_mp3"
mkdir -p "${DATA_DIR}/audio_wav"
mkdir -p "${DATA_DIR}/result"
mkdir -p "${DATA_DIR}/transcribe"
mkdir -p "${DATA_DIR}/video"

echo "Folder \"${DATA_DIR}\" and subfolders created successfully."

# Docker 컨테이너 실행 (바인드 마운트 포함)
docker run -it --rm -p 7860:7860 --gpus all \
  --name "anitts-container-${INPUT_NAME}" \
  -v "${SCRIPT_DIR}:/workspace/AniTTS-Builder-v3" \
  -v "${DATA_DIR}:/workspace/AniTTS-Builder-v3/data" \
  -v "${SCRIPT_DIR}/module/model:/workspace/AniTTS-Builder-v3/module/model" \
  anitts-builder-v3

# 일시 정지 (Windows의 pause 대체)
read -rp "Press Enter to continue..." _
