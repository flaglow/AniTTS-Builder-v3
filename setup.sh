#!/usr/bin/env bash
set -euo pipefail

# 현재 스크립트가 있는 디렉터리로 이동
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Optional: INSTALL_CUML=1 ./setup.sh
INSTALL_CUML="${INSTALL_CUML:-0}"

# Docker 이미지 빌드 (기본: 캐시 사용)
docker build \
  --build-arg INSTALL_CUML="${INSTALL_CUML}" \
  -t anitts-builder-v3 .

# 일시 정지 (Windows의 pause 대체)
read -rp "Press Enter to continue..." _
