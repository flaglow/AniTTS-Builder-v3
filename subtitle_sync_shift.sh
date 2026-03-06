#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'USAGE'
Usage:
  ./subtitle_sync_shift.sh <project_suffix> <input_srt> [shift_ms] [output_srt]

Examples:
  ./subtitle_sync_shift.sh mydata episode01.srt 1500
  ./subtitle_sync_shift.sh mydata episode01.srt -700 episode01.fixed.srt

Notes:
- Default input base dir: data_<project_suffix>/transcribe
- If output_srt is omitted, '<input_name>.shifted.srt' is created next to input.
- This script supports .srt files.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PROJECT_SUFFIX="${1:-}"
INPUT_ARG="${2:-}"
SHIFT_MS="${3:-1500}"
OUTPUT_ARG="${4:-}"

if [[ -z "$PROJECT_SUFFIX" || -z "$INPUT_ARG" ]]; then
  usage
  exit 1
fi

if [[ ! "$PROJECT_SUFFIX" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "[ERROR] Invalid project suffix: $PROJECT_SUFFIX"
  exit 1
fi

if [[ ! "$SHIFT_MS" =~ ^-?[0-9]+$ ]]; then
  echo "[ERROR] shift_ms must be an integer (milliseconds): $SHIFT_MS"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker command not found."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "[ERROR] Docker daemon is not available."
  exit 1
fi

DATA_DIR="$SCRIPT_DIR/data_${PROJECT_SUFFIX}"
TRANSCRIBE_DIR="$DATA_DIR/transcribe"
CONTAINER_NAME="anitts-container-${PROJECT_SUFFIX}"
IMAGE_NAME="${IMAGE_NAME:-anitts-builder-v3}"

if [[ ! -d "$TRANSCRIBE_DIR" ]]; then
  echo "[ERROR] Directory not found: $TRANSCRIBE_DIR"
  exit 1
fi

resolve_input() {
  local arg="$1"
  if [[ -f "$arg" ]]; then
    realpath "$arg"
    return 0
  fi
  if [[ -f "$TRANSCRIBE_DIR/$arg" ]]; then
    realpath "$TRANSCRIBE_DIR/$arg"
    return 0
  fi
  return 1
}

INPUT_HOST_PATH="$(resolve_input "$INPUT_ARG" || true)"
if [[ -z "$INPUT_HOST_PATH" ]]; then
  echo "[ERROR] Input file not found: $INPUT_ARG"
  echo "        Tried:"
  echo "        - $INPUT_ARG"
  echo "        - $TRANSCRIBE_DIR/$INPUT_ARG"
  exit 1
fi

if [[ "${INPUT_HOST_PATH##*.}" != "srt" && "${INPUT_HOST_PATH##*.}" != "SRT" ]]; then
  echo "[ERROR] Only .srt files are supported: $INPUT_HOST_PATH"
  exit 1
fi

if [[ "$INPUT_HOST_PATH" != "$SCRIPT_DIR"/* ]]; then
  echo "[ERROR] Input must be inside project directory: $SCRIPT_DIR"
  echo "        Input: $INPUT_HOST_PATH"
  exit 1
fi

if [[ -z "$OUTPUT_ARG" ]]; then
  input_dir="$(dirname "$INPUT_HOST_PATH")"
  input_stem="$(basename "$INPUT_HOST_PATH" .srt)"
  OUTPUT_HOST_PATH="$input_dir/${input_stem}.shifted.srt"
else
  if [[ "$OUTPUT_ARG" = /* ]]; then
    OUTPUT_HOST_PATH="$OUTPUT_ARG"
  elif [[ "$OUTPUT_ARG" == *"/"* ]]; then
    OUTPUT_HOST_PATH="$SCRIPT_DIR/$OUTPUT_ARG"
  else
    OUTPUT_HOST_PATH="$(dirname "$INPUT_HOST_PATH")/$OUTPUT_ARG"
  fi
  OUTPUT_HOST_PATH="$(realpath -m "$OUTPUT_HOST_PATH")"
fi

if [[ "$OUTPUT_HOST_PATH" != "$SCRIPT_DIR"/* ]]; then
  echo "[ERROR] Output must be inside project directory: $SCRIPT_DIR"
  echo "        Output: $OUTPUT_HOST_PATH"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_HOST_PATH")"

to_container_path() {
  local host_path="$1"
  local rel="${host_path#"$SCRIPT_DIR"/}"
  if [[ "$rel" == "$host_path" ]]; then
    return 1
  fi
  printf '/workspace/AniTTS-Builder-v3/%s' "$rel"
}

INPUT_CONT_PATH="$(to_container_path "$INPUT_HOST_PATH")"
OUTPUT_CONT_PATH="$(to_container_path "$OUTPUT_HOST_PATH")"

PY_CODE='import sys, pysubs2; s=pysubs2.load(sys.argv[1]); s.shift(ms=int(sys.argv[3])); s.save(sys.argv[2]); print(f"[OK] saved: {sys.argv[2]}")'

echo "[INFO] Input : $INPUT_HOST_PATH"
echo "[INFO] Output: $OUTPUT_HOST_PATH"
echo "[INFO] Shift : ${SHIFT_MS} ms"

if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "[INFO] Running in existing container: $CONTAINER_NAME"
  docker exec "$CONTAINER_NAME" \
    python3 -c "$PY_CODE" "$INPUT_CONT_PATH" "$OUTPUT_CONT_PATH" "$SHIFT_MS"
else
  echo "[INFO] Container '$CONTAINER_NAME' not running. Running one-shot container with image '$IMAGE_NAME'."
  docker run --rm \
    -v "$SCRIPT_DIR:/workspace/AniTTS-Builder-v3" \
    "$IMAGE_NAME" \
    python3 -c "$PY_CODE" "$INPUT_CONT_PATH" "$OUTPUT_CONT_PATH" "$SHIFT_MS"
fi

echo "[DONE] Subtitle sync shift completed."
