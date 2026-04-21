#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build and validate the W2V TensorRT release engine from a pre-exported ONNX bundle.

Default flow:
1. Extract artifacts/w2v-stage3-onnx-bundle.tar.gz into artifacts/model-bundle-w2vbert2-stage3-onnx
2. Delete the stale model.plan from the bundle
3. Run the pinned nvcr.io/nvidia/pytorch:25.06-py3 builder in Docker
4. Validate the new engine against ONNX and write the report

Examples:
  bash research/scripts/build_w2v_trt_release_from_bundle.sh
  BUNDLE_PATH=/srv/w2v-stage3-onnx-bundle.tar.gz bash research/scripts/build_w2v_trt_release_from_bundle.sh
  CONFIG_PATH=research/configs/release/tensorrt-fp16-w2vbert2-stage3.toml bash research/scripts/build_w2v_trt_release_from_bundle.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

BUNDLE_PATH="${BUNDLE_PATH:-${PROJECT_ROOT}/artifacts/w2v-stage3-onnx-bundle.tar.gz}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/research/configs/release/tensorrt-fp16-w2vbert2-stage3-ampere_plus-b512.toml}"
DOCKER_IMAGE="${DOCKER_IMAGE:-nvcr.io/nvidia/pytorch:25.06-py3}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/artifacts/.pip-cache/w2v-trt-builder}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

toml_string_value() {
  local key="$1"
  awk -F'"' -v key="$key" '
    $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
      print $2
      exit
    }
  ' "$CONFIG_PATH"
}

require_command docker
require_command tar
require_command sha256sum

if [[ ! -f "$BUNDLE_PATH" ]]; then
  echo "Bundle archive not found: $BUNDLE_PATH" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH" >&2
  exit 1
fi

ENGINE_RELATIVE_PATH="$(toml_string_value engine_path)"
OUTPUT_RELATIVE_PATH="$(toml_string_value output_root)"
if [[ -z "$ENGINE_RELATIVE_PATH" || -z "$OUTPUT_RELATIVE_PATH" ]]; then
  echo "Failed to read engine_path/output_root from $CONFIG_PATH" >&2
  exit 1
fi

ENGINE_PATH="${PROJECT_ROOT}/${ENGINE_RELATIVE_PATH}"
ENGINE_DIR="$(dirname -- "$ENGINE_PATH")"
REPORT_DIR="${PROJECT_ROOT}/${OUTPUT_RELATIVE_PATH}"
REPORT_JSON="${REPORT_DIR}/generic_tensorrt_engine_report.json"
METADATA_PATH="${ENGINE_PATH}.metadata.json"

mkdir -p "${PROJECT_ROOT}/artifacts" "$PIP_CACHE_DIR"

echo "[w2v-trt-build] bundle=${BUNDLE_PATH}"
echo "[w2v-trt-build] config=${CONFIG_PATH}"
echo "[w2v-trt-build] image=${DOCKER_IMAGE}"

rm -rf "$ENGINE_DIR" "$REPORT_DIR"
tar -C "${PROJECT_ROOT}/artifacts" -xzf "$BUNDLE_PATH"

if [[ ! -f "${ENGINE_DIR}/model.onnx" ]]; then
  echo "Extracted bundle is missing model.onnx: ${ENGINE_DIR}/model.onnx" >&2
  exit 1
fi

# The uploaded bundle currently contains an old incompatible model.plan.
# Remove it before the build so a failed build never leaves a stale engine behind.
rm -f "$ENGINE_PATH" "$METADATA_PATH"

docker run \
  --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=/workspace/src \
  -e PIP_CACHE_DIR=/workspace/artifacts/.pip-cache/w2v-trt-builder \
  -e KRYPTONITE_TRT_BUILDER_IMAGE="$DOCKER_IMAGE" \
  -v "${PROJECT_ROOT}:/workspace" \
  -v "${CONFIG_PATH}:/tmp/w2v-trt-build-config.toml:ro" \
  -w /workspace \
  "$DOCKER_IMAGE" \
  bash -lc "set -euo pipefail; python -m pip install --user --disable-pip-version-check 'onnxruntime-gpu>=1.24.4'; python research/scripts/build_generic_tensorrt_engine.py --config /tmp/w2v-trt-build-config.toml"

if [[ ! -f "$ENGINE_PATH" ]]; then
  echo "Build did not produce engine: $ENGINE_PATH" >&2
  exit 1
fi
if [[ ! -f "$METADATA_PATH" ]]; then
  echo "Build did not produce metadata: $METADATA_PATH" >&2
  exit 1
fi
if [[ ! -f "$REPORT_JSON" ]]; then
  echo "Build did not produce report: $REPORT_JSON" >&2
  exit 1
fi
if ! grep -q '"status": "pass"' "$REPORT_JSON"; then
  echo "TensorRT build report status is not pass: $REPORT_JSON" >&2
  exit 1
fi

echo "[w2v-trt-build] build_and_validation=pass"
echo "[w2v-trt-build] engine=${ENGINE_PATH}"
echo "[w2v-trt-build] metadata=${METADATA_PATH}"
echo "[w2v-trt-build] report=${REPORT_JSON}"
echo "[w2v-trt-build] sha256 $(sha256sum "$ENGINE_PATH")"
echo "[w2v-trt-build] sha256 $(sha256sum "$METADATA_PATH")"
