#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALLER_CWD="$PWD"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/code/campp/ms42_release${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'USAGE'
Usage:
  ./run.sh [options]

Host mode:
  Prefetches only the selected model artifacts into repo-local data/models/,
  stages them into the Docker build context, builds a self-contained image,
  runs the container, and writes submission.csv.

Container mode:
  Executes the selected inference pipeline directly inside the container
  without attempting any network downloads.

Core options:
  --model {w2v-trt|campp-pt}  Submission path. Default: w2v-trt.
  --test-csv PATH             Required. Input CSV with filepath column.
  --data-root PATH            Root directory for filepath resolution.
                              Default: directory containing --test-csv.
  --output-dir PATH           Runtime output root. Default: data/submission_entrypoint.
  --device DEVICE             Inference device. Default: cuda.
  --batch-size INT            Override the primary model batch size.
  --top-k INT                 Number of neighbours to write. Default: 10.
  --offline                   Disable network downloads for missing artifacts.
  --dry-run                   Print commands without running them.
  --help                      Show this help message.

Model defaults:
  w2v-trt:
    batch-size=300, num-workers=4, prefetch-factor=1, search-batch-size=4096,
    top-cache-k=300, crop-seconds=6.0, n-crops=3, precision=bf16

  campp-pt:
    batch-size=256, frontend-workers=16, frontend-prefetch=256,
    search-batch-size=4096, top-cache-k=200, class-batch-size=4096, class-top-k=5

Advanced overrides are available via environment variables:
  USE_UV_RUN
  TEMPLATE_CSV ARTIFACTS_MANIFEST SEARCH_DEVICE TOP_K
  W2V_BATCH_SIZE W2V_PRECISION W2V_NUM_WORKERS W2V_PREFETCH_FACTOR W2V_PIN_MEMORY
  W2V_SEARCH_BATCH_SIZE W2V_TOP_CACHE_K W2V_CROP_SECONDS W2V_N_CROPS
  W2V_EDGE_TOP W2V_RECIPROCAL_TOP W2V_RANK_TOP W2V_ITERATIONS
  W2V_LABEL_MIN_SIZE W2V_LABEL_MAX_SIZE W2V_LABEL_MIN_CANDIDATES
  W2V_SHARED_TOP W2V_SHARED_MIN_COUNT W2V_RECIPROCAL_BONUS W2V_DENSITY_PENALTY
  W2V_TENSORRT_OUTPUT_NAME W2V_TENSORRT_PROFILE_INDEX W2V_RUN_ID
  W2V_CHECKPOINT W2V_TENSORRT_ENGINE
  CAMPP_BATCH_SIZE CAMPP_CONFIG CAMPP_RUN_ID CAMPP_CHECKPOINT CAMPP_FRONTEND_WORKERS
  CAMPP_FRONTEND_PREFETCH CAMPP_SEARCH_BATCH_SIZE CAMPP_TOP_CACHE_K
  CAMPP_CLASS_BATCH_SIZE CAMPP_CLASS_TOP_K
  KRYPTONITE_CAMPP_CHECKPOINT_URL KRYPTONITE_W2V_PLAN_URL
  KRYPTONITE_W2V_TEACHER_BUNDLE_URL
  DOCKER_IMAGE DOCKERFILE_PATH DOCKER_BASE_IMAGE
USAGE
}

log() {
  printf '[submission-entrypoint] %s\n' "$*" >&2
}

ensure_host_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run: uv missing; would install via python3 -m pip install --user uv"
    UV_BIN="uv"
    return 0
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    log "python3 is required to bootstrap uv on the host"
    return 1
  fi
  log "uv not found on host; installing via python3 -m pip install --user uv"
  python3 -m pip install --user uv || return 1
  UV_BIN="$(
    python3 - <<'PY'
from pathlib import Path
import shutil
import site

user_bin = Path(site.getuserbase()) / "bin" / "uv"
if user_bin.is_file():
    print(user_bin)
else:
    print(shutil.which("uv") or "")
PY
  )"
  if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
    log "uv bootstrap completed but uv binary was not found"
    return 1
  fi
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[submission-entrypoint] dry-run:' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
    return 0
  fi
  "$@"
}

resolve_host_path() {
  local raw_path="$1"
  if [[ -z "$raw_path" ]]; then
    RESOLVED_HOST_PATH=""
    return 0
  fi
  if [[ "$raw_path" == /* ]]; then
    RESOLVED_HOST_PATH="$raw_path"
    return 0
  fi
  RESOLVED_HOST_PATH="$(
    python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=False))' \
      "${CALLER_CWD}/${raw_path}"
  )"
}

device_requires_gpu() {
  local device="$1"
  case "${device,,}" in
    cuda*|gpu*|nvidia*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

validate_top_k() {
  if [[ ! "$TOP_K" =~ ^[0-9]+$ ]]; then
    log "invalid top-k: ${TOP_K}; expected integer in platform range 10 <= K < 1000"
    return 1
  fi
  if (( TOP_K < 10 || TOP_K >= 1000 )); then
    log "invalid top-k: ${TOP_K}; expected platform range 10 <= K < 1000"
    return 1
  fi
}

check_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      log "dry-run: missing ${label} ignored: ${path}"
      return 0
    fi
    log "missing ${label}: ${path}"
    return 1
  fi
}

check_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      log "dry-run: missing ${label} ignored: ${path}"
      return 0
    fi
    log "missing ${label}: ${path}"
    return 1
  fi
}

check_common_inputs() {
  check_file "test CSV" "$TEST_CSV" || return 1
  check_file "template CSV" "$TEMPLATE_CSV" || return 1
  check_dir "data root" "$DATA_ROOT" || return 1
}

validate_root_submission() {
  local output_dir="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  check_file "root submission" "${PROJECT_ROOT}/submission.csv" || return 1
  mkdir -p "$output_dir" || return 1
  run_cmd "${PYTHON_RUNNER[@]}" "${PROJECT_ROOT}/utils/validate_submission.py" \
    --template-csv "$TEMPLATE_CSV" \
    --submission-csv "${PROJECT_ROOT}/submission.csv" \
    --output-json "$output_dir/submission_validation.json" \
    --k "$TOP_K" || return 1
  sha256sum "${PROJECT_ROOT}/submission.csv" | tee "$output_dir/submission.sha256" >/dev/null
}

ensure_model_artifacts() {
  local model="$1"
  local command=(
    "${PYTHON_RUNNER[@]}"
    "${PROJECT_ROOT}/utils/download_artifacts.py"
    --model "$model"
    --manifest "$ARTIFACTS_MANIFEST"
  )
  if [[ "$OFFLINE" == "1" ]]; then
    command+=(--offline)
  fi
  run_cmd "${command[@]}" || return 1
}

stage_model_artifacts() {
  local model="$1"
  local command=(
    "${PYTHON_RUNNER[@]}"
    "${PROJECT_ROOT}/utils/download_artifacts.py"
    --model "$model"
    --manifest "$ARTIFACTS_MANIFEST"
    --stage-dir "$ARTIFACT_STAGE_DIR"
  )
  run_cmd "${command[@]}" || return 1
}

copy_submission_from_output() {
  local host_output_dir="$1"
  local final_submission=""
  case "$MODEL" in
    w2v-trt)
      final_submission="${host_output_dir}/w2v_trt/submission_${W2V_RUN_ID}.csv"
      ;;
    campp-pt)
      final_submission="${host_output_dir}/campp/submission_${CAMPP_RUN_ID}.csv"
      ;;
  esac
  check_file "host submission" "$final_submission" || return 1
  cp "$final_submission" "${PROJECT_ROOT}/submission.csv" || return 1
}

run_w2v_trt() {
  local output_dir="$OUTPUT_DIR/w2v_trt"
  local run_id="${W2V_RUN_ID}"
  local batch_size="${BATCH_SIZE_OVERRIDE:-$W2V_BATCH_SIZE}"
  log "w2v-trt start run_id=${run_id} output_dir=${output_dir}"
  check_common_inputs || return 1
  check_file "W2V checkpoint metadata" "$W2V_CHECKPOINT/checkpoint_metadata.json" || return 1
  check_file "W2V feature extractor" "$W2V_CHECKPOINT/feature_extractor/preprocessor_config.json" || return 1
  check_file "W2V TensorRT engine" "$W2V_TENSORRT_ENGINE" || return 1
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$output_dir" || return 1
  fi

  local command=(
    "${PYTHON_RUNNER[@]}"
    "${PROJECT_ROOT}/research/scripts/run_teacher_peft_c4_tail.py"
    --checkpoint-path "$W2V_CHECKPOINT"
    --manifest-csv "$TEST_CSV"
    --template-csv "$TEMPLATE_CSV"
    --data-root "$DATA_ROOT"
    --output-dir "$output_dir"
    --experiment-id "$run_id"
    --encoder-backend tensorrt
    --tensorrt-engine-path "$W2V_TENSORRT_ENGINE"
    --tensorrt-output-name "$W2V_TENSORRT_OUTPUT_NAME"
    --tensorrt-profile-index "$W2V_TENSORRT_PROFILE_INDEX"
    --device "$DEVICE"
    --search-device "$SEARCH_DEVICE"
    --precision "$W2V_PRECISION"
    --batch-size "$batch_size"
    --output-top-k "$TOP_K"
    --num-workers "$W2V_NUM_WORKERS"
    --prefetch-factor "$W2V_PREFETCH_FACTOR"
    --search-batch-size "$W2V_SEARCH_BATCH_SIZE"
    --top-cache-k "$W2V_TOP_CACHE_K"
    --crop-seconds "$W2V_CROP_SECONDS"
    --n-crops "$W2V_N_CROPS"
    --edge-top "$W2V_EDGE_TOP"
    --reciprocal-top "$W2V_RECIPROCAL_TOP"
    --rank-top "$W2V_RANK_TOP"
    --iterations "$W2V_ITERATIONS"
    --label-min-size "$W2V_LABEL_MIN_SIZE"
    --label-max-size "$W2V_LABEL_MAX_SIZE"
    --label-min-candidates "$W2V_LABEL_MIN_CANDIDATES"
    --shared-top "$W2V_SHARED_TOP"
    --shared-min-count "$W2V_SHARED_MIN_COUNT"
    --reciprocal-bonus "$W2V_RECIPROCAL_BONUS"
    --density-penalty "$W2V_DENSITY_PENALTY"
    --force-embeddings
  )
  if [[ "$W2V_PIN_MEMORY" == "1" ]]; then
    command+=(--pin-memory)
  else
    command+=(--no-pin-memory)
  fi

  run_cmd "${command[@]}" || return 1
  if [[ "$DRY_RUN" == "1" ]]; then
    log "w2v-trt dry-run complete"
    return 0
  fi

  local final_submission="$output_dir/submission_${run_id}.csv"
  check_file "W2V submission" "$final_submission" || return 1
  cp "$final_submission" "${PROJECT_ROOT}/submission.csv" || return 1
  validate_root_submission "$output_dir" || return 1
  log "w2v-trt complete submission.csv"
}

run_campp() {
  local output_dir="$OUTPUT_DIR/campp"
  local batch_size="${BATCH_SIZE_OVERRIDE:-$CAMPP_BATCH_SIZE}"
  log "campp-pt start run_id=${CAMPP_RUN_ID} output_dir=${output_dir}"
  check_common_inputs || return 1
  check_file "CAM++ config" "$CAMPP_CONFIG" || return 1
  check_file "CAM++ checkpoint" "$CAMPP_CHECKPOINT" || return 1
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$output_dir" || return 1
  fi

  local command=(
    "${PYTHON_RUNNER[@]}"
    "${PROJECT_ROOT}/code/campp/run_ms42_submission.py"
    --config "$CAMPP_CONFIG"
    --manifest-csv "$TEST_CSV"
    --template-csv "$TEMPLATE_CSV"
    --data-root "$DATA_ROOT"
    --output-dir "$output_dir"
    --run-id "$CAMPP_RUN_ID"
    --checkpoint-path "$CAMPP_CHECKPOINT"
    --encoder-backend torch
    --device "$DEVICE"
    --search-device "$SEARCH_DEVICE"
    --batch-size "$batch_size"
    --output-top-k "$TOP_K"
  )
  if [[ -n "$CAMPP_FRONTEND_WORKERS" ]]; then
    command+=(--frontend-workers "$CAMPP_FRONTEND_WORKERS")
  fi
  if [[ -n "$CAMPP_FRONTEND_PREFETCH" ]]; then
    command+=(--frontend-prefetch "$CAMPP_FRONTEND_PREFETCH")
  fi
  if [[ -n "$CAMPP_SEARCH_BATCH_SIZE" ]]; then
    command+=(--search-batch-size "$CAMPP_SEARCH_BATCH_SIZE")
  fi
  if [[ -n "$CAMPP_TOP_CACHE_K" ]]; then
    command+=(--top-cache-k "$CAMPP_TOP_CACHE_K")
  fi
  if [[ -n "$CAMPP_CLASS_BATCH_SIZE" ]]; then
    command+=(--class-batch-size "$CAMPP_CLASS_BATCH_SIZE")
  fi
  if [[ -n "$CAMPP_CLASS_TOP_K" ]]; then
    command+=(--class-top-k "$CAMPP_CLASS_TOP_K")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    command+=(--dry-run)
  fi

  run_cmd "${command[@]}" || return 1
  validate_root_submission "$output_dir" || return 1
  if [[ "$DRY_RUN" == "1" ]]; then
    log "campp-pt dry-run complete"
  else
    log "campp-pt complete submission.csv"
  fi
}

append_env_if_set() {
  local name="$1"
  local value="${!name:-}"
  if [[ -n "$value" ]]; then
    DOCKER_ENV_ARGS+=(--env "${name}=${value}")
  fi
}

append_path_env_if_set() {
  local name="$1"
  local kind="${2:-auto}"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    return 0
  fi
  resolve_host_path "$value" || return 1
  to_container_path "$RESOLVED_HOST_PATH" "$kind" || return 1
  DOCKER_ENV_ARGS+=(--env "${name}=${TO_CONTAINER_PATH_RESULT}")
}

register_host_mount() {
  local path="$1"
  local kind="${2:-auto}"
  local source="$path"
  local leaf=""
  local index
  local target=""

  if [[ "$kind" == "auto" ]]; then
    if [[ -d "$path" ]]; then
      kind="dir"
    else
      kind="file"
    fi
  fi

  case "$kind" in
    dir)
      source="$path"
      ;;
    file)
      source="$(dirname "$path")"
      leaf="$(basename "$path")"
      ;;
    *)
      log "unsupported host mount kind: ${kind}"
      return 1
      ;;
  esac

  for index in "${!DOCKER_MOUNT_SOURCES[@]}"; do
    if [[ "${DOCKER_MOUNT_SOURCES[$index]}" == "$source" ]]; then
      target="${DOCKER_MOUNT_TARGETS[$index]}"
      if [[ "$kind" == "file" ]]; then
        REGISTER_HOST_MOUNT_RESULT="${target}/${leaf}"
      else
        REGISTER_HOST_MOUNT_RESULT="${target}"
      fi
      return 0
    fi
  done

  index="${#DOCKER_MOUNT_SOURCES[@]}"
  target="/hostfs/${index}"
  DOCKER_MOUNT_SOURCES+=("$source")
  DOCKER_MOUNT_TARGETS+=("$target")
  DOCKER_MOUNT_KINDS+=("$kind")
  DOCKER_VOLUME_ARGS+=(-v "${source}:${target}")

  if [[ "$kind" == "file" ]]; then
    REGISTER_HOST_MOUNT_RESULT="${target}/${leaf}"
  else
    REGISTER_HOST_MOUNT_RESULT="${target}"
  fi
}

to_container_path() {
  local path="$1"
  local kind="${2:-auto}"
  if [[ -z "$path" ]]; then
    TO_CONTAINER_PATH_RESULT="$path"
    return 0
  fi
  if [[ "$path" != /* ]]; then
    TO_CONTAINER_PATH_RESULT="$path"
    return 0
  fi
  register_host_mount "$path" "$kind" || return 1
  TO_CONTAINER_PATH_RESULT="$REGISTER_HOST_MOUNT_RESULT"
}

run_host_docker() {
  local host_test_csv
  local host_template_csv
  local host_data_root
  local host_output_dir
  local container_test_csv
  local container_template_csv
  local container_data_root
  local container_output_dir
  local container_command
  local build_command

  command -v docker >/dev/null 2>&1 || {
    log "docker is required for host mode"
    return 1
  }

  ensure_model_artifacts "$MODEL" || return 1
  stage_model_artifacts "$MODEL" || return 1

  DOCKER_ENV_ARGS=()
  DOCKER_VOLUME_ARGS=()
  DOCKER_MOUNT_SOURCES=()
  DOCKER_MOUNT_TARGETS=()
  DOCKER_MOUNT_KINDS=()

  resolve_host_path "$TEST_CSV" || return 1
  host_test_csv="$RESOLVED_HOST_PATH"
  to_container_path "$host_test_csv" file || return 1
  container_test_csv="$TO_CONTAINER_PATH_RESULT"
  resolve_host_path "$DATA_ROOT" || return 1
  host_data_root="$RESOLVED_HOST_PATH"
  to_container_path "$host_data_root" dir || return 1
  container_data_root="$TO_CONTAINER_PATH_RESULT"
  resolve_host_path "$OUTPUT_DIR" || return 1
  host_output_dir="$RESOLVED_HOST_PATH"
  to_container_path "$host_output_dir" dir || return 1
  container_output_dir="$TO_CONTAINER_PATH_RESULT"
  mkdir -p "$host_output_dir" || return 1

  if [[ "$TEMPLATE_CSV" != "$TEST_CSV" ]]; then
    resolve_host_path "$TEMPLATE_CSV" || return 1
    host_template_csv="$RESOLVED_HOST_PATH"
    to_container_path "$host_template_csv" file || return 1
    container_template_csv="$TO_CONTAINER_PATH_RESULT"
  else
    container_template_csv="$container_test_csv"
  fi

  if [[ "$ARTIFACTS_MANIFEST" != "$ARTIFACTS_MANIFEST_DEFAULT" ]]; then
    append_path_env_if_set ARTIFACTS_MANIFEST file || return 1
  fi
  if [[ "$TEMPLATE_CSV" != "$TEST_CSV" ]]; then
    DOCKER_ENV_ARGS+=(--env "TEMPLATE_CSV=${container_template_csv}")
  fi
  append_env_if_set SEARCH_DEVICE
  append_env_if_set TOP_K
  append_env_if_set W2V_BATCH_SIZE
  append_env_if_set W2V_PRECISION
  append_env_if_set W2V_NUM_WORKERS
  append_env_if_set W2V_PREFETCH_FACTOR
  append_env_if_set W2V_PIN_MEMORY
  append_env_if_set W2V_SEARCH_BATCH_SIZE
  append_env_if_set W2V_TOP_CACHE_K
  append_env_if_set W2V_CROP_SECONDS
  append_env_if_set W2V_N_CROPS
  append_env_if_set W2V_EDGE_TOP
  append_env_if_set W2V_RECIPROCAL_TOP
  append_env_if_set W2V_RANK_TOP
  append_env_if_set W2V_ITERATIONS
  append_env_if_set W2V_LABEL_MIN_SIZE
  append_env_if_set W2V_LABEL_MAX_SIZE
  append_env_if_set W2V_LABEL_MIN_CANDIDATES
  append_env_if_set W2V_SHARED_TOP
  append_env_if_set W2V_SHARED_MIN_COUNT
  append_env_if_set W2V_RECIPROCAL_BONUS
  append_env_if_set W2V_DENSITY_PENALTY
  append_env_if_set W2V_TENSORRT_OUTPUT_NAME
  append_env_if_set W2V_TENSORRT_PROFILE_INDEX
  append_env_if_set W2V_RUN_ID
  if [[ "$W2V_CHECKPOINT" != "$W2V_CHECKPOINT_DEFAULT" ]]; then
    append_path_env_if_set W2V_CHECKPOINT dir || return 1
  fi
  if [[ "$W2V_TENSORRT_ENGINE" != "$W2V_TENSORRT_ENGINE_DEFAULT" ]]; then
    append_path_env_if_set W2V_TENSORRT_ENGINE file || return 1
  fi
  if [[ "$CAMPP_CONFIG" != "$CAMPP_CONFIG_DEFAULT" ]]; then
    append_path_env_if_set CAMPP_CONFIG file || return 1
  fi
  append_env_if_set CAMPP_BATCH_SIZE
  append_env_if_set CAMPP_RUN_ID
  if [[ "$CAMPP_CHECKPOINT" != "$CAMPP_CHECKPOINT_DEFAULT" ]]; then
    append_path_env_if_set CAMPP_CHECKPOINT file || return 1
  fi
  append_env_if_set CAMPP_FRONTEND_WORKERS
  append_env_if_set CAMPP_FRONTEND_PREFETCH
  append_env_if_set CAMPP_SEARCH_BATCH_SIZE
  append_env_if_set CAMPP_TOP_CACHE_K
  append_env_if_set CAMPP_CLASS_BATCH_SIZE
  append_env_if_set CAMPP_CLASS_TOP_K
  append_env_if_set KRYPTONITE_CAMPP_CHECKPOINT_URL
  append_env_if_set KRYPTONITE_W2V_PLAN_URL
  append_env_if_set KRYPTONITE_W2V_TEACHER_BUNDLE_URL

  build_command=(
    docker build
    --build-arg "SUBMIT_MODEL=${MODEL}"
    -f "$DOCKERFILE_PATH"
    -t "$DOCKER_IMAGE"
  )
  if [[ -n "$DOCKER_BASE_IMAGE" ]]; then
    build_command+=(--build-arg "BASE_IMAGE=${DOCKER_BASE_IMAGE}")
  fi
  build_command+=("$PROJECT_ROOT")

  container_command=(
    docker run
    --rm
    --network none
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -w /workspace
  )
  if device_requires_gpu "$DEVICE" || device_requires_gpu "$SEARCH_DEVICE"; then
    container_command+=(--gpus all)
  fi
  if [[ ${#DOCKER_VOLUME_ARGS[@]} -gt 0 ]]; then
    container_command+=("${DOCKER_VOLUME_ARGS[@]}")
  fi
  if [[ ${#DOCKER_ENV_ARGS[@]} -gt 0 ]]; then
    container_command+=("${DOCKER_ENV_ARGS[@]}")
  fi
  container_command+=(
    "$DOCKER_IMAGE"
    ./run.sh
    --container-only
    --model "$MODEL"
    --test-csv "$container_test_csv"
    --data-root "$container_data_root"
    --output-dir "$container_output_dir"
    --device "$DEVICE"
  )
  if [[ -n "$BATCH_SIZE_OVERRIDE" ]]; then
    container_command+=(--batch-size "$BATCH_SIZE_OVERRIDE")
  fi
  container_command+=(--top-k "$TOP_K")
  if [[ "$OFFLINE" == "1" ]]; then
    container_command+=(--offline)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    container_command+=(--dry-run)
  fi

  log "host mode image=${DOCKER_IMAGE} model=${MODEL}"
  run_cmd "${build_command[@]}" || return 1
  run_cmd "${container_command[@]}" || return 1
  if [[ "$DRY_RUN" != "1" ]]; then
    copy_submission_from_output "$host_output_dir" || return 1
  fi
}

DRY_RUN=0
OFFLINE=0
CONTAINER_ONLY=0
BATCH_SIZE_OVERRIDE=""
MODEL="${MODEL:-w2v-trt}"
TEST_CSV="${TEST_CSV:-}"
TEMPLATE_CSV="${TEMPLATE_CSV:-}"
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-data/submission_entrypoint}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --test-csv)
      TEST_CSV="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE_OVERRIDE="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --offline)
      OFFLINE=1
      shift
      ;;
    --container-only)
      CONTAINER_ONLY=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      log "unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

case "$MODEL" in
  w2v-trt|campp-pt)
    ;;
  *)
    log "unsupported model: $MODEL"
    usage
    exit 2
    ;;
esac

if [[ -z "$TEST_CSV" ]]; then
  log "missing required --test-csv"
  usage
  exit 2
fi

if [[ -z "$DATA_ROOT" ]]; then
  DATA_ROOT="$(dirname -- "$TEST_CSV")"
fi

TOP_K="${TOP_K:-10}"
validate_top_k || exit 2

TEMPLATE_CSV="${TEMPLATE_CSV:-$TEST_CSV}"
SEARCH_DEVICE="${SEARCH_DEVICE:-$DEVICE}"
USE_UV_RUN="${USE_UV_RUN:-1}"
ARTIFACTS_MANIFEST_DEFAULT="${PROJECT_ROOT}/deployment/artifacts.toml"
ARTIFACTS_MANIFEST="${ARTIFACTS_MANIFEST:-$ARTIFACTS_MANIFEST_DEFAULT}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-${PROJECT_ROOT}/deployment/docker/submission.Dockerfile}"
DOCKER_IMAGE="${DOCKER_IMAGE:-kryptonite-submit:${MODEL}}"
DOCKER_BASE_IMAGE="${DOCKER_BASE_IMAGE:-}"
ARTIFACT_STAGE_DIR="${PROJECT_ROOT}/deployment/docker/.submit-artifacts/current"
KRYPTONITE_ARTIFACTS_ROOT="${KRYPTONITE_ARTIFACTS_ROOT:-}"

W2V_RUN_ID="${W2V_RUN_ID:-W2V1J_TRT_RELEASE}"
W2V_CHECKPOINT_DEFAULT="data/models/w2v_trt/teacher_peft"
W2V_TENSORRT_ENGINE_DEFAULT="data/models/w2v_trt/model.plan"
if [[ -n "$KRYPTONITE_ARTIFACTS_ROOT" ]]; then
  W2V_CHECKPOINT_DEFAULT="${KRYPTONITE_ARTIFACTS_ROOT}/data/models/w2v_trt/teacher_peft"
  W2V_TENSORRT_ENGINE_DEFAULT="${KRYPTONITE_ARTIFACTS_ROOT}/data/models/w2v_trt/model.plan"
fi
W2V_CHECKPOINT="${W2V_CHECKPOINT:-$W2V_CHECKPOINT_DEFAULT}"
W2V_TENSORRT_ENGINE="${W2V_TENSORRT_ENGINE:-$W2V_TENSORRT_ENGINE_DEFAULT}"
W2V_PRECISION="${W2V_PRECISION:-bf16}"
W2V_BATCH_SIZE="${W2V_BATCH_SIZE:-300}"
W2V_NUM_WORKERS="${W2V_NUM_WORKERS:-4}"
W2V_PREFETCH_FACTOR="${W2V_PREFETCH_FACTOR:-1}"
W2V_PIN_MEMORY="${W2V_PIN_MEMORY:-1}"
W2V_SEARCH_BATCH_SIZE="${W2V_SEARCH_BATCH_SIZE:-4096}"
W2V_TOP_CACHE_K="${W2V_TOP_CACHE_K:-300}"
W2V_CROP_SECONDS="${W2V_CROP_SECONDS:-6.0}"
W2V_N_CROPS="${W2V_N_CROPS:-3}"
W2V_EDGE_TOP="${W2V_EDGE_TOP:-10}"
W2V_RECIPROCAL_TOP="${W2V_RECIPROCAL_TOP:-20}"
W2V_RANK_TOP="${W2V_RANK_TOP:-100}"
W2V_ITERATIONS="${W2V_ITERATIONS:-5}"
W2V_LABEL_MIN_SIZE="${W2V_LABEL_MIN_SIZE:-5}"
W2V_LABEL_MAX_SIZE="${W2V_LABEL_MAX_SIZE:-120}"
W2V_LABEL_MIN_CANDIDATES="${W2V_LABEL_MIN_CANDIDATES:-3}"
W2V_SHARED_TOP="${W2V_SHARED_TOP:-20}"
W2V_SHARED_MIN_COUNT="${W2V_SHARED_MIN_COUNT:-0}"
W2V_RECIPROCAL_BONUS="${W2V_RECIPROCAL_BONUS:-0.03}"
W2V_DENSITY_PENALTY="${W2V_DENSITY_PENALTY:-0.02}"
W2V_TENSORRT_OUTPUT_NAME="${W2V_TENSORRT_OUTPUT_NAME:-embedding}"
W2V_TENSORRT_PROFILE_INDEX="${W2V_TENSORRT_PROFILE_INDEX:-0}"

CAMPP_CONFIG_DEFAULT="${PROJECT_ROOT}/code/campp/configs/campp_ms42_release.yaml"
CAMPP_CHECKPOINT_DEFAULT="data/models/campp/campp_encoder.pt"
if [[ -n "$KRYPTONITE_ARTIFACTS_ROOT" ]]; then
  CAMPP_CHECKPOINT_DEFAULT="${KRYPTONITE_ARTIFACTS_ROOT}/data/models/campp/campp_encoder.pt"
fi
CAMPP_CONFIG="${CAMPP_CONFIG:-$CAMPP_CONFIG_DEFAULT}"
CAMPP_RUN_ID="${CAMPP_RUN_ID:-CAMPP_MS42_RELEASE}"
CAMPP_CHECKPOINT="${CAMPP_CHECKPOINT:-$CAMPP_CHECKPOINT_DEFAULT}"
CAMPP_BATCH_SIZE="${CAMPP_BATCH_SIZE:-256}"
CAMPP_FRONTEND_WORKERS="${CAMPP_FRONTEND_WORKERS:-16}"
CAMPP_FRONTEND_PREFETCH="${CAMPP_FRONTEND_PREFETCH:-256}"
CAMPP_SEARCH_BATCH_SIZE="${CAMPP_SEARCH_BATCH_SIZE:-4096}"
CAMPP_TOP_CACHE_K="${CAMPP_TOP_CACHE_K:-200}"
CAMPP_CLASS_BATCH_SIZE="${CAMPP_CLASS_BATCH_SIZE:-4096}"
CAMPP_CLASS_TOP_K="${CAMPP_CLASS_TOP_K:-5}"

if [[ "$USE_UV_RUN" == "1" ]]; then
  if [[ "$CONTAINER_ONLY" != "1" && ! -f "/.dockerenv" ]]; then
    UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PROJECT_ROOT}/artifacts/uv-host-venv}"
    export UV_PROJECT_ENVIRONMENT
    ensure_host_uv || exit $?
    PYTHON_RUNNER=("$UV_BIN" run python)
  else
    PYTHON_RUNNER=(uv run python)
  fi
else
  PYTHON_RUNNER=(python)
fi

if [[ "$CONTAINER_ONLY" != "1" && ! -f "/.dockerenv" ]]; then
  run_host_docker
  exit $?
fi

case "$MODEL" in
  w2v-trt)
    run_w2v_trt
    ;;
  campp-pt)
    run_campp
    ;;
esac
