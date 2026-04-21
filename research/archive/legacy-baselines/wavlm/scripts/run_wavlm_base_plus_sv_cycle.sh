#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$ROOT_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_PATH="research/archive/legacy-baselines/wavlm/configs/wavlm_base_plus_sv.local.yaml"
WAIT_FOR_PID="${1:-}"
RUNTIME_DIR="data/wavlm_runs/wavlm_base_plus_sv/runtime"
RUNTIME_CONFIG="$RUNTIME_DIR/wavlm_base_plus_sv.runtime.yaml"
VALIDATION_RUN="pretrained_validation_wavlm_base_plus_sv"
SUBMISSION_RUN="submission_pretrained_wavlm_base_plus_sv_test_public"

mkdir -p "$RUNTIME_DIR" data/launcher_logs

if [[ -n "$WAIT_FOR_PID" ]]; then
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep 30
  done
fi

HF_HUB_DISABLE_XET=1 .venv/bin/python - <<'PY'
from pathlib import Path
import sys
import torch
import pandas as pd

sys.path.insert(0, str(Path("research/archive/legacy-baselines/wavlm").resolve()))
from common import load_config, write_resolved_config
from retrieval import extract_embeddings

config_path = Path("research/archive/legacy-baselines/wavlm/configs/wavlm_base_plus_sv.local.yaml")
runtime_config_path = Path("data/wavlm_runs/wavlm_base_plus_sv/runtime/wavlm_base_plus_sv.runtime.yaml")
config = load_config(config_path)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for WavLM batch probe.")

from common import load_pretrained_components

device = torch.device("cuda")
feature_extractor, model, _ = load_pretrained_components(config, device)
manifest = pd.read_csv(config["paths"]["experiment_root"] / "prepared" / "val_manifest.csv").head(64)
verification_manifest = pd.read_csv(config["paths"]["experiment_root"] / "prepared" / "val_manifest.csv").head(2048)
candidates = [16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256]
best_batch = int(config["evaluation"]["batch_size"])
best_peak_mb = 0.0

for candidate in candidates:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        extract_embeddings(
            manifest=manifest,
            feature_extractor=feature_extractor,
            model=model,
            data_root=config["paths"]["data_root"],
            sample_rate=int(config["model"]["sample_rate"]),
            mode=str(config["evaluation"]["primary_mode"]),
            chunk_sec=float(config["evaluation"]["chunk_sec"]),
            max_load_len_sec=float(config["evaluation"]["max_load_len_sec"]),
            batch_size=candidate,
            device=device,
            progress_every_rows=0,
            progress_label="wavlm_probe",
        )
        best_batch = candidate
        best_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        break

verified_batch = best_batch
for candidate in [value for value in candidates if value <= best_batch][::-1]:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        extract_embeddings(
            manifest=verification_manifest,
            feature_extractor=feature_extractor,
            model=model,
            data_root=config["paths"]["data_root"],
            sample_rate=int(config["model"]["sample_rate"]),
            mode=str(config["evaluation"]["primary_mode"]),
            chunk_sec=float(config["evaluation"]["chunk_sec"]),
            max_load_len_sec=float(config["evaluation"]["max_load_len_sec"]),
            batch_size=candidate,
            device=device,
            progress_every_rows=0,
            progress_label="wavlm_probe_verify",
        )
        verified_batch = candidate
        break
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        continue

config["evaluation"]["batch_size"] = int(verified_batch)
runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
write_resolved_config(config, runtime_config_path)
print(f"[wavlm_probe] selected_batch_size={best_batch} peak_memory_mb={best_peak_mb:.1f}")
print(f"[wavlm_probe] verified_batch_size={verified_batch}")
print(f"[wavlm_probe] runtime_config={runtime_config_path}")
PY

HF_HUB_DISABLE_XET=1 .venv/bin/python research/archive/legacy-baselines/wavlm/eval_wavlm.py \
  --config "$RUNTIME_CONFIG" \
  --split validation \
  --run-name "$VALIDATION_RUN"

HF_HUB_DISABLE_XET=1 .venv/bin/python research/archive/legacy-baselines/wavlm/build_submission.py \
  --config "$RUNTIME_CONFIG" \
  --csv "data/Для участников/test_public.csv" \
  --topk 10 \
  --run-name "$SUBMISSION_RUN"
