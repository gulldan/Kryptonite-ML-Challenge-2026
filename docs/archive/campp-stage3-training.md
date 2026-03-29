# CAM++ Stage-3: Large-Margin Fine-Tuning

**Ticket:** KRYP-043  
**Depends on:** KRYP-042 (stage-2 run directory or checkpoint)

## Overview

Stage-3 takes the robustness-heavy stage-2 checkpoint and adapts it to a more
target-like operating regime. The main changes are:

| Mechanism | Purpose |
|-----------|---------|
| Longer fixed crops | Closer match to stable enrollment/test segments |
| Margin schedule | Tighten class separation without jumping straight to a large margin |
| Target-like augmentation mix | Retain some robustness while reducing train/infer mismatch |

## 4090 Runtime Defaults

The checked-in stage-3 config narrows the per-step footprint to match longer crops on a 24 GB RTX
4090-class GPU:

- `training.precision = "bf16"` on CUDA, with automatic `fp32` fallback on CPU smoke runs
- `optimizer_name = "adamw"` with cosine decay
- `training.batch_size = 24` and `gradient_accumulation_steps = 2` for an effective batch size of
  48 while preserving headroom for the longer 4.0 -> 6.0 second crop curriculum
- CAM++ keeps `memory_efficient = true`, so Dense-TDNN blocks still use gradient checkpointing

See [training optimization runtime](./training-optimization-runtime.md) for the shared runtime
contract and scheduler behavior.

## Components

| File | Role |
|------|------|
| `src/kryptonite/training/campp/finetune_common.py` | Shared warm-start fine-tuning helpers |
| `src/kryptonite/training/campp/stage3_config.py` | Config dataclasses + TOML loader |
| `src/kryptonite/training/campp/stage3_pipeline.py` | `run_campp_stage3()` training loop |
| `configs/training/campp-stage3.toml` | Production config |
| `configs/training/campp-stage3-smoke.toml` | Local/demo smoke config |
| `scripts/run_campp_stage3_training.py` | CLI entry point |

## Config Surface

```toml
[stage3]
stage2_checkpoint = "artifacts/baselines/campp-stage2/<run-id>"

[stage3.crop_curriculum]
enabled            = true
start_crop_seconds = 4.0
end_crop_seconds   = 6.0
curriculum_epochs  = 3

[stage3.margin_schedule]
enabled      = true
start_margin = 0.30
end_margin   = 0.45
ramp_epochs  = 6

[stage3.hard_negative]
enabled                = false
hard_negative_fraction = 0.25
```

## Target-Like Policy

The checked-in production config keeps augmentation enabled but shifts it away
from the heavy stage-2 distribution:

| Intensity | Probability (start -> end) |
|-----------|-----------------------------|
| clean     | 55% -> 45%                  |
| light     | 30% -> 35%                  |
| medium    | 10% -> 15%                  |
| heavy     | 5% -> 5%                    |

This preserves some corruption exposure while avoiding another robustness-heavy
phase during the final large-margin tuning pass.

## Schedule Artifact

Each run writes `stage3_schedule.json` alongside the checkpoint. It records:

- the warm-start checkpoint source
- the configured crop curriculum
- the configured margin schedule
- the per-epoch crop and margin actually used

This is the primary artifact for later ablations if the stage-3 run degrades
relative to stage-2.

## Running

### Dry run (local Mac, demo data)

1. Produce a compatible stage-2 smoke run.
2. Use the reported stage-2 run directory as the warm-start source:

```bash
uv run python scripts/run_campp_stage3_training.py \
    --config configs/training/campp-stage3-smoke.toml \
    --stage2-checkpoint artifacts/baselines/campp-stage2/<run-id> \
    --device cpu
```

### Production run (gpu-server)

```bash
# On gpu-server: /mnt/storage/Kryptonite-ML-Challenge-2026
uv run python scripts/run_campp_stage3_training.py \
    --config configs/training/campp-stage3.toml \
    --stage2-checkpoint /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/baselines/campp-stage2/<run-id>
```

## Output Contract

The output directory mirrors the stage-2 contract with one additional schedule
artifact:

```text
artifacts/baselines/campp-stage3/<run_id>/
├── campp_stage3_encoder.pt
├── training_summary.json
├── stage3_schedule.json
├── dev_embeddings.npz
├── dev_embedding_metadata.{jsonl,parquet}
├── dev_trials.jsonl
├── dev_scores.jsonl
├── score_summary.json
├── verification_eval_report.{json,md}
├── roc_curve.png / det_curve.png / calibration_curve.png / histogram.png
├── slice_breakdown.{json,html}
├── reproducibility_snapshot.json
└── campp_stage3_report.md
```

If optional hard-negative mining is enabled, the run also writes
`hard_negative_mining_log.jsonl`.
