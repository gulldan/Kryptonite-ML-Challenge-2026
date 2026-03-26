# Training Optimization Runtime

## Goal

`KRYP-044` adds one shared training runtime for optimizer choice, learning-rate scheduling, mixed
precision, and gradient accumulation across the manifest-backed speaker-verification pipelines.

## What Changed

The shared runtime lives in `src/kryptonite/training/optimization_runtime.py` and is now used by:

- CAM++ baseline / stage-1 style runs
- CAM++ stage-2 heavy multi-condition training
- CAM++ stage-3 large-margin fine-tuning
- ERes2NetV2 baseline runs

It centralizes four concerns that were previously open-coded or absent:

- `optimizer_name = "sgd" | "adamw"`
- `scheduler_name = "constant" | "cosine" | "plateau"`
- CUDA autocast for `training.precision = "fp16" | "bf16"` with `GradScaler` on fp16
- `gradient_accumulation_steps` so effective batch size can grow without restoring the old fp32
  memory footprint

## Precision Behavior

- `fp32` runs stay unchanged
- `bf16` enables CUDA autocast without a scaler
- `fp16` enables CUDA autocast with a gradient scaler
- non-CUDA environments intentionally fall back to `fp32` so local smoke runs still work on CPU
  and macOS laptops

## Current 4090 Defaults

The current checked-in CAM++ configs use:

| Config | Precision | Optimizer | Scheduler | Batch | Accumulation | Effective batch |
|--------|-----------|-----------|-----------|-------|--------------|-----------------|
| `campp-baseline.toml` | bf16 | AdamW | cosine | 4 | 1 | 4 |
| `campp-ffsvc2022-restricted-rules.toml` | bf16 | AdamW | cosine | 16 | 2 | 32 |
| `campp-stage2.toml` | bf16 | AdamW | cosine | 32 | 2 | 64 |
| `campp-stage3.toml` | bf16 | AdamW | cosine | 24 | 2 | 48 |

The two stage configs keep CAM++ `memory_efficient = true`, which means the Dense-TDNN stack still
uses gradient checkpointing internally. That remains the main protection against long-crop memory
spikes on a 24 GB card.

## Decisions

- CAM++ stages move to AdamW because the warm-started fine-tuning path benefits from a smaller,
  more stable learning-rate regime than the original smoke-oriented SGD defaults.
- Cosine decay remains the default scheduler because it already matched the repo's staged recipes
  and does not require extra validation targets mid-training.
- `ReduceLROnPlateau` is available for ablations, but it currently keys off mean training loss
  because the repo does not yet run per-epoch dev evaluation inside the training loop itself.

## Limitations

- AMP is only enabled on CUDA in this repository runtime.
- Plateau scheduling monitors mean training loss, not a held-out validation metric.
- Gradient accumulation increases effective batch size, but it does not reduce the cost of the
  forward pass itself; if a single micro-batch does not fit, batch size still must come down.
