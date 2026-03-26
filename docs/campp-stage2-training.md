# CAM++ Stage-2: Heavy Multi-Condition Training

**Ticket:** KRYP-042
**Depends on:** KRYP-041 (stage-1 pretraining — `campp_stage1_encoder.pt`)

## Overview

Stage-2 fine-tunes the CAM++ encoder that was pretrained in stage-1 under significantly
harder acoustic conditions.  Three mechanisms are active simultaneously:

| Mechanism | Purpose |
|-----------|---------|
| Heavy augmentation (corruption bank, multi-severity) | Robustness to noise, reverb, distance, codec, silence |
| Hard-negative speaker mining | Better discrimination between confusable speakers |
| Short-utterance curriculum | Generalisation to short test segments |

## Components

### Code

| File | Role |
|------|------|
| `src/kryptonite/training/campp/stage2_config.py` | Config dataclasses + TOML loader |
| `src/kryptonite/training/campp/stage2_sampler.py` | `Stage2BatchSampler` (hard-negative-aware) |
| `src/kryptonite/training/campp/stage2_pipeline.py` | `run_campp_stage2()` training loop |
| `configs/training/campp-stage2.toml` | Production config |
| `scripts/run_campp_stage2_training.py` | CLI entry point |

### Config sections

#### `[stage2]`

```toml
[stage2]
stage1_checkpoint = "artifacts/baselines/campp-stage1/campp_stage1_encoder.pt"

[stage2.hard_negative]
enabled                    = true
mining_interval_epochs     = 2     # re-mine every N epochs
top_k_per_speaker          = 20    # top-K closest speakers → confusable neighbours
hard_negative_fraction     = 0.5   # fraction of batch slots filled with hard-neg speakers
max_train_rows_for_mining  = 5000  # subsample for speed (null = use all)

[stage2.utterance_curriculum]
enabled             = true
short_crop_seconds  = 1.5   # phase-0 crop
long_crop_seconds   = 4.0   # phase-2 crop
curriculum_epochs   = 7     # approximate phase length
```

## Augmentation Policy

Stage-2 skips the warmup/ramp curriculum entirely (both set to 0 epochs) and
starts immediately in *steady* mode with a heavy-biased intensity distribution:

| Intensity | Probability (start → end) |
|-----------|--------------------------|
| clean     | 5 % → 5 %               |
| light     | 15 % → 10 %             |
| medium    | 35 % → 30 %             |
| heavy     | 45 % → 55 %             |

`max_augmentations_per_sample = 3` — up to three corruption families per utterance
(vs. 2 in stage-1 steady mode).  All five families are active: noise, reverb, distance,
codec, silence.

Coverage of all corruption kinds is verified by inspecting `hard_negative_mining_log.jsonl`
and the training augmentation traces in `training_summary.json`.

## Hard-Negative Mining

Every `mining_interval_epochs` epochs the pipeline:

1. Runs inference over up to `max_train_rows_for_mining` training utterances (subsampled for speed).
2. Computes a **speaker centroid** embedding for each speaker by averaging normalised utterance embeddings.
3. Builds a cosine-similarity matrix across all speaker centroids.
4. For each speaker, averages the top-K highest similarities to other speakers.
5. Converts that mean similarity to a **sampling weight**: `max(1.0, 1.0 + mean_sim × 3.0)`.
6. Calls `Stage2BatchSampler.update_speaker_weights()` to expand the round-robin pool
   proportionally — confusable speakers appear more often in each batch.

Mining results are logged to `<output_root>/hard_negative_mining_log.jsonl`.

## Utterance Curriculum

Training is divided into three equal-length phases:

| Phase | Epochs (20-epoch run) | Crop size |
|-------|----------------------|-----------|
| 0     | 0–6                  | 1.5 s     |
| 1     | 7–13                 | 2.75 s    |
| 2     | 14–19                | 4.0 s     |

The dataloader is rebuilt once at each phase boundary.  Because the sampler uses
a single fixed crop value per batch, all features within a batch always have the
same shape (compatible with `collate_training_examples`).

## Running

### Dry run (local Mac, demo data)

```bash
uv run python scripts/run_campp_stage2_training.py \
    --config configs/training/campp-stage2.toml \
    --project-override 'training.max_epochs=1' \
    --project-override 'runtime.num_workers=0' \
    --project-override 'data.generate_demo_artifacts_if_missing=true' \
    --device cpu
```

### Production run (gpu-server, after approval)

```bash
# On gpu-server: /mnt/storage/Kryptonite-ML-Challenge-2026
uv run python scripts/run_campp_stage2_training.py \
    --config configs/training/campp-stage2.toml
```

## Output Contract

The output directory mirrors the stage-1 contract (same `SpeakerBaselineRunArtifacts`):

```
artifacts/baselines/campp-stage2/<run_id>/
├── campp_stage2_encoder.pt          ← stage-2 checkpoint (encoder + classifier)
├── training_summary.json
├── hard_negative_mining_log.jsonl   ← per-mining-pass statistics
├── dev_embeddings.npz
├── dev_embedding_metadata.{jsonl,parquet}
├── dev_trials.jsonl
├── dev_scores.jsonl
├── score_summary.json
├── verification_eval_report.{json,md}
├── roc_curve.png / det_curve.png / calibration_curve.png / histogram.png
├── slice_breakdown.{json,html}
├── reproducibility_snapshot.json
└── campp_stage2_report.md
```

## Readiness Criteria

- [ ] Robust dev EER improved relative to stage-1 checkpoint
- [ ] Coverage confirmed for all five corruption families (noise, reverb, distance, codec, silence)
- [ ] `hard_negative_mining_log.jsonl` present and non-empty
- [ ] `campp_stage2_encoder.pt` written and loadable
