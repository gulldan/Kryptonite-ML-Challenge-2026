# Production Dataloader

`KRYP-040` replaces the earlier `DataLoader(..., shuffle=True)` baseline path
with a training-side production dataloader that is explicitly shaped for
speaker-recognition recipes.

## What It Does

The loader now provides four guarantees:

- balanced speaker sampling at the batch-planning layer instead of pure row-level shuffle
- variable train crop length across batches while keeping one rectangular crop size inside each batch
- clean/corrupted mixing through the existing augmentation scheduler and corruption-bank stack
- deterministic resume via sampler `state_dict()/load_state_dict()`

## Batch Strategy

The sampler walks speakers in a deterministic round-robin order and draws one
utterance per selected speaker before reusing speakers when the batch is larger
than the speaker pool.

This keeps speaker exposure more even than global row shuffle and avoids adding
padding masks to the current CAM++ / ERes2NetV2 recipes:

- crop duration is sampled once per batch
- every sample in that batch uses the same requested crop duration
- different batches can still cover different train crop lengths within the configured range

## Augmentation Path

The production loader is the integration point for the existing scheduler and
bank work:

- additive noise uses the assembled noise-bank manifest
- reverb uses the RIR-bank room configs plus sibling RIR manifest
- distance uses the far-field plan presets
- codec/channel uses the codec-bank plan presets
- silence uses the existing `[silence_augmentation]` envelope

When a family is unavailable, it simply drops out of the runtime catalog and
the scheduler samples only from the remaining families.

## Resume Contract

The resumable state is deliberately small:

- `epoch`
- `next_batch_index`

The batch plan itself is reconstructed deterministically from seed, epoch, and
manifest ordering, so restoring sampler state is enough to continue from the
same next batch.

Current limitation:

- the training pipelines do not yet persist sampler state into periodic
  checkpoints, so deterministic resumption is available in code and smoke
  tooling now, while automatic crash recovery wiring is a follow-up concern

## Smoke Check

Use the dedicated script for quick inspection on local or GPU machines:

```bash
uv run python scripts/production_dataloader_smoke.py \
  --config configs/deployment/train.toml \
  --train-manifest artifacts/manifests/ffsvc2022-surrogate/train_manifest.jsonl \
  --batches 8
```

The smoke report summarizes:

- batch tensor shapes
- clean vs corrupted counts
- sampled crop durations
- observed speaker coverage
- current sampler resume state
