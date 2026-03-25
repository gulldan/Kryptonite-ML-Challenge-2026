# ERes2NetV2 Baseline

## Goal

Provide a second repo-native speaker-verification baseline that trains from manifests, reuses the
shared audio normalization and Fbank frontend, exports utterance embeddings, and writes cosine
scores for held-out verification pairs on the same contract as CAM++.

## What This Adds

- an Apache-licensed ERes2NetV2 encoder under `src/kryptonite/models/eres2netv2/`, adapted from
  the official `3D-Speaker` implementation:
  - [paper](https://arxiv.org/abs/2406.02167)
  - [reference repo](https://github.com/modelscope/3D-Speaker/tree/main/egs/voxceleb/sv-eres2netv2)
- a manifest-backed baseline pipeline under `src/kryptonite/training/eres2netv2/`
- a reproducible CLI entrypoint:

```bash
uv run python scripts/run_eres2netv2_baseline.py --config configs/training/eres2netv2-baseline.toml
```

## Default Behavior

The checked-in config is intentionally smoke-oriented on the repository demo manifest because the
full competition corpora are not versioned in git.

By default the run will:

1. ensure `artifacts/manifests/demo_manifest.jsonl` exists
2. train ERes2NetV2 classification for 2 epochs on the demo speakers
3. export dev embeddings to `artifacts/baselines/eres2netv2/<run-id>/dev_embeddings.npz`
4. write metadata sidecars:
   - `dev_embedding_metadata.jsonl`
   - `dev_embedding_metadata.parquet`
5. generate simple verification trials from the dev manifest roles when no explicit trials file is
   configured
6. write cosine scores to `dev_scores.jsonl`
7. emit `training_summary.json`, `score_summary.json`, `verification_eval_report.{json,md}`,
   curve artifacts, `reproducibility_snapshot.json`, and a markdown report

## Config Surface

`configs/training/eres2netv2-baseline.toml` is split into:

- `base_config` and `project_overrides`: reuse the existing typed `ProjectConfig`
- `[data]`: train/dev manifests, output root, optional explicit trials manifest
- `[model]`: ERes2NetV2 depth/width/pooling knobs
- `[objective]`: cosine classifier and ArcMargin settings
- `[optimization]`: SGD + warmup/cosine scheduling knobs

For a real dataset run, the usual overrides are:

```bash
uv run python scripts/run_eres2netv2_baseline.py \
  --config configs/training/eres2netv2-baseline.toml \
  --project-override 'paths.project_root="/mnt/storage/Kryptonite-ML-Challenge-2026"' \
  --project-override 'runtime.num_workers=8'
```

and edit `[data]` in the config to point at the actual train/dev manifests on that machine.

For the prepared `gpu-server` surrogate bundle, use:

```bash
uv run python scripts/run_eres2netv2_baseline.py \
  --config configs/training/eres2netv2-ffsvc2022-surrogate.toml \
  --device cuda
```

That config is pinned to the speaker-disjoint dev trials bundle, not the official dev trials file,
because the current EDA backlog explicitly blocks threshold tuning on `official_dev_trials.jsonl`
until the quarantined duplicate references are canonicalized.

## Output Contract

Each run writes a dedicated directory under `artifacts/baselines/eres2netv2/<run-id>/` with:

- `eres2netv2_encoder.pt`: PyTorch checkpoint with model state, classifier state, config snapshot,
  and speaker label map
- `dev_embeddings.npz`: `embeddings` and `point_ids`, compatible with the existing embedding-atlas
  tooling
- `dev_embedding_metadata.{jsonl,parquet}`: utterance metadata aligned with the embedding rows
- `dev_trials.jsonl`: generated trials when no external trial file is configured
- `dev_scores.jsonl`: cosine scores with labels
- `training_summary.json`
- `score_summary.json`
- `verification_eval_report.json`
- `verification_eval_report.md`
- `verification_slice_dashboard.html`
- `verification_roc_curve.jsonl`
- `verification_det_curve.jsonl`
- `verification_calibration_curve.jsonl`
- `verification_score_histogram.json`
- `verification_slice_breakdown.jsonl`
- `reproducibility_snapshot.json`
- `eres2netv2_baseline_report.md`

Because ERes2NetV2 follows the same artifact contract as CAM++, you can compare the two baselines
on the same manifests by diffing `score_summary.json` or plotting both embedding exports through the
existing atlas tooling.

To rebuild the generated verification report manually:

```bash
uv run python scripts/evaluate_verification_scores.py \
  --scores artifacts/baselines/eres2netv2/<run-id>/dev_scores.jsonl \
  --trials artifacts/baselines/eres2netv2/<run-id>/dev_trials.jsonl \
  --metadata artifacts/baselines/eres2netv2/<run-id>/dev_embedding_metadata.parquet
```

## Current Limits

- training currently supports `fp32` only; mixed precision is deferred to the optimizer/scheduler
- the default trial generation is intentionally simple and deterministic; richer scoring backends
  and held-out calibration protocols still come later
- the default config is a smoke baseline, not a claim of production-ready recipe quality
