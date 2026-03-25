# CAM++ Baseline

## Goal

Provide the first repo-native speaker-verification baseline that trains from manifests, reuses the
shared audio normalization and Fbank frontend, exports utterance embeddings, and writes cosine
scores for held-out verification pairs.

## What This Adds

- an Apache-licensed CAM++-style encoder under `src/kryptonite/models/campp/`, adapted from the
  official `3D-Speaker` implementation:
  - [paper](https://arxiv.org/abs/2303.00332)
  - [reference repo](https://github.com/modelscope/3D-Speaker/tree/main/egs/3dspeaker/sv-cam%2B%2B)
- a manifest-backed baseline pipeline under `src/kryptonite/training/campp/`
- a reproducible CLI entrypoint:

```bash
uv run python scripts/run_campp_baseline.py --config configs/training/campp-baseline.toml
```

## Default Behavior

The checked-in config is intentionally smoke-oriented on the repository demo manifest because the
full competition corpora are not versioned in git.

By default the run will:

1. ensure `artifacts/manifests/demo_manifest.jsonl` exists
2. train CAM++ classification for 2 epochs on the demo speakers
3. export dev embeddings to `artifacts/baselines/campp/<run-id>/dev_embeddings.npz`
4. write metadata sidecars:
   - `dev_embedding_metadata.jsonl`
   - `dev_embedding_metadata.parquet`
5. generate simple verification trials from the dev manifest roles when no explicit trials file is
   configured
6. write cosine scores to `dev_scores.jsonl`
7. emit `training_summary.json`, `score_summary.json`, `verification_eval_report.{json,md}`,
   curve artifacts, `reproducibility_snapshot.json`, and a markdown report

## Config Surface

`configs/training/campp-baseline.toml` is split into:

- `base_config` and `project_overrides`: reuse the existing typed `ProjectConfig`
- `[data]`: train/dev manifests, output root, optional explicit trials manifest
- `[model]`: CAM++ width/depth knobs
- `[objective]`: cosine classifier and ArcMargin settings
- `[optimization]`: SGD + warmup/cosine scheduling knobs

For a real dataset run, the usual overrides are:

```bash
uv run python scripts/run_campp_baseline.py \
  --config configs/training/campp-baseline.toml \
  --project-override 'paths.project_root="/mnt/storage/Kryptonite-ML-Challenge-2026"' \
  --project-override 'runtime.num_workers=8'
```

and edit `[data]` in the config to point at the actual train/dev manifests on that machine.

For the strict-rules fallback path, use:

```bash
uv run python scripts/run_campp_baseline.py \
  --config configs/training/campp-ffsvc2022-restricted-rules.toml \
  --device cuda
```

That config is the clean-room fallback contract for this repository: it keeps the run in
`provenance.ruleset = "restricted-rules"` mode, initializes from scratch, and refuses any declared
teacher or pretrained resource.

## Output Contract

Each run writes a dedicated directory under `artifacts/baselines/campp/<run-id>/` with:

- `campp_encoder.pt`: PyTorch checkpoint with model state, classifier state, config snapshot, and
  speaker label map
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
- `verification_error_analysis.json`
- `verification_error_analysis.md`
- `reproducibility_snapshot.json`
- `campp_baseline_report.md`

Because the embedding export follows the atlas format, you can immediately visualize a run with:

```bash
uv run python scripts/build_embedding_atlas.py \
  --embeddings artifacts/baselines/campp/<run-id>/dev_embeddings.npz \
  --metadata artifacts/baselines/campp/<run-id>/dev_embedding_metadata.parquet \
  --output-dir artifacts/eval/embedding-atlas/campp-<run-id>
```

## Current Limits

- training currently supports `fp32` only; mixed precision is deferred to the optimizer/scheduler
- the default trial generation is intentionally simple and deterministic; richer scoring backends
  and held-out calibration protocols still come later
- the default config is a smoke baseline, not a claim of production-ready recipe quality

See [clean-room fallback baseline](./clean-room-fallback-baseline.md) for the explicit restricted
fallback scope, commands, and artifact location.
