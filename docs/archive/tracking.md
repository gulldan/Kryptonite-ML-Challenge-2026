# Tracking

## Current Backend

The repository ships with a local tracking backend by default. It uses only the standard library and writes run data into:

```text
artifacts/tracking/<run_id>/
```

Each run contains:

- `run.json`
- `params.json`
- `metrics.jsonl`
- `artifacts.json`
- `artifacts/` for copied files

## Why Local First

The Linear task mentions MLflow or W&B, but the project still does not have a real train/eval/export stack. Pulling those dependencies in now would add weight without a real call site.

The current design keeps the repository ready for later adapters while already solving:

- unique run ids
- structured params and metrics
- artifact registration
- automatic reproducibility snapshot logging

## Smoke Check

Run:

```bash
uv run python scripts/tracking_smoke.py --config configs/base.toml --kind train
```

The smoke script creates a run, logs params and metrics, writes a reproducibility snapshot, and copies that artifact into the tracking directory.
