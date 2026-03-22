# CI Smoke

## Current Scope

The current GitHub Actions workflow is intentionally narrow and only validates entrypoints that already exist in the repository.

Current checks:

- `uv sync --dev --frozen`
- `uv run ruff check .`
- `uv run ty check`
- `uv run pytest`
- `uv run python scripts/show_config.py --config configs/base.toml --override runtime.seed=123 --override backends.allow_tensorrt=true`
- `uv run python scripts/repro_check.py --config configs/base.toml --self-check`

## Why This Scope Is Limited

The Linear issue mentions smoke checks for audio pipeline, fbank extraction, scoring, ONNX export, and API startup. Those entrypoints do not exist in the repository yet, so the workflow currently focuses on:

- broken imports
- broken config loading
- broken reproducibility hooks
- broken unit tests

When the corresponding modules land, extend the workflow rather than creating a parallel CI pipeline.
