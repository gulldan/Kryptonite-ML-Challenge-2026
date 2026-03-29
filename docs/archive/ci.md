# CI Smoke

## Current Scope

The current GitHub Actions workflow is intentionally narrow and only validates entrypoints that already exist in the repository.

Current checks:

- `uv sync --dev --group train --group tracking --frozen`
- `uv run ruff check .`
- `uv run ty check`
- `uv run pytest`
- `uv run pytest tests/e2e/test_inference_regression_suite.py`
- `uv run python scripts/training_env_smoke.py`
- `uv run python scripts/show_config.py --config configs/base.toml --override runtime.seed=123 --override backends.allow_tensorrt=true`
- `uv run python scripts/repro_check.py --config configs/base.toml --self-check`

## Why This Scope Is Limited

Some downstream entrypoints still depend on real dataset artifacts or export
bundles that are not committed to git, so the workflow currently focuses on:

- broken imports
- broken training-environment imports
- broken config loading
- broken reproducibility hooks
- broken unit tests
- release-surface regressions for `/health`, `/embed`, `/verify`, `/benchmark`, metrics, and
  local/API parity across the logical backend matrix

The Fbank frontend now exists and is covered by unit tests plus
`scripts/fbank_parity_report.py`, but the manifest-backed smoke CLI is still a
manual/dev-path check because the repository does not version a real manifest +
audio fixture for that command. When curated fixtures land, extend the existing
workflow rather than creating a parallel CI pipeline.

The end-to-end serving regression contract is documented in
`docs/end-to-end-regression-suite.md`.
