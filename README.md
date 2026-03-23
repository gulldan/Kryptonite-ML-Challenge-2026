# Kryptonite-ML-Challenge-2026

Monorepo scaffold for the Dataton Kryptonite 2026 speaker-recognition project.

Repository-wide contributor rules live in [AGENTS.md](./AGENTS.md). Local rules for narrower areas live alongside the code they govern.

## Toolchain

- `uv` for environment management, dependencies, lockfiles, and command execution
- `ruff` for formatting and linting
- `ty` for static type checking

`uv sync` is expected to materialize the working environment in the repository-local `.venv`. Treat `.venv` as the canonical environment for this repo on both local machines and `gpu-server`; `uv` cache contents are not the working environment.

## Quick Start

```bash
uv sync --dev --group train --group tracking
uv run python scripts/training_env_smoke.py
uv run python scripts/dataset_inventory_report.py
uv run python scripts/dataset_leakage_report.py
uv run python scripts/validate_manifests.py
uv run ruff format .
uv run ruff check .
uvx ty check
uv run pytest
```

On `gpu-server`, layer TensorRT on top of the locked base environment:

```bash
uv sync --dev --group train --group tracking
uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com/simple tensorrt-cu12
uv run python scripts/training_env_smoke.py --require-gpu
```

The important detail is that TensorRT is layered into the same repo-local `.venv` created by `uv sync`, not into a separate global environment.

## Repository Layout

```text
.
├── apps/
│   └── api/               # thin service entrypoints only
├── artifacts/            # ignored generated outputs
├── assets/               # small curated fixtures and demo assets
├── configs/              # runtime, training, evaluation, deployment config
├── datasets/             # local datasets, ignored by git
├── deployment/           # deployment-oriented manifests and packaging notes
├── docs/                 # architecture, runbooks, model cards
├── notebooks/            # exploration only, never the source of truth
├── scripts/              # reproducible operational entrypoints
├── src/
│   └── kryptonite/       # core package
└── tests/
    ├── e2e/
    ├── integration/
    └── unit/
```

See [docs/repository-layout.md](./docs/repository-layout.md) for module boundaries and naming conventions.
See [docs/configuration.md](./docs/configuration.md) for config overrides and secret handling.
See [docs/reproducibility.md](./docs/reproducibility.md) for seed control and fingerprint checks.
See [docs/ci.md](./docs/ci.md) for the current CI smoke scope.
See [deployment/README.md](./deployment/README.md) for the train/infer container packaging flow.
See [docs/dataset-inventory.md](./docs/dataset-inventory.md) for the repository-level policy on approved, conditional, and blocked data resources.
See [docs/ffsvc2022-surrogate.md](./docs/ffsvc2022-surrogate.md) for the current server-only surrogate dataset strategy.
See [docs/gpu-server-data-sync.md](./docs/gpu-server-data-sync.md) for gpu-server dataset/manifests sync and readiness auditing.
See [docs/training-environment.md](./docs/training-environment.md) for environment groups and setup commands.
See [docs/tracking.md](./docs/tracking.md) for local run tracking and artifact logging.
See [docs/unified-metadata-schema.md](./docs/unified-metadata-schema.md) for the versioned manifest-row contract and validation workflow.
See [docs/audio-normalization.md](./docs/audio-normalization.md) for the 16 kHz mono normalization policy and reproducible CLI flow.

## Naming Conventions

- Python packages, modules, and scripts use `snake_case`.
- Markdown documents use `kebab-case`.
- Configs should be grouped by concern, for example `configs/training/` or `configs/eval/`.
- Local datasets live in `datasets/` and must stay out of git.
- Generated outputs belong in `artifacts/`, not in `src/`, `notebooks/`, or the repository root.

## Quality Gates

For touched Python code, the expected baseline is:

- `uv run ruff format .`
- `uv run ruff check .`
- `uvx ty check`
- `uv run pytest`
