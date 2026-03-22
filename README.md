# Kryptonite-ML-Challenge-2026

Monorepo scaffold for the Dataton Kryptonite 2026 speaker-recognition project.

Repository-wide contributor rules live in [AGENTS.md](./AGENTS.md). Local rules for narrower areas live alongside the code they govern.

## Toolchain

- `uv` for environment management, dependencies, lockfiles, and command execution
- `ruff` for formatting and linting
- `ty` for static type checking

## Quick Start

```bash
uv sync --dev --group train --group tracking
uv run python scripts/training_env_smoke.py
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
See [docs/training-environment.md](./docs/training-environment.md) for environment groups and setup commands.
See [docs/tracking.md](./docs/tracking.md) for local run tracking and artifact logging.

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
