# Repository Layout

## Core Boundaries

- `src/kryptonite/data/` contains dataset manifests, validation, loading, and preprocessing.
- `src/kryptonite/eda/` contains reproducible profiling, leakage checks, duplicate detection, and reusable exploratory helpers.
- `src/kryptonite/features/` contains feature extraction and audio transforms.
- `src/kryptonite/models/` contains model definitions and inference interfaces.
- `src/kryptonite/training/` contains recipes, losses, optimization, and experiment flow.
- `src/kryptonite/eval/` contains metrics, reports, and benchmarks.
- `src/kryptonite/serve/` contains serving adapters and backend wrappers.

## Supporting Areas

- `apps/api/` stays thin and delegates to `src/kryptonite/serve/`.
- `datasets/` is the local dataset root on working machines and on `gpu-server`; it is intentionally git-ignored.
- `scripts/` holds reproducible CLI entrypoints, never one-off root scripts.
- `notebooks/` is exploratory only and must call into reusable code in `src/` or `scripts/`.
- `artifacts/` is for ignored local outputs and should not become a source of truth.
- `assets/` is for small curated files that are intentionally versioned.

## Naming

- Python packages and modules: `snake_case`
- Markdown docs: `kebab-case`
- Config directories: grouped by concern
- Benchmarks and reports: store curated summaries in `docs/`, raw outputs in `artifacts/`
