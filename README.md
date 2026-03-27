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

## Local Demo via Docker Compose

From the repository root, the quickest end-to-end local demo path is:

```bash
docker compose up --build
```

This boots a one-shot artifact generator plus the FastAPI runtime that serves both the API and the
browser demo. Open:

- `http://127.0.0.1:8080/demo`
- `http://127.0.0.1:8080/health`

To stop the stack and remove the generated demo volumes:

```bash
docker compose down -v
```

On `gpu-server`, use the GPU override so the current `feature_statistics` inferencer runs on CUDA:

```bash
docker compose -f compose.yml -f compose.gpu.yml up --build
```

The GPU override swaps the image build to
`deployment/docker/infer.gpu.Dockerfile`, which uses an NVIDIA CUDA runtime
base and points the service at `configs/deployment/infer-gpu.toml`.

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
See [docs/web-demo.md](./docs/web-demo.md) for the browser demo flow and compose-based launch path.
See [docs/dataset-inventory.md](./docs/dataset-inventory.md) for the repository-level policy on approved, conditional, and blocked data resources.
See [docs/ffsvc2022-surrogate.md](./docs/ffsvc2022-surrogate.md) for the current server-only surrogate dataset strategy.
See [docs/gpu-server-data-sync.md](./docs/gpu-server-data-sync.md) for gpu-server dataset/manifests sync and readiness auditing.
See [docs/training-environment.md](./docs/training-environment.md) for environment groups and setup commands.
See [docs/tracking.md](./docs/tracking.md) for local run tracking and artifact logging.
See [docs/unified-metadata-schema.md](./docs/unified-metadata-schema.md) for the versioned manifest-row contract and validation workflow.
See [docs/audio-quality-checks.md](./docs/audio-quality-checks.md) for the manifest-driven audio quality audit flow and flagged-row artifacts.
See [docs/audio-normalization.md](./docs/audio-normalization.md) for the 16 kHz mono normalization policy and reproducible CLI flow.
See [docs/audio-loader.md](./docs/audio-loader.md) for the shared WAV/FLAC/MP3 loading contract used by preprocessing and feature work.
See [docs/audio-vad-trimming.md](./docs/audio-vad-trimming.md) for the optional `none/light/aggressive` trimming modes and the reproducible dev comparison report.
See [docs/audio-fbank-extraction.md](./docs/audio-fbank-extraction.md) for the shared 80-dim log-Mel/Fbank frontend and the offline/online parity smoke workflow.
See [docs/audio-feature-cache.md](./docs/audio-feature-cache.md) for the feature-cache policy, invalidation rules, and CPU/GPU benchmark workflow.
See [docs/audio-chunking-policy.md](./docs/audio-chunking-policy.md) for the unified train/eval/demo utterance chunking contract and pooling rules.
See [docs/audio-noise-bank.md](./docs/audio-noise-bank.md) for additive-noise bank assembly, severity buckets, and reproducible manifest/report generation.
See [docs/audio-rir-bank.md](./docs/audio-rir-bank.md) for room-impulse-response bank assembly, RT60/DRR sanity checks, and reusable room-simulation config generation.
See [docs/audio-codec-simulation.md](./docs/audio-codec-simulation.md) for deterministic FFmpeg-based codec/channel preset generation and preview reporting.
See [docs/audio-far-field-simulation.md](./docs/audio-far-field-simulation.md) for deterministic near/mid/far distance simulation presets, kernel controls, and preview reporting.
See [docs/audio-augmentation-scheduler.md](./docs/audio-augmentation-scheduler.md) for the clean/light/medium/heavy curriculum policy over the assembled corruption banks.
See [docs/audio-embedding-atlas.md](./docs/audio-embedding-atlas.md) for interactive 2D projection of precomputed embeddings plus the manifest-backed baseline export path for immediate dataset inspection.
See [docs/embedding-scoring.md](./docs/embedding-scoring.md) for the shared L2-normalization and cosine scoring contract used by offline verification and the thin HTTP scoring API.
See [docs/export-boundary.md](./docs/export-boundary.md) for the machine-readable `encoder_input -> embedding` contract that keeps decode/VAD/Fbank outside exported ONNX/TensorRT graphs.
See [docs/campp-baseline.md](./docs/campp-baseline.md) for the first repo-native CAM++ baseline path from manifests to checkpoints, embeddings, and cosine scores.
See [docs/campp-stage2-training.md](./docs/campp-stage2-training.md) and [docs/campp-stage3-training.md](./docs/campp-stage3-training.md) for the staged CAM++ fine-tuning flow on top of the baseline checkpoint.
See [docs/campp-hyperparameter-sweep-shortlist.md](./docs/campp-hyperparameter-sweep-shortlist.md) and [docs/campp-model-selection.md](./docs/campp-model-selection.md) for the bounded stage-3 shortlist plus the final-candidate selection and checkpoint-averaging flow.
See [docs/clean-room-fallback-baseline.md](./docs/clean-room-fallback-baseline.md) for the restricted-rules fallback baseline that stays fully train-from-scratch.
See [docs/evaluation-package.md](./docs/evaluation-package.md) for the shared offline verification report contract, ROC/DET artifacts, calibration bins, and per-slice breakdown flow.
See [docs/threshold-calibration.md](./docs/threshold-calibration.md) for the named `balanced/min_dcf/demo/production` threshold profiles and the slice-aware calibration bundle workflow.
See [docs/eres2netv2-baseline.md](./docs/eres2netv2-baseline.md) for the ERes2NetV2 baseline path with the same artifact contract for side-by-side comparison against CAM++.

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
