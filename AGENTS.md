# AGENTS.md

This file defines the repository contract for all coding agents and human contributors.

These rules apply to the whole repository unless a deeper `AGENTS.md` adds narrower local guidance. Local guidance may refine workflow, but it must not override the foundational toolchain choices defined here.

## 1. Foundational Tooling

### Python is standardized on:

- `uv` for environment management, dependency management, lockfiles, and command execution
- `ruff` for formatting and linting
- `ty` for static type checking

### Frontend, if introduced, is standardized on:

- `bun` as the package manager and runtime
- a VoidZero-aligned stack for the app toolchain (lint, format)
- default expectation: `Vite` for dev/build and `Vitest` for tests

### Do not introduce by default:

- `pip`, `virtualenv`, `poetry`, `pip-tools`, `conda`, `requirements*.txt` as the source of truth
- `black`, `isort`, `flake8`, `pylint`, `mypy`, `pyright` as parallel default tools
- `npm`, `pnpm`, `yarn`, `webpack`, `jest`, `create-react-app`, `next.js` unless explicitly approved

If a new tool duplicates `uv`, `ruff`, `ty`, or the Bun/VoidZero frontend stack, the default answer is no.

## 2. Source of Truth

### Python project metadata

- Keep Python project configuration in `pyproject.toml`
- Commit `uv.lock`
- Materialize the managed Python environment in the repository-local `.venv` via `uv sync`; do not rely on an ad hoc global environment or on `uv`'s cache as the working environment
- Configure `ruff` in `pyproject.toml`
- Configure `ty` in `pyproject.toml` once the Python package layout is created

### Frontend project metadata

- Keep frontend package metadata in `apps/web/package.json` when frontend is introduced
- Commit `bun.lock`
- Keep frontend app config local to `apps/web/`

## 3. Planned Repository Layout

This repository should evolve as a monorepo with clear boundaries.

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── apps/
│   ├── api/               # thin Python service layer / serving entrypoints
│   └── web/               # optional Bun + Vite + Vitest frontend
├── src/
│   └── kryptonite/        # core Python package: domain, data, eda, training, inference
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── configs/               # runtime, training, evaluation, and deployment configs
├── scripts/               # operational and reproducible CLI scripts
├── notebooks/             # exploratory entrypoints, never the source of truth
├── docs/                  # architecture, runbooks, model cards
└── artifacts/             # generated local outputs, ignored unless explicitly curated
```

## 4. Architecture Rules

### Core design

- Keep business logic in `src/kryptonite/`
- Keep app entrypoints thin
- Keep orchestration in `apps/` and reusable logic in `src/`
- Keep config separate from code
- Prefer explicit module boundaries over script sprawl
- Keep source files below `600` lines of code; when a file approaches that limit, split it by responsibility into smaller modules or a package

### Data and ML boundaries

- `src/kryptonite/data/` handles manifests, validation, loading, and preprocessing
- `src/kryptonite/eda/` handles dataset profiling, leakage checks, auditing, and reusable exploratory transforms
- `src/kryptonite/features/` handles feature extraction and audio transforms
- `src/kryptonite/models/` handles model definitions and inference interfaces
- `src/kryptonite/training/` handles the generic baseline pipeline (`baseline_pipeline.py`), model-specific thin wrappers, optimization, and experiment flow. New model families delegate to `run_speaker_baseline()` instead of duplicating orchestration
- `src/kryptonite/eval/` handles metrics, reports, and benchmark logic
- `src/kryptonite/serve/` handles serving adapters and backend wrappers

### EDA policy

- EDA is a first-class workstream, not an incidental notebook side effect
- EDA must be kept separate from dataset ingestion and separate from training code
- Reusable EDA logic belongs in `src/kryptonite/eda/`
- Reproducible command entrypoints for EDA belong in `scripts/`
- EDA outputs should land in ignored artifact locations such as `artifacts/eda/` unless a report is intentionally curated into `docs/`
- Leakage checks, duplicate detection, and dataset profiling should be implemented as rerunnable code, not as one-off notebook cells

### Notebook policy

- `notebooks/` is for exploration only
- No production logic should live only in notebooks
- Notebooks may call into `src/kryptonite/eda/`, `src/kryptonite/data/`, and `scripts/`, but must not replace them
- If a notebook yields a reusable step, promote that logic into `src/` or `scripts/`

### Data policy

- Large raw datasets do not belong in git
- Commit manifests, schemas, metadata contracts, and small fixtures
- Local datasets live in `datasets/` under the repository root and must stay git-ignored
- The organizer-provided challenge dataset lives at `datasets/Для участников/` on the GPU machine and local working tree when materialized. Expected organizer files there include `train.csv`, `test_public.csv`, `baseline.onnx`, `train/`, and `test_public/`.
- Generated outputs go to ignored artifact locations unless they are intentionally versioned deliverables

### Experiment history policy

- Every model run, validation run, leaderboard submission candidate, reranking/postprocessing
  run, and hypothesis check must be recorded in `docs/challenge-experiment-history.md`
  during the same work session.
- Do not rely on files in `artifacts/` as the only record of an experiment. The history
  entry must summarize what was run, why it was run, the command/config or code path,
  key local metrics, public leaderboard score when available, artifact paths, and the
  resulting decision.
- Record rejected or failed hypotheses too when they affect the next decision. Mark them
  as rejected/diagnostic and explain why, instead of silently dropping them.
- Important experiment changes and outcomes must be recorded in `docs/challenge-experiment-history.md`, especially anything used for a leaderboard submission, presentation, or strategic decision.
- Each recorded experiment should include the date, short experiment name, key code/config changes, local validation design and metric, public leaderboard score when available, artifact paths, and the decision or lesson learned.
- Public leaderboard scores are external observations because public labels are hidden; record them explicitly instead of implying they are locally reproducible.
- If local validation and public leaderboard diverge, record that gap as an experimental finding, not as a footnote.
- For this hackathon, updating `docs/challenge-experiment-history.md` is mandatory, not
  optional bookkeeping. Any public leaderboard submission must update both the relevant
  experiment section and the top-level `Leaderboard History` table with the exact score,
  submission artifact path, validator status, and decision.
- When a hypothesis is run on `arm11`, record the remote run id, log path, checkpoint path,
  submission path if any, and whether the result was accepted, rejected, or kept for
  fusion.
- Training runs must be reproducible from the written record. For every training run,
  record the full config path, every CLI override, model family, initialization/provenance,
  train/dev manifest paths, dataset/split version, seed, batch size, precision, optimizer,
  scheduler, learning rate, epochs, crop/preprocessing policy, augmentation policy, loss,
  sampler, GPU/device assignment, container/environment, commit SHA or uncommitted diff
  summary, checkpoint path, log path, final train metrics, validation metrics, and the exact
  follow-up evaluation/submission commands.
- Training metrics must be persisted, not only printed. At minimum, keep per-epoch
  `train_loss`, `train_accuracy`, and `learning_rate` in `training_summary.json` and
  `artifacts/tracking/<run_id>/metrics.jsonl`. When a hypothesis depends on whether a
  model is undertrained or overtrained, also save enough intermediate validation evidence
  to judge that hypothesis: per-epoch or selected-epoch dev metrics, checkpoint paths, or
  an explicit note that only final-epoch validation was run.

## 5. Execution Environments

### GPU server workflow

- For GPU-bound work, connect over `ssh` to `gpu-server`
- On `gpu-server`, run repository commands from `/mnt/storage/Kryptonite-ML-Challenge-2026`
- On both local machines and `gpu-server`, the canonical working environment is `/mnt/storage/Kryptonite-ML-Challenge-2026/.venv` (or `./.venv` locally) created by `uv sync`
- Keep local datasets on the GPU machine in `/mnt/storage/Kryptonite-ML-Challenge-2026/datasets`
- Preferred workflow is: commit locally, push, then fetch/pull on `gpu-server` before running GPU jobs
- Do not try to version datasets from `datasets/` or make git depend on them being present

### `arm11` H800 workflow

`arm11` is the current high-memory GPU execution target for this challenge. Use it for
large backbone training, public inference, dense validation, graph/community inference,
and expensive hypothesis checks.

Remote paths:

- SSH host: `arm11`
- Host repository path: `/data/rnd/jupyter/kleshchenok/audio/embbedings`
- Container repository path: `/jupyter/kleshchenok/audio/embbedings`
- Docker container: `MK_RND`
- Organizer dataset path inside the container:
  `/jupyter/kleshchenok/audio/embbedings/datasets/Для участников`

Repository synchronization:

- Keep the local repository and the `arm11` repository synchronized before running remote
  jobs. The normal code-sync command is:

```bash
rsync -a --exclude='.venv/' --exclude='.cache/' --exclude='datasets/' --exclude='artifacts/' --exclude='.pytest_cache/' --exclude='.ruff_cache/' ./ arm11:/data/rnd/jupyter/kleshchenok/audio/embbedings/
```

- Never overwrite or delete the remote `datasets/` directory. It is mounted into the
  container and contains the organizer-provided data.
- Do not rely on `artifacts/` being included in the general repo sync. Copy only the
  specific manifests, checkpoints, submissions, summaries, or reports needed for the
  current experiment.
- After editing `AGENTS.md`, `docs/challenge-experiment-history.md`, configs, scripts,
  or `src/` code needed by a remote job, sync those files to `arm11` before launching.

Container execution:

- Run remote commands inside the container from the container repository path:

```bash
ssh arm11 'docker exec MK_RND bash -lc "cd /jupyter/kleshchenok/audio/embbedings && uv sync --dev --group train"'
```

- Launch long-running jobs detached with `PYTHONUNBUFFERED=1`, a stable run id, a log file
  under `artifacts/logs/`, and a `latest_*` pointer file. Logs must include enough progress
  to identify the current phase, epoch, batch, extraction percentage, or validation stage.
- Use `CUDA_VISIBLE_DEVICES=0` and `CUDA_VISIBLE_DEVICES=1` for independent hypotheses on
  the two H800 GPUs. The current training/inference pipeline is not assumed to be DDP-safe
  unless explicitly implemented and tested.
- Keep both GPUs useful when possible: run independent backbone, inference, validation, or
  reranking hypotheses in parallel, but avoid parallelism that only creates CPU/IO
  contention without reducing wall-clock time.
- Monitor long jobs with:

```bash
ssh arm11 'docker exec MK_RND bash -lc "cd /jupyter/kleshchenok/audio/embbedings && tail -n 80 artifacts/logs/<run>.log && nvidia-smi"'
```

Experiment documentation on `arm11`:

- Every remote launch must be reflected in `docs/challenge-experiment-history.md` in the
  same session: hypothesis, command/config, GPU assignment, remote log path, artifact paths,
  local metrics, public LB score when available, and decision.
- Public LB results from files generated on `arm11` must be added to the top-level
  `Leaderboard History` table immediately after the score is known.
- Sync `docs/challenge-experiment-history.md` back to `arm11` after updating it locally, so
  the remote workspace remains self-describing.
- For training jobs on `arm11`, the history entry must include enough parameters to rerun
  the job without guessing: config file, overrides, manifests, seed, model config, objective,
  optimizer/scheduler, precision, batch sizes, crop policy, VAD/trim policy, GPU id, run id,
  checkpoint path, and the exact C4-tail/public inference command used afterward.

### Branch integration workflow

- Complete the task on a task branch first, including relevant checks and Linear status updates
- After a task is fully complete, merge that task branch into `develop`
- Treat `develop` as the integration branch for completed work; do not leave completed task branches unmerged
- After merging completed work into `develop`, push `develop` and use that updated branch as the base for subsequent integration work on `gpu-server`

## 6. Default Commands

### Python workflow

- Install and sync: `uv sync --dev` (creates or updates the repo-local `.venv`)
- Add runtime dependency: `uv add <package>`
- Add dev dependency: `uv add --group dev <package>`
- Run a module or script: `uv run <command>`
- Format: `uv run ruff format .`
- Lint: `uv run ruff check .`
- Lint with fixes: `uv run ruff check --fix .`
- Type-check: `uvx ty check`

If `ty` is installed as a managed tool, `uv tool run ty check` is also acceptable. Do not rely on a global `ty` binary being present.
If a machine needs optional extras such as TensorRT, layer them into the same repo-local `.venv` after `uv sync`.

### Frontend workflow

When `apps/web` exists:

- Install deps: `bun install`
- Run dev server: `bun run dev`
- Run tests: `bun run test`
- Build: `bun run build`

Frontend scripts must stay inside `apps/web/package.json`; do not add a second package manager.

## 7. Quality Gates

Before considering work complete for touched Python code:

- formatting passes with `ruff format`
- lint passes with `ruff check`
- type checking passes with `uvx ty check` or `uv tool run ty check`
- tests relevant to the touched scope pass

Before considering work complete for touched frontend code:

- `bun` scripts pass for lint/test/build in the touched app
- the frontend remains isolated in `apps/web`
- no backend-only concerns leak into the frontend package

## 8. Agent Behavior Rules

All agents must follow these rules:

- Prefer editing existing structure over inventing a parallel structure
- Do not introduce alternative toolchains that conflict with this file
- Do not create one-off scripts in the repo root
- Do not store important logic only in notebooks
- Do not collapse EDA into training code or hide it inside ad hoc notebooks
- Do not add global state or hidden side effects when a typed interface will do
- Do not mix training logic, serving logic, and exploratory logic in the same module
- Do not put frontend code into Python app folders or Python code into frontend app folders

## 9. Future Local AGENTS Files

When these directories appear, add local `AGENTS.md` files:

- `apps/api/AGENTS.md`
- `apps/web/AGENTS.md`
- `src/kryptonite/AGENTS.md`
- `src/kryptonite/eda/AGENTS.md`
- `notebooks/AGENTS.md`

Those files should narrow local conventions, file ownership, and workflow. They must inherit the toolchain rules from this root file.
