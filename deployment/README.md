# Deployment

Keep deployment manifests, packaging notes, and environment-specific launch material here.

This directory is intentionally separate from `apps/` and `src/` so serving adapters stay thin and reusable.

## Docker Images

The repository ships two container build targets under `deployment/docker/`:

- `train.Dockerfile` installs the locked `train` and `tracking` groups and runs `scripts/training_env_smoke.py --config configs/deployment/train.toml` during build.
- `infer.Dockerfile` installs the locked `infer` and `train` groups, validates `configs/deployment/infer.toml`, and starts the FastAPI adapter from `apps/api/main.py`.

The build-stage smoke is intentionally advisory with respect to datasets, manifests, demo subsets, and model bundles. It verifies the container packaging and locked runtime, but it does not claim deploy readiness by itself.

Build them from the repository root:

```bash
docker build -f deployment/docker/train.Dockerfile -t kryptonite-train .
docker build -f deployment/docker/infer.Dockerfile -t kryptonite-infer .
```

Smoke-check the resulting images:

```bash
docker run --rm kryptonite-train
docker run --rm kryptonite-infer python scripts/infer_smoke.py --config configs/deployment/infer.toml
```

Generate the canonical mini-demo artifact set before strict target-machine validation:

```bash
uv run python scripts/generate_demo_artifacts.py --config configs/deployment/infer.toml
```

## Docker Compose Demo

The repository root now ships a default [`compose.yml`](../compose.yml) so the local mini-demo can
be started with one command from the checkout root:

```bash
docker compose up --build
```

For `gpu-server`, use the GPU override:

```bash
docker compose -f compose.yml -f compose.gpu.yml up --build
```

The stack contains two services:

- `demo-artifacts`: one-shot bootstrap that generates the synthetic model bundle, demo subset,
  manifests, and offline enrollment cache inside named Docker volumes
- `demo`: the FastAPI runtime that serves both the JSON API and the `/demo` browser UI on port
  `8080`

The GPU override changes two things:

- both services request `gpus: all`
- the stack switches to [`configs/deployment/infer-gpu.toml`](../configs/deployment/infer-gpu.toml),
  which sets `runtime.device = "cuda"` and `backends.inference = "torch"`

Open the demo at:

```text
http://127.0.0.1:8080/demo
```

The API health endpoint stays on:

```text
http://127.0.0.1:8080/health
```

Useful lifecycle commands:

```bash
docker compose up --build -d
docker compose logs -f demo
docker compose down
docker compose down -v
```

`docker compose down -v` removes the generated model/cache/manifests volumes and forces a clean
bootstrap on the next run.

## Artifact Contract

Deployment profiles now pin the expected artifact roots inside the repository checkout:

- `configs/deployment/train.toml`
  - `paths.dataset_root = datasets`
  - `paths.manifests_root = artifacts/manifests`
- `configs/deployment/infer.toml`
  - `paths.manifests_root = artifacts/manifests`
  - `deployment.model_bundle_root = artifacts/model-bundle`
  - `deployment.demo_subset_root = artifacts/demo-subset`

The strict mini-demo contract is now concrete:

- `artifacts/manifests/demo_manifest.jsonl`
- `artifacts/model-bundle/model.onnx`
- `artifacts/model-bundle/metadata.json`
- `artifacts/demo-subset/demo_subset.json`
- non-empty `artifacts/demo-subset/enrollment/` and `artifacts/demo-subset/test/`
- non-empty `datasets/` tree with the source demo WAV files

These paths are checked in two modes:

- advisory: default local/bootstrap mode; missing target artifacts are reported but do not fail the run
- strict: target-machine mode; missing datasets/manifests/model bundle/demo subset fail the command or container startup

Enable strict preflight for the training image on `gpu-server`:

```bash
uv run python scripts/generate_demo_artifacts.py --config configs/deployment/infer.toml
docker run --rm \
  -v /mnt/storage/Kryptonite-ML-Challenge-2026/datasets:/app/datasets:ro \
  -v /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/manifests:/app/artifacts/manifests:ro \
  kryptonite-train \
  python scripts/training_env_smoke.py \
    --config configs/deployment/train.toml \
    --require-artifacts \
    --require-gpu
```

Enable strict preflight for the inference/demo image:

```bash
uv run python scripts/generate_demo_artifacts.py --config configs/deployment/infer.toml
docker run --rm \
  -v /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/manifests:/app/artifacts/manifests:ro \
  -v /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/model-bundle:/app/artifacts/model-bundle:ro \
  -v /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/demo-subset:/app/artifacts/demo-subset:ro \
  kryptonite-infer \
  python scripts/infer_smoke.py \
    --config configs/deployment/infer.toml \
    --require-artifacts
```

Run the inference container as a demo service with strict startup validation:

```bash
uv run python scripts/generate_demo_artifacts.py --config configs/deployment/infer.toml
docker run --rm -p 8080:8080 \
  -e KRYP_REQUIRE_DEPLOYMENT_ARTIFACTS=1 \
  -v /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/manifests:/app/artifacts/manifests:ro \
  -v /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/model-bundle:/app/artifacts/model-bundle:ro \
  -v /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/demo-subset:/app/artifacts/demo-subset:ro \
  kryptonite-infer
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/openapi.json
```

The health payload now includes an `artifacts` block so target-machine runs can prove whether startup happened in advisory or strict mode. The FastAPI service also publishes its request contract through the built-in OpenAPI schema.

## Scope Decisions

- Base image versions are pinned in the Dockerfiles via explicit `PYTHON_VERSION` and `UV_IMAGE` arguments.
- Dependency versions stay fixed through `uv.lock`; the build uses `uv sync --frozen` to prevent drift.
- The infer image now layers the shared torch/torchaudio/VAD frontend from the `train` group on top
  of the thin API/runtime dependencies, because the current `feature_statistics` inferencer and
  `generate_demo_artifacts.py` bootstrap both execute the same runtime audio frontend inside the
  container.
- The current GPU path is intentionally the torch-backed runtime frontend. On the validated
  `gpu-server` environment, `onnxruntime` is CPU-only, so the honest GPU mode today is
  `runtime.device = "cuda"` with `backends.inference = "torch"` instead of claiming a CUDA/TensorRT
  engine that is not implemented yet.
- Strict artifact validation is opt-in because the real deploy inputs do not exist on every development machine.
- The generated demo artifact set is intentionally tiny and synthetic so the target-machine deploy path can be validated without checking large raw datasets into git.
