# Deployment

Keep deployment manifests, packaging notes, and environment-specific launch material here.

This directory is intentionally separate from `apps/` and `src/` so serving adapters stay thin and reusable.

## Docker Images

The repository ships two container build targets under `deployment/docker/`:

- `train.Dockerfile` installs the locked `train` and `tracking` groups and runs `scripts/training_env_smoke.py --config configs/deployment/train.toml` during build.
- `infer.Dockerfile` installs the locked `infer` and `train` groups, validates `configs/deployment/infer.toml`, and starts the FastAPI adapter from `apps/api/main.py`.
- `infer.gpu.Dockerfile` installs the same locked runtime on top of an NVIDIA CUDA runtime base so `torch` can execute the current inferencer on `gpu-server`.

The build-stage smoke is intentionally advisory with respect to datasets, manifests, demo subsets, and model bundles. It verifies the container packaging and locked runtime, but it does not claim deploy readiness by itself.

Build them from the repository root:

```bash
docker build -f deployment/docker/train.Dockerfile -t kryptonite-train .
docker build -f deployment/docker/infer.Dockerfile -t kryptonite-infer .
docker build -f deployment/docker/infer.gpu.Dockerfile -t kryptonite-infer-gpu .
```

Smoke-check the resulting images:

```bash
docker run --rm kryptonite-train
docker run --rm kryptonite-infer python scripts/infer_smoke.py --config configs/deployment/infer.toml
docker run --rm --gpus all kryptonite-infer-gpu python scripts/infer_smoke.py --config configs/deployment/infer-gpu.toml
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
DOCKER_BUILDKIT=0 docker build -f deployment/docker/infer.gpu.Dockerfile -t kryptonite-infer-gpu:local .
docker compose -f compose.yml -f compose.gpu.yml up --no-build
```

The stack contains two services:

- `demo-artifacts`: one-shot bootstrap that generates the synthetic model bundle, demo subset,
  manifests, and offline enrollment cache inside named Docker volumes
- `demo`: the FastAPI runtime that serves both the JSON API and the `/demo` browser UI on port
  `8080`

The GPU override changes two things:

- both services request `gpus: all`, build from `deployment/docker/infer.gpu.Dockerfile`, and run with `privileged: true`
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

## Triton Repository Packaging

`KVA-551` adds an optional Triton packaging path for the already-fixed encoder export boundary.

Important scope decision:

- Triton serves only the encoder graph: `encoder_input -> embedding`
- decode / resample / loudness normalization / VAD / chunking / Fbank stay outside Triton
- default repository generation uses the existing ONNX bundle
- TensorRT packaging is supported only when a real `model.plan` already exists

Build the repository from the current deployment config:

```bash
uv run python scripts/build_triton_model_repository.py --config configs/deployment/infer.toml
```

This writes:

- `artifacts/triton-model-repository/kryptonite_encoder/config.pbtxt`
- `artifacts/triton-model-repository/kryptonite_encoder/1/model.onnx`
- `artifacts/triton-model-repository/kryptonite_encoder/metadata.json`
- `artifacts/triton-model-repository/smoke/kryptonite_encoder_infer_request.json`

For TensorRT handoff, supply or materialize an engine and rebuild:

```bash
uv run python scripts/build_triton_model_repository.py \
  --config configs/deployment/infer.toml \
  --backend-mode tensorrt \
  --engine-path artifacts/model-bundle/model.plan
```

Launch Triton with the generated repository mounted as `/models`:

```bash
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:<compatible-tag>}"
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/artifacts/triton-model-repository:/models:ro" \
  "$TRITON_IMAGE" tritonserver --model-repository=/models
```

Smoke the running server with the generated sample request:

```bash
uv run python scripts/triton_infer_smoke.py \
  --repository-root artifacts/triton-model-repository \
  --model-name kryptonite_encoder \
  --server-url http://127.0.0.1:8000
```

## INT8 Feasibility Gate

`KVA-543` does not claim that INT8 is production-ready. Instead it adds one
reproducible decision report that says `go` only after the promoted FP16 path
is real and measured.

Build the report from the checked-in config:

```bash
uv run python scripts/build_int8_feasibility_report.py \
  --config configs/release/int8-feasibility.toml
```

The report checks:

- whether the current model bundle is still a structural stub;
- whether a reference FP16 TensorRT engine exists;
- whether the ONNX path has a saved parity report;
- whether matched FP16 and INT8 verification/stress reports exist;
- whether the saved calibration-set selection covers clean, duration-extreme,
  and corruption traffic without pulling silence-only samples into calibration.

The checked-in config intentionally returns `no_go` until those prerequisites
and measurements exist.

## Scope Decisions

- Base image versions are pinned in the Dockerfiles via explicit `PYTHON_VERSION` and `UV_IMAGE` arguments.
- Dependency versions stay fixed through `uv.lock`; the build uses `uv sync --frozen` to prevent drift.
- The infer image now layers the shared torch/torchaudio/VAD frontend from the `train` group on top
  of the thin API/runtime dependencies, because the current `feature_statistics` inferencer and
  `generate_demo_artifacts.py` bootstrap both execute the same runtime audio frontend inside the
  container.
- The GPU image is split into its own Dockerfile instead of overloading the default CPU image,
  because the validated `gpu-server` Docker runtime requires an NVIDIA CUDA base image for the
  current torch CUDA wheels to initialize correctly inside the container.
- On the currently validated `gpu-server`, CUDA compute inside Docker also requires
  `privileged: true`; `--gpus all` alone is enough for `nvidia-smi`, but not enough for
  `cudaGetDeviceCount()` / `torch.cuda.is_available()`. The root `compose.gpu.yml`
  captures that server-specific workaround explicitly.
- On the same `gpu-server`, `docker compose ... up --build` currently delegates image export to
  `buildx`, and the large CUDA image can stall there for a long time. The validated operational
  path is therefore `DOCKER_BUILDKIT=0 docker build ...` followed by
  `docker compose ... up --no-build`.
- The current GPU path is intentionally the torch-backed runtime frontend. On the validated
  `gpu-server` environment, `onnxruntime` is CPU-only, so the honest GPU mode today is
  `runtime.device = "cuda"` with `backends.inference = "torch"` instead of claiming a CUDA/TensorRT
  engine that is not implemented yet.
- The Triton repository path follows the same honesty rule: the implemented deployable artifact is
  the encoder-only boundary. It does not pretend that decode/VAD/Fbank or a full raw-audio
  TensorRT stack already exist.
- Strict artifact validation is opt-in because the real deploy inputs do not exist on every development machine.
- The generated demo artifact set is intentionally tiny and synthetic so the target-machine deploy path can be validated without checking large raw datasets into git.
