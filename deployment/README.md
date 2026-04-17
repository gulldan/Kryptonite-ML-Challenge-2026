# Deployment

Keep Dockerfiles and deployment-facing notes here. The actual runtime and HTTP logic lives in
`src/kryptonite/serve/`, while `apps/api/main.py` stays a thin CLI wrapper.

## Current Surface

The repository currently ships three maintained Docker targets:

- `train.Dockerfile` installs the locked `train` and `tracking` groups and runs `scripts/training_env_smoke.py --config configs/deployment/train.toml`.
- `infer.Dockerfile` installs the locked `infer` and `train` groups, runs `scripts/infer_smoke.py --config configs/deployment/infer.toml`, and starts `apps/api/main.py`.
- `infer.gpu.Dockerfile` uses the CUDA runtime base, runs `scripts/infer_smoke.py --config configs/deployment/infer-gpu.toml`, and starts the same API entrypoint with `runtime.device = "cuda"`.

The build-stage smoke is advisory by default: it validates packaging, imports, backend selection,
and config wiring without requiring a populated `datasets/` or `artifacts/` tree.

## Build And Smoke

Build from the repository root:

```bash
docker build -f deployment/docker/train.Dockerfile -t kryptonite-train .
docker build -f deployment/docker/infer.Dockerfile -t kryptonite-infer .
docker build -f deployment/docker/infer.gpu.Dockerfile -t kryptonite-infer-gpu .
```

Run the smoke commands:

```bash
docker run --rm kryptonite-train
docker run --rm kryptonite-infer python scripts/infer_smoke.py --config configs/deployment/infer.toml
docker run --rm --gpus all kryptonite-infer-gpu python scripts/infer_smoke.py --config configs/deployment/infer-gpu.toml
```

Run the API directly from the checkout:

```bash
uv run python apps/api/main.py --config configs/deployment/infer.toml --host <bind-host> --port 8080
uv run python apps/api/main.py --config configs/deployment/infer-gpu.toml --host <bind-host> --port 8080
```

## Strict Artifact Mode

Both smoke scripts and the API entrypoint support strict artifact checks:

```bash
uv run python scripts/infer_smoke.py --config configs/deployment/infer.toml --require-artifacts
uv run python apps/api/main.py --config configs/deployment/infer.toml --require-artifacts
```

Strict mode requires the deployment roots declared in the selected config, including:

- `artifacts/manifests/demo_manifest.jsonl`
- `artifacts/model-bundle/model.onnx`
- `artifacts/model-bundle/metadata.json`
- `artifacts/demo-subset/demo_subset.json`
- non-empty `artifacts/demo-subset/enrollment/`
- non-empty `artifacts/demo-subset/test/`
- non-empty `artifacts/enrollment-cache/`

## Compose

The root [`compose.yml`](../compose.yml) starts the advisory CPU service, and
[`compose.gpu.yml`](../compose.gpu.yml) overrides it for the CUDA image:

```bash
docker compose up --build
docker compose -f compose.yml -f compose.gpu.yml up --build
```

The `/demo` route works without a dedicated frontend bundle. If `apps/web/dist` is absent, the
service returns a small fallback HTML page and the JSON endpoints under `/demo/api/*` remain
available.

Strict artifact validation stays opt-in because the real deployment inputs do not exist on every
development machine.
