# Deployment

Keep deployment manifests, packaging notes, and environment-specific launch material here.

This directory is intentionally separate from `apps/` and `src/` so serving adapters stay thin and reusable.

## Docker Images

The repository ships two container build targets under `deployment/docker/`:

- `train.Dockerfile` installs the locked `train` and `tracking` groups and runs `scripts/training_env_smoke.py` during build.
- `infer.Dockerfile` installs the locked `infer` group, validates `configs/deployment/infer.toml`, and starts the thin HTTP adapter from `apps/api/main.py`.

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

Run the inference container as a demo service:

```bash
docker run --rm -p 8080:8080 kryptonite-infer
curl http://127.0.0.1:8080/healthz
```

## Scope Decisions

- Base image versions are pinned in the Dockerfiles via explicit `PYTHON_VERSION` and `UV_IMAGE` arguments.
- Dependency versions stay fixed through `uv.lock`; the build uses `uv sync --frozen` to prevent drift.
- The infer image intentionally uses an ONNX Runtime-only config to avoid shipping the heavier training stack in a demo/runtime container.
