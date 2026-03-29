# Web Demo

## Goal

Expose a dedicated browser UI for the current speaker-verification runtime with a separate frontend
toolchain in `apps/web`.

The demo still runs on top of the existing FastAPI adapter and shared `Inferencer`, so the browser
flow exercises the same runtime/backend selection, enrollment store, audio loading, and
verification code paths as the JSON API. The difference is that the UI itself now lives in a
proper `Bun` + `Vite` + `React` + `TypeScript 7` app and is built into `apps/web/dist`.

## Entry Points

Start the service:

```bash
bun --cwd apps/web install
bun --cwd apps/web run build
uv run python apps/api/main.py --config configs/deployment/infer.toml
```

For split local development:

```bash
uv run python apps/api/main.py --config configs/deployment/infer.toml
bun --cwd apps/web install
bun --cwd apps/web run dev
```

Open the Vite app during local UI work:

```text
http://127.0.0.1:5173
```

The Vite dev server proxies `/demo/api/*`, `/enrollments`, and `/health` back to the FastAPI
runtime on `127.0.0.1:8080`.

Or run the containerized local stack from the repository root:

```bash
docker compose up --build
```

On `gpu-server`, switch to the GPU override:

```bash
DOCKER_BUILDKIT=0 docker build -f deployment/docker/infer.gpu.Dockerfile -t kryptonite-infer-gpu:local .
docker compose -f compose.yml -f compose.gpu.yml up --no-build
```

That override builds `deployment/docker/infer.gpu.Dockerfile`, which keeps the
default CPU image untouched, swaps the demo stack onto a CUDA runtime base, and
applies the `privileged` workaround currently required on `gpu-server` for CUDA
compute inside Docker.

This is also the validated launch sequence on the current `gpu-server`: `buildx`
can hang while exporting the large CUDA image, so the reliable server path is an
explicit legacy `docker build` followed by compose startup without rebuild.

Open the demo:

```text
http://127.0.0.1:8080/demo
```

The root path `/` now serves the same demo page for convenience. Health checks stay on:

- `GET /health`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

`/metrics` exports Prometheus-compatible counters and histograms for HTTP requests, validation
errors, and audio inference operations. Structured JSON logs use the same backend/model-version
context and surface request latency, audio duration, chunk counts, and validation failures.

## Demo API

Browser actions use dedicated JSON endpoints under `/demo/api/*`:

- `GET /demo/api/state`
  returns runtime metadata, current enrollments, and the default threshold reference
- `POST /demo/api/compare`
  uploads two files, embeds them through the shared inferencer, and returns `score`, `decision`,
  `threshold`, `backend`, and `latency_ms`
- `POST /demo/api/enroll`
  uploads one or more enrollment files and persists the resulting enrollment through the runtime
  enrollment store
- `POST /demo/api/verify`
  uploads one probe file and verifies it against a selected enrollment

The browser sends audio as base64-encoded JSON payloads. That keeps the demo self-contained and
avoids an extra multipart dependency in the runtime environment.

## Frontend Toolchain

The product UI lives in `apps/web` with:

- `bun` for dependency install and script execution
- `Vite` for dev/build
- `React` for the UI surface
- `TypeScript 7` via `@typescript/native-preview` and `tsgo`
- `oxlint` and `oxfmt` as the primary JS/TS lint + format tools
- `Vitest` for frontend tests

The backend serves the built frontend bundle from `apps/web/dist`. If the bundle is missing,
`/demo` returns a small HTML page telling the operator to build the frontend first.

## GPU Mode

The current runtime implementation is still the shared `feature_statistics` inferencer, so GPU mode
means:

- the container gets NVIDIA GPU access through Docker Compose;
- the GPU override builds from `deployment/docker/infer.gpu.Dockerfile`;
- the GPU override also enables `privileged: true` on `gpu-server`, because that
  host's Docker security profile otherwise exposes `nvidia-smi` but blocks
  `cudaGetDeviceCount()` for the runtime process;
- `runtime.device` is set to `cuda`;
- the service reports `selected_backend = "torch"` because that is the runtime path that actually
  executes on CUDA today.

This is deliberate. The repository already carries export-boundary and model-bundle metadata for a
future ONNX/TensorRT path, but the implemented raw-audio runtime backend is still torch-based.

## Threshold Resolution

The demo resolves its default threshold in this order:

1. latest `verification_threshold_calibration.json` with a `demo` profile
2. latest `verification_eval_report.json` decision threshold
3. built-in fallback `0.995`

The active source is surfaced in `/demo/api/state` and in the UI so the operator can tell whether
the page is using a calibrated operating point or a fallback.

## Current Limits

- The browser flow is meant for local/manual demonstration, not for production auth or bulk upload.
- Upload transport is JSON base64 for simplicity; very large files are not the intended use case.
- The fallback threshold is only a conservative demo default. A real calibrated
  `verification_threshold_calibration.json` should take precedence when available.
