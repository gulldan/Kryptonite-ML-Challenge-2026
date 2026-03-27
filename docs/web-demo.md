# Web Demo

## Goal

Expose a lightweight browser UI for the current speaker-verification runtime without introducing a
separate frontend toolchain.

The demo is intentionally built on top of the existing FastAPI adapter and shared
`Inferencer`, so the browser flow exercises the same runtime/backend selection, enrollment store,
audio loading, and verification code paths as the JSON API.

## Entry Points

Start the service:

```bash
uv run python apps/api/main.py --config configs/deployment/infer.toml
```

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
