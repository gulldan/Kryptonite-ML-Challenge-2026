# End-to-End Regression Suite

## Goal

Freeze one release-oriented regression contract for the serving path:

- raw audio -> embedding;
- raw audio -> verify score;
- raw audio -> benchmark latency payload;
- metrics and validation-error telemetry;
- local `Inferencer` parity against the HTTP API.

The suite lives in `tests/e2e/test_inference_regression_suite.py`.

## What It Covers

The regression suite generates the repository demo artifacts and then runs the
same top-level flow against the thin FastAPI adapter that production/demo entrypoints use.

Per backend selection (`torch`, `onnxruntime`, `tensorrt`) it verifies:

- `GET /health` and `GET /openapi.json` expose the expected release surface;
- `POST /embed` matches `Inferencer.embed_audio_paths(...)` exactly on the same audio;
- `POST /verify` matches `Inferencer.verify_audio_paths(...)` exactly on the same audio;
- a positive verification score stays above a negative cross-speaker score;
- `POST /benchmark` returns the expected latency-oriented schema;
- Prometheus counters record `embed`, `verify`, `benchmark`, and validation failures.

## Why The Backend Matrix Is Stubbed

The repository currently keeps one shared raw-audio embedding implementation:
`feature_statistics`.

The runtime `selected_backend` flag is already part of the release contract,
telemetry labels, and deploy metadata, but the CI environment does not provide
real TensorRT. The regression suite therefore stubs the backend-availability
probe while keeping the actual raw-audio frontend + scoring path real. This
lets CI catch API/telemetry/backend-label drift without pretending that GitHub
Actions is a production inference host.

## How To Run

```bash
uv run pytest tests/e2e/test_inference_regression_suite.py
```

For the full repository gate:

```bash
uv run pytest
```

## CI Integration

The existing `python-smoke.yml` workflow now runs this suite explicitly and is
wired for:

- pull requests;
- pushes to `develop`, `master`, and `codex/**`;
- nightly scheduled runs.

## Current Limits

- This suite validates the serving contract and backend selection metadata, not
  real ONNX Runtime vs TensorRT kernel-level numerical parity.
- Latency assertions are schema/telemetry checks only; they intentionally avoid
  brittle wall-clock thresholds on shared CI runners.
