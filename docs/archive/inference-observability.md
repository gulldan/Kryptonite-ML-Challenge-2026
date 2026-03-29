# Inference Observability

## Goal

Expose lightweight service telemetry for the FastAPI infer/demo stack without adding a second
runtime dependency chain.

The current implementation lives in `src/kryptonite/serve/telemetry.py` and is wired through
`src/kryptonite/serve/http.py`.

## Config

Serving configs now expose a dedicated `[telemetry]` section:

```toml
[telemetry]
enabled = true
structured_logs = true
metrics_enabled = true
metrics_path = "/metrics"
```

`structured_logs` controls JSON log emission.

`metrics_enabled` controls the Prometheus-compatible endpoint.

`metrics_path` lets the adapter move the scrape endpoint without changing code.

## Structured Logs

The service emits JSON events with a shared runtime context:

- `service`
- `backend`
- `implementation`
- `model_version`

Current event types:

- `service_start`
- `http_request`
- `validation_error`
- `inference_operation`

`inference_operation` logs include:

- `operation`
- `stage`
- `audio_count`
- `total_audio_duration_seconds`
- `total_chunk_count`
- `latency_ms`

Demo endpoints also surface request-level decision/normalization flags in the same event.

## Metrics

`GET /metrics` exposes Prometheus text format with:

- `kryptonite_http_requests_total`
- `kryptonite_http_request_duration_seconds`
- `kryptonite_validation_errors_total`
- `kryptonite_inference_operations_total`
- `kryptonite_inference_operation_duration_seconds`
- `kryptonite_inference_audio_duration_seconds_total`
- `kryptonite_inference_input_audios_total`
- `kryptonite_inference_chunks_total`

Labels are intentionally low-cardinality:

- HTTP metrics use `method`, `path`, and `status`
- inference metrics use `operation`, `stage`, `backend`, and `model_version`

## Runtime Contract

`GET /health` now reports the active telemetry settings and model version, so deploy smoke checks
can confirm that logs and metrics are wired to the expected bundle before traffic hits the service.
