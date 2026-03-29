# API App

This directory is reserved for thin serving entrypoints and transport adapters.

Do not place business logic here. Import reusable behavior from `src/kryptonite/serve/` and adjacent core modules.

The FastAPI entrypoint lives in `apps/api/main.py` and delegates runtime validation and request handling to `src/kryptonite/serve/`.

Current adapter scope:

- `GET /health` as the primary runtime/artifact health endpoint
- compatibility aliases: `GET /healthz` and `GET /readyz`
- `GET /metrics` for Prometheus-compatible service telemetry
- `GET /demo` (and `/`) plus `/demo/api/*` for the dedicated browser demo frontend
- `POST /score/pairwise` and `POST /score/one-to-many` for embedding-based cosine scoring
- startup-time loading of the offline enrollment cache plus `POST /verify` and `GET /enrollments`
- `POST /enroll` and `POST /demo/api/enroll` as runtime-store-backed enrollment updates
- `POST /embed` and `POST /benchmark` for raw-audio runtime inference and latency smoke checks
- generated OpenAPI schema at `/openapi.json` and interactive docs at `/docs`

Read next:

- `docs/release-runbook.md` for launch, health checks and runtime diagnostics
- `docs/model-task-contract.md` for the canonical serving/task surface
- `docs/reference/audio-pipeline.md` for preprocessing and frontend assumptions
- `docs/archive/embedding-scoring.md` if you need the legacy request/response deep dive
