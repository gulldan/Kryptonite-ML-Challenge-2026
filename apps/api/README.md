# API App

This directory is reserved for thin serving entrypoints and transport adapters.

Do not place business logic here. Import reusable behavior from `src/kryptonite/serve/` and adjacent core modules.

The initial HTTP entrypoint lives in `apps/api/main.py` and delegates runtime validation and request handling to `src/kryptonite/serve/`.

Current adapter scope:

- `GET /healthz` and `GET /readyz` for runtime/artifact metadata
- `POST /score/pairwise` and `POST /score/one-to-many` for embedding-based cosine scoring
- `POST /enroll`, `POST /verify`, and `GET /enrollments` for the current in-memory verification flow

See `docs/embedding-scoring.md` for request/response examples and scope limits.
