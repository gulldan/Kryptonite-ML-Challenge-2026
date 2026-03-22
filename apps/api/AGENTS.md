# AGENTS.md

Local rules for `apps/api/`.

- Keep files here thin: wiring, dependency injection, transport adapters, and startup only.
- Put reusable business logic in `src/kryptonite/`, not in API entrypoints.
- Keep API configuration in `configs/` and serving logic in `src/kryptonite/serve/`.
- Put API-facing tests in `tests/integration/` or `tests/e2e/`.
