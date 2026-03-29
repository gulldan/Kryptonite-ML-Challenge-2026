ARG PYTHON_VERSION=3.12.11
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.9.6
ARG BUN_IMAGE=oven/bun:1.3.11

FROM ${UV_IMAGE} AS uv

FROM ${BUN_IMAGE} AS web-builder

WORKDIR /app/apps/web

COPY apps/web/package.json apps/web/bun.lock ./
RUN bun install --frozen-lockfile

COPY apps/web ./
RUN bun run build

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

COPY pyproject.toml uv.lock README.md ./
COPY apps ./apps
COPY configs ./configs
COPY deployment ./deployment
COPY scripts ./scripts
COPY src ./src
COPY --from=web-builder /app/apps/web/dist ./apps/web/dist

RUN uv sync --frozen --no-dev --group infer --group train
RUN .venv/bin/python scripts/infer_smoke.py --config configs/deployment/infer.toml

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}" \
    KRYP_REQUIRE_DEPLOYMENT_ARTIFACTS=0 \
    PYTHONUNBUFFERED=1

COPY --from=builder /app /app

EXPOSE 8080

CMD ["python", "apps/api/main.py", "--config", "configs/deployment/infer.toml", "--host", "0.0.0.0", "--port", "8080"]
