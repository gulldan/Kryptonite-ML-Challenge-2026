ARG PYTHON_VERSION=3.12.11
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7

FROM ${UV_IMAGE} AS uv

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

RUN uv sync --frozen --no-dev --group train --group tracking
RUN .venv/bin/python scripts/training_env_smoke.py --config configs/deployment/train.toml

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}" \
    KRYP_REQUIRE_DEPLOYMENT_ARTIFACTS=0 \
    PYTHONUNBUFFERED=1

COPY --from=builder /app /app

CMD ["python", "scripts/training_env_smoke.py", "--config", "configs/deployment/train.toml"]
