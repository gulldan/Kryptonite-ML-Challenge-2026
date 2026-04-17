ARG CUDA_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu24.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.7

FROM ${UV_IMAGE} AS uv

FROM ${CUDA_IMAGE} AS runtime

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    KRYP_REQUIRE_DEPLOYMENT_ARTIFACTS=0 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libsndfile1 \
        python3 \
        python3-venv \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md ./
COPY apps ./apps
COPY configs ./configs
COPY deployment ./deployment
COPY scripts ./scripts
COPY src ./src

RUN uv sync --frozen --no-dev --group infer --group train --python /usr/bin/python3
RUN python scripts/infer_smoke.py --config configs/deployment/infer-gpu.toml

EXPOSE 8080

CMD ["python", "apps/api/main.py", "--config", "configs/deployment/infer-gpu.toml", "--port", "8080"]
