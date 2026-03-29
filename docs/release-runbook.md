# Release Runbook

## Scope

Этот документ теперь совмещает runtime/demo guide и operational runbook.
Его задача — быстро ответить на вопросы: как поднять стек, что проверить, куда смотреть при проблемах,
и как не перепутать demo-grade runtime с финальным learned-model release.

## Fast path

Локально поднять demo/API:

```bash
uv sync --dev --group train --group tracking
docker compose up --build
```

Открыть:

- `http://127.0.0.1:8080/demo`
- `http://127.0.0.1:8080/health`
- `http://127.0.0.1:8080/metrics`

## Active Runtime Map

Primary configs:

- CPU: `configs/deployment/infer.toml`
- GPU server: `configs/deployment/infer-gpu.toml`
- training/runtime smoke: `configs/deployment/train.toml`

Primary endpoints:

- `GET /health`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /demo/api/state`
- `POST /demo/api/compare`
- `POST /demo/api/enroll`
- `POST /demo/api/verify`
- `POST /embed`
- `POST /enroll`
- `POST /verify`
- `POST /score/pairwise`
- `POST /score/one-to-many`

Critical interpretation rule:

- `requested_backend` = what config asked for
- `selected_backend` = what runtime actually resolved
- `inferencer.implementation` = what is really producing embeddings

For the checked-in repository state, `feature_statistics` remains the expected runtime implementation until a real learned export path is promoted.

## Preflight Checklist

1. Sync environment:

```bash
uv sync --dev --group train --group tracking
```

2. Ensure demo/runtime artifacts exist or rebuild them:

```bash
uv run python scripts/generate_demo_artifacts.py --config configs/deployment/infer.toml
```

3. If model metadata changed, rebuild enrollment cache:

```bash
uv run python scripts/build_enrollment_cache.py \
  --config configs/deployment/infer.toml \
  --manifest artifacts/manifests/demo_manifest.jsonl \
  --output-dir artifacts/enrollment-cache \
  --model-metadata artifacts/model-bundle/metadata.json
```

4. Before strict runs, verify that the active `verification_threshold_calibration.json` is the intended one.

## Local Validation Runbook

```bash
uv run python scripts/infer_smoke.py --config configs/deployment/infer.toml --require-artifacts
```

If you need deeper release validation, the detailed benchmark/export notes are in the archive:

- [archive/end-to-end-regression-suite.md](./archive/end-to-end-regression-suite.md)
- [archive/inference-stress-test.md](./archive/inference-stress-test.md)
- [archive/final-benchmark-pack.md](./archive/final-benchmark-pack.md)
- [archive/submission-release-bundle.md](./archive/submission-release-bundle.md)

## GPU Server Launch Runbook

Run on `gpu-server` from `/mnt/storage/Kryptonite-ML-Challenge-2026`.

```bash
uv sync --dev --group train --group tracking
uv run python scripts/training_env_smoke.py --config configs/deployment/train.toml --require-gpu
uv run python scripts/infer_smoke.py --config configs/deployment/infer-gpu.toml --require-artifacts
DOCKER_BUILDKIT=0 docker build -f deployment/docker/infer.gpu.Dockerfile -t kryptonite-infer-gpu:local .
docker compose -f compose.yml -f compose.gpu.yml up --no-build
```

## Monitoring And Incident Triage

Primary checks:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/demo/api/state
curl -s http://127.0.0.1:8080/metrics | head
```

Interpretation:

- if `status != "ok"`, inspect runtime/artifact preflight first
- always read `requested_backend`, `selected_backend`, and `inferencer.implementation` together
- `threshold.origin_path` should point at the intended `verification_threshold_calibration.json`
- enrollment cache mismatches usually mean model bundle changed without rebuilding the cache

## Rollback Procedure

1. Stop the running stack.
2. Restore a matched snapshot of model bundle, enrollment cache and threshold artifact.
3. Revalidate with strict smoke.
4. Restart the service.
5. Re-check `/health` and `/demo/api/state`.

Rollback is complete only after model version, enrollment-cache compatibility and threshold origin all match the intended previous release.

## Known Limits

- top-level runbook is now intentionally short; export/deployment deep-dive moved to archive
- a healthy runtime still proves integration shape more than final learned-model quality
- threshold activation remains artifact-sensitive and must be controlled explicitly
- the checked-in stack is suitable for demo, smoke and operational validation, not for overclaiming production readiness
