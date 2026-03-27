# Release Runbook

## Scope

This runbook covers the current Kryptonite inference/demo stack:

- environment bring-up;
- artifact preflight;
- release validation;
- candidate promotion;
- monitoring and incident triage;
- rollback.

It is intentionally honest about the current implementation boundary: the
runtime contract is production-like, but the active raw-audio embedding path is
still `feature_statistics` unless a real learned bundle is explicitly promoted.

## Active Runtime Map

Primary configs:

- CPU: `configs/deployment/infer.toml`
- GPU server: `configs/deployment/infer-gpu.toml`
- training/runtime environment smoke: `configs/deployment/train.toml`

Active artifact roots:

- manifests: `artifacts/manifests`
- model bundle: `artifacts/model-bundle`
- demo subset: `artifacts/demo-subset`
- enrollment cache: `artifacts/enrollment-cache`

Primary operational endpoints:

- `GET /health`
- `GET /openapi.json`
- `GET /metrics`
- `GET /demo/api/state`

Important inspection rule:

- `selected_backend` tells you what the deployment config requested;
- `inferencer.implementation` tells you what is actually generating embeddings.

For the current repository state, that second field is the authoritative one.

## Preflight Checklist

Before any rollout or release validation:

1. Sync the repo-local environment:

```bash
uv sync --dev --group train --group tracking
```

2. Rebuild the smoke/demo artifacts unless a real frozen candidate bundle is
   being promoted:

```bash
uv run python scripts/generate_demo_artifacts.py --config configs/deployment/infer.toml
```

3. If the model-bundle metadata changed, rebuild the enrollment cache against
   the active metadata:

```bash
uv run python scripts/build_enrollment_cache.py \
  --config configs/deployment/infer.toml \
  --manifest artifacts/manifests/demo_manifest.jsonl \
  --output-dir artifacts/enrollment-cache \
  --model-metadata artifacts/model-bundle/metadata.json
```

4. For target-machine runs, enable strict artifact validation with
   `--require-artifacts` or `KRYP_REQUIRE_DEPLOYMENT_ARTIFACTS=1`.

## Local Validation Runbook

Use this before a merge or before promoting a candidate on `gpu-server`.

1. Validate the runtime and artifact contract:

```bash
uv run python scripts/infer_smoke.py \
  --config configs/deployment/infer.toml \
  --require-artifacts
```

2. Run the release-oriented regression suites:

```bash
uv run pytest \
  tests/e2e/test_inference_regression_suite.py \
  tests/e2e/test_inference_stress_report.py
```

3. Generate the operational stress report:

```bash
uv run python scripts/inference_stress_report.py \
  --config configs/deployment/infer.toml \
  --output-root artifacts/release/current/stress
```

4. If a real candidate is under review, build the frozen comparison pack from a
   release-specific TOML config:

```bash
uv run python scripts/build_final_benchmark_pack.py \
  --config configs/eval/final-benchmark-pack.toml
```

Do not use the checked-in example TOML as-is for a real release. Replace the
placeholder candidate paths first.

## GPU Server Launch Runbook

Use this path on `gpu-server` from
`/mnt/storage/Kryptonite-ML-Challenge-2026`.

1. Sync the environment and, when required, layer the NVIDIA wheel into the
   same repo-local `.venv`:

```bash
uv sync --dev --group train --group tracking
uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com/simple tensorrt-cu12
uv run python scripts/training_env_smoke.py --config configs/deployment/train.toml --require-gpu
```

2. Validate the inference runtime in strict mode:

```bash
uv run python scripts/infer_smoke.py \
  --config configs/deployment/infer-gpu.toml \
  --require-artifacts
```

3. Start the validated GPU container path:

```bash
DOCKER_BUILDKIT=0 docker build -f deployment/docker/infer.gpu.Dockerfile -t kryptonite-infer-gpu:local .
docker compose -f compose.yml -f compose.gpu.yml up --no-build
```

4. Confirm the live service:

```bash
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/demo/api/state
curl -s http://127.0.0.1:8080/metrics | head
```

## Candidate Promotion And Release Refresh

There is no dedicated promotion script yet, so treat rollout as an explicit,
auditable artifact swap.

Recommended pattern:

1. Snapshot the currently active artifacts.
2. Copy the new candidate bundle into the active roots.
3. Rebuild or restore the matching enrollment cache.
4. Copy the active threshold calibration into a dedicated current-release
   directory under `artifacts/`.
5. Re-run strict smoke and stress validation before exposing traffic.

Example skeleton:

```bash
export SNAPSHOT_ROOT="artifacts/release/backups/$(date -u +%Y%m%dT%H%M%SZ)"
export CANDIDATE_ROOT="artifacts/final/campp"

mkdir -p "$SNAPSHOT_ROOT" artifacts/release/current

rsync -a artifacts/model-bundle/ "$SNAPSHOT_ROOT/model-bundle/" || true
rsync -a artifacts/enrollment-cache/ "$SNAPSHOT_ROOT/enrollment-cache/" || true
cp artifacts/release/current/verification_threshold_calibration.json \
  "$SNAPSHOT_ROOT/verification_threshold_calibration.json" || true

rsync -a --delete "$CANDIDATE_ROOT/model-bundle/" artifacts/model-bundle/

if [ -d "$CANDIDATE_ROOT/enrollment-cache" ]; then
  rsync -a --delete "$CANDIDATE_ROOT/enrollment-cache/" artifacts/enrollment-cache/
else
  uv run python scripts/build_enrollment_cache.py \
    --config configs/deployment/infer.toml \
    --manifest artifacts/manifests/demo_manifest.jsonl \
    --output-dir artifacts/enrollment-cache \
    --model-metadata artifacts/model-bundle/metadata.json
fi

cp "$CANDIDATE_ROOT/verification_threshold_calibration.json" \
  artifacts/release/current/verification_threshold_calibration.json
```

Operational note: the demo threshold resolver walks `artifacts/` and picks the
newest `verification_threshold_calibration.json` by file mtime. The copied
`artifacts/release/current/` file is therefore the safest place to anchor the
active threshold intentionally.

## Monitoring And Incident Triage

Primary checks:

```bash
curl -s http://127.0.0.1:8080/health | jq '{status, selected_backend, inferencer, model_bundle, artifacts, enrollment_cache}'
curl -s http://127.0.0.1:8080/demo/api/state | jq '{threshold, service}'
curl -s http://127.0.0.1:8080/metrics | head -40
```

Interpretation:

- `status != "ok"` means runtime or artifact preflight is degraded;
- `selected_backend` and `inferencer.implementation` must be read together;
- `model_bundle.model_version` must match the intended rollout target;
- `threshold.origin_path` from `/demo/api/state` must point at the intended
  calibration artifact;
- enrollment-cache compatibility failures usually mean the model bundle was
  changed without rebuilding or restoring a matching cache.

Common failure patterns:

- startup failure with missing artifacts:
  regenerate demo artifacts or restore the frozen candidate bundle, then rerun
  `scripts/infer_smoke.py --require-artifacts`;
- wrong threshold in the demo:
  copy the intended calibration JSON back into
  `artifacts/release/current/verification_threshold_calibration.json` so it is
  the newest visible calibration artifact;
- backend confusion:
  inspect both `selected_backend` and `inferencer.implementation` before
  assuming ONNX Runtime or torch is actually serving embeddings;
- runtime latency regression:
  rerun `scripts/inference_stress_report.py` and compare the new report to the
  frozen benchmark pack before changing thresholds or configs.

## Rollback Procedure

Rollback should restore a matched set of artifacts, not just one file.

1. Stop the running service or compose stack.

```bash
docker compose down
```

2. Restore the previous artifact snapshot:

```bash
export SNAPSHOT_ROOT="artifacts/release/backups/<previous-timestamp>"

rsync -a --delete "$SNAPSHOT_ROOT/model-bundle/" artifacts/model-bundle/
rsync -a --delete "$SNAPSHOT_ROOT/enrollment-cache/" artifacts/enrollment-cache/
```

3. Restore the previous active threshold bundle so it becomes the newest visible
   calibration artifact again:

```bash
cp "$SNAPSHOT_ROOT/verification_threshold_calibration.json" \
  artifacts/release/current/verification_threshold_calibration.json
```

If the snapshot does not contain a frozen enrollment cache, rebuild it against
the restored `artifacts/model-bundle/metadata.json` before restart.

4. Revalidate before restart:

```bash
uv run python scripts/infer_smoke.py \
  --config configs/deployment/infer.toml \
  --require-artifacts

uv run python scripts/inference_stress_report.py \
  --config configs/deployment/infer.toml \
  --output-root artifacts/release/rollback-validation
```

5. Restart the service and confirm:

```bash
uv run python apps/api/main.py --config configs/deployment/infer.toml --require-artifacts
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/demo/api/state
```

Rollback is complete only after health, threshold origin, model version, and
enrollment-cache compatibility all match the intended previous release.

## Known Limits

- This runbook assumes one active release per checkout rooted at `artifacts/`.
- Threshold activation is mtime-sensitive until a dedicated release registry
  exists.
- The current runtime contract is ahead of the current learned-export contract;
  do not confuse operational green lights with proof that an exported production
  encoder is live.
