# Inference Stress Test

## Goal

Freeze one reproducible stress matrix for the serving path so release QA stops
depending on ad hoc manual poking.

The stress run validates three classes of behavior:

- same-speaker verification on deterministic extreme-duration and corrupted inputs;
- batch-burst latency behavior on mixed short/long/corrupted payloads;
- process/CUDA memory peaks observed during the serving-path workload;
- malformed-input handling for missing files, corrupt audio containers, invalid
  stage values, and schema failures.

## What It Exercises

The reusable implementation lives in `src/kryptonite/serve/stress_report.py`,
and the CLI entrypoint is `scripts/inference_stress_report.py`.

The run generates deterministic WAV inputs under
`artifacts/inference-stress/inputs/` and then drives the live FastAPI adapter
through:

- `POST /enroll` to create a runtime enrollment reference;
- `POST /verify` for clean, short, long, noisy, echoed, clipped, and silence-only probes;
- `POST /benchmark` for burst sizes such as `1`, `4`, `8`, and `16`;
- malformed `POST /verify` requests that must fail with `400` or `422`, not `500`.

The report writes:

- `artifacts/inference-stress/report/inference_stress_report.json`
- `artifacts/inference-stress/report/inference_stress_report.md`

## How To Run

Against the active infer config and backend:

```bash
uv run python scripts/inference_stress_report.py
```

Example with explicit burst sizes and JSON output:

```bash
uv run python scripts/inference_stress_report.py \
  --batch-size 1 \
  --batch-size 4 \
  --batch-size 8 \
  --output json
```

The script intentionally reuses the same runtime startup checks as the FastAPI
service. If the active model bundle metadata is stale or missing the exported
boundary contract, the run fails immediately instead of silently bypassing that
release blocker.

## Hard Limits Captured

The report records the currently validated serving envelope:

- the `demo` chunking contract from config
  - full utterance threshold
  - chunk window size
  - chunk overlap
- the shortest and longest probe durations that completed successfully;
- the largest batch burst that completed successfully;
- the largest observed total chunk count in a burst run;
- the peak process RSS and, when CUDA is active, peak CUDA allocated/reserved memory;
- the expected malformed-input status codes.

## Why This Is Separate From Offline Quality Eval

This stress report is intentionally operational:

- it validates that the serving path stays alive under difficult inputs;
- it documents current hard limits and failure behavior;
- it does **not** replace offline EER/minDCF evaluation on frozen clean or
  corrupted dev suites.

If the active config points at the final release model bundle, the same script
tests that candidate. If the runtime is operating without a model bundle, the
report still validates the raw-audio frontend, chunking policy, enrollment
flow, API error contract, and operational memory envelope.
