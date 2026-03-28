# INT8 Feasibility

## Goal

`KVA-543` turns INT8 promotion into an explicit `go` / `no_go` workflow instead
of an implicit wishlist item.

The report answers one question:

- does the current export surface justify building and carrying an INT8
  TensorRT engine on top of the already-promoted FP16 path?

The repository now encodes that answer through:

- a tracked calibration-set catalog in [`assets/int8/calibration_catalog.json`](../assets/int8/calibration_catalog.json);
- a machine-readable decision config in
  [`configs/release/int8-feasibility.toml`](../configs/release/int8-feasibility.toml);
- a reproducible builder script in
  [`scripts/build_int8_feasibility_report.py`](../scripts/build_int8_feasibility_report.py).

## What The Report Checks

The decision only returns `go` when all of the following are true:

- the model bundle is not a structural stub;
- a reference FP16 TensorRT engine exists;
- the ONNX path has a saved parity report;
- an INT8 engine exists for the same promoted candidate;
- matched FP16 and INT8 verification/stress reports exist;
- INT8 quality degradation stays inside the configured EER and minDCF gates;
- INT8 latency and memory deltas stay inside the configured operational gates.

## Saved Calibration Set Spec

The tracked catalog is intentionally small and deterministic. It preserves the
same traffic classes the release stress suite already exercises:

- clean baseline controls;
- short and long duration extremes;
- corruption coverage for noise, echo, and clipping.

The default checked-in selection excludes the silence-only sample from actual
INT8 calibration while keeping it in the wider stress suite.

## Current Checked-In Decision

The checked-in config is expected to return `no_go` today. That is the honest
outcome for the current repository state because:

- the promoted FP16 TensorRT engine is not materialized yet;
- the ONNX parity workflow is now defined, but its release artifact still has to
  be generated locally into `artifacts/release/current/`;
- matched FP16 vs INT8 verification/stress artifacts are not frozen yet;
- the local demo model bundle remains a structural ONNX stub when only demo
  artifacts are present.

The workflow is still useful in that state because it preserves the decision
contract and the calibration-set spec before any engine work starts.

## Usage

Build the report from the checked-in config:

```bash
uv run python scripts/build_int8_feasibility_report.py \
  --config configs/release/int8-feasibility.toml
```

The command writes:

```text
artifacts/release-decisions/kryptonite-2026-int8-feasibility/
├── int8_feasibility.json
├── int8_feasibility.md
└── sources/
    └── int8_feasibility_config.toml
```

## Scope Limits

- The workflow does not claim that INT8 is live in the runtime today.
- The workflow does not invent benchmark evidence when FP16/INT8 reports are
  missing.
- The workflow is designed to return a truthful `no_go` until the promoted FP16
  export path is real enough to justify quantization.
