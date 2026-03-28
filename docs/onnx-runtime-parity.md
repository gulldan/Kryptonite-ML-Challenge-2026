# ONNX Runtime Parity

## Goal

`KVA-539` promotes the exported `CAM++` ONNX bundle from "graph materializes and
smoke-check passes" to "the ONNX Runtime path matches the active PyTorch
reference on the current dev protocol."

The parity workflow compares:

- the same raw-audio frontend frozen into the export boundary;
- the same chunking and pooling policy used by the runtime;
- the same clean dev trials plus deterministic probe-side corruptions;
- the final cosine scores and verification metrics produced by PyTorch vs
  ONNX Runtime.

## Command

Build the report from the tracked release config:

```bash
uv run python scripts/build_onnx_parity_report.py \
  --config configs/release/onnx-parity.toml
```

## Generated Outputs

The command writes under `artifacts/release/current/`:

- `onnx_parity_report.json`
- `onnx_parity_report.md`
- `onnx_parity_audio_rows.jsonl`
- `onnx_parity_trial_rows.jsonl`

When the report passes and `promote_validated_backend = true`, the writer also
updates the target model-bundle metadata in place to:

- mark `inference_package.validated_backends.onnxruntime = true`
- clear the `runtime_backends_promotion_blocker`
- save the parity report path into `export_validation`

## Scope

This workflow proves numerical parity for the exported encoder under the
runtime-owned frontend. It does not claim that the repository already serves via
ONNX Runtime end-to-end; later packaging/runtime tasks still decide when that
backend becomes selectable in production.
