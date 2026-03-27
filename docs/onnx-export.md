# ONNX Export

## Goal

`KVA-538` materializes a real encoder-only ONNX bundle for the frozen `CAM++`
family without replacing the checked-in demo bundle that still drives current
runtime and integration smoke checks.

## Scope

The export covers only the `encoder_input -> embedding` graph boundary already
frozen in [docs/export-boundary.md](./export-boundary.md).

What stays outside the graph:

- raw audio decode
- channel fold-down / resample
- optional loudness normalization
- optional VAD trimming
- chunking
- Fbank extraction

That split is intentional. `KVA-538` proves that the chosen `CAM++` checkpoint
can be exported with the modern ONNX exporter, dynamic time axis, and stable
tensor names. It does not claim full runtime parity yet; that promotion is
reserved for `KVA-539`.

## Command

Export a checkpoint file or a completed CAM++ run directory:

```bash
uv run python scripts/export_campp_onnx.py \
  --config configs/base.toml \
  --checkpoint artifacts/baselines/campp/<run-id> \
  --output-root artifacts/model-bundle-campp-onnx
```

The script accepts either:

- a direct checkpoint file such as `.../campp_encoder.pt`
- a run directory that contains one of the known CAM++ checkpoint names

## Generated Artifacts

The export bundle is written under `artifacts/model-bundle-campp-onnx/` by
default:

- `model.onnx`
- `metadata.json`
- `export_boundary.json`
- `export_report.json`
- `export_report.md`

The bundle intentionally uses a separate root from `artifacts/model-bundle/` so
the existing demo artifact path remains stable for current deploy-contract and
API smoke tests.

## Validation

Each export run performs:

- `torch.onnx.export(..., dynamo=True)` with stable `encoder_input` /
  `embedding` names
- dynamic time-axis export metadata
- `onnx.checker.check_model(...)`
- one deterministic ONNX Runtime smoke input against the original PyTorch model
  on the same feature tensor

The smoke report records max/mean absolute error for that single-sample check.
That is enough to detect broken exports early, but it is not the broader parity
matrix that later tasks must prove.

## Implementation Note

`CAM++` originally used a `segment_pooling` path that expanded pooled segments,
reshaped them, and sliced back to the original frame count. That formulation was
numerically fine but fragile for `torch.export` symbolic shapes. The exporter now
uses an index-based gather that preserves the exact legacy result while avoiding
the symbolic-shape guard failures triggered by the reshape-based form.

## Next Step

Use this bundle as the source for:

- `KVA-539` ONNX Runtime parity work
- `KVA-540` TensorRT engine build
- Triton packaging via [docs/triton-deployment.md](./triton-deployment.md)
