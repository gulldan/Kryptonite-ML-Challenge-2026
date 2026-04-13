# TensorRT FP16 Engine

## Goal

`KVA-540` closes the gap between the parity-promoted ONNX bundle and the
existing TensorRT/Triton handoff by materializing a real `model.plan` artifact
from the encoder-only CAM++ export.

The workflow does three things in one run:

- builds a TensorRT FP16 engine from the promoted ONNX bundle;
- compares TensorRT encoder outputs against the source PyTorch checkpoint on
  deterministic feature-tensor samples;
- records latency speedup vs PyTorch and promotes `tensorrt` in model metadata
  only when the configured quality and speed gates pass.

## Frozen Workflow

Use the checked-in config:

```bash
uv run python scripts/build_tensorrt_fp16_engine.py \
  --config configs/release/tensorrt-fp16.toml
```

The checked-in config assumes:

- ONNX Runtime parity has already been promoted for
  `artifacts/model-bundle-campp-onnx/metadata.json`;
- the engine should be written to
  `artifacts/model-bundle-campp-onnx/model.plan`;
- the human-readable report should be written to
  `artifacts/release/current/fp16/`.

## Generated Artifacts

One successful run writes:

```text
artifacts/model-bundle-campp-onnx/model.plan
artifacts/release/current/fp16/tensorrt_fp16_engine_report.json
artifacts/release/current/fp16/tensorrt_fp16_engine_report.md
artifacts/release/current/fp16/sources/tensorrt_fp16_config.toml
```

It also updates `artifacts/model-bundle-campp-onnx/metadata.json` with:

- `inference_package.artifacts.tensorrt_engine_file`;
- `inference_package.validated_backends.tensorrt = true` only when the report
  passes;
- `export_validation.tensorrt_fp16_*` bookkeeping fields.

## Validation Contract

The workflow validates the encoder boundary directly:

- input: feature tensors with shape `[batch, frames, mel_bins]`;
- output: embeddings with shape `[batch, embedding_dim]`;
- accuracy gates: mean absolute diff and cosine distance vs the source PyTorch
  checkpoint;
- latency gate: TensorRT must beat PyTorch by the configured speedup ratio.

This deliberately avoids claiming raw-audio runtime parity. Raw-audio frontend
steps still live outside the engine boundary fixed in
[docs/export-boundary.md](./export-boundary.md).

## Environment Requirements

Run this on `gpu-server`, not on the local macOS environment.

Minimum requirements:

- CUDA-capable GPU;
- TensorRT Python package available inside the repo-local `.venv`;
- ONNX Runtime and torch already synced in the same `.venv`.

If TensorRT extras are missing, the workflow fails early with an explicit error
instead of silently writing placeholder metadata.
