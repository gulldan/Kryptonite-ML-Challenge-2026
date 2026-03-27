# Export Boundary

`KRYP-059` fixes one explicit export contract for the repo: the exported graph starts at
`encoder_input` and ends at `embedding`. The waveform frontend stays outside the engine in the
runtime layer.

## Scope Decision

The supported export boundary is currently:

- boundary mode: `encoder_only`
- runtime-owned pre-engine steps:
  - audio decode
  - mono fold-down
  - resample to the configured sample rate
  - optional bounded loudness normalization
  - optional VAD trimming
  - utterance chunking
  - log-Mel/Fbank extraction
- engine-owned steps:
  - speaker encoder forward pass from `encoder_input`
- runtime-owned post-engine steps:
  - chunk embedding pooling
  - enrollment averaging
  - embedding normalization for scoring
  - cosine scoring and thresholding

This keeps unsupported decode/VAD/Fbank ops out of future ONNX/TensorRT graphs and makes the
frontend/runtime contract explicit.

## Contract Shape

- input tensor name: `encoder_input`
- input layout: `BTF` (`batch`, `frames`, `mel_bins`)
- output tensor name: `embedding`
- output layout: `BE` (`batch`, `embedding_dim`)
- time axis can stay dynamic when `export.dynamic_axes = true`

The model bundle metadata now carries a machine-readable `export_boundary` block. Runtime startup
loads that block and rejects the bundle if the active `audio_load_request`, `features`, or
`chunking` config no longer matches the contract.

## Reproducible Report

Generate the current contract from config:

```bash
uv run python scripts/export_boundary_report.py --config configs/base.toml
```

This writes:

- `artifacts/export-boundary/export_boundary.json`
- `artifacts/export-boundary/export_boundary.md`

These reports are the handoff artifact for the next export tasks (`KRYP-060+`) so ONNX/TensorRT
work starts from the same frontend assumptions that runtime and enrollment cache already use.
