# Triton Deployment

## Goal

Package the already-fixed export boundary into a Triton model repository so the project can serve
`encoder_input -> embedding` through Triton's KServe `/v2/models/<name>/infer` API.

This is an optional deployment path. It complements the FastAPI runtime instead of replacing it.

## Scope Decision

The Triton repository serves only the encoder graph.

Runtime-owned frontend steps remain outside Triton:

- audio decode
- mono fold-down / resample
- bounded loudness normalization
- optional VAD trimming
- waveform chunking
- log-Mel / Fbank extraction

That split is intentional and matches the repository's canonical export contract in
`docs/export-boundary.md`.

## Generated Repository

Build the default ONNX-backed repository:

```bash
uv run python scripts/build_triton_model_repository.py --config configs/deployment/infer.toml
```

The builder writes:

- `artifacts/triton-model-repository/kryptonite_encoder/config.pbtxt`
- `artifacts/triton-model-repository/kryptonite_encoder/1/model.onnx`
- `artifacts/triton-model-repository/kryptonite_encoder/metadata.json`
- `artifacts/triton-model-repository/smoke/kryptonite_encoder_infer_request.json`

The generated `config.pbtxt` is derived from model-bundle metadata plus the export-boundary
contract, so Triton input/output names stay aligned with the runtime handoff:

- input: `encoder_input`
- layout: `BTF`
- output: `embedding`
- layout: `BE`

## Launch

Run Triton with the generated repository mounted as `/models`:

```bash
TRITON_IMAGE="${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:<compatible-tag>}"
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/artifacts/triton-model-repository:/models:ro" \
  "$TRITON_IMAGE" tritonserver --model-repository=/models
```

The repository is backend-agnostic at the folder level. By default it packages the ONNX bundle,
which is the only artifact this repo can build and validate locally today.

## Smoke

Probe the running Triton server with the generated sample request:

```bash
uv run python scripts/triton_infer_smoke.py \
  --repository-root artifacts/triton-model-repository \
  --model-name kryptonite_encoder \
  --server-url http://127.0.0.1:8000
```

The generated request hits:

```text
POST /v2/models/kryptonite_encoder/infer
```

and checks that Triton returns a non-empty output tensor with the configured output name and shape.

## TensorRT Handoff

TensorRT packaging is supported only when a real plan file already exists. The builder can switch
the same repository layout from ONNX to TensorRT:

```bash
uv run python scripts/build_triton_model_repository.py \
  --config configs/deployment/infer.toml \
  --backend-mode tensorrt \
  --engine-path artifacts/model-bundle/model.plan
```

In that mode the builder writes:

- `artifacts/triton-model-repository/kryptonite_encoder/1/model.plan`
- `platform: "tensorrt_plan"` in `config.pbtxt`

What this does not claim:

- there is no repo-native export step that materializes `model.plan`
- there is no raw-audio TensorRT deployment path yet
- there is no claim that the current `gpu-server` path has replaced the torch runtime with TensorRT

## Limits

- The packaged Triton model is encoder-only, not end-to-end raw audio.
- The default demo bundle is a structural ONNX stub used for deploy-contract validation, not a
  production SV encoder.
- Full raw-audio parity still lives in the FastAPI + `Inferencer` runtime path.
