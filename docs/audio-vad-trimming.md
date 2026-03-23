# Audio VAD And Trimming

## Goal

Make leading/trailing silence handling explicit and reproducible instead of
letting every downstream experiment guess whether it should keep raw pauses,
lightly trim them, or cut more aggressively.

The current repository contract exposes three loader-level modes:

- `none`: keep the decoded waveform unchanged
- `light`: trim only confident leading/trailing silence with generous padding
- `aggressive`: trim more tightly for ablation and production selection

The implementation is now backed by the official Silero VAD v6 ONNX model. We
use the official packaged ONNX asset and official timestamp post-processing, but
keep provider selection explicit in our code instead of relying on the package's
CPU-forced convenience loader.

The runtime dependency is installed through the repository `train` environment
group, so the expected setup command remains:

```bash
uv sync --dev --group train --group tracking
```

## Config

The active mode lives in `configs/base.toml`:

```toml
[vad]
mode = "none"
backend = "silero_vad_v6_onnx"
provider = "auto"
```

Supported values:

- `none`
- `light`
- `aggressive`

Supported backends:

- `silero_vad_v6_onnx`

Supported providers:

- `auto`
- `cpu`
- `cuda`

Override it per run when needed:

```bash
uv run python scripts/show_config.py \
  --config configs/base.toml \
  --override vad.mode=light
```

## Loader Usage

The shared loader now accepts VAD mode through `AudioLoadRequest.from_config(...)`:

```python
from kryptonite.config import load_project_config
from kryptonite.data import AudioLoadRequest, load_audio

config = load_project_config(config_path="configs/base.toml")
request = AudioLoadRequest.from_config(config.normalization, vad=config.vad)

loaded = load_audio(
    "datasets/ffsvc2022-surrogate/raw/dev/dev/ffsvc22_dev_000001.wav",
    project_root=config.paths.project_root,
    request=request,
)
```

The returned `LoadedAudio` metadata now records:

- `vad_mode`
- `vad_backend`
- `vad_provider`
- `trim_applied`
- `trim_reason`
- `trim_start_seconds`
- `trim_end_seconds`
- `trimmed_leading_seconds`
- `trimmed_trailing_seconds`

If the VAD does not find reliable speech boundaries, the loader keeps the
original waveform and reports why instead of silently returning an empty slice.

## Dev Comparison Report

Use the comparison CLI on the held-out dev manifest:

```bash
uv run python scripts/vad_trimming_report.py \
  --config configs/base.toml \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl \
  --output-dir artifacts/eda/vad-trimming
```

The default run writes:

- `artifacts/eda/vad-trimming/vad_trimming_report.json`
- `artifacts/eda/vad-trimming/vad_trimming_report.md`
- `artifacts/eda/vad-trimming/vad_trimming_rows.jsonl`

`vad_trimming_report.md` is the human-readable comparison for `none`, `light`,
and `aggressive`. `vad_trimming_rows.jsonl` is the row-level handoff artifact if
baseline or threshold-tuning work needs to inspect which utterances changed the
most.

## Current Limits

- only leading/trailing trimming is applied; interior pauses stay untouched
- the repo currently keeps `light` and `aggressive` as preset parameter bundles
  around the same Silero V6 ONNX backend
- `provider = "auto"` falls back to CPU when `CUDAExecutionProvider` is not
  available in the active `onnxruntime` build
