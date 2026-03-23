# Audio Loader

## Goal

Provide one explicit audio-loading contract for preprocessing and feature work
instead of letting each downstream stage guess how to open files, resample
waveforms, or resolve manifest paths.

The loader now covers the current repository contract:

- supported source formats: `WAV`, `FLAC`, `MP3`
- channel-first `float32` tensors in memory
- optional explicit resampling
- optional deterministic mono fold-down
- optional loader-time `none/light/aggressive` Silero VAD v6 ONNX trimming
- windowed reads for long files through `start_seconds` and `duration_seconds`
- direct loading from unified manifest rows or manifest JSONL files

## Public API

Use the helpers from `kryptonite.data`:

```python
from kryptonite.config import load_project_config
from kryptonite.data import AudioLoadRequest, iter_manifest_audio, load_audio

config = load_project_config(config_path="configs/base.toml")
request = AudioLoadRequest.from_config(config.normalization, vad=config.vad)

single = load_audio(
    "datasets/ffsvc2022-surrogate/raw/dev/dev/ffsvc22_dev_000001.wav",
    project_root=config.paths.project_root,
    request=request,
)

batch = list(
    iter_manifest_audio(
        "artifacts/manifests/ffsvc2022-surrogate/train_manifest.jsonl",
        project_root=config.paths.project_root,
        request=request,
    )
)
```

`AudioLoadRequest.from_config(...)` binds the loader to the same `16 kHz` /
mono preprocessing policy already described in
[audio-normalization.md](./audio-normalization.md), and optionally to the
configured VAD/trimming mode from
[audio-vad-trimming.md](./audio-vad-trimming.md). For ablations or future
datasets, the request can be constructed manually.

## Contract Notes

- segment windows are applied at the source sample rate before resampling, so
  long files can be inspected without decoding the full recording
- mono fold-down is an arithmetic mean over channels; other channel-layout
  conversions are rejected explicitly
- `light` and `aggressive` use the Silero VAD v6 ONNX backend and only trim
  leading/trailing silence; they keep
  interior pauses so downstream chunking and augmentation still see realistic
  pause structure
- manifest loading validates rows through `ManifestRow.from_mapping(...)`, so
  missing canonical fields fail before the model pipeline consumes corrupted
  metadata

## Limits

- the current loader supports preserving the original channel layout or
  collapsing to `mono`; arbitrary channel remapping is intentionally left for a
  later task if a dataset actually needs it
- windowed reads are synchronous and file-based; this task does not introduce a
  streaming iterator or async dataloader layer
