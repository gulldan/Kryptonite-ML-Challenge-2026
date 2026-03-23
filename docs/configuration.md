# Configuration

## Goal

Keep runtime parameters in versioned config files and keep secret values out of git.

## Files

- `configs/base.toml` contains the versioned default profile
- `configs/schema.json` documents the expected shape
- `.env.example` lists supported secret variables
- `scripts/repro_check.py` performs a reproducibility self-check for a config

## Overrides

Use dotted overrides in `key=value` form:

```bash
uv run python scripts/show_config.py --config configs/base.toml \
  --override runtime.seed=123 \
  --override training.batch_size=32 \
  --override backends.allow_tensorrt=true
```

Supported value forms:

- integers, booleans, floats
- quoted TOML strings
- plain strings, which are treated as raw text

## Secrets

Copy `.env.example` to `.env` on each machine and fill in only the values you need.

The config stores environment variable names, not secret values. At runtime, the loader resolves:

- `WANDB_API_KEY`
- `MLFLOW_TRACKING_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`

By default, the CLI masks secret values when printing them.

## Reproducibility Hooks

The config includes a `reproducibility` section with:

- `deterministic`
- `pythonhashseed`
- `fingerprint_paths`

Run the self-check with:

```bash
uv run python scripts/repro_check.py --config configs/base.toml --self-check
```

This launches two short subprocess runs with the same seed and `PYTHONHASHSEED`, then verifies:

- identical control metadata
- identical fingerprints for tracked inputs
- equal probe outputs from seeded randomness

## Tracking

The config also includes a `tracking` section. Right now the default backend is `local`, which writes:

- run metadata
- params
- metrics
- copied artifact files

into `artifacts/tracking/<run_id>/`.

## Audio Normalization

The base config also includes a `normalization` section for the manifest-driven audio rewrite flow:

- `target_sample_rate_hz`
- `target_channels`
- `output_format`
- `output_pcm_bits_per_sample`
- `peak_headroom_db`
- `dc_offset_threshold`
- `clipped_sample_threshold`

The defaults currently define the canonical preprocessing bundle for this repo:

- `16 kHz`
- `mono`
- `PCM16 WAV`
- `1 dB` peak headroom

Use them through `scripts/normalize_audio_dataset.py`, and override only when a specific experiment needs a different derived bundle.

## VAD / Trimming

The base config also includes a `vad` section for loader-time silence trimming:

- `mode`
- `backend`
- `provider`

Supported values are:

- `none`
- `light`
- `aggressive`

The only supported backend right now is:

- `silero_vad_v6_onnx`

Provider values are:

- `auto`
- `cpu`
- `cuda`

The base profile keeps `mode = "none"` so raw waveform behavior stays stable by
default. Use config overrides for ablations or production-oriented comparisons:

```bash
uv run python scripts/vad_trimming_report.py \
  --config configs/base.toml \
  --override vad.mode=light \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl
```
