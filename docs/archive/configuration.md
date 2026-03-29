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

## Backend Selection

Serving configs use `backends.inference` to express intent, not just a fixed
runtime label.

- `auto` asks the service to resolve the highest eligible backend from the
  fallback chain `tensorrt -> onnxruntime -> torch`;
- `torch`, `onnxruntime`, and `tensorrt` request a specific runtime path and
  fail fast when that path is not validated or not available.

At runtime, `GET /health` reports both:

- `requested_backend`: what the config asked for;
- `selected_backend`: what the service actually resolved after validation and
  fallback.

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
- `loudness_mode`
- `target_loudness_dbfs`
- `max_loudness_gain_db`
- `max_loudness_attenuation_db`

The defaults currently define the canonical preprocessing bundle for this repo:

- `16 kHz`
- `mono`
- `PCM16 WAV`
- `1 dB` peak headroom
- `loudness_mode = "none"` by default, so gain staging stays explicit

Use them through `scripts/normalize_audio_dataset.py`, and override only when a specific experiment needs a different derived bundle.

The loudness-related normalization fields support the current bounded RMS policy
described in
[audio-normalization.md](./audio-normalization.md):

- `loudness_mode`: `none` or `rms`
- `target_loudness_dbfs`: target RMS level when the mode is enabled
- `max_loudness_gain_db`: upper bound on amplification
- `max_loudness_attenuation_db`: upper bound on attenuation

Use the dedicated comparison CLI before enabling it in a real run:

```bash
uv run python scripts/loudness_normalization_report.py \
  --config configs/base.toml \
  --override normalization.loudness_mode=rms \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl
```

See the same document for the full contract and report outputs.

## VAD / Trimming

The base config also includes a `vad` section for loader-time silence trimming:

- `mode`
- `backend`
- `provider`
- `min_output_duration_seconds`
- `min_retained_ratio`

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

The trim guards keep loader-time boundary trimming from collapsing utterances
too aggressively:

- `min_output_duration_seconds` rejects a trim if the remaining clip would be
  shorter than this value
- `min_retained_ratio` rejects a trim if the remaining clip would keep less
  than this fraction of the original duration

The base profile keeps `mode = "none"` so raw waveform behavior stays stable by
default. Use config overrides for ablations or production-oriented comparisons:

```bash
uv run python scripts/vad_trimming_report.py \
  --config configs/base.toml \
  --override vad.mode=light \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl
```

## Silence Augmentation

The base config also includes a `silence_augmentation` section for the
standalone silence/pause robustness transform:

- `enabled`
- `max_leading_padding_seconds`
- `max_trailing_padding_seconds`
- `max_inserted_pauses`
- `min_inserted_pause_seconds`
- `max_inserted_pause_seconds`
- `pause_ratio_min`
- `pause_ratio_max`
- `min_detected_pause_seconds`
- `max_perturbed_pause_seconds`
- `analysis_frame_ms`
- `silence_threshold_dbfs`

The defaults keep it disabled so baseline waveforms and existing reports do not
start changing implicitly, while still leaving a reusable parameter envelope for
the shared scheduler. The intended workflow is:

1. keep the base profile stable with `enabled = false`
2. enable the transform explicitly through overrides for ablations
3. feed the same config block into the training scheduler instead of
   inventing one-off augmentation knobs elsewhere

Use the dedicated ablation CLI when you need a reproducible before/after report:

```bash
uv run python scripts/silence_augmentation_report.py \
  --config configs/base.toml \
  --override silence_augmentation.enabled=true \
  --override silence_augmentation.max_inserted_pauses=2 \
  --override silence_augmentation.pause_ratio_max=1.4 \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl
```

See [audio-silence-augmentation.md](./audio-silence-augmentation.md) for the
full contract and artifact layout.

## Augmentation Scheduler

The base config also includes an `augmentation_scheduler` section for the
training-side curriculum that mixes clean, light, medium, and heavy corruption
across epochs:

- `enabled`
- `warmup_epochs`
- `ramp_epochs`
- `max_augmentations_per_sample`
- `clean_probability_start` / `clean_probability_end`
- `light_probability_start` / `light_probability_end`
- `medium_probability_start` / `medium_probability_end`
- `heavy_probability_start` / `heavy_probability_end`
- `family_weights.noise`
- `family_weights.reverb`
- `family_weights.distance`
- `family_weights.codec`
- `family_weights.silence`

The scheduler reads the existing bank manifests under `artifacts/corruptions/`
and derives per-epoch sampling policies plus coverage logs. Family weights act
as global priors; the scheduler still applies stage-specific dampening so codec,
distance, and strong room effects ramp in later than noise and silence.

Use the dedicated report CLI to sanity-check the curriculum before wiring it
into a full training recipe:

```bash
uv run python scripts/augmentation_scheduler_report.py \
  --config configs/base.toml \
  --epochs 10 \
  --samples-per-epoch 512
```

See [audio-augmentation-scheduler.md](./audio-augmentation-scheduler.md) for
the manifest contract, stage policy, and artifact layout.

## Corrupted Dev Suites

Frozen corrupted dev suites are configured via the versioned plan file:

- `configs/corruption/corrupted-dev-suites.toml`

The builder reuses the shared `normalization` and `silence_augmentation`
settings from the base config while reading the reusable corruption banks and
plans already materialized under `artifacts/corruptions/`.

Run it with:

```bash
uv run python scripts/build_corrupted_dev_suites.py \
  --config configs/base.toml \
  --plan configs/corruption/corrupted-dev-suites.toml
```

See [audio-corrupted-dev-suites.md](./audio-corrupted-dev-suites.md) for the
suite catalog, artifact layout, and design constraints.
