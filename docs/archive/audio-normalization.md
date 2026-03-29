# Audio Normalization

## Goal

Make audio preprocessing reproducible and explicit so no stage relies on hidden loader defaults.

The repository contract for derived training-ready audio is:

- resample every active row to `16 kHz`
- downmix every active row to `mono`
- optionally apply bounded RMS loudness normalization
- write a deterministic `PCM16 WAV` bundle into `artifacts/preprocessed/`
- keep source clipping and decode issues auditable
- quarantine rows that cannot be read, decoded, or written cleanly

## Configuration

Normalization options are controlled via `configs/base.toml`:

```toml
[normalization]
target_sample_rate_hz = 16000
target_channels = 1
output_format = "wav"
output_pcm_bits_per_sample = 16
peak_headroom_db = 1.0
dc_offset_threshold = 0.01
clipped_sample_threshold = 0.999
loudness_mode = "none"
target_loudness_dbfs = -27.0
max_loudness_gain_db = 20.0
max_loudness_attenuation_db = 12.0
```

Supported `loudness_mode` values:

- `none` — loudness is not changed
- `rms` — bounded gain alignment to target RMS

For a controlled A/B check:

```bash
uv run python scripts/show_config.py \
  --config configs/base.toml \
  --override normalization.loudness_mode=rms \
  --override normalization.target_loudness_dbfs=-27.0
```

## Canonical entry point

```bash
uv run python scripts/normalize_audio_dataset.py \
  --config configs/base.toml \
  --source-manifests-root artifacts/manifests/ffsvc2022-surrogate \
  --output-root artifacts/preprocessed/ffsvc2022-surrogate
```

The script rewrites manifests to the derived bundle, carries forward existing quarantine
rows, and writes provenance for reproducibility.

## Output Layout

- `artifacts/preprocessed/ffsvc2022-surrogate/audio/`
- `artifacts/preprocessed/ffsvc2022-surrogate/manifests/`
- `artifacts/preprocessed/ffsvc2022-surrogate/reports/audio_normalization_report.json`
- `artifacts/preprocessed/ffsvc2022-surrogate/reports/audio_normalization_report.md`

Rewritten rows add provenance fields such as:

- `source_audio_path`
- `source_sample_rate_hz`
- `source_num_channels`
- `source_rms_dbfs`
- `normalized_rms_dbfs`
- `source_dc_offset_ratio`
- `source_clipped_sample_ratio`
- `normalization_profile`
- `normalization_resampled`
- `normalization_downmixed`
- `normalization_dc_offset_removed`
- `normalization_peak_scaled`
- `normalization_loudness_mode`
- `normalization_loudness_applied`
- `normalization_loudness_gain_db`
- `normalization_loudness_gain_clamped`
- `normalization_loudness_peak_limited`
- `normalization_loudness_degradation_check_passed`
- `loudness_mode`
- `loudness_target_dbfs`
- `loudness_applied`
- `loudness_gain_db`
- `loudness_gain_clamped`
- `loudness_peak_limited`
- `loudness_skip_reason`
- `pre_loudness_rms_dbfs`
- `post_loudness_rms_dbfs`
- `loudness_alignment_error`

## Loudness comparison artifact

```bash
uv run python scripts/loudness_normalization_report.py \
  --config configs/base.toml \
  --override normalization.loudness_mode=rms \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl \
  --output-dir artifacts/eda/loudness-normalization
```

This writes:

- `artifacts/eda/loudness-normalization/loudness_normalization_report.json`
- `artifacts/eda/loudness-normalization/loudness_normalization_report.md`
- `artifacts/eda/loudness-normalization/loudness_normalization_rows.jsonl`

## Loader contract

`AudioLoadRequest.from_config(...)` applies loudness as a bounded scale-only transform after:

1. source windowing
2. channel fold-down
3. resampling
4. optional VAD/trimming
5. optional loudness gain stage

The contract is to preserve waveform shape and only apply global gain under explicit
bounds. It is intentionally not a perceptual LUFS or compressor stage.

## Validation

After generating a derived bundle:

```bash
uv run python scripts/validate_manifests.py \
  --config configs/base.toml \
  --manifests-root artifacts/preprocessed/ffsvc2022-surrogate/manifests
```

For EDA/profile checks on derived manifests:

```bash
uv run python scripts/dataset_profile_report.py \
  --config configs/base.toml \
  --override paths.manifests_root=\"artifacts/preprocessed/ffsvc2022-surrogate/manifests\" \
  --output-dir artifacts/eda/dataset-profile-normalized
```
