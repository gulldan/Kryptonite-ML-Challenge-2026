# Audio Normalization

## Goal

Make audio normalization reproducible and explicit instead of relying on backend-default resampling or ad hoc one-off conversion commands.

The current repository policy for derived training-ready audio is:

- resample every active row to `16 kHz`
- downmix every active row to `mono`
- write a deterministic `PCM16 WAV` bundle under `artifacts/preprocessed/`
- keep source clipping as an auditable source-side flag instead of silently dropping rows
- remove material DC offset when it crosses the configured threshold
- quarantine rows that cannot be read, decoded, or written cleanly

## Entry Point

Run:

```bash
uv run python scripts/normalize_audio_dataset.py \
  --config configs/base.toml \
  --source-manifests-root artifacts/manifests/ffsvc2022-surrogate \
  --output-root artifacts/preprocessed/ffsvc2022-surrogate
```

The script reads the active manifests from the source root, normalizes each unique `audio_path` once, rewrites the manifests to point at the derived audio bundle, carries forward existing quarantine rows, and writes a report plus a fresh manifest inventory.

## Output Layout

The default run writes:

- `artifacts/preprocessed/ffsvc2022-surrogate/audio/` for normalized audio files
- `artifacts/preprocessed/ffsvc2022-surrogate/manifests/` for rewritten manifests and copied auxiliary metadata
- `artifacts/preprocessed/ffsvc2022-surrogate/reports/audio_normalization_report.json`
- `artifacts/preprocessed/ffsvc2022-surrogate/reports/audio_normalization_report.md`

The rewritten manifest rows keep the unified schema fields and add normalization provenance such as:

- `source_audio_path`
- `source_sample_rate_hz`
- `source_num_channels`
- `source_dc_offset_ratio`
- `source_clipped_sample_ratio`
- `normalization_profile`
- `normalization_resampled`
- `normalization_downmixed`
- `normalization_dc_offset_removed`
- `normalization_peak_scaled`

## Policy Notes

- Source clipping is preserved as a source-quality observation. The derived waveform is peak-scaled only to avoid writing a newly clipped output file.
- Missing or undecodable audio does not stay in active manifests. Those rows are appended to the derived `quarantine_manifest.jsonl` with `quarantine_stage=audio_normalization`.
- Existing quarantine rows are carried forward so downstream stages can see the complete exclusion set in one place.

## Validation

After generating a derived bundle, validate it with:

```bash
uv run python scripts/validate_manifests.py \
  --config configs/base.toml \
  --manifests-root artifacts/preprocessed/ffsvc2022-surrogate/manifests
```

For profile or EDA checks against the normalized bundle, point the existing scripts at the derived manifests root with a config override:

```bash
uv run python scripts/dataset_profile_report.py \
  --config configs/base.toml \
  --override paths.manifests_root=\"artifacts/preprocessed/ffsvc2022-surrogate/manifests\" \
  --output-dir artifacts/eda/dataset-profile-normalized
```
