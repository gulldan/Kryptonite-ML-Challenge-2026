# Audio Quality Checks

## Goal

Turn audio-quality auditing into a reproducible manifest-driven step that emits
both a summary report and concrete flagged rows for downstream preprocessing.

The current checks cover:

- missing or unreadable files
- zero-signal waveforms
- near-clipping peaks
- short and long duration tails
- high silence ratios
- low loudness
- sample-rate and channel mismatches
- DC offset risk

## Entry Point

Run:

```bash
uv run python scripts/dataset_audio_quality_report.py \
  --config configs/base.toml \
  --output-dir artifacts/eda/dataset-audio-quality
```

The script scans active manifests under the configured `paths.manifests_root`
and inspects deduplicated audio rows through the shared audio I/O layer.

## Output Artifacts

The default run writes:

- `artifacts/eda/dataset-audio-quality/dataset_audio_quality.json`
- `artifacts/eda/dataset-audio-quality/dataset_audio_quality.md`
- `artifacts/eda/dataset-audio-quality/dataset_audio_quality_rows.jsonl`
- `artifacts/eda/dataset-audio-quality/dataset_audio_quality_flagged_rows.jsonl`

`dataset_audio_quality_rows.jsonl` contains one deduplicated row per active
audio item with derived metrics and `quality_flags`.

`dataset_audio_quality_flagged_rows.jsonl` is the filtered subset where
`quality_flags` is non-empty. This is the handoff artifact for later
preprocessing, quarantine, or ablation tasks.

## Format Support

Waveform-derived metrics are computed for all currently supported input formats:

- `WAV`
- `FLAC`
- `MP3`

The checks reuse the same shared decoder contract as the repository audio
loader, so quality flags are not restricted to WAV-only corpora anymore.

## Notes

- `all/train/dev` overlap is deduplicated by canonical row identity before
  summary aggregation.
- Trial and quarantine manifests are excluded from active profiling.
- Missing files and decode failures still appear in the row artifacts with
  `audio_read_error` or `missing_audio_file` flags so they can be gated before
  normalization or training.
