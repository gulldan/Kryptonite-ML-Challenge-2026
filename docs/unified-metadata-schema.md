# Unified Metadata Schema

## Goal

Give every active data manifest a single versioned row contract so downstream preprocessing,
EDA, split audits, and loaders stop guessing field names.

The current contract version is:

```text
kryptonite.manifest.v1
```

## Canonical Row Shape

Every data-manifest row must carry:

- `schema_version`: exact manifest schema version string
- `record_type`: currently `utterance`
- `dataset`: logical manifest dataset name such as `ffsvc2022-surrogate`
- `source_dataset`: upstream dataset/source identifier
- `speaker_id`: speaker identifier used for split and verification logic
- `audio_path`: project-relative path to the audio file

Standardized optional fields:

- `utterance_id`
- `session_id`
- `split`
- `role`
- `language`
- `device`
- `channel`
- `snr_db`
- `rir_id`
- `duration_seconds`
- `sample_rate_hz`
- `num_channels`

Dataset-specific audit fields may stay at the top level as extra keys, but they must not override
the canonical keys above.

## Compatibility Rules

- Legacy aliases are still readable by internal consumers:
  - `session_index` is normalized into `session_id`
  - `channels` is normalized into `num_channels`
  - `snr` is normalized into `snr_db`
- New manifest writers must emit canonical field names, not the legacy aliases.
- Trial-only JSONL files are not validated against the utterance schema because they use a
  different contract (`label`, `left_audio`, `right_audio`).

## Validation

Run the repository validator from the project root:

```bash
uv run python scripts/validate_manifests.py
```

Use `--no-strict` if you want the JSON report without failing the command:

```bash
uv run python scripts/validate_manifests.py --no-strict
```

The validator:

- scans `artifacts/manifests/` (or the configured manifests root)
- skips trial-only JSONL files
- rejects rows with missing `schema_version`, wrong `record_type`, or missing canonical fields
- reports row-level field errors with manifest path and line number

## Current Producers

- `scripts/prepare_ffsvc2022_surrogate.py`
- `scripts/generate_demo_artifacts.py`

Both now emit `kryptonite.manifest.v1` rows.

For reproducibility and handoff:

- each generated manifest JSONL now gets a deterministic CSV sidecar next to it
- manifest producers write `duration_seconds`, `sample_rate_hz`, and `num_channels` when that
  audio metadata is known at generation time
- each manifest bundle also writes a deterministic inventory JSON with relative paths, row counts,
  speaker counts, and SHA-256 checksums for the generated manifest/list files
