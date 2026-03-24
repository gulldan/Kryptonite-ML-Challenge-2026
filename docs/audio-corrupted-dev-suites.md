# Corrupted Dev Suites

## Goal

Freeze a small set of evaluation-ready corrupted dev bundles so every scoring or
ablation pipeline can point at the same deterministic stress sets instead of
re-sampling corruptions ad hoc.

The current canonical suites are:

- `dev_snr`
- `dev_reverb`
- `dev_codec`
- `dev_distance`
- `dev_channel`
- `dev_silence`

## Inputs

The builder consumes:

- a clean source dev manifest
- zero or more source trial manifests to copy into each suite
- the reusable corruption banks and plans already assembled elsewhere in the repo

The versioned plan lives at
[`configs/corruption/corrupted-dev-suites.toml`](../configs/corruption/corrupted-dev-suites.toml).

## Command

```bash
uv run python scripts/build_corrupted_dev_suites.py \
  --config configs/base.toml \
  --plan configs/corruption/corrupted-dev-suites.toml
```

Override bank locations explicitly when running against alternate artifact roots
or smoke fixtures:

```bash
uv run python scripts/build_corrupted_dev_suites.py \
  --plan configs/corruption/corrupted-dev-suites.toml \
  --noise-manifest artifacts/corruptions/noise-bank/manifests/noise_bank_manifest.jsonl \
  --rir-manifest artifacts/corruptions/rir-bank/manifests/rir_bank_manifest.jsonl \
  --room-config-manifest artifacts/corruptions/rir-bank/manifests/room_simulation_configs.jsonl
```

## Output Layout

By default the build lands in `artifacts/eval/corrupted-dev-suites/`:

- one directory per suite, each with:
  - rendered audio under `audio/`
  - `dev_manifest.jsonl` plus CSV sidecar
  - copied trial manifests translated to the suite audio basenames
  - `manifest_inventory.json`
  - `suite_summary.json`
  - `suite_summary.md`
- one root catalog:
  - `corrupted_dev_suites_catalog.json`
  - `corrupted_dev_suites_catalog.md`

## Scope Notes

- The suite seed is fixed in the plan, so per-utterance corruption selection is
  stable across rebuilds.
- `dev_codec` and `dev_channel` are rendered from the codec-bank plan so the
  source transform stays explicit, not reverse-engineered from report text.
- `dev_distance` is rendered from the far-field plan for the same reason.
- `dev_reverb` uses the room-config manifest to choose a room bucket and then a
  deterministic RIR from the compatible `rir_id` pool.
- `dev_silence` derives light/medium/heavy variants from the shared
  `silence_augmentation` envelope in `configs/base.toml`; the builder forces
  `enabled = true` only inside the generated suites.

## Known Limitations

- The builder writes WAV outputs for every suite, even if the clean source
  manifest used another extension.
- The fixed suites are generation artifacts and should stay ignored; commit the
  plan, code, and docs, not the rendered audio payloads.
