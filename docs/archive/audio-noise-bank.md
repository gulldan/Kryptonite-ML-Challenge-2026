# Audio Noise Bank

## Goal

Build one reproducible additive-noise bank that later augmentation tasks can consume without
guessing source structure, severity buckets, or normalization policy.

The authoritative source plan is [`configs/corruption/noise-bank.toml`](../configs/corruption/noise-bank.toml).
The reproducible command is:

```bash
uv run python scripts/build_noise_bank.py
```

This assembles approved additive-noise sources into `artifacts/corruptions/noise-bank/` with:

- normalized mono audio under `audio/`
- a machine-readable manifest at `manifests/noise_bank_manifest.jsonl`
- quarantine rows at `manifests/noise_bank_quarantine.jsonl`
- JSON and Markdown summaries under `reports/`

## Scope

This task only covers additive-noise sources for the future corruption bank:

- `stationary`
- `babble`
- `music`
- `impulsive`
- `low_snr`

RIR handling stays separate and belongs to the room-simulation workstream.

## Classification Policy

The plan currently treats:

- `MUSAN speech` as `babble`
- `MUSAN music` as `music`
- `MUSAN noise` as `stationary` by default, with keyword overrides for `impulsive` and `low_snr`
- OpenSLR SLR28 point-source noises as `low_snr`
- OpenSLR SLR28 isotropic noises as `stationary`

Each row also receives a severity bucket with a recommended SNR range:

- `light`: `15-25 dB`
- `medium`: `8-15 dB`
- `heavy`: `0-8 dB`

These severity values are sampling guidance for later augmentation steps. They do not claim that a
source file is intrinsically "light" or "heavy" in isolation.

## Missing Sources

The command does not fail when approved corpora are absent. Instead, it writes the report with
explicit `missing` source roots so local smoke runs and CI can stay deterministic before MUSAN or
SLR28 are materialized.

Once the corpora are present under `datasets/`, rerun the command and the manifest will populate
without changing the code path.
