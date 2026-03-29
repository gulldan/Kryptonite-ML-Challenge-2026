# Audio RIR Bank

## Goal

Build one reproducible room-impulse-response bank that later far-field, room, and reverberation
augmentation tasks can consume without guessing source structure, room-size buckets, or RT60/DRR
coverage.

The authoritative source plan is [`configs/corruption/rir-bank.toml`](../configs/corruption/rir-bank.toml).
The reproducible command is:

```bash
uv run python scripts/build_rir_bank.py
```

This assembles approved OpenSLR SLR28 RIR sources into `artifacts/corruptions/rir-bank/` with:

- normalized mono audio under `audio/`
- a machine-readable RIR manifest at `manifests/rir_bank_manifest.jsonl`
- aggregated room presets at `manifests/room_simulation_configs.jsonl`
- quarantine rows at `manifests/rir_bank_quarantine.jsonl`
- JSON and Markdown summaries under `reports/`

## Scope

This task covers room impulse responses and derived room presets for the future room-simulation
stages:

- room size buckets: `small`, `medium`, `large`
- field buckets: `near`, `mid`, `far`
- reverberation buckets: `short`, `medium`, `long`
- direct-to-reverb conditions: `high`, `medium`, `low`

The current plan is intentionally limited to the approved OpenSLR SLR28 corpus from the dataset
inventory. Additive noise handling remains in the separate
[noise-bank workflow](./audio-noise-bank.md).

## Classification Policy

The plan currently treats:

- OpenSLR real RIR folders as the `real` room family
- OpenSLR simulated RIR folders as the `simulated` room family
- path tokens like `smallroom` and `largeroom` as explicit room-size hints

Field coverage is inferred from estimated direct-to-reverberant ratio unless a path rule overrides
it:

- `near`: `DRR >= 6 dB`
- `mid`: `0 dB <= DRR < 6 dB`
- `far`: `DRR < 0 dB`

RT60 buckets are derived from the normalized tail estimate:

- `short`: `< 0.45 s`
- `medium`: `0.45-0.9 s`
- `long`: `>= 0.9 s`

The generated room config manifest groups matching RIR entries by
`room_size x field x rt60_bucket x direct_condition` so downstream augmentation code can sample a
stable preset instead of rebuilding these slices ad hoc.

## Sanity Checks

The Markdown report includes both:

- statistical coverage tables for room sizes, fields, RT60 buckets, and direct conditions
- an ASCII envelope preview for representative `near`, `mid`, and `far` examples

This keeps sanity checks lightweight enough for CI and local smoke runs while still surfacing
obvious problems in the RIR tail shape or bucket distribution.

## Missing Sources

The command does not fail when approved corpora are absent. Instead, it writes the report with
explicit `missing` source roots so local smoke runs and CI can stay deterministic before SLR28 is
materialized under `datasets/`.

Once the corpus is present under `datasets/rirs_noises/`, rerun the command and the manifest plus
room-config presets will populate without changing the code path.
