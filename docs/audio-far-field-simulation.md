# Audio Far-Field Simulation

## Goal

Build one reproducible far-field simulation bank that later augmentation and evaluation stages can
consume without re-deriving distance presets, off-axis losses, or direct-to-reverb heuristics by
hand.

The authoritative source plan is
[`configs/corruption/far-field-bank.toml`](../configs/corruption/far-field-bank.toml). The
reproducible command is:

```bash
uv run python scripts/build_far_field_bank.py
```

This renders deterministic control artifacts into `artifacts/corruptions/far-field-bank/` with:

- a synthetic source probe under `probe/`
- per-preset preview audio under `previews/`
- normalized impulse-response previews under `kernels/`
- a machine-readable manifest at `manifests/far_field_bank_manifest.jsonl`
- JSON and Markdown summaries under `reports/`

## Scope

This task covers fixed near/mid/far presets for the future distance-robustness augmentation stages:

- distance-driven attenuation
- off-axis high-frequency loss
- room-coupled reflections and late reverb
- low direct-to-reverb ratio for far-field stress conditions

The workflow deliberately uses a synthetic probe waveform so the smoke path stays deterministic and
license-clean while still producing control examples that downstream work can inspect and version.

## Preset Policy

The baseline plan keeps three field presets:

- `near`: close-talking reference with mostly direct energy
- `mid`: meeting-room style capture with moderate room coupling
- `far`: low-direct hall-style capture with stronger reverberant tail

Each preset records the same explicit controls:

- `distance_meters`
- `off_axis_angle_deg`
- `attenuation_db`
- `target_drr_db`
- `reverb_rt60_seconds`
- `lowpass_hz`

Those parameters are the source of truth for later augmentation scheduling. The rendered previews
are inspection artifacts, not the future training sampler itself.

## Signal Model

Each preset is rendered by:

1. generating a synthetic probe waveform with voiced, chirp, and transient content;
2. building a deterministic impulse response with:
   - a physically derived direct-path arrival delay from `distance_meters`,
   - explicit early reflections from the plan,
   - a seeded exponentially decaying late tail scaled to the requested DRR;
3. applying frequency shaping for low-pass distance loss and additional high-shelf attenuation from
   the off-axis angle.

The report records both target and realized DRR, arrival delay, RMS delta, and spectral-rolloff
delta so sanity checks remain lightweight and reproducible.
