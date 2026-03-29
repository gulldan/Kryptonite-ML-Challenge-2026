# Audio Augmentation Scheduler

`KRYP-032` adds one training-side curriculum scheduler that mixes clean, light,
medium, and heavy corruption instead of treating every augmentation bank as a
flat pool.

The implementation lives in `src/kryptonite/training/augmentation_scheduler/`
and is meant to be the orchestration layer over the already-built primitives:

- additive noise bank
- room / reverb configs
- far-field distance presets
- codec / channel presets
- silence / pause robustness transform

## Policy

The scheduler uses three stages:

1. `warmup`
   - defaults to the first `warmup_epochs`
   - keeps a high clean ratio
   - caps recipes at one corruption per sample
   - downweights codec, far-field, and strong reverb
2. `ramp`
   - defaults to the next `ramp_epochs`
   - linearly moves from the start ratios to the end ratios
   - allows up to two corruptions per sample
3. `steady`
   - covers the remaining epochs
   - holds the late-epoch clean/light/medium/heavy mix
   - keeps all configured families available

The base config lives in `configs/base.toml`:

```toml
[augmentation_scheduler]
enabled = true
warmup_epochs = 2
ramp_epochs = 3
max_augmentations_per_sample = 2
clean_probability_start = 0.70
clean_probability_end = 0.25
light_probability_start = 0.25
light_probability_end = 0.30
medium_probability_start = 0.05
medium_probability_end = 0.25
heavy_probability_start = 0.0
heavy_probability_end = 0.20

[augmentation_scheduler.family_weights]
noise = 1.0
reverb = 1.0
distance = 0.9
codec = 0.8
silence = 0.6
```

The start and end probabilities must each sum to `1.0`.

## Manifest Inputs

By default the report tool reads these artifacts when they exist:

- `artifacts/corruptions/noise-bank/manifests/noise_bank_manifest.jsonl`
- `artifacts/corruptions/rir-bank/manifests/room_simulation_configs.jsonl`
- fallback: `artifacts/corruptions/rir-bank*/manifests/room_simulation_configs.jsonl`
- `artifacts/corruptions/far-field-bank/manifests/far_field_bank_manifest.jsonl`
- `artifacts/corruptions/codec-bank/manifests/codec_bank_manifest.jsonl`

Families with missing or empty manifests stay visible in the report as missing
coverage instead of silently pretending to be active.

## Silence Integration

The scheduler does not invent a parallel silence config. It reuses the existing
`[silence_augmentation]` block as the maximum envelope and derives:

- `light` silence recipes by scaling the padding / pause limits down
- `medium` recipes as the bridge profile
- `heavy` recipes from the full configured envelope

The standalone silence ablation remains the waveform-level check; the scheduler
is only responsible for curriculum orchestration and coverage accounting.

## CLI

Generate a coverage report with:

```bash
uv run python scripts/augmentation_scheduler_report.py \
  --config configs/base.toml \
  --epochs 10 \
  --samples-per-epoch 512
```

Optional manifest overrides are available when a bank lives outside the default
artifact roots:

```bash
uv run python scripts/augmentation_scheduler_report.py \
  --room-config-manifest artifacts/corruptions/rir-bank-smoke/manifests/room_simulation_configs.jsonl
```

## Outputs

The default output directory is `artifacts/reports/augmentation-scheduler/` and
contains:

- `augmentation_scheduler_report.json`
- `augmentation_scheduler_report.md`
- `augmentation_scheduler_epochs.jsonl`

The JSON and Markdown summaries surface:

- candidate availability per family
- missing families
- overall clean/light/medium/heavy counts
- overall family/severity counts
- per-epoch target ratios versus sampled empirical ratios
- top sampled augmentation labels per epoch

This is the intended smoke boundary before the next step wires the scheduler
directly into a production dataloader or training recipe.
