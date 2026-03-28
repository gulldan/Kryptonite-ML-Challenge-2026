# EPIC-04 Corruption Bank Closeout

`KVA-471` is the closeout note for the repository block that assembles the
reusable corruption banks, the training-side augmentation curriculum, and the
fixed corrupted-dev evaluation bundles.

The child tasks already provide the low-level builders and docs. This note
exists to make the full block auditable as one entrypoint instead of forcing the
next phase to reconstruct the corruption stack from several independent
documents, plans, and test modules.

## Scope

`EPIC-04 - Corruption bank и robust augmentation` covers:

- `KVA-505 / KRYP-027`: reproducible additive noise bank
- `KVA-506 / KRYP-028`: room impulse response bank and derived room presets
- `KVA-507 / KRYP-029`: deterministic codec/channel simulation bank
- `KVA-508 / KRYP-030`: deterministic far-field distance simulation bank
- `KVA-509 / KRYP-031`: silence and pause robustness augmentation primitive
- `KVA-510 / KRYP-032`: training-side augmentation scheduler and coverage report
- `KVA-511 / KRYP-033`: deterministic corrupted dev suites for evaluation

## Deliverables

- [docs/audio-noise-bank.md](./audio-noise-bank.md)
  Approved additive-noise sources, normalization policy, severity buckets, and
  the manifest/report contract rooted in
  `configs/corruption/noise-bank.toml`.
- [docs/audio-rir-bank.md](./audio-rir-bank.md)
  Reusable RIR bank with room-size, field, RT60, and direct-condition buckets,
  plus derived room-simulation configs rooted in
  `configs/corruption/rir-bank.toml`.
- [docs/audio-codec-simulation.md](./audio-codec-simulation.md)
  Deterministic FFmpeg-based codec/channel preset catalog with reproducibility
  metadata and preview hashes rooted in
  `configs/corruption/codec-bank.toml`.
- [docs/audio-far-field-simulation.md](./audio-far-field-simulation.md)
  Deterministic near/mid/far distance simulation presets with explicit DRR,
  attenuation, low-pass, and off-axis controls rooted in
  `configs/corruption/far-field-bank.toml`.
- [docs/audio-silence-augmentation.md](./audio-silence-augmentation.md)
  Waveform-level silence/pause robustness primitive with explicit config knobs,
  audit fields, and ablation reporting.
- [docs/audio-augmentation-scheduler.md](./audio-augmentation-scheduler.md)
  One curriculum scheduler over the corruption banks with warmup/ramp/steady
  stages, manifest coverage accounting, and per-epoch reporting.
- [docs/audio-corrupted-dev-suites.md](./audio-corrupted-dev-suites.md)
  Frozen `dev_snr`, `dev_reverb`, `dev_codec`, `dev_distance`, `dev_channel`,
  and `dev_silence` bundles rooted in
  `configs/corruption/corrupted-dev-suites.toml`.

## What This Unlocks

- `EPIC-05` and `EPIC-06`: baselines and staged training now build against one
  explicit corruption stack instead of ad hoc augmentation choices embedded in
  each recipe.
- `EPIC-07`: scoring, calibration, and enrollment evaluation can compare clean
  versus corrupted behavior on fixed stress suites rather than resampled
  perturbations.
- `EPIC-09` and `EPIC-10`: export and demo work inherit deterministic corrupted
  bundles for parity, smoke, and regression checks.

## Validation

Fast builder smoke for the bank-level artifacts:

```bash
uv run python scripts/build_noise_bank.py
uv run python scripts/build_rir_bank.py
uv run python scripts/build_codec_bank.py
uv run python scripts/build_far_field_bank.py
uv run python scripts/augmentation_scheduler_report.py --config configs/base.toml --epochs 10 --samples-per-epoch 512
```

Data-backed augmentation and evaluation bundle flow:

```bash
uv run python scripts/silence_augmentation_report.py \
  --config configs/base.toml \
  --override silence_augmentation.enabled=true \
  --override silence_augmentation.max_leading_padding_seconds=0.15 \
  --override silence_augmentation.max_trailing_padding_seconds=0.20 \
  --override silence_augmentation.max_inserted_pauses=2 \
  --override silence_augmentation.pause_ratio_min=0.9 \
  --override silence_augmentation.pause_ratio_max=1.4 \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl \
  --output-dir artifacts/eda/silence-augmentation
uv run python scripts/build_corrupted_dev_suites.py --config configs/base.toml --plan configs/corruption/corrupted-dev-suites.toml
```

Repo-local smoke coverage for the full EPIC-04 contract:

```bash
uv run pytest \
  tests/unit/test_noise_bank.py \
  tests/unit/test_rir_bank.py \
  tests/unit/test_codec_bank.py \
  tests/unit/test_far_field_bank.py \
  tests/unit/test_silence_augmentation.py \
  tests/unit/test_augmentation_scheduler.py \
  tests/unit/test_corrupted_dev_suites.py \
  tests/unit/test_epic_04_corruption_bank_docs.py
```

## Remaining Risks

- The noise and RIR banks intentionally tolerate missing corpora so local smoke
  and CI stay deterministic before the approved datasets are materialized under
  `datasets/`; full coverage still depends on the real corpora being present.
- The codec bank and corrupted-dev suite builders depend on a working `ffmpeg`
  installation, so those test paths may skip on lean environments even when the
  Python-side contract remains valid.
- The scheduler closes the orchestration gap, but the final model-quality impact
  still depends on the downstream training recipes that consume these manifests
  and severity policies.
