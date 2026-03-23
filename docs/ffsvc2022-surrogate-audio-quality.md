# FFSVC 2022 Surrogate Audio Quality

## Scope

This note curates the first full audio-quality EDA run for the active
`ffsvc2022-surrogate` manifests. The full machine-readable report lives under
`artifacts/eda/dataset-audio-quality/`; this document captures the decisions the
preprocessing and training pipeline should inherit from that run.

Report provenance:

- generated on `gpu-server`
- generated at `2026-03-23T16:53:42+00:00`
- source manifests:
  - `artifacts/manifests/ffsvc2022-surrogate/train_manifest.jsonl`
  - `artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl`
  - `artifacts/manifests/ffsvc2022-surrogate/all_manifest.jsonl`
  - `artifacts/manifests/demo_manifest.jsonl`

## How To Reproduce

Run this on `gpu-server` from `/mnt/storage/Kryptonite-ML-Challenge-2026`:

```bash
PYTHONPATH=src .venv/bin/python scripts/dataset_audio_quality_report.py
```

Artifacts:

- `artifacts/eda/dataset-audio-quality/dataset_audio_quality.json`
- `artifacts/eda/dataset-audio-quality/dataset_audio_quality.md`

## What Was Analyzed

- 68,547 deduplicated active rows
- 68,541 `ffsvc2022-surrogate` rows plus 6 demo rows
- 68,547/68,547 rows had waveform-derived metrics
- `all/train/dev` overlap was deduplicated, collapsing 68,541 redundant rows from the aggregate manifests

## Key Findings

### 1. Resampling is mandatory, not optional

Observed sample-rate mix on active rows:

- `16000`: 35,234 rows
- `44100`: 27,107 rows
- `48000`: 6,206 rows

Implication:

- preprocessing must explicitly resample to `16 kHz`
- resampling behavior cannot be left to implicit backend defaults
- any feature extraction benchmark must log pre/post resample assumptions

### 2. Silence-heavy recordings dominate the surrogate

Silence findings on active rows:

- 51,312 rows (`74.9%`) have `high_silence_ratio` (`>= 50%` silent 100 ms windows)
- 17,037 rows (`24.9%`) have `moderate_silence_ratio` (`20-50%` silent windows)
- combined, 68,349 rows (`99.7%`) have at least `20%` silence
- silence ratio `p95` is `100%`

Split view:

- `train`: silence mean `66.5%`, `p95=100%`
- `dev`: silence mean `69.3%`, `p95=100%`
- `demo`: silence mean `0%`

Implication:

- VAD / trimming should exist as a first-class preprocessing mode
- ablations should compare `no VAD`, `light trimming`, and `aggressive trimming`
- augmentation policy should assume large leading/trailing silence is normal in the surrogate

### 3. Loudness is extremely low almost everywhere

Loudness findings on active rows:

- mean loudness: `-44.38 dBFS`
- 64,198 rows (`93.7%`) are `very_low_loudness` (`<= -35 dBFS`)
- 3,745 rows (`5.5%`) are `low_loudness` (`-35:-30 dBFS`)
- combined, 67,943 rows (`99.1%`) are below `-30 dBFS`

Split view:

- `train`: mean `-44.33 dBFS`
- `dev`: mean `-44.61 dBFS`
- `demo`: mean `-12.13 dBFS`

Implication:

- loudness normalization or bounded gain staging should be part of the preprocessing contract
- feature extraction should be checked against quiet-input failure modes
- demo audio is much hotter than the surrogate, so demo-only smoke checks are not representative here

### 4. Channel format is already stable

Observed channels:

- `1`: 68,547 rows

Implication:

- mono fold-down logic is still worth keeping in the loader for future datasets
- `ffsvc2022-surrogate` itself does not currently force a multi-channel policy decision

### 5. Duration tail exists, but it is not the main pain point

Observed duration stats:

- total duration: `221,010.99 s`
- `train` median: `2.98 s`, `p95=5.44 s`, max `12.62 s`
- `dev` median: `3.10 s`, `p95=5.61 s`, max `13.16 s`
- global histogram mass sits almost entirely in `2-5 s`

Implication:

- chunking/truncation policy should still be explicit
- the bigger risk is silence and level normalization, not runaway long files

## Representative Examples

Examples from the generated report consistently show the same cluster:

- non-16k sample rate
- very low loudness
- high silence ratio

Typical examples:

- `datasets/ffsvc2022-surrogate/raw/dev/dev/ffsvc22_dev_000001.wav`
  - `44.1 kHz`, `-44.74 dBFS`, `66.7%` silence
- `datasets/ffsvc2022-surrogate/raw/dev/dev/ffsvc22_dev_000006.wav`
  - `48 kHz`, `-53.84 dBFS`, `100%` silence
- `datasets/ffsvc2022-surrogate/raw/dev/dev/ffsvc22_dev_000011.wav`
  - `44.1 kHz`, `-47.39 dBFS`, `80.8%` silence

## Pipeline Consequences

The current surrogate strongly argues for the following defaults in upcoming pipeline work:

1. Always resample to `16 kHz` before feature extraction.
2. Keep preprocessing configurable, but implement VAD / trimming early because the silence tail is overwhelming.
3. Add loudness normalization or bounded gain control before robust training experiments.
4. Track silence ratio and loudness-derived quality flags as explicit artifacts, not just one-off EDA observations.
5. Treat demo-only checks as insufficient proxies for surrogate training data quality.

## Limits

- Silence is estimated on `100 ms` windows using a `-45 dBFS` RMS threshold.
- The current report is waveform-derived for WAV inputs; if future manifests introduce other formats, non-WAV rows will need a compatible analyzer.
- The current implementation computes PCM statistics in-repo and keeps the quality report independent from external DSP/runtime packages.
