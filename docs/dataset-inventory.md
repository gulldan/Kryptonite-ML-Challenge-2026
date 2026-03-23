# Dataset Inventory

## Goal

Keep a single repository-level decision record for candidate data resources before they enter
manifests, augmentation banks, or training configs.

The authoritative machine-readable source is
`configs/data-inventory/allowed-sources.toml`. The reproducible report command is:

```bash
uv run python scripts/dataset_inventory_report.py
```

This writes a local inspection report under `artifacts/reports/dataset-inventory/` with:

- the current `approved` / `conditional` / `blocked` policy decision
- the expected scope of each resource (`train`, `dev`, `noise`, `rir`, `codec`, and so on)
- the current local materialization state for the configured paths

Important: these statuses are repository policy decisions for this project. They are not legal
advice and they do not replace upstream terms.

## Decision Rules

- `approved`: the repository can use this resource now under the documented constraints.
- `conditional`: the source is viable only with additional restrictions, overlap checks, or manual
  access steps.
- `blocked`: do not integrate the resource until the missing access or policy gap is resolved.

## Source Matrix

| Source | Scope | Status | Why |
| --- | --- | --- | --- |
| Dataton Kryptonite 2026 raw challenge data | `train`, `dev`, `eval`, `challenge` | `blocked` | The real challenge data is still not public or materialized in the checkout, so train and eval readiness cannot depend on it yet. |
| FFSVC 2022 development and evaluation bundle | `dev`, `surrogate`, `verification` | `conditional` | Good engineering surrogate for pipeline bring-up, but the official page says the development set is not allowed for training. |
| FFSVC 2020 main challenge data | `train`, `external`, `in-domain` | `conditional` | In-domain and useful, but the official path is manual email request and the terms are not yet captured in-repo. |
| FFSVC 2020 supplementary set | `external`, `in-domain`, `augmentation` | `conditional` | Publicly downloadable, but still challenge-context data with overlap risk against other FFSVC splits. |
| VoxCeleb 1 and VoxCeleb 2 | `train`, `external`, `pretraining` | `blocked` | The primary Oxford VGG pages no longer provide the URLs, audio files, or identifying metadata downloads. |
| MUSAN | `noise`, `augmentation` | `approved` | OpenSLR provides a public CC BY 4.0 noise corpus suitable for additive corruption work. |
| OpenSLR Room Impulse Response and Noise Database | `rir`, `noise`, `augmentation` | `approved` | OpenSLR provides a public Apache 2.0 RIR and noise bank that matches the planned room-simulation work. |
| Synthetic codec and channel simulation profiles | `codec`, `augmentation`, `channel` | `approved` | Prefer deterministic FFmpeg-based transforms over introducing another third-party corpus. |

## Datasheets

### Dataton Kryptonite 2026 raw challenge data

- Status: `blocked`
- Reason: the current repository docs still describe the real Dataton data as unavailable.
- References:
  - [FFSVC 2022 surrogate note](./ffsvc2022-surrogate.md)
  - [GPU server data sync note](./gpu-server-data-sync.md)
- Implication: keep the dataset adapter boundary clean and do not claim train readiness until the
  official manifests exist.

### FFSVC 2022 development and evaluation bundle

- Status: `conditional`
- Role: engineering surrogate for manifests, speaker-disjoint splits, trial generation, and scoring.
- Official basis:
  - [FFSVC 2022 dataset page](https://ffsvc.github.io/dataset/)
  - [FFSVC 2022 site](https://ffsvc.github.io/)
- In-repo references:
  - [Acquisition plan](../configs/data-acquisition/ffsvc2022-surrogate.toml)
  - [Surrogate workflow note](./ffsvc2022-surrogate.md)
- Restriction: do not treat the dev set as final challenge-legal training data.

### FFSVC 2020 main challenge data

- Status: `conditional`
- Official basis:
  - [FFSVC 2022 dataset page](https://ffsvc.github.io/dataset/)
- Reason: the official page says the concluded challenge now requires email request for the
  `FFSVC2020` dataset, so access is not yet reproducible from the repo alone.
- Implication: capture the request path, terms, and overlap policy before integration.

### FFSVC 2020 supplementary set

- Status: `conditional`
- Official basis:
  - [FFSVC 2022 dataset page](https://ffsvc.github.io/dataset/)
- Reason: the supplementary set is still publicly downloadable, but it remains challenge-context
  data with non-trivial speaker-overlap risk.
- Implication: require an explicit overlap audit before mixing it with any other FFSVC source.

### VoxCeleb 1 and VoxCeleb 2

- Status: `blocked`
- Official basis:
  - [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
  - [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
  - [VGG dataset privacy notice](https://www.robots.ox.ac.uk/~vgg/terms/url-lists-privacy-notice.html)
- Reason: the primary Oxford VGG pages still expose benchmark metadata and terms, but they no
  longer host the URLs, audio files, or identifying metadata downloads.
- Implication: do not depend on unofficial mirrors.

### MUSAN

- Status: `approved`
- Official basis:
  - [OpenSLR 17](https://www.openslr.org/17/)
- Reason: open and permissive noise corpus that matches the planned additive augmentation work.
- Implication: use it as the default source for the future noise bank.

### OpenSLR Room Impulse Response and Noise Database

- Status: `approved`
- Official basis:
  - [OpenSLR 28](https://www.openslr.org/28/)
- Reason: open RIR and noise bank aligned with the planned room and far-field simulation stages.
- Implication: use it as the default starting point for `RIR` and reverberation experiments.

### Synthetic codec and channel simulation profiles

- Status: `approved`
- Official basis:
  - [FFmpeg legal page](https://ffmpeg.org/legal.html)
- Reason: codec and channel degradations can be generated deterministically from already-approved
  source audio, which is safer than introducing another external corpus by default.
- Implication: log the exact FFmpeg build flags before release if GPL-only codecs or filters are
  enabled.
