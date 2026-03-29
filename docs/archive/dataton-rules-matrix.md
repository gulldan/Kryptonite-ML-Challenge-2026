# Dataton Rules Matrix

## Goal

Keep one competition-facing decision record for what the repo currently treats as:

- `allow`
- `deny`
- `unknown`

with an explicit owner, clarification channel, and risk register for every ambiguous area around
external data, pretrained models, pseudo-labels, teacher/student transfer, post-processing, and
augmentation corpora.

This document is the curated entrypoint. The machine-readable source of truth is
`configs/data-inventory/dataton-rules-matrix.toml`.

## Why This Exists Separately From The Dataset Inventory

`docs/dataset-inventory.md` answers a repo-internal question: which resources are operationally
approved, conditional, or blocked from a provenance and reproducibility standpoint.

This rules matrix answers a different question: what we can responsibly treat as challenge-legal
today, given the public Dataton materials available on `2026-03-28`.

Those two views must stay separate so internal engineering shortcuts do not get misread as final
competition policy.

## Current Working Read

As of `2026-03-28`, the public Dataton site confirms the task and dates, but also says the
evaluation criteria will only be published on `2026-04-11`, the start date of the competition.
That creates a hard distinction between what the repo can engineer and what the final challenge
recipe can safely assume.

Current working decisions:

- `allow`: official Dataton challenge data once released; repo-internal post-processing such as
  scoring, AS-norm, TAS-norm, and threshold calibration; synthetic codec/channel/acoustic
  transforms applied to already-authorized parent audio.
- `deny`: unofficial mirrors or provenance-unclear dataset/checkpoint downloads.
- `unknown` until organizers clarify on `2026-04-11`: external speaker datasets, external
  pretrained checkpoints, pseudo-labeling over official challenge audio, teacher/student transfer
  with an external teacher, and external RIR/noise corpora for the final submission path.

## Reproducible Artifact

Render the current matrix and risk register with:

```bash
uv run python scripts/build_dataton_rules_matrix.py
```

This writes:

```text
artifacts/reports/dataton-rules-matrix/
├── dataton_rules_matrix.json
└── dataton_rules_matrix.md
```

The generated report is the handoff artifact for `KVA-479`. It records:

- the current allow/deny/unknown table
- the official sources reviewed on `2026-03-28`
- repo references backing each decision
- the open-risk register and mitigation steps

## External Sources Reviewed

- [Dataton Kryptonite.Tembr official site](https://dataton-kryptonite.ru/)
- [Official rules PDF link from the Dataton site footer](https://cloud.mail.ru/public/gQL5/cXf5NkPpu)
- [FFSVC dataset page](https://ffsvc.github.io/dataset/)
- [OpenSLR 17 MUSAN](https://www.openslr.org/17/)
- [OpenSLR 28 RIR and Noise Database](https://www.openslr.org/28/)
- [FFmpeg legal overview](https://ffmpeg.org/legal.html)
- [VoxCeleb primary page](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)

## Repo References

- `docs/dataset-inventory.md`
- `docs/clean-room-fallback-baseline.md`
- `docs/ffsvc2022-surrogate.md`
- `docs/audio-noise-bank.md`
- `docs/audio-rir-bank.md`
- `docs/audio-codec-simulation.md`
- `docs/release-postmortem.md`

## Next Review Point

Refresh this matrix immediately after the organizers publish the official criteria on
`2026-04-11`, and propagate the resulting decision changes into the dataset inventory, clean-room
fallback docs, and any active experiment configs that still depend on unknown items.
