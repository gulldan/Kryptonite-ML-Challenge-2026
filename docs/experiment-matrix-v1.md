# Experiment Matrix v1

`KRYP-005` freezes the first honest experiment matrix for the repository.

The source of truth is
`configs/training/experiment-matrix-v1.toml`, which rebuilds into a
machine-readable report under
`artifacts/planning/kryptonite-2026-experiment-matrix-v1/`.

## Why This Exists

The ticket asked for one minimal launch set covering:

- `CAM++`
- `ERes2NetV2`
- baseline augmentation
- heavy augmentation
- large-margin fine-tuning
- `AS-norm`
- `TAS-norm`
- teacher `PEFT`
- distillation

The repository state on `2026-03-28` changes how that list should be read:

- the repo-native student lane is already reproducible end-to-end;
- offline `AS-norm` / `TAS-norm` workflows are already checked in;
- teacher-heavy work is explicitly de-scoped from the must-have path by
  `KVA-531`, `KVA-533`, `KVA-536`, and the release postmortem.

So `v1` is not a flat backlog dump. It is a sequenced matrix that prioritizes
the student path first, keeps one comparison baseline, makes cheap offline
scoring work explicit, and isolates stretch teacher work instead of letting it
sprawl into the critical path.

## Matrix

| Seq | Experiment | Linear | Priority | State | Budget (GPU-hours) | Expected effect |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | CAM++ baseline augment / stage-1 pretraining | `KVA-519` | `P0` | `ready gpu` | `4-8` | Main student anchor and warm-start checkpoint for the CAM++ lane |
| 2 | ERes2NetV2 comparison baseline | `KVA-513` | `P1` | `ready gpu` | `4-6` | Side-by-side compact family comparison against CAM++ |
| 3 | CAM++ heavy multi-condition training | `KVA-520` | `P0` | `ready gpu` | `12-18` | Main robustness gain on corrupted slices |
| 4 | CAM++ large-margin fine-tuning | `KVA-521` | `P0` | `ready gpu` | `6-10` | Recover target-like verification quality after stage-2 |
| 5 | `AS-norm` baseline | `KVA-527` | `P1` | `ready offline` | `0-0.5` | Cheap normalized-score uplift on top of the selected student artifacts |
| 6 | `TAS-norm` go/no-go experiment | `KVA-528` | `P2` | `ready offline` | `0.5-1.5` | Explicit test for a learned score head on a larger verification split |
| 7 | WavLM / w2v-BERT teacher in `PEFT` mode | `KVA-531` | `P3` | `deferred stretch` | `20-32` | Potential upside source for later teacher/student work |
| 8 | Distillation into a compact student | `KVA-533` | `P3` | `deferred stretch` | `12-20` | Potential student gain after the teacher path is proven |

## Budget Reading

- Ready-now lane (`P0`-`P2`): `26.5-44.0 GPU-hours`
- Deferred stretch lane (`P3`): `32-52 GPU-hours`
- Full matrix: `58.5-96.0 GPU-hours`

These are planning estimates for one `24 GB RTX 4090`-class GPU on
`gpu-server`, not measured runtime benchmarks. They are derived from the
checked-in epoch counts, crop lengths, augmentation intensity, and whether the
row is encoder training versus offline scoring.

## Recommended Sequence

1. Run the CAM++ student lane first: `KVA-519 -> KVA-520 -> KVA-521`.
2. Keep one comparison anchor via `KVA-513`, but do not branch the main path
   around it unless it clearly wins on the same manifest contract.
3. Apply `AS-norm` only after a concrete student artifact exists.
4. Re-run `TAS-norm` only on a larger verification split; the current checked-in
   smoke result is already `no_go`.
5. Keep `KVA-531` and `KVA-533` out of the must-have lane until the student
   family and export path are both frozen.

## Evidence

- Student baselines and staged CAM++ training:
  `docs/campp-baseline.md`, `docs/campp-stage2-training.md`,
  `docs/campp-stage3-training.md`, `docs/eres2netv2-baseline.md`
- Offline score normalization:
  `docs/as-norm-baseline.md`, `docs/tas-norm-experiment.md`,
  `docs/cohort-embedding-bank.md`
- Scope control for teacher/distillation:
  `docs/final-family-decision.md`,
  `docs/release-postmortem.md`,
  `configs/release/final-family-decision.toml`,
  `configs/release/release-postmortem-v2.toml`

## Rebuild

```bash
uv run python scripts/build_experiment_matrix.py \
  --config configs/training/experiment-matrix-v1.toml
```
