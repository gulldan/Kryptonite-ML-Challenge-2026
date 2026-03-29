# CAM++ Distillation

`KVA-533` turns teacher-guided distillation from planning text into a runnable
repository workflow.

The scope is intentionally narrow:

- keep the student family fixed to `CAM++`, because
  `configs/release/final-family-decision.toml` already freezes it as the
  production-student family;
- reuse the checked-in stage-3 schedule as the warm-start anchor instead of
  inventing a second student recipe;
- keep the teacher frozen and use it only as supervision, not as another train
  branch that can sprawl into the critical path;
- emit a built-in baseline-vs-distilled comparison so the ticket does not rely
  on manual report stitching.

## What It Runs

The checked-in path lives in:

- `configs/training/campp-distillation.toml`
- `scripts/run_campp_distillation.py`
- `src/kryptonite/training/campp/distillation_config.py`
- `src/kryptonite/training/campp/distillation_runtime.py`
- `src/kryptonite/training/campp/distillation_pipeline.py`

The pipeline:

1. loads the existing `CAM++ stage-3` config as the student baseline contract;
2. warm-starts the student from a completed `campp_stage3` checkpoint;
3. loads one frozen teacher checkpoint produced by `docs/teacher-peft.md`;
4. feeds the same waveform crop into both branches:
   the student sees repo-native Fbanks, the teacher sees the PEFT feature
   extractor inputs;
5. combines three losses:
   supervised `ArcMargin` classification, direct embedding alignment, and
   pairwise cosine-score alignment inside the batch;
6. exports the distilled student artifacts with the same dev-embedding and
   verification layout as the other CAM++ runs;
7. re-evaluates the undistilled student checkpoint on the same dev manifest and
   writes a comparison report.

## Default Config

The checked-in config uses:

- `CAM++ stage-3` as the recipe base
- `bf16`
- micro-batch `16`
- gradient accumulation `2`
- `4` epochs
- `AdamW` with a lower `1e-4` learning rate than the base stage-3 fine-tuning
- loss weights:
  - classification `1.0`
  - embedding alignment `0.35`
  - score alignment `0.15`

That keeps the first runnable version conservative: the teacher nudges the
student, but the supervised student objective still anchors the update.

## Command

```bash
uv sync --dev --group train --group tracking
uv run python scripts/training_env_smoke.py
uv run python scripts/run_campp_distillation.py \
  --config configs/training/campp-distillation.toml \
  --student-checkpoint /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/baselines/campp-stage3/<run-id> \
  --teacher-checkpoint /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/baselines/teacher-peft/<run-id> \
  --device cuda
```

## Output Layout

Each run writes under `artifacts/baselines/campp-distillation/<run-id>/`:

- `campp_distilled_encoder.pt`
- `training_summary.json`
- `distillation_summary.json`
- `distillation_schedule.json`
- `dev_embeddings.npz`
- `dev_embedding_metadata.jsonl`
- `dev_embedding_metadata.parquet`
- `dev_trials.jsonl`
- `dev_scores.jsonl`
- `score_summary.json`
- `verification_report.json`
- `verification_report.md`
- `baseline_reference/`
- `baseline_comparison.json`
- `baseline_comparison.md`
- `campp_distillation_report.md`
- `reproducibility_snapshot.json`

The `baseline_reference/` directory is intentionally local to the run so the
comparison always uses the same manifest contract as the distilled checkpoint.

## Limits

- The first repository-native version supports only a `CAM++` student. That is
  deliberate, not an omission: the repo does not freeze `ERes2NetV2` as an
  exportable student family yet.
- Teacher supervision uses embedding alignment plus pairwise score alignment.
  It does not distill teacher classifier logits because that would couple the
  recipe more tightly to the teacher's speaker table and make checkpoint reuse
  brittle.
- This workflow is still a stretch lane. A runnable distillation path is now
  checked in, but it still must not block the must-have export/parity chain.
