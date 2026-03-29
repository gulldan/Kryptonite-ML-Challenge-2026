# CAM++ Clean/Corrupted Consistency

`KVA-534` turns the consistency-loss idea into one runnable repository workflow.

The scope is intentionally narrow:

- keep the student family fixed to `CAM++`, because
  `configs/release/final-family-decision.toml` already freezes it as the
  production-student family;
- reuse the checked-in stage-3 recipe as the warm-start anchor instead of
  inventing a parallel student branch;
- make the invariance objective explicit on clean/corrupted pairs of the same
  utterance crop;
- emit built-in clean-dev and robust-dev ablations so the ticket does not rely
  on manual report stitching.

## What It Runs

The checked-in path lives in:

- `configs/training/campp-consistency.toml`
- `scripts/run_campp_consistency.py`
- `src/kryptonite/training/campp/consistency_config.py`
- `src/kryptonite/training/campp/consistency_runtime.py`
- `src/kryptonite/training/campp/consistency_pipeline.py`
- `src/kryptonite/training/campp/consistency_ablation.py`

The pipeline:

1. loads the existing `CAM++ stage-3` config as the baseline contract;
2. warm-starts from a completed `campp_stage3` checkpoint;
3. samples one crop per utterance, keeps the clean crop as the anchor, and
   applies repo-native corruption recipes to a paired copy of that crop;
4. combines four losses:
   clean supervised `ArcMargin` classification, optional corrupted-view
   classification, direct clean/corrupted embedding alignment, and clean-vs-
   corrupted pairwise score-matrix alignment inside the corrupted subset;
5. exports the consistency-tuned checkpoint with the same dev-embedding and
   verification layout as the other CAM++ runs;
6. re-evaluates the original stage-3 checkpoint on the same clean dev contract;
7. re-scores both checkpoints on the frozen corrupted-dev suites and writes one
   ablation report.

## Why Crop Then Corrupt

The repository applies the corruption recipe after the train crop is selected.
That is deliberate: `KVA-534` is about invariance between clean and corrupted
views of the *same utterance content*, not about comparing unrelated random
segments from the same file.

Some corruption families can change the frame count. The runtime trims or pads
the corrupted crop back to the clean-crop length before Fbank extraction so the
student always sees aligned feature tensors.

## Default Config

The checked-in config uses:

- `CAM++ stage-3` as the recipe base
- `bf16`
- micro-batch `16`
- gradient accumulation `2`
- `4` epochs
- `AdamW` with a lower `1e-4` learning rate than stage-3
- slightly corruption-heavier scheduler weights than stage-3 so the run
  actually observes paired corrupted views often enough
- loss weights:
  - clean classification `1.0`
  - corrupted classification `0.5`
  - embedding consistency `0.25`
  - score consistency `0.1`
- robust-dev weighting:
  - clean `0.25`
  - corrupted `0.75`

That keeps the first runnable version conservative: the clean supervised signal
still anchors the update, while the pairwise objective only nudges the model
toward corruption invariance.

## Command

```bash
uv sync --dev --group train --group tracking
uv run python scripts/training_env_smoke.py
uv run python scripts/run_campp_consistency.py \
  --config configs/training/campp-consistency.toml \
  --student-checkpoint /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/baselines/campp-stage3/<run-id> \
  --device cuda
```

## Output Layout

Each run writes under `artifacts/baselines/campp-consistency/<run-id>/`:

- `campp_consistency_encoder.pt`
- `training_summary.json`
- `consistency_summary.json`
- `consistency_schedule.json`
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
- `robust_dev_ablation/`
- `campp_consistency_report.md`
- `reproducibility_snapshot.json`

## Limits

- This workflow still supports only the export-frozen `CAM++` family.
- It assumes the corruption-bank manifests and corrupted-dev suite catalog
  already exist. If the scheduler cannot sample any corrupted pairs, the run
  fails fast instead of pretending the consistency objective was active.
- This remains stretch work. The repository now has a reproducible ablation for
  `KVA-534`, but it still must not block the must-have export/parity chain.
