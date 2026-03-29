# AS-norm Baseline

## Goal

Provide one reproducible `AS-norm` path for offline verification evaluation on top of:

- an existing raw verification `scores.jsonl`
- the aligned eval embedding export (`.npz` + metadata)
- a frozen cohort bank produced by [`docs/cohort-embedding-bank.md`](cohort-embedding-bank.md)

## Current Scope

The baseline is intentionally offline-first:

- it runs from `scripts/evaluate_verification_scores.py`
- it rewrites a normalized score file before metrics/report generation
- it records a small normalization summary JSON next to the report artifacts

The formula is the standard symmetric `AS-norm` variant:

```text
score_as_norm = 0.5 * (
  (raw_score - mean_left_topk) / std_left_topk +
  (raw_score - mean_right_topk) / std_right_topk
)
```

Per-side cohort statistics come from the top-`k` cosine scores against the frozen cohort bank.

## Usage

```bash
uv run python scripts/evaluate_verification_scores.py \
  --scores artifacts/baselines/campp/<run-id>/dev_scores.jsonl \
  --trials artifacts/baselines/campp/<run-id>/dev_trials.jsonl \
  --metadata artifacts/baselines/campp/<run-id>/dev_embedding_metadata.parquet \
  --embeddings artifacts/baselines/campp/<run-id>/dev_embeddings.npz \
  --cohort-bank artifacts/eval/cohort-bank/campp-<run-id> \
  --score-normalization as-norm \
  --as-norm-top-k 100 \
  --output-dir artifacts/eval/as-norm/campp-<run-id>
```

This writes:

- `verification_scores_as_norm.jsonl`
- `verification_score_normalization_summary.json`
- the normal verification report bundle (`verification_eval_report.*`, ROC/DET curves, slice dashboard, error analysis)

## Guardrails

- AS-norm requires resolvable `left_id` / `right_id` identifiers in the score rows.
- Eval embedding metadata must align with the exported embedding matrix.
- Cohort-score standard deviations are floored with a small epsilon to keep tiny smoke runs numerically stable.

If AS-norm does not help a slice, keep the report and summary anyway. The task outcome is still useful because it makes the comparison explicit and reproducible.
