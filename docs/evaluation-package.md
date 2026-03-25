# Evaluation Package

## Goal

Provide one reusable verification-evaluation package that consumes baseline score artifacts and
emits a full offline report with:

- `EER`
- normalized `minDCF`
- ROC operating points
- DET operating points
- score histograms
- calibration bins from a simple Platt-style logistic fit
- per-slice breakdowns driven by embedding metadata

## CLI

Use the existing score-evaluation entrypoint:

```bash
uv run python scripts/evaluate_verification_scores.py \
  --scores artifacts/baselines/eres2netv2/<run-id>/dev_scores.jsonl \
  --trials artifacts/baselines/eres2netv2/<run-id>/dev_trials.jsonl \
  --metadata artifacts/baselines/eres2netv2/<run-id>/dev_embedding_metadata.parquet
```

By default the CLI writes artifacts into the same directory as the score file. Use
`--output-dir` to redirect them elsewhere.

## Output Contract

The report writer produces these files:

- `verification_eval_report.json`
- `verification_eval_report.md`
- `verification_slice_dashboard.html`
- `verification_roc_curve.jsonl`
- `verification_det_curve.jsonl`
- `verification_calibration_curve.jsonl`
- `verification_score_histogram.json`
- `verification_slice_breakdown.jsonl`

Baseline pipelines now emit the same files automatically into each run directory next to
`score_summary.json`.

## Slice Breakdown

The default per-slice report groups trials by:

- `dataset`
- `channel`
- `role_pair`
- `duration_bucket`
- `noise_slice`
- `reverb_slice`
- `channel_slice`
- `distance_slice`
- `silence_slice`

Override or extend this with repeated `--slice-field` flags. The implementation supports pairwise
fields such as `dataset` / `channel`, explicit side selectors such as `left_device`, derived
corruption-aware fields such as `noise_slice`, and `duration_bucket`.

## Calibration Notes

The calibration section fits a lightweight logistic transform over the provided score file only.
That makes it useful for offline diagnostics and relative comparisons between runs, but it is not a
replacement for a held-out calibration protocol or a production scorer. The dedicated scoring /
calibration backend task can later plug in stricter train-calibrate-eval splits on top of this
artifact contract.
