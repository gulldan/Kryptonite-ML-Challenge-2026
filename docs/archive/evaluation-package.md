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
- thresholded error analysis for false accepts / false rejects, hard examples, domain failures,
  and speaker confusion patterns

## CLI

Use the existing score-evaluation entrypoint when you only need the shared verification report:

```bash
uv run python scripts/evaluate_verification_scores.py \
  --scores artifacts/baselines/eres2netv2/<run-id>/dev_scores.jsonl \
  --trials artifacts/baselines/eres2netv2/<run-id>/dev_trials.jsonl \
  --metadata artifacts/baselines/eres2netv2/<run-id>/dev_embedding_metadata.parquet
```

Use the threshold-calibration entrypoint when you also need named demo/production operating
points:

```bash
uv run python scripts/calibrate_verification_thresholds.py \
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
- `verification_error_analysis.json`
- `verification_error_analysis.md`

Baseline pipelines now emit the same files automatically into each run directory next to
`score_summary.json`.

The dedicated threshold-calibration workflow additionally writes:

- `verification_threshold_calibration.json`
- `verification_threshold_calibration.md`

## Error Analysis

When the score rows carry trial identifiers or you provide `--trials`, the writer also emits a
thresholded error-analysis bundle. By default it uses the global `EER` threshold so false accepts
and false rejects stay comparable for baseline review. The artifact summarizes:

- hardest false accepts
- hardest false rejects
- slice-level/domain failure rows with error rates
- recurrent speaker confusion pairs
- speaker-specific false-reject fragility
- priority weak spots to carry into the next training stage

## Slice Breakdown

The default per-slice report groups trials by:

- `dataset`
- `channel`
- `role_pair`
- `duration_bucket`
- `noise_slice`
- `reverb_slice`
- `rt60_slice`
- `codec_slice`
- `channel_slice`
- `distance_slice`
- `silence_ratio_bucket`
- `silence_slice`

Override or extend this with repeated `--slice-field` flags. The implementation supports pairwise
fields such as `dataset` / `channel`, explicit side selectors such as `left_device`, derived
corruption-aware fields such as `noise_slice`, explicit reverb/codec buckets such as `rt60_slice`
and `codec_slice`, and silence-aware fields such as `silence_ratio_bucket`.

## Calibration Notes

The calibration section fits a lightweight logistic transform over the provided score file only.
That makes it useful for offline diagnostics and relative comparisons between runs, but it is not a
replacement for a held-out calibration protocol or a production scorer.

For named thresholds and deployment-oriented operating points, use the dedicated calibration
workflow documented in [threshold-calibration.md](./threshold-calibration.md).
