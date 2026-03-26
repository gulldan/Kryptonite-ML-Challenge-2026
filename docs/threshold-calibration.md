# Threshold Calibration

## Goal

Freeze reproducible operating points on top of one verification score file so downstream demo and
serving work can reference named thresholds instead of ad hoc score cutoffs.

The calibration flow now emits four global profiles:

- `balanced`: the threshold closest to the offline `EER` crossing
- `min_dcf`: the threshold minimizing normalized detection cost for the selected `p_target`
- `demo`: a permissive false-accept budget tuned for interactive demos
- `production`: a stricter false-accept budget for safer deployment defaults

Optional slice-aware thresholds can also be emitted for specific metadata views such as
`channel`, `duration_bucket`, or corruption-derived slices when there is enough support.

## CLI

Use the dedicated calibration entrypoint:

```bash
uv run python scripts/calibrate_verification_thresholds.py \
  --scores artifacts/baselines/campp/<run-id>/dev_scores.jsonl \
  --trials artifacts/baselines/campp/<run-id>/dev_trials.jsonl \
  --metadata artifacts/baselines/campp/<run-id>/dev_embedding_metadata.parquet \
  --threshold-slice-field channel \
  --threshold-slice-field duration_bucket
```

The command also writes the shared evaluation-package artifacts, including ROC/DET curves and the
existing markdown/json report bundle.

## Default Budgets

The default named budgets are intentionally simple and easy to override:

- `demo`: `FAR <= 0.05`
- `production`: `FAR <= 0.01`

These defaults favor:

- fewer false rejects during live demos, where operator supervision is available
- stricter false-accept control for production-like verification decisions

Override them with `--demo-target-far` and `--production-target-far` if a run needs different
tradeoffs.

## Output Contract

The calibration writer adds:

- `verification_threshold_calibration.json`
- `verification_threshold_calibration.md`

The JSON artifact is the machine-readable handoff contract for later backend/demo tasks. It
contains:

- the global profile definitions that were used
- the selected global thresholds and realized FAR/FRR/TAR/TRR
- optional slice-aware threshold groups with the same named profiles

## Slice-Aware Thresholds

Slice-aware thresholds are disabled by default. Enable them with repeated
`--threshold-slice-field` flags and set a realistic support floor with `--min-slice-trials`.

The writer skips slice groups that:

- do not meet the minimum trial count
- lack either positive or negative examples

This keeps the bundle from pretending that tiny slices are calibrated when they are not.

## Notes

- The calibration operates on the provided score file only; it does not create a separate
  train/calibrate/eval split by itself.
- If `AS-norm` is enabled, the command calibrates the normalized score file and keeps the raw score
  path in the evaluation bundle for traceability.
- The later serving layer can consume either the global profiles or the slice-aware profiles from
  the JSON bundle without re-running offline calibration logic.
