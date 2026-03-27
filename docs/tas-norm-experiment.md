# TAS-norm Experiment

## Goal

Run one reproducible `TAS-norm` go/no-go experiment on top of the existing offline
verification stack without silently changing runtime scoring.

## Current Scope

The repository implements a pragmatic, repo-native `TAS` variant:

- keep the cohort bank frozen;
- derive per-side cohort statistics in the same family as `AS-norm`;
- train a lightweight logistic head over:
  - raw score
  - `AS-norm` score
  - mean/std summaries of the left/right cohort statistics
- evaluate the learned head on a deterministic held-out split and emit an explicit
  `go` / `no_go` report.

This is intentionally a controlled experiment, not a full learnable-impostor
embedding reproduction.

## Checked-in Decision Config

Use the checked-in config:

```bash
uv run python scripts/build_tas_norm_experiment.py \
  --config configs/release/tas-norm-experiment.toml
```

The config points at the current repo-smoke `CAM++` artifacts and writes into:

- `artifacts/release-decisions/kryptonite-2026-tas-norm/tas_norm_experiment.json`
- `artifacts/release-decisions/kryptonite-2026-tas-norm/tas_norm_experiment.md`
- `artifacts/release-decisions/kryptonite-2026-tas-norm/tas_norm_model.json`
- `artifacts/release-decisions/kryptonite-2026-tas-norm/verification_scores_as_norm_eval.jsonl`
- `artifacts/release-decisions/kryptonite-2026-tas-norm/verification_scores_tas_norm.jsonl`

## Current Checked-in Result

Running the checked-in config currently produces:

- decision: `no_go`
- eval winner: `as-norm`
- held-out split: `4` trials (`2` positive / `2` negative)
- `raw` eval: `EER=0.0`, `minDCF=0.0`
- `AS-norm` eval: `EER=0.0`, `minDCF=0.0`
- `TAS-norm` eval: `EER=0.5`, `minDCF=0.5`

The `no_go` verdict is driven by two explicit facts:

- the smoke split is far too small to justify promoting a learned score head;
- both `raw` and `AS-norm` already saturate the held-out smoke set, while the current
  trainable head overfits the tiny train split and generalizes worse on eval.

That is still a useful artifact: it records that the next time `TAS-norm` is revisited,
it should happen on a larger verification set, not by silently modifying the current
release path.
