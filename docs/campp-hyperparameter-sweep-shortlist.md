# CAM++ Stage-3 Hyperparameter Sweep Shortlist

**Ticket:** KRYP-045  
**Depends on:** KRYP-042 / KRYP-043 artifacts and frozen corrupted dev suites

## Goal

Keep hyperparameter exploration bounded and reproducible instead of turning the
next training step into an open-ended grid search.

The checked-in shortlist runner evaluates a small set of stage-3 candidates
across these decision axes:

- margin schedule
- crop length / evaluation chunking
- micro-batch size
- augmentation severity mix
- eval pooling

The winner is chosen by a **robust-dev objective**, not by clean-dev metrics
alone.

## Components

| File | Role |
|------|------|
| `src/kryptonite/training/campp/sweep_shortlist_config.py` | Typed TOML loader for the shortlist contract |
| `src/kryptonite/training/campp/sweep_shortlist.py` | Candidate orchestration, robust-suite eval, leaderboard writing |
| `configs/training/campp-stage3-sweep-shortlist.toml` | Checked-in bounded candidate set and ranking weights |
| `scripts/run_campp_sweep_shortlist.py` | CLI entry point |

## Ranking Contract

The checked-in config uses:

- clean-dev weight: `0.25`
- corrupted-suite weight: `0.75`
- EER weight inside the final score: `0.70`
- minDCF weight inside the final score: `0.30`

For each candidate:

1. run CAM++ stage-3 training from the shared stage-2 checkpoint
2. keep the normal clean-dev report from the stage-3 pipeline
3. re-export embeddings and re-score every frozen corrupted dev suite
4. average the corrupted-suite metrics
5. compute:

```text
selection_score =
  0.70 * weighted_eer +
  0.30 * weighted_min_dcf
```

where `weighted_*` is the clean/corrupted weighted combination above.

This keeps clean-dev visible, but it cannot dominate the decision.

## Running

### Production shortlist on `gpu-server`

```bash
# On gpu-server: /mnt/storage/Kryptonite-ML-Challenge-2026
uv run python scripts/run_campp_sweep_shortlist.py \
    --config configs/training/campp-stage3-sweep-shortlist.toml \
    --stage2-checkpoint /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/baselines/campp-stage2/<run-id>
```

### Run only a subset first

```bash
uv run python scripts/run_campp_sweep_shortlist.py \
    --config configs/training/campp-stage3-sweep-shortlist.toml \
    --stage2-checkpoint /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/baselines/campp-stage2/<run-id> \
    --candidate balanced_mean \
    --candidate robust_5p5
```

### Smoke / structural validation

Use a tiny local shortlist config plus `--device cpu` in unit tests or ad hoc
tmp manifests. The repository test suite already covers this path end-to-end.

## Config Notes

`configs/training/campp-stage3-sweep-shortlist.toml` keeps the shortlist bounded
to six candidates and documents the budget explicitly. Each candidate reuses the
repo-native override style:

- `project_overrides` for base-project knobs such as batch size, eval chunking,
  pooling, and augmentation probabilities
- structured `margin_schedule` override for stage-3 margin changes
- structured `crop_curriculum` override for stage-3 crop changes

The runner also redirects every candidate into a sweep-specific artifact root so
normal single-run stage-3 outputs stay separate from shortlist experiments.

## Output Contract

One shortlist run writes:

```text
artifacts/sweeps/campp-stage3-shortlist/<run_id>/
├── campp_stage3_sweep_shortlist_report.json
├── campp_stage3_sweep_shortlist_report.md
└── runs/
    ├── <candidate-id>/<stage3-run-id>/...
    │   ├── campp_stage3_encoder.pt
    │   ├── verification_eval_report.{json,md}
    │   ├── robust_dev/
    │   │   └── <suite-id>/
    │   │       ├── dev_embeddings.npz
    │   │       ├── suite_trials.jsonl
    │   │       ├── score_summary.json
    │   │       └── verification_eval_report.{json,md}
    │   └── campp_stage3_report.md
    └── ...
```

The shortlist report is the handoff artifact for the next task. It includes:

- full candidate leaderboard
- clean vs robust metrics
- per-suite breakdown for every candidate
- explicit winner and ranking formula

## Scope Limits

- The shortlist is intentionally serial and bounded; it is not a generic
  distributed sweeper.
- It assumes frozen corrupted dev suites already exist and are versioned by
  catalog, not regenerated on every candidate run.
- The objective optimizes offline verification quality only. Latency and memory
  stay separate concerns for later model-selection / deployment tasks.
