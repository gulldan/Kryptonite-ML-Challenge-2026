# CAM++ Stage-3 Model Selection

**Ticket:** KRYP-046  
**Depends on:** KRYP-045 shortlist report and the referenced stage-3 run artifacts

## Goal

Turn the shortlist leaderboard into a reproducible final-candidate decision,
instead of making the last checkpoint choice manually.

The checked-in selector does two things:

- keeps the shortlist ranking objective as the primary decision rule
- optionally evaluates uniform checkpoint averages over the top-ranked,
  checkpoint-compatible candidates

## Why Averaging Instead of EMA

The current stage-3 checkpoint contract stores one final checkpoint per run and
does not carry an EMA shadow model. Because of that, repo-native model
selection can implement **post-hoc checkpoint averaging**, but not EMA
reconstruction.

If EMA becomes necessary later, it has to be added to the training pipeline
itself and written into the run artifacts during training.

## Components

| File | Role |
|------|------|
| `src/kryptonite/training/campp/model_selection_config.py` | Typed TOML loader for the selection step |
| `src/kryptonite/training/campp/model_selection.py` | Winner selection, checkpoint averaging, re-evaluation, final artifact writing |
| `configs/training/campp-stage3-model-selection.toml` | Checked-in selection contract |
| `scripts/run_campp_model_selection.py` | CLI entry point |

## Selection Contract

The selector reads the shortlist report from KRYP-045 and inherits its ranking
weights:

- clean-dev weight
- corrupted-suite weight
- EER weight
- minDCF weight

That means:

1. the raw shortlist winner is always a candidate
2. each configured averaged variant reuses the **same** clean/robust suite set
3. every averaged variant is ranked with the **same** weighted objective as the
   shortlist winner

This keeps KRYP-046 aligned with KRYP-045 instead of silently introducing a new
ranking criterion.

## Averaging Contract

The checked-in config enables:

- `top2_uniform_average`
- `top3_uniform_average`

Each variant:

1. takes the top-N shortlist candidates in rank order
2. checks checkpoint compatibility:
   same `model_config`, same `speaker_to_index`, same tensor shapes
3. averages floating-point tensors uniformly
4. copies non-floating tensors from the best-ranked source checkpoint
5. evaluates the averaged checkpoint with the shortlist winner's eval config

If compatibility fails or the shortlist has too few candidates, the variant is
skipped and the reason is written into the final report.

## Running

### Production run on `gpu-server`

```bash
# On gpu-server: /mnt/storage/Kryptonite-ML-Challenge-2026
uv run python scripts/run_campp_model_selection.py \
    --config configs/training/campp-stage3-model-selection.toml
```

### Smoke / structural validation

The unit test exercises the full path on tiny CPU fixtures:

```bash
uv run pytest tests/unit/test_campp_model_selection.py
```

## Output Contract

One selector run writes:

```text
artifacts/model-selection/campp-stage3/<run_id>/
├── campp_model_selection_report.json
├── campp_model_selection_report.md
├── variants/
│   └── topN_uniform_average/
│       ├── campp_stage3_encoder.pt
│       ├── variant_report.md
│       └── <suite-id>/
│           ├── dev_embeddings.npz
│           ├── score_summary.json
│           └── verification_eval_report.{json,md}
└── final_candidate/
    ├── campp_stage3_encoder.pt
    └── final_candidate_selection.json
```

The top-level report includes:

- shortlist source report
- raw winner vs averaged variants
- skipped-variant reasons
- per-suite breakdown for every evaluated variant
- stable `final_candidate` checkpoint path for the next stage

## Scope Limits

- This step does not rerun the shortlist or change its weights.
- This step does not implement EMA recovery.
- Latency and memory remain separate concerns for the later release/deploy
  tasks.
