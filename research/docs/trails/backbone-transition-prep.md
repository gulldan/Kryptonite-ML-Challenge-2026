# Backbone Transition Prep

Date: 2026-04-11

Hypothesis: the current baseline encoder is saturated; the next meaningful gain should
come from a stronger backbone evaluated through the already-confirmed C4 postprocess
tail, rather than from more tuning of the old encoder.

Prepared code/config:

- Participant training manifest builder:
  `scripts/build_participant_training_manifests.py`
- Reusable manifest conversion logic:
  `src/kryptonite/data/participant_manifests.py`
- First ERes2NetV2 candidate config:
  `configs/training/eres2netv2-participants-candidate.toml`
- Generic checkpoint-to-C4-tail runner for CAM++/ERes2NetV2 checkpoints:
  `scripts/run_torch_checkpoint_c4_tail.py`

Generated manifests:

- `artifacts/manifests/participants_fixed/train_manifest.jsonl`: `659804` rows
- `artifacts/manifests/participants_fixed/dev_manifest.jsonl`: `13473` rows
- `artifacts/manifests/participants_fixed/manifest_inventory.json`

Checks:

- `scripts/validate_manifests.py --manifests-root artifacts/manifests/participants_fixed --strict`
  passed: `673277` valid rows, `0` invalid rows.
- `pytest tests/unit/test_eda.py -q` passed: `6 passed`.
- `ruff check` passed for touched Python files.

Important note:

- A CPU full-data smoke command was accidentally started without `max_train_rows`; it was
  stopped and produced no usable experiment result. This is recorded as a process lesson:
  do not smoke-test full participant configs on CPU without explicit row limits.

Next intended run:

```bash
uv run --group train python scripts/run_baseline.py \
  --model eres2netv2 \
  --config configs/training/eres2netv2-participants-candidate.toml \
  --device cuda
```

After checkpoint training, evaluate through C4 tail:

```bash
uv run --group train python scripts/run_torch_checkpoint_c4_tail.py \
  --model eres2netv2 \
  --checkpoint-path artifacts/baselines/eres2netv2-participants/<run-id> \
  --manifest-csv artifacts/eda/baseline_fixed_dense_shifted_v2_honest/dense_gallery_manifest.csv \
  --output-dir artifacts/backbone_eval/eres2netv2-candidate/<run-id> \
  --experiment-id E1_eres2netv2_c4_dense_shifted_v2 \
  --shift-mode v2
```

Decision gate:

- Public submission should wait until the new backbone beats current C4 by at least
  `+0.008...+0.010` on honest dense shifted v2 after the full C4 tail, unless it is
  explicitly being tested as a fusion candidate.
