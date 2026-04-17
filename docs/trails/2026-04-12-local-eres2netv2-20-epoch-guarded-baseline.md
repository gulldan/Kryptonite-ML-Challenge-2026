# 2026-04-12 — Local ERes2NetV2 20-Epoch Guarded Baseline

Hypothesis:

- A longer ERes2NetV2 participant baseline may improve the P1 backbone if the run is
  allowed up to 20 epochs but guarded against obvious train-side overfitting.
- This is not a replacement for public-like validation. The local guard is intentionally
  conservative: full dev retrieval after every epoch is too expensive for a local RTX
  4090 launch, so the run uses train-loss patience and a train-accuracy ceiling, then
  restores the best train-loss state before checkpoint/export.

Implementation:

- Added generic baseline early stopping fields under `[optimization]`:
  `early_stopping_enabled`, `early_stopping_monitor`, `early_stopping_min_delta`,
  `early_stopping_patience_epochs`, `early_stopping_min_epochs`,
  `early_stopping_restore_best`, and `early_stopping_stop_train_accuracy`.
- Updated the shared baseline training loop to snapshot the best model/classifier state,
  restore it before writing the final checkpoint, and persist early-stopping metadata in
  `training_summary.json`.
- Added config:
  `configs/training/eres2netv2-participants-20epoch-local-guarded.toml`.

Config summary:

- Model family: `ERes2NetV2`, from scratch, 192-dim embedding.
- Train manifest:
  `artifacts/manifests/participants_fixed/train_manifest.jsonl` (`659804` rows).
- Dev manifest:
  `artifacts/manifests/participants_fixed/dev_manifest.jsonl` (`13473` rows).
- Seed: inherited from `configs/base.toml`, `42`.
- Device: local RTX 4090, `CUDA_VISIBLE_DEVICES=0`, `bf16`.
- Batch size: `32`; gradient accumulation steps: `2`; effective batch size: `64`;
  eval batch size: `64`.
- Max epochs: `20`.
- Optimizer/scheduler: SGD, LR `0.12`, min LR `0.00005`, momentum `0.9`,
  weight decay `0.0001`, cosine scheduler, warmup `2`, grad clip `5.0`.
- Crop/preprocessing: train crop `2.0-6.0s`, one crop, eval chunks `6.0s` with
  `1.5s` overlap, mean pooling, VAD disabled.
- Loss/objective: ArcMargin, scale `32.0`, margin `0.3`, classifier hidden dim `192`.
- Early stopping: monitor `train_loss`, min delta `0.0005`, patience `3`, min epochs
  `8`, restore best state, stop immediately after `train_accuracy >= 0.995`.

Local checks before launch:

- Config load check passed and resolved to `max_epochs=20`,
  `early_stopping_enabled=true`, monitor `train_loss`, stop accuracy `0.995`.
- `uv run ruff check` passed on the touched training files and new early-stopping test.
- `uvx ty check` passed on the touched training modules.
- New focused test passed:
  `uv run pytest tests/unit/test_eres2netv2_baseline.py::test_eres2netv2_baseline_early_stopping_records_and_restores_best -q`.
- Existing full `tests/unit/test_eres2netv2_baseline.py` still has a pre-existing failure:
  the smoke test expects `slice_dashboard_path` and `error_analysis_*` fields on
  `WrittenVerificationEvaluationReport`, but the current report dataclass does not expose
  those attributes. This was not changed as part of the 20-epoch guarded run.

Launch:

- First local foreground validation with batch size `64` reached `epoch=1/20` but failed
  with CUDA OOM on the RTX 4090: the process used about `20.27 GiB`, with only
  `188.94 MiB` free, while trying to allocate another `134 MiB`.
- Config was adjusted to `training.batch_size=32` and
  `gradient_accumulation_steps=2` to preserve effective batch `64`.

```bash
RUN_ID=eres2netv2_e20_guarded_local_20260412T104013Z
mkdir -p artifacts/logs
setsid -f bash -lc 'cd <repo-root> && PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 uv run --group train python scripts/run_eres2netv2_baseline.py --config configs/training/eres2netv2-participants-20epoch-local-guarded.toml --device cuda --output json >> artifacts/logs/eres2netv2_e20_guarded_local_20260412T104013Z.log 2>&1'
pgrep -n -f "run_eres2netv2_baseline.py --config configs/training/eres2netv2-participants-20epoch-local-guarded.toml" \
  > artifacts/logs/${RUN_ID}.pid
printf '%s\n' "${RUN_ID}" > artifacts/logs/latest_eres2netv2_e20_guarded_local
```

Superseded failed detached attempt:

```bash
RUN_ID=eres2netv2_e20_guarded_local_20260412T104013Z
mkdir -p artifacts/logs
nohup bash -lc 'cd <repo-root> && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --group train python scripts/run_eres2netv2_baseline.py --config configs/training/eres2netv2-participants-20epoch-local-guarded.toml --device cuda --output json' \
  > artifacts/logs/${RUN_ID}.log 2>&1 &
echo $! > artifacts/logs/${RUN_ID}.pid
printf '%s\n' "${RUN_ID}" > artifacts/logs/latest_eres2netv2_e20_guarded_local
```

Aborted/mis-targeted artifacts:

- Run wrapper id: `eres2netv2_e20_guarded_local_20260412T104013Z`.
- Log path: `artifacts/logs/eres2netv2_e20_guarded_local_20260412T104013Z.log`.
- PID path: `artifacts/logs/eres2netv2_e20_guarded_local_20260412T104013Z.pid`.
- Output root prefix:
  `artifacts/baselines/eres2netv2-participants-20epoch-local-guarded/`.
- Tracking experiment: `eres2netv2-participants-20epoch-local-guarded`.
- No detached ERes2NetV2 run is active. A foreground validation attempt created the run
  root `artifacts/baselines/eres2netv2-participants-20epoch-local-guarded/20260412T104117Z-7fa4c946c8ec`
  before failing with CUDA OOM.

Decision:

- Aborted as the wrong target after clarification: the requested run is the original
  organizer baseline, not ERes2NetV2.
- Do not use this ERes2NetV2 section as an active experiment record.
