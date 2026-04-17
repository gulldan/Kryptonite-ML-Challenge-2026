# 2026-04-12 — Local Organizer Baseline 20-Epoch Early-Stopping Run

Hypothesis:

- The first organizer baseline may benefit from allowing up to 20 epochs if training is
  guarded by validation retrieval early stopping instead of always stopping at the
  original short schedule.
- Unlike the aborted ERes2NetV2 guard above, this run monitors the organizer baseline's
  speaker-disjoint validation `precision@10`, so it is a real overfitting guard.

Implementation:

- Updated `baseline/train.py` to support:
  - best checkpoint by configurable validation metric;
  - early stopping by validation metric with `min_delta`, `patience`, and `min_epochs`;
  - optional train-accuracy stop threshold;
  - optional `ReduceLROnPlateau`;
  - optional gradient clipping;
  - per-epoch `metrics.jsonl`;
  - final `training_summary.json`.
- Added config:
  `baseline/configs/participants_baseline_fixed_20epoch_earlystop.json`.
- Added `pandas==2.2.2` to the `train` dependency group because organizer
  `baseline/train.py` imports pandas and the repo-local `uv` environment did not have it.
- Added `faiss-cpu==1.13.2` to the `train` dependency group because organizer
  `baseline/src/metrics.py` imports `faiss` for validation retrieval metrics.

Config summary:

- Model family: original organizer ECAPA-style baseline in `baseline/src/model.py`.
- Train CSV: `datasets/Для участников/train.csv`.
- Split policy: organizer fixed speaker-disjoint split, `train_ratio=0.98`,
  `min_val_utts=11`, seed `2026`.
- Output dir: `artifacts/baseline_fixed_participants_e20_earlystop`.
- Device: local RTX 4090, `CUDA_VISIBLE_DEVICES=0`.
- Max epochs: `20`.
- Batch size: `256`.
- Optimizer: AdamW, LR `0.001`, weight decay `0.00001`.
- Audio policy: train chunk `3.0s`, validation chunk `6.0s`, sample rate `16000`.
- Validation metric: `precision@10`.
- Early stopping: metric `precision@10`, mode `max`, min delta `0.0005`, patience `3`,
  min epochs `6`, restore best checkpoint.
- Scheduler: ReduceLROnPlateau, factor `0.5`, patience `1`, threshold `0.0005`,
  min LR `0.00001`.
- Grad clip: `5.0`.

Local checks before launch:

- Config load check passed: `epochs=20`, `early_stopping_enabled=true`,
  `early_stopping_metric=precision@10`, scheduler `reduce_on_plateau`.
- `uv run ruff check baseline/train.py tests/unit/test_organizer_baseline_fixed.py`
  passed.
- `uv run pytest tests/unit/test_organizer_baseline_fixed.py -q` passed:
  `4 passed`.
- `PYTHONPATH=baseline uvx ty check baseline/train.py` passed.

Launch:

- First detached launch failed before training with
  `ModuleNotFoundError: No module named 'faiss'`. This was an environment gap against
  `baseline/requirements.txt`, not a model/training result. Added `faiss-cpu==1.13.2`
  and relaunched.

```bash
RUN_ID=organizer_baseline_e20_earlystop_local_20260412T104626Z
mkdir -p artifacts/logs
setsid -f bash -lc 'cd <repo-root> && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --project <repo-root> --group train bash -lc "cd baseline && python train.py --config configs/participants_baseline_fixed_20epoch_earlystop.json" >> artifacts/logs/organizer_baseline_e20_earlystop_local_20260412T104626Z.log 2>&1'
pgrep -n -f "train.py --config configs/participants_baseline_fixed_20epoch_earlystop.json" \
  > artifacts/logs/${RUN_ID}.pid
printf '%s\n' "${RUN_ID}" > artifacts/logs/latest_organizer_baseline_e20_earlystop_local
```

Artifacts:

- Run wrapper id: `organizer_baseline_e20_earlystop_local_20260412T104626Z`.
- Training process status: stopped manually at user request after epoch `10`
  completed and while epoch `11` was in progress.
- Log path:
  `artifacts/logs/organizer_baseline_e20_earlystop_local_20260412T104626Z.log`.
- PID path:
  `artifacts/logs/organizer_baseline_e20_earlystop_local_20260412T104626Z.pid`.
- Model checkpoint:
  `artifacts/baseline_fixed_participants_e20_earlystop/model.pt`.
- Metrics JSONL:
  `artifacts/baseline_fixed_participants_e20_earlystop/metrics.jsonl`.
- Training summary, written after the manual interruption:
  `artifacts/baseline_fixed_participants_e20_earlystop/training_summary.json`.
- ONNX:
  `artifacts/baseline_fixed_participants_e20_earlystop/model_embeddings_epoch10_center_opset20.onnx`.
- Public embeddings:
  `artifacts/baseline_fixed_participants_e20_earlystop/test_public_emb_epoch10_center_opset20.npy`.
- Public submission:
  `artifacts/baseline_fixed_participants_e20_earlystop/submission_epoch10_center_opset20.csv`.
- Upload-friendly copy with the same SHA256:
  `artifacts/baseline_fixed_participants_e20_earlystop/submission.csv`.
- Submission validation:
  `artifacts/baseline_fixed_participants_e20_earlystop/submission_epoch10_center_opset20_validation.json`.
- Upload-copy validation:
  `artifacts/baseline_fixed_participants_e20_earlystop/submission_validation.json`.
- Public inference log:
  `artifacts/logs/organizer_baseline_e20_epoch10_public_center_infer_20260412.log`.

Monitor snapshot:

- 2026-04-12 10:48 UTC: run active on local RTX 4090, epoch `1/20`, around
  `64/2577` training batches processed, GPU utilization `100%`, memory used about
  `13.7 GiB`.
- 2026-04-12 11:42 UTC: run still active on local RTX 4090, training epoch `10/20`
  around the halfway point. `metrics.jsonl` has `9` completed epochs. Best validation
  `precision@10` is epoch `7`: `0.922089` with train loss `0.211063` and train accuracy
  `0.945035`. Epoch `8` dropped to `0.918600`; epoch `9` recovered only to `0.921361`,
  which is not an improvement under `min_delta=0.0005`. Early-stopping bad epochs are
  now `2`; `ReduceLROnPlateau` lowered LR from `0.001` to `0.0005`. If epoch `10` does
  not exceed `0.922589`, the configured patience should stop the run after epoch `10`
  and keep the epoch-7 checkpoint.
- 2026-04-12 11:52 UTC: user requested stopping training and generating a public
  submission to measure Public LB. The training process was terminated during epoch
  `11/20` before epoch-11 validation/checkpointing. The last completed validation was
  epoch `10`, which became the best checkpoint:
  - epoch `10`;
  - validation `precision@10 = 0.928308`;
  - train loss `0.069959`;
  - train accuracy `0.980654`;
  - learning rate `0.0005`.
  The run summary was written manually from
  `artifacts/baseline_fixed_participants_e20_earlystop/metrics.jsonl` because the
  normal train-script finalizer does not execute after manual termination.

Stop command:

```bash
RUN_ID=$(cat artifacts/logs/latest_organizer_baseline_e20_earlystop_local)
PID=$(cat artifacts/logs/${RUN_ID}.pid)
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
kill -TERM -- -"$PGID"
```

Public submission generation:

```bash
cd <repo-root>/baseline
uv run --project <repo-root> --group train \
  python convert_to_onnx.py \
    --config configs/participants_baseline_fixed_20epoch_earlystop.json \
    --pt ../artifacts/baseline_fixed_participants_e20_earlystop/model.pt \
    --out ../artifacts/baseline_fixed_participants_e20_earlystop/model_embeddings_epoch10_center_opset20.onnx \
    --opset 20

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
uv run --project <repo-root> --group train \
  python inference_onnx.py \
    --onnx_path ../artifacts/baseline_fixed_participants_e20_earlystop/model_embeddings_epoch10_center_opset20.onnx \
    --csv "../datasets/Для участников/test_public.csv" \
    --data_base_dir "../datasets/Для участников" \
    --output_emb ../artifacts/baseline_fixed_participants_e20_earlystop/test_public_emb_epoch10_center_opset20.npy \
    --output_indices ../artifacts/baseline_fixed_participants_e20_earlystop/submission_epoch10_center_opset20.csv \
    --batch_size 64 \
    --num_workers 8 \
    --sample_rate 16000 \
    --chunk_seconds 6.0 \
    --num_crops 1 \
    --device cuda
```

Public inference result:

- ONNX Runtime provider: `CUDAExecutionProvider`.
- Public rows embedded: `134697`.
- Embedding dimension: `192`.
- Inference plus FAISS indexing wall time: `135.241s`.
- Submission SHA256:
  `a6e2428590e909d132f84deb3535cfe874a36ee58c9f73493e45b477afb3896a`.
  The same hash applies to
  `artifacts/baseline_fixed_participants_e20_earlystop/submission.csv`.
- ONNX SHA256:
  `fd66fac9bdab090bfa050727d11e0ba763166cc6ba6eb074f53c10a7269f0ca2`.
- Embeddings SHA256:
  `d9b0de9495c792fdf15a1faaf181edf4bcc88e708b9e28fd8f41f387cbd73398`.

Submission validation:

```bash
uv run python scripts/validate_submission.py \
  --template-csv "datasets/Для участников/test_public.csv" \
  --submission-csv artifacts/baseline_fixed_participants_e20_earlystop/submission_epoch10_center_opset20.csv \
  --output-json artifacts/baseline_fixed_participants_e20_earlystop/submission_epoch10_center_opset20_validation.json \
  --k 10
```

- Validator status: `passed=True`, `errors=0`.
- Public LB score from external upload: `0.1046`.
- Public LB deltas:
  - `+0.0267` vs original organizer baseline `0.0779`;
  - `+0.0022` vs `baseline_fixed_participants` `0.1024`;
  - `-0.0203` vs C4 graph branch `0.1249`;
  - `-0.1364` vs current P1 ERes2NetV2 best `0.2410`.

Decision:

- Rejected as a dead-end branch. The gap between local validation `0.928308` and public
  LB `0.1046` confirms that longer guarded training of the original organizer baseline
  mostly over-optimizes the local split/domain and does not solve the public retrieval
  failure.
- Do not continue this baseline family as a production path. Keep only as a diagnostic
  control for sanity checks against `baseline_fixed_participants`.
