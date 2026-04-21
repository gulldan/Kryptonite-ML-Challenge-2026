# 2026-04-15 — W2V1f aggressive micro-batch relaunch after worker-feature fix

- Config retune derived from the post-fix benchmark and synced to `remote`:
  - `configs/training/w2vbert2-mfa-lora-stage1.toml`
    - `training.batch_size 32 -> 96`
    - `training.eval_batch_size` kept at `16`
    - `optimization.gradient_accumulation_steps` kept at `1`
  - `configs/training/w2vbert2-mfa-joint-ft-stage2.toml`
    - `training.batch_size 32 -> 96`
    - `training.eval_batch_size` kept at `16`
    - `optimization.gradient_accumulation_steps` kept at `1`
  - `configs/training/w2vbert2-mfa-lmft-stage3.toml`
    - `training.batch_size 16 -> 48`
    - `training.eval_batch_size` kept at `16`
    - `optimization.gradient_accumulation_steps` kept at `1`
- Relaunch command executed inside `container`:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 nohup \
  uv run --group train python scripts/run_w2vbert2_sv_moonshot.py \
    --run-id W2V1f_w2vbert2_mfa_lora_lmft_worker_bs96_20260415T103758Z \
    --gpu-label gpu1 \
    --report-json artifacts/reports/w2vbert2/W2V1f_w2vbert2_mfa_lora_lmft_worker_bs96_20260415T103758Z_summary.json \
  > artifacts/logs/W2V1f_w2vbert2_mfa_lora_lmft_worker_bs96_20260415T103758Z.log 2>&1 < /dev/null &
```

- Relaunch artifacts:
  - run id: `W2V1f_w2vbert2_mfa_lora_lmft_worker_bs96_20260415T103758Z`
  - log: `artifacts/logs/W2V1f_w2vbert2_mfa_lora_lmft_worker_bs96_20260415T103758Z.log`
  - pid file:
    `artifacts/logs/W2V1f_w2vbert2_mfa_lora_lmft_worker_bs96_20260415T103758Z.pid`
  - summary target:
    `artifacts/reports/w2vbert2/W2V1f_w2vbert2_mfa_lora_lmft_worker_bs96_20260415T103758Z_summary.json`
  - stage1 local run id: `20260415T103816Z-b2c32e479771`
  - stage1 output root:
    `artifacts/baselines/w2vbert2-mfa-lora-stage1/20260415T103816Z-b2c32e479771`
- First live snapshot after the aggressive relaunch:
  - `batch_size=96`, `eval_batch_size=16`, `accumulation=1`, `epochs=4`,
    `train_batches=7961`
  - batch `1/7961`: `loss=18.123117`, `ex_per_sec=22.93`, `eta=9h15m20s`,
    `mem=2.44/7.44GiB`
  - GPU1 snapshot at the same time:
    - `92%` utilization
    - `8305 MiB` allocated
    - `291.42 W`
- Short sustained utilization sample over the next ~10 seconds:
  - GPU1 utilization samples: `92%, 100%, 100%, 100%, 100%`
  - GPU1 memory stayed stable around `8309 MiB`
  - GPU1 power stayed around `294-297 W`
- Status at this write:
  - the run is active on `gpu1`
  - stage1 checkpoints and per-epoch metrics are still pending
  - no public inference / submission has been launched from this branch yet
  - public leaderboard score pending
- Decision: keep `W2V1f_w2vbert2_mfa_lora_lmft_worker_bs96_20260415T103758Z` as the
  active moonshot run on `gpu1`. The utilization problem is treated as resolved at the
  pipeline level; further speed work should now target steady-state examples/sec and
  downstream stage2/3 memory headroom rather than the old CPU preprocessing stall.
