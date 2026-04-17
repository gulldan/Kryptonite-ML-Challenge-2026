# 2026-04-15 — W2V1b batch retune and relaunch on remote

- Trigger: the first detached moonshot launch underused `gpu1` during stage1 startup and
  early training (roughly `3.7-4.9 GiB` on the H100, with visible headroom), so the
  batch profile was re-evaluated before spending more wall-clock on the initial
  conservative micro-batches.
- Action taken before relaunch:
  - stopped the kept first run `W2V1_w2vbert2_mfa_lora_lmft_20260415T090107Z` by killing
    process group `515262`
  - left the unrelated GPU0 ERes2Net workload untouched
  - ran a real one-step benchmark on `gpu1` inside `container` using the actual teacher PEFT
    code path, real manifests, real 3 s / 6 s crops, and a forward+backward optimizer
    step for each candidate batch size.
- Batch-search results on `gpu1`:
  - stage1 `configs/training/w2vbert2-mfa-lora-stage1.toml` (LoRA + frozen backbone, 3 s):
    tested `8,16,32,64,96,128,160`; no OOM in tested range.
    Representative points: `8 -> 6.54 ex/s, 2.71 GiB peak`; `32 -> 30.16 ex/s, 3.67 GiB`;
    `160 -> 40.91 ex/s, 8.70 GiB`.
  - stage2 `configs/training/w2vbert2-mfa-joint-ft-stage2.toml` (full FT, 3 s):
    tested `4,8,12,16,20,24,28,32`; no OOM in tested range.
    Representative points: `4 -> 6.65 ex/s, 10.40 GiB peak`;
    `16 -> 18.29 ex/s, 10.79 GiB`; `32 -> 33.68 ex/s, 9.68 GiB`.
  - stage3 `configs/training/w2vbert2-mfa-lmft-stage3.toml` (full FT, 6 s LMFT):
    tested `2,4,6,8,10,12,14,16`; no OOM in tested range.
    Representative points: `2 -> 2.96 ex/s, 10.39 GiB peak`;
    `10 -> 11.34 ex/s, 10.76 GiB`; `16 -> 13.31 ex/s, 10.40 GiB`.
- Interpretation: the original stage configs were not memory-limited. The main speed loss
  came from tiny micro-batches plus heavy gradient accumulation. The safe retune is to
  increase the micro-batch while preserving the original effective batch size of each
  stage.
- Config changes applied and synced to `remote`:
  - stage1: `training.batch_size 8 -> 32`,
    `optimization.gradient_accumulation_steps 4 -> 1`,
    `training.eval_batch_size 2 -> 16`
  - stage2: `training.batch_size 4 -> 32`,
    `optimization.gradient_accumulation_steps 8 -> 1`,
    `training.eval_batch_size 2 -> 16`
  - stage3: `training.batch_size 2 -> 16`,
    `optimization.gradient_accumulation_steps 8 -> 1`,
    `training.eval_batch_size 2 -> 16`
- Effective-batch rationale:
  - stage1 stays at effective batch `32` (`8x4 -> 32x1`)
  - stage2 stays at effective batch `32` (`4x8 -> 32x1`)
  - stage3 stays at effective batch `16` (`2x8 -> 16x1`)
  - this removes accumulation overhead without changing the intended optimization scale of
    the three-stage recipe.
- New detached relaunch:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 nohup \
  uv run --group train python scripts/run_w2vbert2_sv_moonshot.py \
    --run-id W2V1b_w2vbert2_mfa_lora_lmft_bs32_20260415T091257Z \
    --gpu-label gpu1 \
    --report-json artifacts/reports/w2vbert2/W2V1b_w2vbert2_mfa_lora_lmft_bs32_20260415T091257Z_summary.json \
  > artifacts/logs/W2V1b_w2vbert2_mfa_lora_lmft_bs32_20260415T091257Z.log 2>&1 < /dev/null &
```

- Relaunch artifacts:
  - run id: `W2V1b_w2vbert2_mfa_lora_lmft_bs32_20260415T091257Z`
  - log: `artifacts/logs/W2V1b_w2vbert2_mfa_lora_lmft_bs32_20260415T091257Z.log`
  - pid file: `artifacts/logs/W2V1b_w2vbert2_mfa_lora_lmft_bs32_20260415T091257Z.pid`
  - latest pointer:
    `artifacts/logs/latest_W2V1_w2vbert2_mfa_lora_lmft.txt`
  - final summary target:
    `artifacts/reports/w2vbert2/W2V1b_w2vbert2_mfa_lora_lmft_bs32_20260415T091257Z_summary.json`
- Status after relaunch check:
  - log contains the new run banner and stage1 command handoff
  - active stage1 worker process on GPU1:
    `pid 518760` under the new run tree
  - `nvidia-smi` showed GPU1 process `518760` using about `4.87 GiB` with roughly `92%`
    utilization during the post-relaunch check
  - the moderate VRAM footprint is expected here because stage1 still uses a frozen
    backbone + LoRA + gradient checkpointing path; the important signal is that GPU1 is
    now busy rather than sitting nearly idle.
- Decision: keep `W2V1b_w2vbert2_mfa_lora_lmft_bs32_20260415T091257Z` as the active
  moonshot run. Only escalate further to larger effective batches if stage1 epoch timing
  is still unacceptable after this retune.
