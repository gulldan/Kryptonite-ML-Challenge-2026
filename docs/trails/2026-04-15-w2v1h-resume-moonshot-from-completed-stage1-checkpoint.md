# 2026-04-15 — W2V1h resume moonshot from completed stage1 checkpoint

- Context:
  - after the optimized full restart `W2V1g...` was launched, it became clear that the
    previous run `W2V1f...` had already completed stage1 training and preserved a valid
    reusable checkpoint, so recomputing stage1 was unnecessary wall-clock waste
  - user explicitly asked to avoid recalculating already finished stages when possible
- Reused stage1 artifacts from `W2V1f_w2vbert2_mfa_lora_lmft_worker_bs96_20260415T103758Z`:
  - checkpoint:
    `artifacts/baselines/w2vbert2-mfa-lora-stage1/20260415T103816Z-b2c32e479771/teacher_peft`
  - training summary:
    `artifacts/baselines/w2vbert2-mfa-lora-stage1/20260415T103816Z-b2c32e479771/training_summary.json`
  - score summary:
    `artifacts/baselines/w2vbert2-mfa-lora-stage1/20260415T103816Z-b2c32e479771/score_summary.json`
- Code change for orchestration:
  - updated `scripts/run_w2vbert2_sv_moonshot.py` to accept
    `--stage1-checkpoint <dir|run_dir|checkpoint_metadata.json>`
  - the wrapper now records a reused stage1 in the final summary and continues directly
    into stage2 + stage3 without retraining stage1
- Remote corrective action on `remote`:
  - stopped the unnecessary full restart `W2V1g_w2vbert2_mfa_lora_lmft_opt_20260415T174703Z`
    after it had re-entered stage1 training
  - launched the resume run:
    `W2V1h_w2vbert2_mfa_lora_lmft_resume_s2_20260415T175719Z`
- Resume command executed inside `container`:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 nohup \
  uv run --group train python scripts/run_w2vbert2_sv_moonshot.py \
    --run-id W2V1h_w2vbert2_mfa_lora_lmft_resume_s2_20260415T175719Z \
    --gpu-label gpu1 \
    --stage1-checkpoint artifacts/baselines/w2vbert2-mfa-lora-stage1/20260415T103816Z-b2c32e479771/teacher_peft \
    --report-json artifacts/reports/w2vbert2/W2V1h_w2vbert2_mfa_lora_lmft_resume_s2_20260415T175719Z_summary.json \
  > artifacts/logs/W2V1h_w2vbert2_mfa_lora_lmft_resume_s2_20260415T175719Z.log 2>&1 < /dev/null &
```

- Resume artifacts:
  - run id: `W2V1h_w2vbert2_mfa_lora_lmft_resume_s2_20260415T175719Z`
  - log:
    `artifacts/logs/W2V1h_w2vbert2_mfa_lora_lmft_resume_s2_20260415T175719Z.log`
  - pid file:
    `artifacts/logs/W2V1h_w2vbert2_mfa_lora_lmft_resume_s2_20260415T175719Z.pid`
  - summary target:
    `artifacts/reports/w2vbert2/W2V1h_w2vbert2_mfa_lora_lmft_resume_s2_20260415T175719Z_summary.json`
- Initial log snapshot:
  - wrapper logged `reuse stage1 checkpoint=.../teacher_peft`
  - stage2 launched from the reused stage1 checkpoint with:
    `scripts/run_teacher_peft_finetune.py --config configs/training/w2vbert2-mfa-joint-ft-stage2.toml --init-checkpoint ... --merge-lora --init-classifier-from-checkpoint --output json`
- Decision:
  - keep `W2V1h...` as the active run on `gpu1`
  - this removes the redundant stage1 retrain and preserves the shared optimized
    scoring / verification runtime for stage2 and stage3 follow-up evaluation
