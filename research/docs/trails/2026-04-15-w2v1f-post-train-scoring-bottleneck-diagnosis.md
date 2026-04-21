# 2026-04-15 — W2V1f post-train scoring bottleneck diagnosis

- Trigger: after `W2V1f...` finished stage1 training, the run stopped emitting logs at
  `phase=scoring start` while the wrapper process remained alive. GPU1 fell to `0%`
  utilization, but one Python process kept consuming about `280%` CPU.
- Confirmed runtime state before intervention:
  - kept run id:
    `W2V1f_w2vbert2_mfa_lora_lmft_worker_bs96_20260415T103758Z`
  - stage1 local run id:
    `20260415T103816Z-b2c32e479771`
  - stage1 had already completed:
    `epoch 4/4 complete loss=1.927715 accuracy=0.926306`
  - stage1 artifacts already written:
    `training_summary.json`, `teacher_peft/`, `dev_embeddings.npz`,
    `cohort_embeddings.npz`, `cohort_summary.json`
  - generated trial manifest:
    `artifacts/baselines/w2vbert2-mfa-lora-stage1/20260415T103816Z-b2c32e479771/dev_trials.jsonl`
    with `trial_count=90754128` and on-disk size about `13 GiB`
  - last visible log line before the stall diagnosis:
    `[teacher-peft] phase=scoring start`
- Root cause found in code:
  - `src/kryptonite/training/speaker_baseline.py` materialized the entire all-pairs
    trial set as Python dicts, then materialized the entire scored set as another Python
    dict list, and normalized gathered embeddings repeatedly inside `cosine_score_pairs`
    instead of normalizing the base embedding matrix once.
  - after scoring, the teacher PEFT path planned to reload the full score file into
    Python dict rows for verification metrics and report generation, which would create
    another large memory and wall-clock spike.
  - the long post-train phases had only boundary logs, not intra-phase progress.
- Decision: stop `W2V1f...` and replace the post-train verification runtime before
  spending more wall-clock on the old `O(N^2)` Python materialization path.
