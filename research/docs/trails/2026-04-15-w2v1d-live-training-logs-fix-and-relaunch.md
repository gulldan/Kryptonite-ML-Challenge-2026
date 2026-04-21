# 2026-04-15 — W2V1d live training logs fix and relaunch

- Problem found during remote monitoring: the moonshot wrapper only wrote the outer
  stage-start banners to the detached log. The inner `teacher_peft` progress prints were
  not visible during the run, which made stage1 impossible to monitor in real time.
- Root cause:
  - `src/kryptonite/training/teacher_peft/pipeline.py` did not emit enough progress logs
    for long-running remote training and post-train phases.
  - `scripts/run_w2vbert2_sv_moonshot.py` executed stage scripts with
    `subprocess.run(..., capture_output=True)`, so child progress was buffered until stage
    completion instead of being streamed into the detached log.
- Code changes applied:
  - added `src/kryptonite/training/teacher_peft/progress.py`
  - updated `src/kryptonite/training/teacher_peft/pipeline.py` to emit progress to
    `stderr` for:
    - teacher init summary
    - epoch start / epoch complete
    - batch progress with loss, avg loss, avg accuracy, lr, examples/sec, elapsed, eta,
      and CUDA memory
    - checkpoint / embedding export / trials / cohort / scoring / verification / report
      phase boundaries
  - updated `scripts/run_w2vbert2_sv_moonshot.py` to stream child `stderr` live while
    still capturing child `stdout` as JSON for stage chaining.
- Local validation after the patch:
  - `ruff format` passed on the touched files
  - `ruff check` passed on the touched files
  - `pytest tests/unit/test_teacher_peft_pipeline.py tests/unit/test_teacher_peft_config.py tests/unit/test_teacher_peft_checkpoint.py`
    passed (`4 passed`)
  - `python scripts/run_w2vbert2_sv_moonshot.py --help` succeeded
- Remote relaunch sequence:
  - stopped silent run `W2V1c_w2vbert2_mfa_lora_lmft_logs_20260415T101923Z`
    by killing process group `519237`
  - synced the patched pipeline, progress helper, and moonshot runner to `remote`
  - relaunched as:
    `W2V1d_w2vbert2_mfa_lora_lmft_live_logs_20260415T102217Z`
- Relaunch artifacts:
  - run id: `W2V1d_w2vbert2_mfa_lora_lmft_live_logs_20260415T102217Z`
  - log: `artifacts/logs/W2V1d_w2vbert2_mfa_lora_lmft_live_logs_20260415T102217Z.log`
  - pid file:
    `artifacts/logs/W2V1d_w2vbert2_mfa_lora_lmft_live_logs_20260415T102217Z.pid`
  - summary target:
    `artifacts/reports/w2vbert2/W2V1d_w2vbert2_mfa_lora_lmft_live_logs_20260415T102217Z_summary.json`
- First live log snapshot after relaunch:
  - teacher init line reported:
    - stage1 local run id `20260415T102235Z-4755140b7cf3`
    - `train_rows=764165`, `dev_rows=13473`, `speakers=14021`
    - `batch_size=32`, `eval_batch_size=16`, `accumulation=1`, `epochs=4`
    - `train_batches=23881`
    - `trainable_params=14924160`, `total_params=595417280`
    - stage1 output root:
      `artifacts/baselines/w2vbert2-mfa-lora-stage1/20260415T102235Z-4755140b7cf3`
  - first batch line:
    - `epoch 1/4 batch 1/23881`
    - `loss=17.750872`
    - `ex_per_sec=14.57`
    - `eta=14h34m10s`
    - `mem=2.44/3.96GiB`
- GPU state at the same snapshot:
  - GPU1 process `519702` running the stage1 worker
  - GPU1 memory around `4.84 GiB`
  - GPU1 utilization around `68%`
  - GPU1 power around `190 W`
- Decision: keep `W2V1d_w2vbert2_mfa_lora_lmft_live_logs_20260415T102217Z` as the active
  moonshot run because observability is now fixed. Future tuning decisions should use the
  new batch-level log stream rather than blind wall-clock guesses.
