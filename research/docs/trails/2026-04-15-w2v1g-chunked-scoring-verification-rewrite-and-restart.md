# 2026-04-15 — W2V1g chunked scoring / verification rewrite and restart

- Code changes applied locally and synced to `remote`:
  - `src/kryptonite/training/speaker_baseline.py`
    - `load_or_generate_trials(...)` now writes the trial manifest in streaming chunks
      instead of building one giant Python list
    - added progress logs for trial generation
    - added `score_trials_detailed(...)` with:
      - chunked trial reads from JSONL
      - one-time embedding normalization before pair scoring
      - preallocated `labels` / `scores` arrays
      - streaming `dev_scores.jsonl` writes
      - progress logs with processed trial count, throughput, elapsed time, ETA, and
        missing-embedding count
  - `src/kryptonite/eval/verification_metrics.py`
    - added exact array-based verification metrics and operating-point computation
    - enabled downsampling of stored ROC/DET points for very large score sets while
      keeping summary metrics exact
  - `src/kryptonite/eval/verification_report.py`
    - added `build_verification_evaluation_report_from_arrays(...)`
    - report generation now consumes in-memory `labels` / `scores` arrays instead of
      reloading giant JSONL score files into Python dict rows
    - added progress logs for verification metrics / curve / calibration phases
  - `src/kryptonite/training/baseline_pipeline.py` and
    `src/kryptonite/training/teacher_peft/pipeline.py`
    - switched shared post-train evaluation to the new chunked scoring +
      array-based verification path
  - `scripts/run_w2vbert2_sv_moonshot.py`
    - now logs stage completion for stage1, stage2, and stage3 after each successful
      handoff
- Local validation after the rewrite:
  - `ruff format` applied to touched files
  - `ruff check` passed on:
    `src/kryptonite/training/speaker_baseline.py`,
    `src/kryptonite/training/baseline_pipeline.py`,
    `src/kryptonite/training/teacher_peft/pipeline.py`,
    `src/kryptonite/eval/verification_metrics.py`,
    `src/kryptonite/eval/verification_report.py`,
    `src/kryptonite/eval/__init__.py`,
    `scripts/run_w2vbert2_sv_moonshot.py`
  - `pytest tests/unit/test_speaker_baseline.py tests/unit/test_teacher_peft_pipeline.py tests/unit/test_teacher_peft_config.py tests/unit/test_teacher_peft_checkpoint.py tests/unit/test_verification_metrics.py`
    passed (`8 passed`)
  - `python scripts/run_w2vbert2_sv_moonshot.py --help` succeeded
- Remote sequence on `remote`:
  - synced the rewritten scoring / verification runtime and moonshot wrapper to
    `<remote-repo>`
  - stopped the old scoring-stalled run by killing process group `529713`
  - launched the optimized moonshot as:
    `W2V1g_w2vbert2_mfa_lora_lmft_opt_20260415T174703Z`
- Relaunch command executed inside `container`:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 nohup \
  uv run --group train python scripts/run_w2vbert2_sv_moonshot.py \
    --run-id W2V1g_w2vbert2_mfa_lora_lmft_opt_20260415T174703Z \
    --gpu-label gpu1 \
    --report-json artifacts/reports/w2vbert2/W2V1g_w2vbert2_mfa_lora_lmft_opt_20260415T174703Z_summary.json \
  > artifacts/logs/W2V1g_w2vbert2_mfa_lora_lmft_opt_20260415T174703Z.log 2>&1 < /dev/null &
```

- Relaunch artifacts:
  - run id: `W2V1g_w2vbert2_mfa_lora_lmft_opt_20260415T174703Z`
  - log: `artifacts/logs/W2V1g_w2vbert2_mfa_lora_lmft_opt_20260415T174703Z.log`
  - pid file:
    `artifacts/logs/W2V1g_w2vbert2_mfa_lora_lmft_opt_20260415T174703Z.pid`
  - summary target:
    `artifacts/reports/w2vbert2/W2V1g_w2vbert2_mfa_lora_lmft_opt_20260415T174703Z_summary.json`
  - stage1 local run id:
    `20260415T174725Z-6e694055dc36`
  - stage1 output root:
    `artifacts/baselines/w2vbert2-mfa-lora-stage1/20260415T174725Z-6e694055dc36`
- First live snapshot after the optimized relaunch:
  - `batch_size=96`, `eval_batch_size=16`, `accumulation=1`, `epochs=4`,
    `train_batches=7961`
  - batch `1/7961`: `loss=18.123117`, `ex_per_sec=17.04`, `eta=12h27m23s`,
    `mem=2.44/7.44GiB`
  - GPU1 snapshot after removing an accidental smoke-process leftover:
    - `100%` utilization
    - `8309 MiB` allocated
    - about `300 W`
- Additional remote note:
  - a temporary foreground smoke check was started and then terminated after it became
    clear that `run_teacher_peft.py` accepts only base project overrides, not teacher
    data-section overrides such as `data.max_train_rows`. The stray smoke workers were
    killed before the final `W2V1g...` snapshot, so the active restart is clean.
- Status at this write:
  - `W2V1g...` is the active moonshot run on `gpu1`
  - the optimized chunked trial/scoring/verification runtime is now the path that will
    execute after stage1/2/3 training finishes
  - public leaderboard score still pending
- Decision: keep `W2V1g_w2vbert2_mfa_lora_lmft_opt_20260415T174703Z` as the active run.
  The next checkpoint is the first post-train phase on the new runtime, where the log
  should now show `[trials]`, `[score]`, and `[verification]` progress instead of a
  silent CPU-bound gap.
