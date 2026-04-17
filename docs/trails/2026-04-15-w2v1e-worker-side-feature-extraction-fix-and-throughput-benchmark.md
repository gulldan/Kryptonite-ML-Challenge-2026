# 2026-04-15 — W2V1e worker-side feature extraction fix and throughput benchmark

- Trigger: after the logging fix, the active moonshot run was still underfeeding `gpu1`.
  The first visible stage1 batch in `W2V1d...` reported only `14.57 ex/s`, GPU1 stayed
  around `68%` utilization, and H100 memory remained below `5 GiB`, so the pipeline was
  still bottlenecked outside the model itself.
- Root cause found:
  - `facebook/w2v-bert-2.0` uses the HF `SeamlessM4TFeatureExtractor` and produces
    `input_features` plus `attention_mask`, not just padded raw waveforms.
  - the old teacher PEFT path built those features in the main training process after the
    `DataLoader` returned waveforms, so the GPU repeatedly waited on CPU-side feature
    extraction between optimizer steps.
- Code changes applied:
  - `src/kryptonite/training/teacher_peft/data.py`
    - `WaveformTrainingBatch` now carries `model_inputs` instead of raw waveforms
    - `collate_waveform_examples(...)` now runs the HF feature extractor inside the
      worker `collate_fn`
  - `src/kryptonite/training/teacher_peft/pipeline.py`
    - instantiates the feature extractor before building the `DataLoader`
    - passes the feature extractor into `collate_waveform_examples(...)`
    - enables `persistent_workers` when `runtime.num_workers > 0`
    - moves `batch.model_inputs` to CUDA with `non_blocking=True`
- Local validation after this refactor:
  - `ruff format` passed on the touched files
  - `ruff check` passed on the touched files
  - `pytest tests/unit/test_teacher_peft_pipeline.py tests/unit/test_teacher_peft_config.py tests/unit/test_teacher_peft_checkpoint.py`
    passed (`4 passed`)
- Remote sequence on `remote`:
  - stopped the previously active live-log run after diagnosis
  - synced the updated `data.py` and `pipeline.py` to
    `<remote-repo>`
  - relaunched the moonshot as
    `W2V1e_w2vbert2_mfa_lora_lmft_worker_feats_20260415T102812Z`
- Diagnostic run artifacts:
  - run id: `W2V1e_w2vbert2_mfa_lora_lmft_worker_feats_20260415T102812Z`
  - log: `artifacts/logs/W2V1e_w2vbert2_mfa_lora_lmft_worker_feats_20260415T102812Z.log`
  - pid file:
    `artifacts/logs/W2V1e_w2vbert2_mfa_lora_lmft_worker_feats_20260415T102812Z.pid`
  - stage1 local run id: `20260415T102830Z-73d684885eaf`
  - stage1 output root:
    `artifacts/baselines/w2vbert2-mfa-lora-stage1/20260415T102830Z-73d684885eaf`
- First post-fix live snapshot:
  - `batch_size=32`, `eval_batch_size=16`, `accumulation=1`, `train_batches=23881`
  - batch `1/23881`: `loss=17.750872`, `ex_per_sec=11.99`, `eta=17h42m34s`,
    `mem=2.44/3.96GiB`
  - `nvidia-smi` snapshot at the same point showed GPU1 around `75%` utilization and
    roughly `272 W`
  - host CPU shifted from one overloaded main process to eight busy loader workers,
    confirming that the feature extraction bottleneck had moved to the intended place
- Dedicated one-step throughput benchmark after the worker-feature fix:
  - stage1 `configs/training/w2vbert2-mfa-lora-stage1.toml`
    - tested `32,64,96,128,160`
    - representative points:
      - `32 -> 18.14 ex/s, 3.66 GiB peak`
      - `64 -> 37.32 ex/s, 4.98 GiB peak`
      - `96 -> 50.51 ex/s, 6.03 GiB peak`
      - `128 -> 43.16 ex/s, 7.63 GiB peak`
      - `160 -> 49.80 ex/s, 8.79 GiB peak`
    - best tested point: `batch_size=96`
  - stage2 `configs/training/w2vbert2-mfa-joint-ft-stage2.toml`
    - tested `32,48,64,80,96`
    - representative points:
      - `32 -> 17.01 ex/s, 10.40 GiB peak`
      - `48 -> 33.30 ex/s, 10.39 GiB peak`
      - `64 -> 35.45 ex/s, 10.04 GiB peak`
      - `80 -> 38.69 ex/s, 10.07 GiB peak`
      - `96 -> 43.49 ex/s, 10.53 GiB peak`
    - best tested point: `batch_size=96`
  - stage3 `configs/training/w2vbert2-mfa-lmft-stage3.toml`
    - tested `16,24,32,40,48`
    - representative points:
      - `16 -> 7.80 ex/s, 10.40 GiB peak`
      - `24 -> 11.58 ex/s, 10.75 GiB peak`
      - `32 -> 13.79 ex/s, 10.04 GiB peak`
      - `40 -> 15.07 ex/s, 10.76 GiB peak`
      - `48 -> 16.56 ex/s, 11.11 GiB peak`
    - best tested point: `batch_size=48`
- Decision:
  - accept the worker-side feature extraction refactor as the real throughput fix
  - reject `W2V1e...` as a kept training run; it was stopped after diagnosis so the tuned
    micro-batches could be applied immediately
  - use the benchmark maxima `96 / 96 / 48` for the next detached launch on `gpu1`
