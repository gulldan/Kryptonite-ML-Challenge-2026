# Backbone Training Completion And C4-Tail Eval Prep

Date: 2026-04-12

Training results on `remote`:

- `campp_h100_b1024_20260411_200846`
  - Output root: `artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317`
  - Final train loss: `0.816177`
  - Final train accuracy: `0.987965`
  - Dev score gap: `0.522243`
  - Status: completed successfully, checkpoint/report written.
- `eres2netv2_h100_b128_20260411_200735`
  - Output root: `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee`
  - Final train loss: `0.964109`
  - Final train accuracy: `0.994018`
  - Dev score gap: `0.568871`
  - Status: training, checkpoint, embeddings, and score summary completed; final report
    generation was still running at check time.

Training reproducibility parameters:

- Shared environment and data contract:
  - Remote host: `<redacted>`
  - Docker container: `container`
  - Host repository path: `<remote-repo>`
  - Container repository path: `<repo-root>`
  - Container setup command: `uv sync --dev --group train`
  - Base config: `configs/base.toml`
  - Code state recorded after run: base commit
    `012d87f2ea6a37cccccf327ff22bcdb43e139131` plus uncommitted training/logging
    changes in `scripts/run_torch_checkpoint_c4_tail.py`,
    `src/kryptonite/training/optimization_runtime.py`, and
    `src/kryptonite/training/speaker_baseline.py`.
  - Seed: `42`; deterministic mode: `true`; `PYTHONHASHSEED=42` contract from
    `configs/base.toml`.
  - Train manifest:
    `artifacts/manifests/participants_fixed/train_manifest.jsonl`
  - Dev manifest:
    `artifacts/manifests/participants_fixed/dev_manifest.jsonl`
  - Split/provenance: participant train split from `baseline_fixed`; no VoxBlink2 data
    or checkpoint; both backbones initialized `from_scratch`.
  - Audio/features: mono `16 kHz`, 80-bin log-fbank, 25 ms frame, 10 ms shift, FFT `512`,
    Hann window, `cmvn_mode="none"`.
  - Training crops: one random crop per utterance, crop seconds sampled per batch from
    `2.0` to `6.0`; short utterances use repeat padding through the chunking pipeline.
  - Eval/dev embedding crops during training report: max/full chunk `6.0s`, chunk
    overlap `1.5s`, embedding pooling `mean`.
  - VAD/trim during training dataloading: `vad.mode="none"`; no waveform trim in the
    training loader.
  - Augmentation during these runs: no waveform augmentation was active in
    `ManifestSpeakerDataset`; `silence_augmentation.enabled=false`. The base
    `augmentation_scheduler` config existed but was not consumed by this production
    dataloader path.
  - Sampler: `BalancedSpeakerBatchSampler`; batches select balanced speakers and one
    utterance request per selected speaker where possible.
  - Loss/classifier: cosine classifier plus `ArcMarginLoss`.
  - Precision: `bf16` autocast on CUDA.
  - Epochs: `10`.
  - Dataloader workers: `6`, `prefetch_factor=4`, persistent workers enabled.

- `eres2netv2_h100_b128_20260411_200735` exact training run:
  - Config: `configs/training/eres2netv2-participants-candidate.toml`
  - GPU: `CUDA_VISIBLE_DEVICES=0`
  - CLI overrides: `training.batch_size=128`, `training.eval_batch_size=128`
  - Effective batch size: `128`; effective eval batch size: `128`.
  - Model config: `embedding_size=192`, `m_channels=64`, `base_width=26`, `scale=2`,
    `expansion=2`, `num_blocks=[3,4,6,3]`, `pooling_func="TSTP"`,
    `two_embedding_layers=false`.
  - Objective: `classifier_blocks=0`, `classifier_hidden_dim=192`, `scale=32.0`,
    `margin=0.3`, `easy_margin=false`.
  - Optimizer/scheduler: `sgd`, cosine schedule, `learning_rate=0.12`,
    `min_learning_rate=0.00005`, `momentum=0.9`, `weight_decay=0.0001`,
    `warmup_epochs=2`, `gradient_accumulation_steps=1`, `grad_clip_norm=5.0`.
  - Train log: `artifacts/logs/eres2netv2_h100_b128_20260411_200735.log`
  - Output root:
    `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee`
  - Checkpoint:
    `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt`
  - Training command:

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_baseline.py \
  --model eres2netv2 \
  --config configs/training/eres2netv2-participants-candidate.toml \
  --project-override training.batch_size=128 \
  --project-override training.eval_batch_size=128 \
  --device cuda
```

  - Public C4-tail command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group train python scripts/run_torch_checkpoint_c4_tail.py \
  --model eres2netv2 \
  --checkpoint-path artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee \
  --experiment-id P1_eres2netv2_h100_b128_public_c4 \
  --shift-mode none \
  --device cuda \
  --search-device cuda \
  --batch-size 1024
```

  - Public C4-tail effective defaults not shown in the command:
    `precision=bf16`, `search_batch_size=2048`, `top_cache_k=100`,
    `crop_seconds=6.0`, `n_crops=3`, `trim=true`, `edge_top=10`,
    `shared_min_count=0`, `seed=42`.
  - Public LB: `0.2410`.

- `campp_h100_b1024_20260411_200846` exact training run:
  - Config: `configs/training/campp-participants-candidate.toml`
  - GPU: `CUDA_VISIBLE_DEVICES=1`
  - CLI overrides: `training.batch_size=1024`, `training.eval_batch_size=1024`
  - Effective batch size: `1024`; effective eval batch size: `1024`.
  - Model config: `embedding_size=512`, `growth_rate=32`, `bottleneck_scale=4`,
    `init_channels=128`, `head_channels=32`, `head_res_blocks=[2,2]`,
    `block_layers=[12,24,16]`, `block_kernel_sizes=[3,3,3]`,
    `block_dilations=[1,2,2]`, `memory_efficient=true`.
  - Objective: `classifier_blocks=0`, `classifier_hidden_dim=512`, `scale=32.0`,
    `margin=0.2`, `easy_margin=false`.
  - Optimizer/scheduler: `adamw`, cosine schedule, `learning_rate=0.0015`,
    `min_learning_rate=0.00005`, `momentum=0.9`, `weight_decay=0.0001`,
    `warmup_epochs=1`, `gradient_accumulation_steps=1`, `grad_clip_norm=5.0`.
  - Train log: `artifacts/logs/campp_h100_b1024_20260411_200846.log`
  - Output root:
    `artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317`
  - Checkpoint:
    `artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317/campp_encoder.pt`
  - Training command:

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
uv run --group train python scripts/run_baseline.py \
  --model campp \
  --config configs/training/campp-participants-candidate.toml \
  --project-override training.batch_size=1024 \
  --project-override training.eval_batch_size=1024 \
  --device cuda
```

  - Public C4-tail command:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --group train python scripts/run_torch_checkpoint_c4_tail.py \
  --model campp \
  --checkpoint-path artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317/campp_encoder.pt \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/campp/20260411T200858Z-757aa9406317 \
  --experiment-id P2_campp_h100_b1024_public_c4 \
  --shift-mode none \
  --device cuda \
  --search-device cuda \
  --batch-size 1024
```

  - Public C4-tail effective defaults not shown in the command:
    `precision=bf16`, `search_batch_size=2048`, `top_cache_k=100`,
    `crop_seconds=6.0`, `n_crops=3`, `trim=true`, `edge_top=10`,
    `shared_min_count=0`, `seed=42`.
  - Public LB: `0.1753`.

Important remote-path fix:

- `dense_gallery_manifest.csv` contained local absolute paths under
  `<repo-root>`.
- For `remote`, wrote
  `artifacts/eda/baseline_fixed_dense_shifted_v2_honest/dense_gallery_manifest.remote.csv`
  with `resolved_path` rewritten to `<repo-root>`.

Next evaluation:

- Run C4-tail dense shifted v2 eval first for CAM++ on free GPU1.
- Run C4-tail dense shifted v2 eval for ERes2NetV2 after the final report process releases
  GPU0, or run it on GPU1 if GPU0 remains occupied by reporting.

Active C4-tail evaluation:

- Experiment id: `E2_campp_h100_b1024_c4_dense_shifted_v2`
- Run id: `campp_c4_dense_shifted_v2_20260412_061705`
- Checkpoint:
  `artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317/campp_encoder.pt`
- Manifest:
  `artifacts/eda/baseline_fixed_dense_shifted_v2_honest/dense_gallery_manifest.remote.csv`
- Output dir:
  `artifacts/backbone_eval/campp-candidate/20260411T200858Z-757aa9406317`
- Log: `artifacts/logs/campp_c4_dense_shifted_v2_20260412_061705.log`
- Status at launch: alive on GPU1, manifest rows `38500`, embedding extraction started.

Result:

- `E2_campp_h100_b1024_c4_dense_shifted_v2`
  - `p10 = 0.3552`
  - `top1_accuracy = 0.7721818181818182`
  - `top1_score_mean = 0.7131559252738953`
  - `top10_mean_score_mean = 0.5882105231285095`
  - `embedding_s = 907.987723`
  - `search_s = 0.169722`
  - `rerank_s = 2.48215`
  - Summary:
    `artifacts/backbone_eval/campp-candidate/20260411T200858Z-757aa9406317/E2_campp_h100_b1024_c4_dense_shifted_v2_summary.json`

Active follow-up evaluation:

- Experiment id: `E3_eres2netv2_h100_b128_c4_dense_shifted_v2`
- Run id: `eres2netv2_c4_dense_shifted_v2_20260412_063505`
- Checkpoint:
  `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt`
- Manifest:
  `artifacts/eda/baseline_fixed_dense_shifted_v2_honest/dense_gallery_manifest.remote.csv`
- Output dir:
  `artifacts/backbone_eval/eres2netv2-candidate/20260411T200748Z-15ced4a6d3ee`
- Status at launch: GPU0 was free; C4-tail evaluation started on GPU0.

Public leaderboard candidate generation:

- Hypothesis: trained participant backbones should be tested on public LB through the same
  C4-tail graph/retrieval postprocess, because dense shifted local `P@10` is not enough
  to decide whether the new embeddings transfer to the public pool.
- ERes2NetV2 dense-val eval was stopped early to prioritize public submission generation.
- Public manifest:
  `artifacts/eda/backbone_public/test_public_manifest.remote.csv`
  (`134697` rows, `resolved_path` rewritten for the `remote` container).
- `P1_eres2netv2_h100_b128_public_c4`
  - Checkpoint:
    `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt`
  - Output dir:
    `artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee`
  - Log: `artifacts/logs/eres2netv2_public_c4_20260412_063759.log`
  - Status at launch: alive on GPU0, public extraction started.
  - Follow-up: stopped at about `10%` extraction because running both public extraction
    processes in parallel was CPU/IO-bound (`~21 rows/s` each), delaying the first
    leaderboard candidate. ERes2NetV2 public generation should be restarted after the
    CAM++ public submission is produced.
  - Follow-up 2: restarted immediately after confirming the server has enough CPU/disk
    capacity and the user prefers both public candidates to run at once. New run id:
    `eres2netv2_public_c4_restart_20260412_065600`.
  - Result artifact:
    `artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/submission_P1_eres2netv2_h100_b128_public_c4.csv`
  - Validator: passed, `134697` rows, `k=10`, `0` errors.
  - Runtime summary: `embedding_s=4785.937889`, `search_s=0.642428`,
    `rerank_s=9.132642`.
  - Public confidence summary: `top1_score_mean=0.7809972167015076`,
    `top10_mean_score_mean=0.7344830632209778`.
  - Public LB: `0.2410`.
- `P2_campp_h100_b1024_public_c4`
  - Checkpoint:
    `artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317/campp_encoder.pt`
  - Output dir:
    `artifacts/backbone_public/campp/20260411T200858Z-757aa9406317`
  - Log: `artifacts/logs/campp_public_c4_20260412_063800.log`
  - Status at launch: alive on GPU1, public extraction started.
  - Follow-up: kept running as the first public candidate because CAM++ already has
    `P@10=0.3552` on dense shifted v2.
  - Result artifact:
    `artifacts/backbone_public/campp/20260411T200858Z-757aa9406317/submission_P2_campp_h100_b1024_public_c4.csv`
  - Validator: passed, `134697` rows, `k=10`, `0` errors.
  - Runtime summary: `embedding_s=5150.635756`, `search_s=0.791136`,
    `rerank_s=11.580039`.
  - Public confidence summary: `top1_score_mean=0.7773728966712952`,
    `top10_mean_score_mean=0.7309617400169373`.
  - Public LB: `0.1753`.

Conclusion:

- The public leaderboard confirms the backbone switch is a real improvement over the
  previous best C4 baseline (`0.1249`).
- `ERes2NetV2 + C4-tail` is the new production candidate: `0.2410`, a `+0.1161`
  absolute gain over `C4_b8_labelprop_mutual10`.
- `CAM++ + C4-tail` improved over C4 too (`0.1753`) but underperformed ERes2NetV2 by
  `0.0657` absolute on public despite a strong dense shifted local `P@10=0.3552`.
- The dense shifted local result did not rank CAM++ and ERes2NetV2 correctly because
  ERes2NetV2 dense-val was interrupted before completion. Finish ERes2NetV2 dense-val
  and add public LB as the authoritative rank check for this cycle.
