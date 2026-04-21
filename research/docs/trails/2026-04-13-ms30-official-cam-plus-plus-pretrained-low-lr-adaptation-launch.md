# 2026-04-13 — MS30 Official CAM++ Pretrained Low-LR Adaptation Launch

Hypothesis:

- The strong `MS1_modelscope_campplus_voxceleb_default` branch should be treated as the
  reference geometry, not the local-fbank converted-weight path.
- A conservative stage-wise adaptation from that pretrained CAM++ checkpoint may improve
  challenge/public alignment if the official frontend is used during both training and
  public inference.
- This first run tests only the supervised participant-train stage with light public-shift
  augmentation. Filtered public pseudo-label self-training is deliberately left for the
  next stage after confirming this checkpoint does not destroy the MS1 neighborhood
  structure.

Remote launch:

- Run id: `MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z`.
- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=0`.
- Local code changes synced to remote before launch: `configs/schema.json`,
  `configs/training/campp-ms1-official-participants-lowlr.toml`,
  `scripts/run_campp_finetune.py`, `scripts/README.md`,
  `src/kryptonite/config.py`, `src/kryptonite/features/fbank.py`,
  `src/kryptonite/features/cache.py`, `tests/unit/test_fbank.py`, and this history file.
- Training config: `configs/training/campp-ms1-official-participants-lowlr.toml`.
- Init checkpoint:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt`.
- Train manifest: `artifacts/manifests/participants_fixed/train_manifest.jsonl`.
- Dev manifest: `artifacts/manifests/participants_fixed/dev_manifest.jsonl`.
- Model: CAM++ `512d`, ModelScope VoxCeleb pretrained initialization.
- Frontend: `features.frontend="official_campp"`, implemented as
  `torchaudio.compliance.kaldi.fbank(..., dither=0.0)` followed by utterance mean
  normalization.
- Crop/preprocess policy: no VAD, fixed 6s training crops with repeat padding for short
  utterances; dev export uses 6s chunks and mean pooling.
- Augmentation policy: light public-shift scheduler with silence/pause, codec/channel,
  distance, noise/reverb if banks are present, and mild speed perturbation.
- Optimizer/scheduler: AdamW, LR `1e-4`, min LR `1e-5`, weight decay `5e-5`, cosine
  schedule, warmup `1` epoch, grad clip `5.0`.
- Precision/batch: bf16, batch `256`, eval batch `256`, max epochs `8`, early stopping on
  train loss after min epoch `4`, stop train accuracy threshold `0.9975`.
- Output root: `artifacts/baselines/campp-ms1-official-participants-lowlr/`.
- Tracking root: `artifacts/tracking/`.
- Public output root:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms30_lowlr_official_aug_20260413T2018Z/`.
- Remote log:
  `artifacts/logs/MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z.log`.
- Latest pointer:
  `artifacts/logs/latest_MS30_campp_ms1_official_lowlr_train_aug_public_c4.txt`.

Launch command:

```bash
cd <repo-root>
RUN_ID=MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --group train \
  python scripts/run_campp_finetune.py \
  --config configs/training/campp-ms1-official-participants-lowlr.toml \
  --init-checkpoint artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt \
  --device cuda \
  --output json

CHECKPOINT=$(python - <<'PY'
from pathlib import Path
root = Path("artifacts/baselines/campp-ms1-official-participants-lowlr")
runs = sorted((path for path in root.iterdir() if (path / "campp_encoder.pt").is_file()), key=lambda path: path.stat().st_mtime)
print(runs[-1] / "campp_encoder.pt")
PY
)

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --group train \
  python scripts/run_official_campp_tail.py \
  --checkpoint-path "$CHECKPOINT" \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms30_lowlr_official_aug_20260413T2018Z \
  --experiment-id "$RUN_ID" \
  --encoder-backend torch \
  --device cuda \
  --search-device cuda \
  --batch-size 512 \
  --search-batch-size 2048 \
  --top-cache-k 200 \
  --mode segment_mean \
  --eval-chunk-seconds 6.0 \
  --segment-count 3 \
  --long-file-threshold-seconds 6.0
```

Status at launch: running, PID `463160`. First training progress reached
`epoch=1/8 batch=1/2578` with GPU0 memory around `23.2 GiB`.

Completion:

- Training completed on remote GPU0 on 2026-04-14. Final epoch metrics:
  `train_loss=0.748006`, `train_accuracy=0.978814`, `learning_rate=1e-5`.
- Final checkpoint:
  `artifacts/baselines/campp-ms1-official-participants-lowlr/20260413T202123Z-d45cc7d9936e/campp_encoder.pt`.
- Training summary:
  `artifacts/baselines/campp-ms1-official-participants-lowlr/20260413T202123Z-d45cc7d9936e/training_summary.json`.
- The original chained wrapper stopped after training because `${OUT}` did not exist before
  redirecting `selected_checkpoint.txt`. Public tail was rerun standalone without
  retraining.
- Public tail rerun log:
  `artifacts/logs/MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z_public_tail_resume.log`.
- Public tail command:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --group train \
  python scripts/run_official_campp_tail.py \
  --checkpoint-path artifacts/baselines/campp-ms1-official-participants-lowlr/20260413T202123Z-d45cc7d9936e/campp_encoder.pt \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms30_lowlr_official_aug_20260413T2018Z \
  --experiment-id MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z \
  --encoder-backend torch \
  --device cuda \
  --search-device cuda \
  --batch-size 512 \
  --search-batch-size 2048 \
  --top-cache-k 200 \
  --mode segment_mean \
  --eval-chunk-seconds 6.0 \
  --segment-count 3 \
  --long-file-threshold-seconds 6.0 \
  --frontend-pack-dir artifacts/cache/campp-official-public-ms1-v1-pack
```

Public tail result:

- Embedding extraction from packed official frontend cache: `162.070616s`.
- Exact top-k search: `0.943175s`.
- C4 rerank: `9.600895s`.
- C4 validator: passed, `134697/134697` rows, `0` errors.
- C4 graph stats: `top1_score_mean=0.6888875`, `top10_mean_score_mean=0.6260672`,
  `label_count=14236`, `label_used_share=0.816321`, Gini@10 `0.336347`,
  max in-degree@10 `68`.
- Submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms30_lowlr_official_aug_20260413T2018Z/submission_MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z_c4.csv`.
- Validation report:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms30_lowlr_official_aug_20260413T2018Z/submission_MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z_c4_validation.json`.
- Summary:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms30_lowlr_official_aug_20260413T2018Z/MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z_summary.json`.
- Submission SHA-256:
  `d18faab64f94eb74ad40fc3539b86a21e76ca05b1a09f4700699b52a9bda3eb3`.
- Public LB: `0.6953` from external upload, reported by user on 2026-04-14.
- Decision: accepted as new public best. The low-LR supervised adaptation improves MS1 by
  `+0.1258` absolute (`0.6953` vs `0.5695`) and should become the new safe CAM++ base
  for filtered public pseudo-label self-training and conservative fusion.
