# 2026-04-14 — MS34 Strict-Consensus Pseudo + Clean Finish Launch

Hypothesis:

- MS32's gain proves pseudo-label self-training is useful, but the MS31-only pseudo pool
  may still include noisy public components.
- Build a stricter pseudo pool from the MS32 teacher graph and keep only nodes in
  non-bloated teacher clusters where MS31, MS30, and optional MS1 exact neighbours agree.
- Run a short low-LR pseudo stage, then a short clean finish on real participant labels
  with no on-the-fly augmentation to reduce pseudo-label drift.

Repository changes for this run:

- Added `scripts/build_strict_consensus_pseudo_manifests.py`.
- Added `configs/training/campp-ms32-strict-consensus-pseudo-lowlr.toml`.
- Added `configs/training/campp-ms34-clean-finish-real-labels.toml`.
- Local code state at launch: commit `c37af3d` plus uncommitted additions above and the
  existing uncommitted VoxBlink/CN-Celeb documentation/script changes noted in this file.

Strict-consensus pseudo manifest:

```bash
cd <repo-root>
uv run --group train python scripts/build_strict_consensus_pseudo_manifests.py \
  --teacher-indices artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/indices_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy \
  --teacher-scores artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/scores_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy \
  --confirmation-indices artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_voxblink2_aug_20260413T2029Z/indices_MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z_top200.npy \
  --confirmation-indices artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms30_lowlr_official_aug_20260413T2018Z/indices_MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z_top200.npy \
  --confirmation-submission-csv artifacts/backbone_public/campp/default_model_submission.csv \
  --public-manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --base-train-manifest artifacts/manifests/participants_fixed/train_manifest.jsonl \
  --output-dir artifacts/manifests/pseudo_ms32_consensus \
  --experiment-id ms32_strict_consensus \
  --min-pseudo-cluster-size 8 \
  --max-pseudo-cluster-size 60 \
  --teacher-margin-compare-rank 20 \
  --teacher-margin-quantile 0.25 \
  --confirmation-top-k 20 \
  --min-confirmed-same-cluster 2 \
  --min-confirming-models 2 \
  --min-selected-rows-per-cluster 6 \
  --min-cluster-selected-share 0.60
```

- Builder completed before training.
- Teacher cluster graph: `5694` clusters, p50 size `17`, p95 `62`, p99 `81`, max `479`,
  split oversized rows `3018`, cluster_used_share `0.9437`.
- Confirmation shares: MS31 `0.8955`, MS30 `0.8767`, MS1 exact CSV `0.7312`.
- Margin threshold: top1 minus top20 teacher score `0.0942879`.
- Pseudo rows: `68833`; pseudo clusters: `2452`; mixed train rows: `728637`.
- Pseudo manifest:
  `artifacts/manifests/pseudo_ms32_consensus/ms32_strict_consensus_pseudo_manifest.jsonl`.
- Mixed manifest:
  `artifacts/manifests/pseudo_ms32_consensus/ms32_strict_consensus_mixed_train_manifest.jsonl`.
- Row audit:
  `artifacts/manifests/pseudo_ms32_consensus/ms32_strict_consensus_row_audit.csv`.

Remote launch:

- Run id: `MS34_ms32_strict_consensus_pseudo_clean_20260414T1616Z`.
- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=1`.
- Detached PID: `485712`.
- Log: `artifacts/logs/MS34_ms32_strict_consensus_pseudo_clean_20260414T1616Z.log`.
- PID file: `artifacts/logs/MS34_ms32_strict_consensus_pseudo_clean_20260414T1616Z.pid`.
- Latest pointer: `artifacts/logs/latest_MS34_ms32_strict_consensus_pseudo_clean.txt`.

Pseudo-stage training:

- Config path: `configs/training/campp-ms32-strict-consensus-pseudo-lowlr.toml`.
- Init checkpoint:
  `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt`.
- Train manifest:
  `artifacts/manifests/pseudo_ms32_consensus/ms32_strict_consensus_mixed_train_manifest.jsonl`.
- Dev manifest: `artifacts/manifests/participants_fixed/dev_manifest.jsonl`,
  capped by `max_dev_rows=1024`.
- Model family: CAM++ `512d`, official ModelScope/3D-Speaker architecture.
- Frontend/crop: `features.frontend="official_campp"`, fixed 6s train crops, no VAD,
  eval 6s chunks with mean pooling.
- Seed/environment: runtime seed `42`, remote container `container`, repo path
  `<repo-root>`, bf16 precision, batch `256`.
- Optimizer/scheduler: AdamW, LR `3e-5`, min LR `3e-6`, weight decay `5e-5`, cosine,
  warmup `1`, max epochs `2`, grad clip `5.0`.
- Objective: ArcMargin scale `32.0`, margin `0.2`.
- Augmentation: conservative, max one augmentation per sample; clean probability
  `0.75 -> 0.50`, light `0.20 -> 0.30`, medium `0.05 -> 0.15`, heavy `0.0 -> 0.05`.
- Pseudo-stage completed 2 epochs.
- Epoch metrics: epoch 1 train loss `8.830437`, train accuracy `0.645242`;
  epoch 2 train loss `4.295517`, train accuracy `0.940896`.
- Score gap: `0.61312`.
- Output root:
  `artifacts/baselines/campp-ms32-strict-consensus-pseudo-lowlr/20260414T161910Z-a8428f5d9063/`.
- Checkpoint:
  `artifacts/baselines/campp-ms32-strict-consensus-pseudo-lowlr/20260414T161910Z-a8428f5d9063/campp_encoder.pt`.
- Training summary:
  `artifacts/baselines/campp-ms32-strict-consensus-pseudo-lowlr/20260414T161910Z-a8428f5d9063/training_summary.json`.

Clean-finish stage:

- Config path: `configs/training/campp-ms34-clean-finish-real-labels.toml`.
- Init checkpoint: the pseudo-stage checkpoint written to
  `artifacts/logs/MS34_ms32_strict_consensus_pseudo_clean_20260414T1616Z_pseudo_checkpoint.txt`
  when the pseudo stage completes.
- Train manifest: `artifacts/manifests/participants_fixed/train_manifest.jsonl`.
- Dev manifest: `artifacts/manifests/participants_fixed/dev_manifest.jsonl`,
  capped by `max_dev_rows=1024`.
- Model/frontend/crop: same official CAM++ frontend and fixed 6s crops.
- Precision/batch: bf16, batch `256`, eval batch `256`.
- Optimizer/scheduler: AdamW, LR `1e-5`, min LR `2e-6`, weight decay `5e-5`, cosine,
  warmup `1`, max epochs `2`, grad clip `5.0`.
- Augmentation policy: disabled (`augmentation_scheduler.enabled=false`,
  `silence_augmentation.enabled=false`).
- Clean checkpoint and public tail: pending.
- Clean finish completed 2 epochs.
- Epoch metrics: epoch 1 train loss `10.879819`, train accuracy `0.487163`;
  epoch 2 train loss `8.263546`, train accuracy `0.801037`.
- Score gap: `0.617363`.
- Output root:
  `artifacts/baselines/campp-ms34-clean-finish-real-labels/20260414T172208Z-d482435b2354/`.
- Checkpoint:
  `artifacts/baselines/campp-ms34-clean-finish-real-labels/20260414T172208Z-d482435b2354/campp_encoder.pt`.
- Training summary:
  `artifacts/baselines/campp-ms34-clean-finish-real-labels/20260414T172208Z-d482435b2354/training_summary.json`.

Planned public tail:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --group train \
  python scripts/run_official_campp_tail.py \
  --checkpoint-path "$CLEAN_CHECKPOINT" \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms34_strict_consensus_clean_20260414T1616Z \
  --experiment-id MS34_ms32_strict_consensus_pseudo_clean_20260414T1616Z \
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

Status:

- Launched at `2026-04-14T16:18:18Z`.
- Builder completed and pseudo-stage completed.
- Clean finish completed and public tail completed at `2026-04-14T18:21:26Z`.
- Public tail used packed official frontend cache hits for all `134697` rows.
- Public tail metrics: embedding `157.135s`, exact top-k search `0.7388s`,
  C4 rerank `9.659s`, exact `top10_mean_score_mean=0.7336834`,
  C4 `top10_mean_score_mean=0.7225211`, C4 `top1_score_mean=0.7765661`,
  label_used_share `0.8824`, Gini@10 `0.333266`, max in-degree `54`.
- Validator: passed, `134697/134697` rows, `k=10`, `error_count=0`.
- Remote submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms34_strict_consensus_clean_20260414T1616Z/submission.csv`
  (short copy of
  `submission_MS34_ms32_strict_consensus_pseudo_clean_20260414T1616Z_c4.csv`).
- Local upload copy:
  `artifacts/submissions/MS34_ms32_strict_consensus_pseudo_clean_submission.csv`.
- SHA-256:
  `ba9fb0e3d2088dee3c0e911cf9b21cff32a929e338d815520256cbe495a1da8a`.
- Public LB score: `0.6791`.
- Decision: rejected. It keeps MS32-like hubness (`Gini@10 0.3333`, max in-degree `54`)
  and raises local public graph score sharply versus MS32 C4 (`0.7225` vs `0.6564`),
  but hidden LB is much worse than MS32 `0.7379` and below MS31 `0.7018`.
- Finding: this is a clear local/public divergence. The strict-consensus pseudo filter
  and/or clean finish made neighbourhoods look cleaner under public graph diagnostics while
  damaging the hidden scoring geometry. Do not treat `top10_mean_score_mean` alone as a
  sufficient selection metric for late-stage MS32 variants.
