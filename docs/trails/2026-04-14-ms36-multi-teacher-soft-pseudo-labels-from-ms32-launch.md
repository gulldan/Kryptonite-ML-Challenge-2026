# 2026-04-14 — MS36 Multi-Teacher Soft Pseudo Labels From MS32 Launch

Hypothesis:

- MS32 proved that public pseudo-label self-training transfers, while MS34 showed that
  stricter hard consensus plus a clean finish can damage hidden geometry.
- This run tests the opposite direction: keep uncertainty from a committee of public
  teachers instead of forcing one hard pseudo label per row.
- Teachers: MS32, MS31, MS30, MS1 exact-reference TensorRT cache, and H7c official
  3D-Speaker ERes2Net-large. H7c is included as the orthogonal geometry source, not as a
  direct public-score candidate.
- Student starts from the MS32 checkpoint. Real participant rows use hard ArcMargin;
  public pseudo rows use confidence-weighted soft CE over up to three pseudo clusters.
  There is no MS34-style clean finish.

Repository changes for this run:

- Added `scripts/build_multi_teacher_soft_pseudo_manifests.py`.
- Added `scripts/run_campp_soft_pseudo_finetune.py`.
- Added `src/kryptonite/eda/soft_pseudo_stability.py`.
- Added `configs/training/campp-ms36-multiteacher-soft-pseudo-lowlr.toml`.
- Local code state at launch: commit `a0232f1` plus the uncommitted MS35 GLL changes,
  the MS36 additions above, and this experiment-history entry. Only the MS36 files were
  synced to remote for this launch to avoid overwriting unrelated local work.

Remote smoke before launch:

- Full-row top-20 smoke completed on remote before the detached run.
- Teacher paths and shapes were validated for all five caches.
- Smoke fused graph: top1 teacher agreement mean `4.2827`, top10 agreement mean `3.5221`.
- Smoke cluster graph: `18665` clusters, p50 size `4`, p95 `26`, p99 `40`, max `115`.
- Smoke pseudo output: `129711` pseudo rows, `13680` pseudo clusters, mixed train rows
  `789515`, confidence mean `0.7154`, p10 `0.5098`, p50 `0.7208`.
- Smoke artifacts: `artifacts/tmp/ms35_smoke/` on remote. This path is diagnostic only;
  the real run writes to `artifacts/manifests/pseudo_ms36_multiteacher/`.

Full multi-teacher soft pseudo manifest command:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --group train \
  python scripts/build_multi_teacher_soft_pseudo_manifests.py \
  --public-manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --base-train-manifest artifacts/manifests/participants_fixed/train_manifest.jsonl \
  --output-dir artifacts/manifests/pseudo_ms36_multiteacher \
  --experiment-id ms36_multiteacher_soft \
  --label-prefix pseudo_ms36_soft_ \
  --dataset-name participants_ms36_multiteacher_soft_pseudo \
  --teacher name=MS32,indices=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/indices_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy,scores=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/scores_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy,weight=1.25 \
  --teacher name=MS31,indices=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_voxblink2_aug_20260413T2029Z/indices_MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z_top200.npy,scores=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_voxblink2_aug_20260413T2029Z/scores_MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z_top200.npy,weight=1.0 \
  --teacher name=MS30,indices=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms30_lowlr_official_aug_20260413T2018Z/indices_MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z_top200.npy,scores=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms30_lowlr_official_aug_20260413T2018Z/scores_MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z_top200.npy,weight=0.9 \
  --teacher name=MS1,indices=artifacts/backbone_public/modelscope_campplus_voxceleb_default/full_reference_compare_20260413T2245/indices_MS11_full_tensorrt_cache_readonly_exact_20260413T2245_top100.npy,scores=artifacts/backbone_public/modelscope_campplus_voxceleb_default/full_reference_compare_20260413T2245/scores_MS11_full_tensorrt_cache_readonly_exact_20260413T2245_top100.npy,weight=0.9 \
  --teacher name=H7c,indices=artifacts/backbone_public/official_3dspeaker_eres2net_large_20260413T/full_b128_bf16/indices_H7c_eres2net_large_3dspeaker_pretrained_public_c4_b128_bf16_20260413T0418Z_top200.npy,scores=artifacts/backbone_public/official_3dspeaker_eres2net_large_20260413T/full_b128_bf16/scores_H7c_eres2net_large_3dspeaker_pretrained_public_c4_b128_bf16_20260413T0418Z_top200.npy,weight=0.7 \
  --source-top-k 200 \
  --top-cache-k 300 \
  --shared-edge-top 50 \
  --shared-top 50 \
  --cluster-edge-top 30 \
  --cluster-reciprocal-top 80 \
  --cluster-rank-top 300 \
  --cluster-shared-min-count 3 \
  --soft-rank-top 80 \
  --soft-top-clusters 3 \
  --min-pseudo-cluster-size 8 \
  --max-pseudo-cluster-size 90 \
  --min-teacher-agreement 0.40 \
  --min-confidence 0.50 \
  --min-stability 0.35 \
  --stability-drop-teacher H7c
```

Soft-pseudo training plan:

- Run id: `MS36_multiteacher_soft_pseudo_public_c4_20260414T1913Z`.
- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=0`.
- Detached PID: `493545`.
- Log: `artifacts/logs/MS36_multiteacher_soft_pseudo_public_c4_20260414T1913Z.log`.
- PID file: `artifacts/logs/MS36_multiteacher_soft_pseudo_public_c4_20260414T1913Z.pid`.
- Latest pointer: `artifacts/logs/latest_MS36_multiteacher_soft_pseudo_public_c4.txt`.
- Config path: `configs/training/campp-ms36-multiteacher-soft-pseudo-lowlr.toml`.
- Init checkpoint:
  `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt`.
- Train manifest:
  `artifacts/manifests/pseudo_ms36_multiteacher/ms36_multiteacher_soft_mixed_train_manifest.jsonl`.
- Model family: CAM++ `512d`, official ModelScope/3D-Speaker architecture.
- Frontend/crop: `features.frontend="official_campp"`, fixed 6s train crops, no VAD,
  eval 6s chunks with mean pooling.
- Seed/environment: runtime seed `42`, remote container `container`, repo path
  `<repo-root>`, bf16 precision, batch `256`.
- Optimizer/scheduler: AdamW, LR `2e-5`, min LR `2e-6`, weight decay `5e-5`, cosine,
  warmup `1`, max epochs `2`, grad clip `5.0`.
- Objective: hard ArcMargin scale `32.0`, margin `0.2` on real participant labels;
  confidence-weighted soft CE on pseudo rows with soft-loss multiplier `0.55` and
  `pseudo_hard_loss_weight=0.0`.
- Augmentation: conservative official-CAM++ public-shift augmentation, max one augmentation
  per sample, clean probability `0.80 -> 0.55`, no meaningful heavy branch.
- Training summary/result JSON:
  `artifacts/logs/MS36_multiteacher_soft_pseudo_public_c4_20260414T1913Z_train_result.json`
  after training completes.

Post-train public tail command:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --group train \
  python scripts/run_official_campp_tail.py \
  --checkpoint-path "$CHECKPOINT" \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms36_multiteacher_soft_MS36_multiteacher_soft_pseudo_public_c4_20260414T1913Z \
  --experiment-id MS36_multiteacher_soft_pseudo_public_c4_20260414T1913Z \
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

- Launched detached on remote GPU0 at `2026-04-14T19:13:18Z`.
- Failed at `2026-04-14T19:21Z` during the stability-dropout variant after the first full
  top-300 fuse finished. Error:
  `ValueError: Row 9 has only 291 fused candidates.`
- Root cause: the dropout stability variant removed H7c but still requested fused top-300.
  With four teachers and high overlap, some rows have fewer than 300 unique candidates.
  The main graph build had already written top-300 caches, but the run never reached
  manifest writing or training.
