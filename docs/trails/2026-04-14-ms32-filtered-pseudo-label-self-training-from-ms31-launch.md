# 2026-04-14 — MS32 Filtered Pseudo-Label Self-Training From MS31 Launch

Hypothesis:

- `MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z` is the current
  strongest public branch at external LB `0.7018`. The next low-risk improvement is
  stage-wise public self-training from that checkpoint, not a scratch model.
- Use the public graph only as pseudo-label structure: no hidden public labels and no
  VoxBlink2 data are used. VoxBlink2 remains only the augmentation-style reference inherited
  from MS31.
- Keep pseudo-label filtering conservative by training only on clusters with size `[8,80]`,
  avoiding singleton noise and oversized public communities.

Pseudo-label pool:

- Pool id: `MS32a_ms31_clusterfirst_shared4_penalty020_top200_20260414T0551Z`.
- Source public cache:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_voxblink2_aug_20260413T2029Z/indices_MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z_top200.npy`
  and matching `scores_..._top200.npy`.
- Cluster-first settings: `edge_top=20`, `reciprocal_top=50`, `rank_top=200`,
  `iterations=8`, `cluster_min_size=5`, `cluster_max_size=160`,
  `cluster_min_candidates=3`, `shared_top=50`, `shared_min_count=4`,
  `split_edge_top=8`, `self_weight=0.0`, `label_size_penalty=0.20`.
- Output dir:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_clusterfirst_pseudo_pool_20260414T0551Z/`.
- Validator: passed.
- Pool metrics: `cluster_count=8102`, `cluster_used_share=0.8907176849`,
  cluster size p50 `5`, p95 `64`, p99 `92`, max `391`,
  `top10_mean_score_mean=0.6191661`, Gini@10 `0.3650773`, max in-degree `64`.
- Cluster assignments:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_clusterfirst_pseudo_pool_20260414T0551Z/clusters_MS32a_ms31_clusterfirst_shared4_penalty020_top200_20260414T0551Z.csv`.

Filtered pseudo manifest:

```bash
cd <repo-root>
uv run --group train python scripts/build_pseudo_label_manifests.py \
  --clusters-csv artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_clusterfirst_pseudo_pool_20260414T0551Z/clusters_MS32a_ms31_clusterfirst_shared4_penalty020_top200_20260414T0551Z.csv \
  --public-manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --base-train-manifest artifacts/manifests/participants_fixed/train_manifest.jsonl \
  --output-dir artifacts/manifests/pseudo_ms31 \
  --experiment-id ms31_filtered \
  --min-cluster-size 8 \
  --max-cluster-size 80 \
  --label-prefix pseudo_ms31_ \
  --dataset-name participants_ms31_pseudo \
  --public-audio-prefix 'datasets/Для участников'
```

- Pseudo rows: `104361`.
- Pseudo clusters: `3173`.
- Mixed train rows: `764165`.
- Pseudo manifest:
  `artifacts/manifests/pseudo_ms31/ms31_filtered_pseudo_manifest.jsonl`.
- Mixed manifest:
  `artifacts/manifests/pseudo_ms31/ms31_filtered_mixed_train_manifest.jsonl`.

Remote launch:

- Run id: `MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z`.
- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=0`.
- Detached PID: `478094`.
- Config path: `configs/training/campp-ms31-official-pseudo-filtered-lowlr.toml`.
- Init checkpoint:
  `artifacts/baselines/campp-ms1-official-participants-voxblink2-augment/20260413T203245Z-b87036ccb3db/campp_encoder.pt`.
- Train manifest:
  `artifacts/manifests/pseudo_ms31/ms31_filtered_mixed_train_manifest.jsonl`.
- Dev manifest: `artifacts/manifests/participants_fixed/dev_manifest.jsonl`, capped by
  `max_dev_rows=1024` to avoid all-pairs dev-trial blowup after checkpointing.
- Output root: `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/`.
- Public tail output root:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/`.
- Remote log:
  `artifacts/logs/MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z.log`.
- PID file:
  `artifacts/logs/MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z.pid`.
- Latest pointer:
  `artifacts/logs/latest_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4.txt`.

Training parameters:

- Model family: CAM++ `512d`, official ModelScope/3D-Speaker architecture.
- Frontend: `features.frontend="official_campp"`,
  `torchaudio.compliance.kaldi.fbank(dither=0.0)` with utterance mean normalization.
- Crop/preprocess policy: no VAD, fixed 6s train crops, eval 6s chunks with mean pooling.
- Precision/batch: bf16, train batch `256`, eval batch `256`, max epochs `4`.
- Optimizer/scheduler: AdamW, LR `5e-5`, min LR `5e-6`, weight decay `5e-5`, cosine,
  warmup `1`, grad clip `5.0`.
- Objective: ArcMargin scale `32.0`, margin `0.2`.
- Augmentation policy: conservative public-shift mix, max `2` augmentations/sample,
  clean probability `0.65 -> 0.35`, light `0.30 -> 0.35`, medium `0.05 -> 0.20`,
  heavy `0.00 -> 0.10`; family weights noise `1.10`, reverb `0.85`, distance `0.90`,
  codec `1.05`, silence `1.00`, speed `0.35`.

Post-train public tail command:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --group train \
  python scripts/run_official_campp_tail.py \
  --checkpoint-path "$CHECKPOINT" \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z \
  --experiment-id MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z \
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

- Launched detached on remote GPU0 at `2026-04-14T05:53:02Z`.
- Cluster export and pseudo manifest completed successfully before training.
- Training completed 4 epochs at `2026-04-14T08:01Z`. Final epoch loss `1.724956`,
  train accuracy `0.943825`, LR `5e-6`. Epoch history: epoch 1 loss `7.614209`,
  acc `0.682823`; epoch 2 loss `2.365201`, acc `0.934833`; epoch 3 loss `1.750846`,
  acc `0.945141`; epoch 4 loss `1.724956`, acc `0.943825`.
- Speaker classes during self-training: `14021` = `10848` original train speakers +
  `3173` pseudo public clusters.
- Checkpoint:
  `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt`.
- Training summary:
  `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/training_summary.json`.
- Public tail completed at `2026-04-14T08:06:30Z` using packed frontend cache hits for all
  `134697` public rows.
- Public C4 tail metrics: embedding `165.545s`, exact top-k search `0.900s`,
  C4 rerank `9.825s`, `c4_top10_mean_score_mean=0.6563528`,
  `c4_top1_score_mean=0.7203525`, label_used_share `0.8914`, Gini@10 `0.332575`,
  max in-degree `57`.
- Validator: passed, `134697/134697` rows, `k=10`, `error_count=0`.
- Submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/submission.csv`
  (short copy of
  `submission_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_c4.csv`).
- SHA-256:
  `239118b2d431f20a85ef1127e2bd681abf95314714e81b6009298d9b42939ffc`.
- Public LB score: `0.7379`.
- Decision: accepted as the new public best. Local public-graph metrics were substantially
  stronger than MS31 (`top10_mean_score_mean 0.6564` vs `0.6207`, Gini@10 `0.3326` vs
  `0.3380`, max in-degree `57` vs `91`), and the hidden public LB confirmed the gain:
  `+0.0361` absolute over MS31 `0.7018`, `+0.1684` over MS1 `0.5695`, and `+0.6600`
  over the organizer baseline `0.0779`.
