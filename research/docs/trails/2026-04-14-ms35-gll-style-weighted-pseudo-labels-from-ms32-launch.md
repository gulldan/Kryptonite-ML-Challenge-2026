# 2026-04-14 — MS35 GLL-Style Weighted Pseudo Labels From MS32 Launch

Hypothesis:

- If MS32 already found the right official CAM++ representation family, then MS34 regressed
  because strict pseudo-label purification and the real-label clean finish discarded useful
  public structure or over-rotated the embedding geometry.
- The low-risk next check is therefore not a new backbone, new official frontend, or new
  retrieval tail. Start from the exact MS32 checkpoint, rebuild the public pseudo pool from
  MS32 teacher embeddings, and replace hard include/exclude filtering with weighted
  pseudo supervision.
- The intended signal is small: preserve MS32 neighbourhood geometry while letting lower
  confidence public rows contribute with lower weight, pseudo-only label smoothing, and an
  online assigned-label confidence gate.

Repository changes for this run:

- Added `scripts/build_weighted_pseudo_label_manifests.py`.
- Added pseudo-aware training metadata propagation through manifest rows and batches:
  `pseudo_sample_weight`, `pseudo_verified`, `source_dataset`, and `split`.
- Extended `ArcMarginLoss` and the shared training loop with per-sample weights,
  pseudo-only label smoothing, and GLL-style confidence gating.
- Added `configs/training/campp-ms35-gll-weighted-pseudo-lowlr.toml`.
- Local code state at launch: commit `a0232f1` plus the uncommitted MS35 changes above and
  this experiment-history entry.

Weighted pseudo manifest:

```bash
cd <repo-root>
uv run --group train python scripts/build_weighted_pseudo_label_manifests.py \
  --teacher-indices artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/indices_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy \
  --teacher-scores artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/scores_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy \
  --confirmation-indices artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms31_voxblink2_aug_20260413T2029Z/indices_MS31_campp_ms1_official_voxblink2_aug_public_c4_20260413T2029Z_top200.npy \
  --confirmation-indices artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms30_lowlr_official_aug_20260413T2018Z/indices_MS30_campp_ms1_official_lowlr_train_aug_public_c4_20260413T2018Z_top200.npy \
  --confirmation-submission-csv artifacts/backbone_public/campp/default_model_submission.csv \
  --public-manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --base-train-manifest artifacts/manifests/participants_fixed/train_manifest.jsonl \
  --output-dir artifacts/manifests/pseudo_ms35_gll \
  --experiment-id ms35_gll_weighted \
  --min-pseudo-cluster-size 5 \
  --max-pseudo-cluster-size 120 \
  --confidence-quantile 0.18 \
  --min-confidence 0.25 \
  --min-confirming-models 1 \
  --min-graph-support 4 \
  --min-reciprocal-support 1 \
  --label-prefix pseudo_ms35_gll_ \
  --dataset-name participants_ms35_gll_pseudo \
  --public-audio-prefix 'datasets/Для участников'
```

- Builder completed successfully before training.
- Teacher cluster graph: `5690` clusters, p50 size `17`, p95 `62`, p99 `81`, max
  `479`, split oversized rows `2311`, cluster_used_share `0.9481`.
- Flexible confidence threshold: `0.2839105` from `confidence_quantile=0.18` and
  `min_confidence=0.25`.
- Confirmation shares: MS31 `0.8955`, MS30 `0.8767`, MS1 exact CSV `0.7312`.
- Pseudo rows: `115156`; pseudo clusters: `3813`; mixed train rows: `774960`.
- Selection counts: confidence-selected rows `107660`, verification-selected rows `111833`
  with overlap between the two criteria.
- Pseudo weights over selected rows: mean `0.7358`, p10 `0.5585`, p50 `0.7512`,
  p90 `0.8824`.
- Pseudo manifest:
  `artifacts/manifests/pseudo_ms35_gll/ms35_gll_weighted_pseudo_manifest.jsonl`.
- Mixed manifest:
  `artifacts/manifests/pseudo_ms35_gll/ms35_gll_weighted_mixed_train_manifest.jsonl`.
- Row audit:
  `artifacts/manifests/pseudo_ms35_gll/ms35_gll_weighted_row_audit.csv`.

Remote launch:

- Run id: `MS35_ms32_gll_weighted_pseudo_public_c4_20260414T1904Z`.
- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=1`.
- Detached PID: `491780`.
- Log:
  `artifacts/logs/MS35_ms32_gll_weighted_pseudo_public_c4_20260414T1904Z.log`.
- PID file:
  `artifacts/logs/MS35_ms32_gll_weighted_pseudo_public_c4_20260414T1904Z.pid`.
- Latest pointer:
  `artifacts/logs/latest_MS35_ms32_gll_weighted_pseudo_public_c4.txt`.

Training plan:

- Config path: `configs/training/campp-ms35-gll-weighted-pseudo-lowlr.toml`.
- Init checkpoint:
  `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt`.
- Train manifest:
  `artifacts/manifests/pseudo_ms35_gll/ms35_gll_weighted_mixed_train_manifest.jsonl`.
- Dev manifest: `artifacts/manifests/participants_fixed/dev_manifest.jsonl`,
  capped by `max_dev_rows=1024`.
- Model family: CAM++ `512d`, official ModelScope/3D-Speaker architecture.
- Frontend/crop: `features.frontend="official_campp"`, fixed 6s train crops, no VAD,
  eval 6s chunks with mean pooling.
- Seed/environment: runtime seed `42`, remote container `container`, repo path
  `<repo-root>`, bf16 precision, batch `256`.
- Optimizer/scheduler: AdamW, LR `1.5e-5`, min LR `2e-6`, weight decay `5e-5`,
  cosine, warmup `1`, max epochs `2`, grad clip `5.0`.
- Objective: ArcMargin scale `32.0`, margin `0.2`; real rows label smoothing `0.0`,
  pseudo rows label smoothing `0.12`; pseudo loss multiplier `0.80`; GLL threshold
  ramps from `0.0` to `0.05`, with verified pseudo rows always eligible.
- Augmentation: same official frontend and fixed 6s crops as MS32, conservative
  MS31-style public-shift augmentation with no heavy branch and weak speed perturbation
  (`speed` family weight `0.10`).
- No clean-finish stage.

Post-train public tail command:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --group train \
  python scripts/run_official_campp_tail.py \
  --checkpoint-path "$CHECKPOINT" \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --data-root 'datasets/Для участников' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms35_gll_weighted_pseudo_20260414T1904Z \
  --experiment-id MS35_ms32_gll_weighted_pseudo_public_c4_20260414T1904Z \
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

- Launched detached on remote GPU1 at `2026-04-14T19:09:10Z`.
- Weighted pseudo manifest builder completed; training started with epoch `1/2`,
  `3028` batches per epoch.
- Status check at `2026-04-14T19:21Z`: an accidental duplicate process tree from the
  first malformed launch command was found and terminated. Canonical tracked PID `491780`
  remains active; GPU1 memory dropped to about `23.2GiB` and utilization remained `100%`.
  Training was at epoch `1/2`, batch `453/3028`, loss `14.351500`, accuracy `0.056317`.
- Training completed both epochs. Epoch 1 loss `11.104657`, accuracy `0.410490`;
  epoch 2 loss `7.502000`, accuracy `0.778071`; score gap `0.634104`.
- Checkpoint:
  `artifacts/baselines/campp-ms35-gll-weighted-pseudo-lowlr/20260414T191007Z-65020ae8c987/campp_encoder.pt`.
- Training summary:
  `artifacts/baselines/campp-ms35-gll-weighted-pseudo-lowlr/20260414T191007Z-65020ae8c987/training_summary.json`.
- Public tail completed at `2026-04-14T20:26:33Z`.
- Public tail used packed official frontend cache hits for all `134697` rows.
- Public tail metrics: embedding `161.304s`, exact top-k search `0.777s`,
  C4 rerank `9.659s`, exact `top10_mean_score_mean=0.7503617`, C4
  `top10_mean_score_mean=0.7400894`, C4 `top1_score_mean=0.7918007`,
  label_used_share `0.8768`, Gini@10 `0.3298`, max in-degree `55`.
- Validator: exact and C4 both passed.
- Submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms35_gll_weighted_pseudo_20260414T1904Z/submission.csv`.
- Upload copy:
  `artifacts/submissions/MS35_ms32_gll_weighted_pseudo_submission.csv`.
- SHA-256:
  `48bd2be5bda616dd91f32617a1fda8a87f330c6d1978c4b50d023caffb4ad39a`.
- Public LB score: `0.6884`.
- Decision: rejected as a replacement for MS32. The run is `+0.0093` above rejected MS34
  but still `-0.0495` below MS32 `0.7379` and `-0.0134` below MS31 `0.7018`.
  Weighted/GLL pseudo-label continuation made public graph diagnostics look excellent
  (`top10_mean_score_mean=0.7401`, Gini@10 `0.3298`, max in-degree `55`), but hidden LB
  again rejected the local graph signal. Finding: after MS32, more aggressive pseudo-label
  reuse around MS32 teacher clusters is not automatically safer than strict filtering; the
  current safe branch remains MS32, and future pseudo work needs a better hidden-transfer
  guard than public C4 geometry alone.
