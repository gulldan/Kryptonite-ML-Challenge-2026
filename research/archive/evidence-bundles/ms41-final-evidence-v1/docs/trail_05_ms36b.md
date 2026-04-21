# 2026-04-14 — MS36b Multi-Teacher Soft Pseudo Fast Retry

Rationale:

- The first MS36 launch failed because the stability-dropout branch was too strict about
  requiring exactly top-300 fused candidates after dropping one teacher.
- The expensive/fragile piece is not needed for the first hypothesis check. The core test
  is multi-teacher soft supervision vs single-teacher hard pseudo labels.
- Retry with a narrower graph: source top-160, fused top-200, cluster rank top-200,
  no teacher-dropout stability, `min_stability=0.0`.

Remote launch:

- Run id: `MS36b_multiteacher_soft_pseudo_fast_public_c4_20260414T1922Z`.
- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=0`.
- Detached PID: `494118`.
- Log:
  `artifacts/logs/MS36b_multiteacher_soft_pseudo_fast_public_c4_20260414T1922Z.log`.
- PID file:
  `artifacts/logs/MS36b_multiteacher_soft_pseudo_fast_public_c4_20260414T1922Z.pid`.
- Latest pointer: `artifacts/logs/latest_MS36_multiteacher_soft_pseudo_public_c4.txt`.

Builder command differences from MS36:

- `--source-top-k 160`
- `--top-cache-k 200`
- `--shared-edge-top 30`
- `--shared-top 40`
- `--cluster-edge-top 20`
- `--cluster-reciprocal-top 60`
- `--cluster-rank-top 200`
- `--cluster-shared-min-count 2`
- `--soft-rank-top 70`
- no `--stability-drop-teacher`
- `--min-stability 0.0`

Training and public tail:

- Same config: `configs/training/campp-ms36-multiteacher-soft-pseudo-lowlr.toml`.
- Same init checkpoint:
  `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt`.
- Same loss weights: real hard loss `1.0`, pseudo soft loss `0.55`,
  pseudo hard loss `0.0`.
- Same post-train public C4 tail through packed official frontend cache.

Status:

- Launched detached on remote GPU0 at `2026-04-14T19:22:09Z`.
- Builder completed after writing fused top-200 caches:
  `artifacts/manifests/pseudo_ms36_multiteacher/indices_ms36_multiteacher_soft_top200.npy`
  and
  `artifacts/manifests/pseudo_ms36_multiteacher/scores_ms36_multiteacher_soft_top200.npy`.
- Fused graph diagnostics: teacher count `5`, top1 agreement mean `4.6256`, top10
  agreement mean `4.3386`.
- Cluster graph diagnostics: `5214` clusters before pseudo filtering, p50 size `17`,
  p95 `66`, p99 `95`, max `606`; `15` oversized clusters split, cluster used share
  `0.9161`.
- Pseudo manifest output: `114129` pseudo rows, `3271` pseudo clusters, `773933` mixed
  training rows, confidence mean `0.7689`, p10 `0.5897`, p50 `0.7564`.
- Training started on GPU0 after builder completion. First logged batch:
  epoch `1/2`, batch `1/3024`, loss `22.481453`, accuracy `0.0`, throughput
  `58.5` examples/s.
- Training completed successfully. Checkpoint:
  `artifacts/baselines/campp-ms36-multiteacher-soft-pseudo-lowlr/20260414T192542Z-29e95947d4f1/campp_encoder.pt`.
- Training summary:
  `artifacts/baselines/campp-ms36-multiteacher-soft-pseudo-lowlr/20260414T192542Z-29e95947d4f1/training_summary.json`.
- Final training metrics: epoch 1 loss `14.588967`, acc `0.376638`, LR `2e-5`;
  epoch 2 loss `8.223330`, acc `0.851156`, LR `2e-6`; final hard loss
  `7.013661`, soft loss `2.199399`.
- Public C4 tail completed and validator passed. C4 diagnostics:
  `top10_mean_score_mean=0.7735468`, `top1_score_mean=0.8223155`,
  label_used_share `0.8894`, Gini@10 `0.3220`, max in-degree `42`,
  rerank time `9.756s`.
- Exact public tail validator also passed. Exact diagnostics:
  `top10_mean_score_mean=0.7836804`, `top1_score_mean=0.8264713`,
  Gini@10 `0.4463`, max in-degree `104`.
- C4 submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms36b_multiteacher_soft_MS36b_multiteacher_soft_pseudo_fast_public_c4_20260414T1922Z/submission_MS36b_multiteacher_soft_pseudo_fast_public_c4_20260414T1922Z_c4.csv`,
  SHA-256 `2c16e42bea22bedfb241473d05dafdcefc89c3db7ab6f6de3ca3ac111fac4623`.
- Exact submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms36b_multiteacher_soft_MS36b_multiteacher_soft_pseudo_fast_public_c4_20260414T1922Z/submission_MS36b_multiteacher_soft_pseudo_fast_public_c4_20260414T1922Z_exact.csv`,
  SHA-256 `dccbd9b3099038008bc7c9b902a209096b7e97e12ba62ef78d9c4ca527bc881b`.
- Completed at `2026-04-14T20:34:02Z`. Decision before public LB: keep as a
  submission candidate, but do not infer hidden improvement from the local C4 metric alone
  because MS34 showed a strong local/hidden divergence.
- Public LB result for the C4 submission: `0.6906`.
- Decision after public LB: rejected as an MS32 replacement. The result is `-0.0473`
  below MS32 `0.7379`, `-0.0112` below MS31 `0.7018`, slightly above MS35 `0.6884`,
  and above MS34 `0.6791`, but the key finding is negative: the best-looking local C4
  diagnostic in this family (`0.7735`) did not transfer to hidden public scoring.
- Lesson: after MS32, stronger pseudo-pool confidence, weighted pseudo labels, and
  multi-teacher soft uncertainty all fail the hidden/public transfer test in their current
  form. Do not spend more submission budget on MS32-derived pseudo refinements unless a
  new validation signal explains this divergence.
