# 2026-04-14 — MS33 Tail Ablations On MS32

Hypothesis:

- MS32 is the current strongest encoder/checkpoint branch, but the C4 label-propagation
  tail may be over-constraining a high-quality local neighbourhood graph.
- Before longer training, test three cheap public submission candidates from the cached
  MS32 embeddings: exact top-k without C4, a weaker conservative C4, and reciprocal/local
  density ranking only with hard label-propagation selection disabled.

Remote launch:

- Run id: `MS33_ms32_tail_ablations_20260414T1616Z`.
- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=0`.
- Detached PID: `484542`.
- Log: `artifacts/logs/MS33_ms32_tail_ablations_20260414T1616Z.log`.
- Latest pointer: `artifacts/logs/latest_MS33_ms32_tail_ablations.txt`.
- Source checkpoint recorded for provenance:
  `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt`.
- Source embeddings:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/embeddings_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z.npy`.
- Command path: `scripts/run_official_campp_tail.py` with
  `--embeddings-path`, `--top-cache-k 200`, `--search-device cuda`,
  `--mode segment_mean`, `--eval-chunk-seconds 6.0`, `--segment-count 3`,
  `--long-file-threshold-seconds 6.0`.

Variants and results:

- `MS33a_ms32_exact_no_c4_20260414T1616Z`: `--skip-c4`. Validator passed.
  Exact `top10_mean_score_mean=0.6671717`, Gini@10 `0.4454046`, max in-degree `137`.
  Submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms33_ms32_exact_no_c4_20260414T1616Z/submission.csv`.
- `MS33b_ms32_weak_c4_conservative_20260414T1616Z`: `edge_top=6`,
  `reciprocal_top=15`, `rank_top=80`, `iterations=3`, label sizes `[6,80]`,
  `label_min_candidates=4`, `shared_min_count=1`, reciprocal bonus `0.02`,
  density penalty `0.015`. Validator passed. C4 `top10_mean_score_mean=0.6565515`,
  label_used_share `0.7685`, Gini@10 `0.3477857`, max in-degree `66`. Submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms33_ms32_weak_c4_conservative_20260414T1616Z/submission.csv`.
- `MS33c_ms32_reciplocal_only_20260414T1616Z`: label usage disabled by
  `label_min_size=1000000` and `label_min_candidates=1000000`, leaving only
  reciprocal/local-density adjusted ranking. Validator passed. C4-path
  `top10_mean_score_mean=0.6635115`, label_used_share `0.0`, Gini@10 `0.3096333`,
  max in-degree `54`. Submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms33_ms32_reciplocal_only_20260414T1616Z/submission.csv`.
  Short local copy for upload:
  `artifacts/submissions/MS33c_ms32_reciplocal_only_submission.csv`.
  SHA-256:
  `16156cf223878569be64ca876ec9cbf1dd0bccdb09d61cb43ee2878d08c777c4`.
  Public LB score: `0.6980`.

Status:

- Completed at `2026-04-14T16:18:07Z`.
- MS33c public LB result: `0.6980`, which is below MS32 C4 `0.7379` and slightly below
  MS31 C4 `0.7018`.
- Decision: reject reciprocal/local-only as an MS32 replacement. The hubness reduction
  looks attractive locally, but hidden LB says the hard C4 label-propagation selection in
  MS32 is still carrying useful speaker structure. MS33a exact no-C4 and MS33b weak C4
  remain unsubmitted diagnostics rather than priority candidates.
