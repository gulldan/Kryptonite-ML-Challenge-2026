# 2026-04-15 — MS40 Rowwise Tail Router

Hypothesis:

- Earlier static fusion showed that global neighbour mixing can add noisy evidence and hurt
  public score. The row-wise alternative is to choose the retrieval policy per query based
  on confidence and agreement features instead of applying one global tail to every row.
- For low graph confidence, suspicious label size, or weak reciprocal support, exact or
  weak C4 may be safer than full C4. For high graph confidence and strong agreement,
  full C4 or class-aware C4 should remain safe. Soup is allowed only when row-wise
  consensus favours it.

Remote run:

- Run id: `MS40_rowwise_tail_router_20260415T0611Z`.
- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=1`.
- Log: `artifacts/logs/MS40_rowwise_tail_router_20260415T0611Z.log`.
- Output directory:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms40_rowwise_tail_router_20260415T0611Z/`.
- Short submission copy:
  `artifacts/submissions/MS40_rowwise_tail_router_20260415T0611Z_submission.csv`.
- Row diagnostics:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms40_rowwise_tail_router_20260415T0611Z/MS40_rowwise_tail_router_20260415T0611Z_row_diagnostics.parquet`.

Inputs:

- Manifest: `artifacts/eda/participants_public_baseline/test_public_manifest.csv`.
- Template: `datasets/Для участников/test_public.csv`.
- MS32 top-k graph:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/indices_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy`
  and matching scores file.
- Candidate policies:
  `classaware_c4=MS41`, `full_c4=MS32 C4`, `soup_c4=MS38a_i095 C4`,
  `weak_c4=MS33b`, `reciprocal_only=MS33c`, `exact=MS32 exact`.
- Class entropy source:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/class_probs_MS41_ms32_classaware_c4_weak_20260415T0530Z_classcache_top5.npy`.
- Duration source: `artifacts/eda/participants_audio6/file_stats.parquet`.

Command:

```bash
cd <repo-root>
RUN_ID=MS40_rowwise_tail_router_20260415T0611Z
OUT=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms40_rowwise_tail_router_20260415T0611Z
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --group train \
  python scripts/run_rowwise_tail_router.py \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv "datasets/Для участников/test_public.csv" \
  --indices-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/indices_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy \
  --scores-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/scores_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy \
  --candidate-csv full_c4=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/submission_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_c4.csv \
  --candidate-csv exact=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/submission_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_exact.csv \
  --candidate-csv weak_c4=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms33_ms32_weak_c4_conservative_20260414T1616Z/submission_MS33b_ms32_weak_c4_conservative_20260414T1616Z_c4.csv \
  --candidate-csv reciprocal_only=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms33_ms32_reciplocal_only_20260414T1616Z/submission_MS33c_ms32_reciplocal_only_20260414T1616Z_c4.csv \
  --candidate-csv soup_c4=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms38_weight_soup_20260415T0531Z/MS38a_i095_campp_ms31_ms32_interp_20260415T0531Z/submission_MS38a_i095_campp_ms31_ms32_interp_20260415T0531Z_c4.csv \
  --candidate-csv classaware_c4=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/submission_MS41_ms32_classaware_c4_weak_20260415T0530Z.csv \
  --class-probs-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/class_probs_MS41_ms32_classaware_c4_weak_20260415T0530Z_classcache_top5.npy \
  --file-stats-parquet artifacts/eda/participants_audio6/file_stats.parquet \
  --output-dir "$OUT" \
  --experiment-id "$RUN_ID"
```

Result:

- Runtime: `18.754458s`.
- Validator: passed, `134697/134697` rows, `0` errors.
- SHA-256:
  `b7171cffe374ccf232358a33f445a42fc22585d516796fde7c3abb32bbb4ace1`.
- Graph diagnostics from MS32 top-k: `label_usable_share=0.8914`,
  top1-top10 margin p10 `0.04678`, p50 `0.08365`, reciprocal support p50 `8`.
- Selected policy shares: class-aware C4 `79.33%`, full C4 `7.04%`,
  weak C4 `6.22%`, exact `3.72%`, soup C4 `3.30%`, reciprocal-only `0.38%`.
- Row-wise reason counts: low-entropy class-aware agreement `91617`,
  strong graph full C4 `9483`, weak C4 low graph confidence `8376`,
  strong graph class-aware agreement `7744`, class-aware default `7495`,
  extreme low graph confidence exact `5013`, soup consensus advantage `4451`,
  reciprocal low graph confidence `518`.
- Overlap vs MS41: mean `9.353/10`, p50 `10`, p10 `8`, top1 equal `93.68%`.
- Overlap vs MS32 full C4: mean `8.962/10`, p50 `10`, p10 `7`, top1 equal `90.72%`.
- Hubness: Gini@10 `0.3473`, max in-degree `108`.

Decision:

- Public LB: `0.7441`, reported by user on 2026-04-15 after submitting
  `artifacts/submissions/MS40_rowwise_tail_router_20260415T0611Z_submission.csv`.
- Delta: `+0.6662` vs organizer baseline `0.0779`, `+0.0062` vs MS32 `0.7379`,
  `+0.0045` vs MS38a_i095 `0.7396`, and `-0.0032` vs MS41 `0.7473`.
- Reject as a replacement for MS41. The condition-aware router is useful evidence and
  does beat the older MS32/weight-soup public scores, but it underperforms the simpler
  MS41 class-aware C4 branch. The higher max in-degree (`108`) was a valid risk signal.
  Next router attempt should be calibrated against MS41 specifically rather than using
  broad consensus gates that route too many rows away from the best branch.
