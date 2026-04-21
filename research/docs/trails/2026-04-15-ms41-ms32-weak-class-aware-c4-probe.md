# 2026-04-15 — MS41 MS32 Weak Class-Aware C4 Probe

Hypothesis:

- The old P1 logit diagnostics showed that hard class assignment and class-first retrieval
  are unsafe, while weak classifier posterior evidence can be useful as a graph edge
  feature. `H3b_p1_classaware_c4_weak` was the best local continuation because it changed
  enough neighbours to be meaningful without creating large hubness.
- Transfer the same weak posterior-edge idea to the current safe CAM++ family by using
  `MS32` cached top-200 neighbours and the MS32 classifier head. Do not build buckets and
  do not perform class-first retrieval; only add a small bonus inside the existing top-200
  edge set before the standard C4 label-propagation tail.

Code change:

- Added `--class-cache-only` to `scripts/run_classifier_first_tail.py` so posterior top-k
  caches can be computed from a checkpoint and cached embeddings without running or
  writing a hard class-first retrieval candidate.
- Local verification before remote sync:

```bash
uv run ruff format scripts/run_classifier_first_tail.py
uv run ruff check scripts/run_classifier_first_tail.py
```

Run id:

- `MS41_ms32_classaware_c4_weak_20260415T0530Z`

Remote execution:

- Execution environment: `<redacted>`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=1`.
- Detached PID: `502938`.
- Log:
  `artifacts/logs/MS41_ms32_classaware_c4_weak_20260415T0530Z.log`.
- Latest pointer:
  `artifacts/logs/latest_MS41_ms32_classaware_c4_weak.txt`.
- Code state: base commit `8458459` plus local script change adding
  `--class-cache-only`; script synced directly to `remote` before launch.
- Source checkpoint:
  `artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt`.
- Source MS32 embeddings:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/embeddings_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z.npy`.
- Source MS32 top-200 cache:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/indices_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy`
  and
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/scores_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy`.

Commands:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --group train \
  python scripts/run_classifier_first_tail.py \
  --checkpoint-path artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt \
  --embeddings-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/embeddings_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z.npy \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z \
  --experiment-id MS41_ms32_classaware_c4_weak_20260415T0530Z_classcache \
  --device cuda \
  --class-batch-size 4096 \
  --class-top-k 5 \
  --class-scale 32.0 \
  --class-cache-only

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --group train \
  python scripts/run_class_aware_graph_tail.py \
  --indices-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/indices_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy \
  --scores-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/scores_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy \
  --class-indices-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/class_indices_MS41_ms32_classaware_c4_weak_20260415T0530Z_classcache_top5.npy \
  --class-probs-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/class_probs_MS41_ms32_classaware_c4_weak_20260415T0530Z_classcache_top5.npy \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z \
  --experiment-id MS41_ms32_classaware_c4_weak_20260415T0530Z \
  --class-overlap-top-k 3 \
  --class-overlap-weight 0.03 \
  --same-top1-bonus 0.01 \
  --same-query-topk-bonus 0.005 \
  --edge-top 10 \
  --reciprocal-top 20 \
  --rank-top 100 \
  --iterations 5 \
  --label-min-size 5 \
  --label-max-size 120 \
  --label-min-candidates 3 \
  --shared-top 20 \
  --shared-min-count 0 \
  --reciprocal-bonus 0.03 \
  --density-penalty 0.02
```

Artifacts:

- Posterior cache:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/class_indices_MS41_ms32_classaware_c4_weak_20260415T0530Z_classcache_top5.npy`
  and
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/class_probs_MS41_ms32_classaware_c4_weak_20260415T0530Z_classcache_top5.npy`.
- Summary:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/MS41_ms32_classaware_c4_weak_20260415T0530Z_summary.json`.
- Submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/submission_MS41_ms32_classaware_c4_weak_20260415T0530Z.csv`.
- Short submission copy:
  `artifacts/submissions/MS41_ms32_classaware_c4_weak_20260415T0530Z_submission.csv`.
- Overlap comparison:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms41_ms32_classaware_c4_weak_20260415T0530Z/MS41_ms32_classaware_c4_weak_20260415T0530Z_vs_MS32_overlap.json`.

Result:

- Completed at `2026-04-15T05:32:11Z`.
- Class cache: computed in `0.850s`; classifier classes `14021`, top-k `5`, scale `32.0`.
- Weak score adjustment: `class_overlap_top_k=3`, `class_overlap_weight=0.03`,
  same-top1 `0.01`, same-query-top3 `0.005`.
- Pre-C4 adjusted top1 changed share: `0.083647`.
- Validator: passed, `134697/134697` rows.
- C4 metrics: `top10_mean_score_mean=0.6859419`, `top1_score_mean=0.7525114`,
  label_used_share `0.9095`, label count `10592`, Gini@10 `0.3436619`,
  max in-degree `56`.
- Class edge diagnostics: posterior-overlap mean `0.1383`, p95 `0.9913`,
  same-top1 edge share `0.1670`, same-query-top3 edge share `0.3023`.
- Overlap vs MS32 C4: mean `8.6278/10`, median `9`, top1 equal `0.8776`,
  same neighbour set `0.4681`, exact same order `0.2548`.
- Submission SHA-256:
  `8b58013c3a710ef7e4c9f2fc5466ee9b2918d2ee271b5eaaa095b4976e194e84`.
- Public LB: `0.7473`.

Decision:

- Accept as the new public best and current safe branch.
- The perturbation profile matches the intended cheap probe: only `8.36%` pre-C4 top1
  changes and `8.63/10` mean overlap against MS32 after C4.
- Local C4 score improves substantially over MS32 (`0.6859` vs `0.6564`) and max in-degree
  stays controlled (`56` vs MS32 `57`). Gini is slightly worse (`0.3437` vs `0.3326`),
  but the hidden public result is better than MS32 by `+0.0094`, so the weak posterior
  edge signal is confirmed useful for this branch.
- This narrows the lesson from the old logit diagnostics: hard class assignment remains
  unsafe, but weak classifier posterior evidence can transfer when it is constrained to
  existing high-similarity edges.
