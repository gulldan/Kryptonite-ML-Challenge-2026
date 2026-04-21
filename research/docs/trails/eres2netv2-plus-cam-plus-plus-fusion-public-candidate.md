# ERes2NetV2 + CAM++ Fusion Public Candidate

Date: 2026-04-12

Hypothesis:

- `ERes2NetV2 + C4-tail` is now the safe branch at public LB `0.2410`, but CAM++ may
  still carry complementary neighbor evidence despite being weaker alone (`0.1753`).
- Fuse the public top-neighbor graphs from both trained backbones, then reuse the same
  C4-style label propagation tail. This tests whether backbone complementarity improves
  the transductive speaker-community graph without retraining.

Implementation:

- New reusable helper:
  `src/kryptonite/eda/fusion.py`
- New CLI:
  `scripts/run_backbone_fusion_c4_tail.py`
- Fusion method: exact top-200 from each embedding space, row-wise union, rank/robust-score
  fusion, then top-100 fused cache into existing `LabelPropagationConfig`.
- Fusion config:
  - `left_name=eres2netv2`, `left_weight=0.75`
  - `right_name=campp`, `right_weight=0.25`
  - `source_top_k=200`
  - `top_cache_k=100`
  - `rank_weight=1.0`
  - `score_z_weight=0.15`
- C4-tail config:
  - `edge_top=10`
  - `reciprocal_top=20`
  - `rank_top=100`
  - `iterations=5`
  - `label_min_size=5`
  - `label_max_size=120`
  - `label_min_candidates=3`
  - `shared_min_count=0`
  - `reciprocal_bonus=0.03`
  - `density_penalty=0.02`

Remote command:

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_backbone_fusion_c4_tail.py \
  --left-embeddings artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/embeddings_P1_eres2netv2_h100_b128_public_c4.npy \
  --right-embeddings artifacts/backbone_public/campp/20260411T200858Z-757aa9406317/embeddings_P2_campp_h100_b1024_public_c4.npy \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/fusion/20260412T085023Z \
  --experiment-id F1_eres075_cam025_rankscore_public_c4 \
  --left-weight 0.75 \
  --right-weight 0.25 \
  --source-top-k 200 \
  --top-cache-k 100 \
  --search-device cuda \
  --search-batch-size 2048
```

Result before public LB:

- Experiment id: `F1_eres075_cam025_rankscore_public_c4`
- Submission:
  `artifacts/backbone_public/fusion/20260412T085023Z/submission_F1_eres075_cam025_rankscore_public_c4.csv`
- Summary:
  `artifacts/backbone_public/fusion/20260412T085023Z/F1_eres075_cam025_rankscore_public_c4_summary.json`
- Validator: passed, `134697` rows, `k=10`, `0` errors.
- Runtime: `left_search_s=2.422092`, `right_search_s=0.809668`,
  `fusion_elapsed_s=60.584358`, `rerank_s=9.081155`.
- Fusion overlap: source top-200 overlap `p50=66`, `p95=100`; enough disagreement exists
  for fusion to be a real hypothesis, not a duplicate of ERes2NetV2.
- Public graph diagnostics: `label_used_share=0.63449`, `label_size_max=267`,
  `indegree_gini_10=0.46110`, `indegree_max_10=153`.

Decision:

- Public LB: `0.2305`.
- The run is below `P1_eres2netv2_h100_b128_public_c4 = 0.2410` by `0.0105`, so it is
  rejected as the production candidate.
- Interpretation: naive rank/robust-score fusion injected too much noisy CAM++ neighbor
  evidence into the ERes2NetV2 graph. The top-200 overlap confirmed complementarity, but
  that complementarity was not clean enough under the current fusion weights and C4 tail.
- Keep `P1_eres2netv2_h100_b128_public_c4` as the safe branch.
- Do not run a broad blind fusion sweep before improving calibration. Any later fusion
  should be gated by local/public rank checks and should try more conservative CAM++
  influence, for example CAM++ only as a tie-breaker or mutual-confirmation bonus.
