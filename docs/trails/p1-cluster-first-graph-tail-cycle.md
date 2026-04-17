# P1 Cluster-First Graph Tail Cycle

Date: 2026-04-12

Hypothesis:

- The public task is transductive over the full `test_public` pool, and each hidden
  speaker should have at least `K+1` utterances. `P1_eres2netv2_h100_b128_public_c4`
  is the safe branch, but the C4 tail is still only a local label-propagation preference
  over top-10 mutual edges.
- A broader mutual-kNN graph with shared-neighbor filtering and cluster-first retrieval
  may recover speaker communities better than C4, provided oversized graph components
  are controlled.

Implementation:

- New reusable code: `src/kryptonite/eda/community.py`
  - `ClusterFirstConfig`
  - `cluster_first_rerank()`
  - weighted mutual edge builder with shared-neighbor counts
  - optional label-size penalty to prevent broad graph label collapse
  - cluster assignment export for pseudo-label/self-training follow-up
- New CLI: `scripts/run_cluster_first_tail.py`
- Source embeddings:
  `artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/embeddings_P1_eres2netv2_h100_b128_public_c4.npy`
- Manifest:
  `artifacts/eda/backbone_public/test_public_manifest.remote.csv`
- Top-k cache generated once on `remote`:
  `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g1/indices_G1_p1_clusterfirst_mutual50_shared2_top300_top300.npy`
  and matching `scores_*.npy`.

Remote runs:

| Run | Config summary | Validator | Key graph diagnostics | Artifact | Decision |
| --- | --- | --- | --- | --- | --- |
| `G1_p1_clusterfirst_mutual50_shared2_top300` | top-300 cache, `edge_top=50`, `reciprocal_top=100`, `shared_min_count=2`, no size penalty | passed | `cluster_used_share=0.0260`, p99 cluster size `679.26`, max cluster `22352`, Gini@10 `0.3600` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g1/` | Rejected/diagnostic: graph collapsed into giant communities and mostly fell back. |
| `G2_p1_clusterfirst_mutual20_shared4_top300` | cached top-300, `edge_top=20`, `reciprocal_top=50`, `shared_min_count=4`, no size penalty | passed | `cluster_used_share=0.3176`, p99 `110.84`, max `8964`, Gini@10 `0.3519` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g2_shared4/` | Diagnostic: stricter graph helps, but oversized labels remain too large. |
| `G3_p1_clusterfirst_mutual20_shared4_penalty_top300` | G2 + `self_weight=1.0`, `label_size_penalty=0.35` | passed | `cluster_used_share=0.0018`, p99 `1.0`, max `55`, Gini@10 `0.3348` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g3_penalty/` | Rejected/diagnostic: penalty is too strong and destroys communities into singletons. |
| `G4_p1_clusterfirst_mutual20_shared4_penalty010_top300` | G2 + `label_size_penalty=0.10` | passed | `cluster_used_share=0.5965`, p50 same-cluster candidates `5`, p99 `101`, max `1602`, Gini@10 `0.3519` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g4_penalty/` | Kept as conservative cluster-first candidate. |
| `G5_p1_clusterfirst_mutual20_shared4_penalty015_top300` | G2 + `label_size_penalty=0.15` | passed | `cluster_used_share=0.6985`, p50 same-cluster candidates `7`, p99 `93.77`, max `1020`, Gini@10 `0.3503`, max in-degree `64` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g5_penalty/` | Kept as balanced high-upside candidate. |
| `G6_p1_clusterfirst_mutual20_shared4_penalty020_top300` | G2 + `label_size_penalty=0.20` | passed | `cluster_used_share=0.7552`, p50 same-cluster candidates `7`, p99 `84.66`, max `521`, Gini@10 `0.3460`, max in-degree `65` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g6_penalty/` | Submitted; public LB `0.2369`; rejected as production candidate. |

Logged reruns for candidate artifacts:

- `G5` log:
  `artifacts/logs/20260412T093233Z_p1_clusterfirst_g5_penalty.log`
- `G6` log:
  `artifacts/logs/20260412T093233Z_p1_clusterfirst_g6_penalty.log`
- Process note: `G2`-`G4` were foreground diagnostics and their terminal logs were not
  persisted; this was corrected by rerunning the retained candidates `G5` and `G6` with
  `tee` logs.

Representative retained command (`G6`):

```bash
cd <repo-root>
PYTHONUNBUFFERED=1 uv run --group train python scripts/run_cluster_first_tail.py \
  --indices-path artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g1/indices_G1_p1_clusterfirst_mutual50_shared2_top300_top300.npy \
  --scores-path artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g1/scores_G1_p1_clusterfirst_mutual50_shared2_top300_top300.npy \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g6_penalty \
  --experiment-id G6_p1_clusterfirst_mutual20_shared4_penalty020_top300 \
  --top-cache-k 300 \
  --edge-top 20 \
  --reciprocal-top 50 \
  --rank-top 300 \
  --iterations 8 \
  --cluster-min-size 5 \
  --cluster-max-size 160 \
  --cluster-min-candidates 3 \
  --shared-top 50 \
  --shared-min-count 4 \
  --split-edge-top 8 \
  --self-weight 0.0 \
  --label-size-penalty 0.20
```

Comparison to safe `P1+C4`:

- `P1+C4` remains the only scored safe branch: public LB `0.2410`.
- `P1+C4` submission hubness from its CSV: Gini@10 `0.3510`, max in-degree `69`.
- `G5`/`G6` are structurally different but not wild: mean top-10 neighbor overlap with
  `P1+C4` is `6.32` for `G5` and `6.25` for `G6`; p50 overlap is `7` for both.
- `G6` reduces Gini@10 to `0.3460` and raises cluster usage to `0.7552`, but mean
  submitted-neighbor cosine falls from `P1+C4` `0.73448` to `0.72667`. This is exactly
  the intended high-upside tradeoff: trust community structure more than local pairwise
  score.

Decision:

- Submitted `G6` as the high-upside public slot:
  `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g6_penalty/submission_G6_p1_clusterfirst_mutual20_shared4_penalty020_top300.csv`
- Public LB: `0.2369`.
- Decision: rejected as production candidate because it is below
  `P1_eres2netv2_h100_b128_public_c4 = 0.2410` by `0.0041`.
- Keep `P1_eres2netv2_h100_b128_public_c4` as the production fallback.
- `G5` remains available as a more conservative unsubmitted diagnostic:
  `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g5_penalty/submission_G5_p1_clusterfirst_mutual20_shared4_penalty015_top300.csv`
- Use exported cluster assignment CSVs from `G5`/`G6` as the first pseudo-label pool for
  the next self-training experiment, after filtering oversized clusters and low-neighbor
  support rows.
- Next direction: stop spending public slots on graph-only variants over P1. The graph
  tail is useful but the public miss confirms the next likely bottleneck is representation
  quality, so move to an orthogonal pretrained encoder before fusion or self-training.
