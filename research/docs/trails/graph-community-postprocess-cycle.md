# Graph / Community Postprocess Cycle

Hypothesis: because public retrieval is transductive over the full test pool and each
speaker has at least `K+1` utterances, graph/community structure should add signal beyond
pairwise cosine ranking.

Implementation:

- Reusable code: `src/kryptonite/eda/community.py`
- Public runner: `scripts/run_public_graph_community.py`
- Input embeddings: `artifacts/eda/public_ablation_cycle/embeddings_B4_trim_3crop.npy`
- Output directory: `artifacts/eda/public_graph_community/`

Runs checked:

| Date | Experiment | Local honest shifted v2 | Public LB | Validator | Decision |
| --- | --- | ---: | ---: | --- | --- |
| 2026-04-11 | `C4_b8_labelprop_mutual10` | `0.2413` | `0.1249` | passed | Keep as current best. |
| 2026-04-11 | `C5_b8_labelprop_mutual10_shared2` | `0.2395` | not submitted | passed | Next public candidate if another submission is available. |
| 2026-04-11 | `C6_b8_labelprop_mutual15` | `0.2387` | not submitted | passed | Submit only after C5 or if broader graph looks necessary. |
| 2026-04-11 | `C1/C2/C3` connected-components variants | effectively B8 fallback | not submitted | passed | Rejected/diagnostic: mutual-kNN connected components collapse into giant components, so component guard falls back instead of giving useful communities. |

Current best public submission artifact:

- `artifacts/eda/public_graph_community/submission_C4_b8_labelprop_mutual10.csv`

Graph-cycle lesson:

- Label propagation over a mutual top-10 graph improves public from B8 `0.1223`
  to C4 `0.1249`.
- Connected-component community detection is too brittle on this graph because it forms
  giant components; deterministic weighted label propagation is a better first graph
  postprocess for this backbone.
- The public gain is real but small, so graph postprocess is useful as a safe layer,
  while the larger gap to leaderboard leaders still points to a stronger encoder/backbone
  plus graph/community inference.
