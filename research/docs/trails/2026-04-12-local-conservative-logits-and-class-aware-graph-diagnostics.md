# 2026-04-12 — Local Conservative Logits and Class-Aware Graph Diagnostics

Context:

- `H1_p1_classifier_first_top1bucket` rejected hard bucket assignment because it moved
  too far away from safe P1+C4 and created severe hubness.
- Follow-up question: are P1 classifier logits harmful in general, or only harmful when
  used as hard global buckets?

Implementation:

- Extended `src/kryptonite/eda/classifier_first.py` with:
  - `bucket_backfill=False` support for conservative top-k-only logits rerank;
  - `class_adjusted_topk()` for soft posterior-overlap score adjustment on an existing
    embedding top-k cache.
- Added `scripts/run_class_aware_graph_tail.py` for class-aware score adjustment before
  C4-style label propagation.
- All runs below reused cached local P1 public artifacts and did not perform audio
  extraction.

Inputs:

- Safe P1 submission:
  `artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/submission_P1_eres2netv2_h100_b128_public_c4.csv`
- P1 top-100 cache:
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/indices_H1_p1_classifier_first_top1bucket_top100.npy`
  and
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/scores_H1_p1_classifier_first_top1bucket_top100.npy`
- P1 classifier top-5 cache:
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/class_indices_H1_p1_classifier_first_top1bucket_top5.npy`
  and
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/class_probs_H1_p1_classifier_first_top1bucket_top5.npy`

Runs:

| Experiment | Local method | Validator | Gini@10 | Max in-degree | Mean overlap vs P1+C4 | Top1 same vs P1+C4 | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `H2_p1_logits_top100_conservative` | Conservative logits rerank restricted to P1 top-100, `bucket_backfill=false`, min class candidates `10`, class top-3. | passed | `0.5362` | `245` | `4.88/10` | `0.5337` | Rejected as public candidate. It is safer than hard H1 but still creates too much hubness and changes too much of safe P1. |
| `H3_p1_classaware_c4` | Class-aware score adjustment on P1 top-100 with posterior overlap weight `0.08`, same-top1 `0.02`, same-query-top3 `0.01`, then C4 labelprop. | passed | `0.3830` | `97` | `6.96/10` | `0.6637` | Kept as diagnostic. It proves logits are usable as soft graph weights, but hubness is still above P1. |
| `H3b_p1_classaware_c4_weak` | Weaker class-aware score adjustment on P1 top-100 with posterior overlap weight `0.03`, same-top1 `0.01`, same-query-top3 `0.005`, then C4 labelprop. | passed | `0.3619` | `71` | `7.89/10` | `0.7874` | Best local continuation. It is close to P1 hubness (`0.3510`, max `69`) while perturbing enough neighbors to be a plausible public slot if a low-risk logits test is desired. |

Artifacts:

- H2 submission:
  `artifacts/backbone_public/classifier_first/20260412T_local_h2_conservative/submission_H2_p1_logits_top100_conservative.csv`
- H2 summary:
  `artifacts/backbone_public/classifier_first/20260412T_local_h2_conservative/H2_p1_logits_top100_conservative_summary.json`
- H3 submission:
  `artifacts/backbone_public/class_aware_graph/20260412T_local_h3/submission_H3_p1_classaware_c4.csv`
- H3 summary:
  `artifacts/backbone_public/class_aware_graph/20260412T_local_h3/H3_p1_classaware_c4_summary.json`
- H3b submission:
  `artifacts/backbone_public/class_aware_graph/20260412T_local_h3b_weak/submission_H3b_p1_classaware_c4_weak.csv`
- H3b summary:
  `artifacts/backbone_public/class_aware_graph/20260412T_local_h3b_weak/H3b_p1_classaware_c4_weak_summary.json`
- Comparison JSON:
  `artifacts/backbone_public/classifier_first/20260412T_local_h2_h3_comparison.json`

Decision:

- Reject direct logits reranking (`H2`) for public: it still behaves like an assignment
  shortcut and increases hubness too much.
- Keep `H3b_p1_classaware_c4_weak` as the only reasonable local logits candidate. It
  supports the narrower conclusion: P1 classifier logits can be useful only as weak
  graph-edge evidence, not as hard class assignment.
- If using a public slot for a logits hypothesis, submit H3b rather than H1/H2/H3.
  Otherwise wait for the orthogonal WavLM branch and use the same weak class-aware graph
  idea only after checking complementarity.

Current decision:

- New safe public candidate: `P3_eres2netv2_g6_pseudo_ft_public_c4 = 0.2861`.
- Pseudo-label self-training is confirmed: P3 improves over safe P1 by `+0.0451`
  absolute on public. Next ERes2NetV2-side work should start from P3 artifacts, not P1:
  conservative P1+P3 fusion, P3 graph-tail tuning, and P3+orthogonal-pretrained fusion.
- Do not submit direct H1/H2/H3/H4/H5 pretrained HF xvector outputs. H2 collapsed after
  short naive fine-tuning; H1/H4 are useful orthogonal WavLM fusion candidates; H3/H5 are
  compressed Wav2Vec2 diagnostics.
- Let `wavlm_e1_domain_ft_20260412T130254Z` run as the main pretrained-first branch.
  After it finishes, run public C4 tail from its saved `hf_model/`, check pairwise cosine
  collapse, P1 overlap, Gini@10, max in-degree, and then test `P1 + WavLM-E1` fusion if
  the geometry is not collapsed.
