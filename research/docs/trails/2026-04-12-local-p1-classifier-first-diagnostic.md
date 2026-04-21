# 2026-04-12 — Local P1 Classifier-First Diagnostic

Hypothesis:

- The P1 ERes2NetV2 classifier head may contain useful train-speaker posterior signal
  that the public C4 path discarded by using only embeddings.
- If public is close to a closed-set/transductive speaker assignment problem, a
  class-aware retrieval pass should expose that by grouping public clips through
  classifier top-k classes before fallback embedding retrieval.

Implementation:

- Added reusable class-aware rerank logic in `src/kryptonite/eda/classifier_first.py`.
- Added CLI entrypoint `scripts/run_classifier_first_tail.py`.
- Copied only the required P1 artifacts from `remote` to local:
  - checkpoint:
    `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt`
  - public embeddings:
    `artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/embeddings_P1_eres2netv2_h100_b128_public_c4.npy`
- Ran locally on RTX 4090 using cached P1 embeddings, not audio extraction.

Command:

```bash
uv run --group train python scripts/run_classifier_first_tail.py \
  --checkpoint-path artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt \
  --embeddings-path artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/embeddings_P1_eres2netv2_h100_b128_public_c4.npy \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv datasets/Для\ участников/test_public.csv \
  --output-dir artifacts/backbone_public/classifier_first/20260412T_local_h1 \
  --experiment-id H1_p1_classifier_first_top1bucket \
  --indices-path artifacts/backbone_public/classifier_first/20260412T_local_h1/indices_H1_p1_classifier_first_top1bucket_top100.npy \
  --scores-path artifacts/backbone_public/classifier_first/20260412T_local_h1/scores_H1_p1_classifier_first_top1bucket_top100.npy \
  --top-cache-k 100 \
  --class-batch-size 4096 \
  --search-batch-size 2048
```

Local result:

- Validator: passed.
- Submission:
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/submission_H1_p1_classifier_first_top1bucket.csv`
- Summary:
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/H1_p1_classifier_first_top1bucket_summary.json`
- Classifier classes in checkpoint: `10848`; `speaker_to_index_count=10848`.
- `class_used_share=0.9913`.
- Top1 class bucket stats: `7398` non-empty classes, p50 size `3`, p95 `40.15`,
  p99 `167`, max `26811`.
- Hubness: `Gini@10=0.5953`, max in-degree `474`.
- Mean submitted score fell to `0.5605` from P1+C4 `0.7345`.
- Overlap with safe P1+C4 submission: mean `2.35/10`, p50 `2`, p10 `0`, p90 `5`;
  top1 neighbor unchanged only `19.92%` of rows.

Decision:

- Keep as diagnostic; do not spend a public slot on this raw hard class-first candidate
  unless a more conservative variant first reduces hubness.
- The classifier posterior is not immediately a clean closed-set assignment signal:
  the largest predicted class bucket contains `26811` public rows, which is incompatible
  with the expected ~12 rows per true speaker and likely indicates domain/miscalibration
  or overconfident common-class collapse.
- Next safe use of logits is as a soft edge feature inside P1 graph/community or a
  conservative rerank within existing P1 top-100, not as a hard top1 bucket assignment.
