# 2026-04-12 — Public Gap Diagnostic: Order Leak And Validation Mismatch

Context:

- User reported that the public leaderboard has reached about `0.71`, while current best
  local branch `P3_eres2netv2_g6_pseudo_ft_public_c4` is only `0.2861`.
- Goal: check whether the gap is caused by a submission/id bug, a public CSV ordering
  leak, or a deeper representation/validation problem.

Checks:

- Submission format/id path:
  - All scored public submissions checked locally have validator pass status:
    `134697/134697` rows, `K=10`, no duplicate/self/out-of-range neighbor indices.
  - `test_public.csv` is exactly `test_public/000000.flac` through
    `test_public/134696.flac` in row order, so neighbor integers are row indices.
- Train-order leak:
  - `train.csv` is fully grouped by speaker in row order: `11053` consecutive speaker
    runs for `11053` speaker ids.
  - On train order alone, simple neighbor heuristics would score very high:
    forward `i+1..i+10` gives `P@10 ~= 0.912`; symmetric `i±1..i±5` gives
    `P@10 ~= 0.951`.
- Public-order leak probe:
  - Generated diagnostic public submissions under `artifacts/diagnostics/order_leak/`:
    `submission_order_forward10.csv`, `submission_order_backward10.csv`, and
    `submission_order_symmetric5.csv`.
  - All three pass `scripts/validate_submission.py`.
  - Public audio statistics do not show row adjacency structure: lag-1 correlations for
    duration/RMS/peak/silence/rolloff/centroid are approximately `0`.
  - P1 public embeddings also do not show row adjacency structure: lag-1/2/5/10 cosine
    distributions are effectively the same as random pairs. Example P1 lag-1 mean
    `0.1949`, random mean `0.1967`; lag-1 p50 `0.1459`, random p50 `0.1474`.
  - Existing P1/P3/E1 submissions almost never select nearby row indices:
    P3 has only `0.000154` of neighbor slots within `±10` and `0.000756` within `±50`.

Interpretation:

- There is no evidence of a simple public row-order leak, despite train being grouped.
  The order-only submissions are kept as optional cheap LB probes only because they pass
  validation and directly test the hypothesis.
- No local evidence points to a submission row/id bug.
- Follow-up public probe: direct pretrained H1 WavLM scored only `0.1228`, so raw
  off-the-shelf WavLM embeddings also do not explain the `0.71` leaderboard gap.
- Follow-up user report: default ModelScope CAM++ VoxCeleb model
  `iic/speech_campplus_sv_en_voxceleb_16k` scored `0.5695` without challenge
  fine-tuning. This sharply points to backbone/provenance rather than submission format
  as the main gap.
- The main gap is more likely that the current representation is too weak for hidden
  public clustering and current local validation is not a faithful public proxy. The
  strongest immediate direction is domain-adapted/pretrained speaker-recognition
  backbones or stronger pseudo-label/fusion over public embeddings, not raw pretrained
  inference and not more trust in train-derived speaker-disjoint validation.
