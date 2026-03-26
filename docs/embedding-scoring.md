# Embedding Scoring

Shared L2 normalization and cosine scoring now live in `src/kryptonite/models/scoring.py`.

The goal is to keep one contract for:

- offline verification scoring in baseline/training pipelines;
- pairwise and one-to-many score computation;
- thin HTTP scoring endpoints used by the current infer adapter.

## Core API

Available helpers:

- `l2_normalize_embeddings(...)`
- `average_normalized_embeddings(...)`
- `cosine_score_pairs(...)`
- `cosine_score_matrix(...)`
- `rank_cosine_scores(...)`

Rules:

- inputs may be a single embedding vector (`[dim]`) or a batch (`[batch, dim]`);
- all values must be finite;
- zero-norm embeddings are rejected instead of being silently accepted;
- pairwise scoring requires aligned shapes;
- one-to-many scoring requires matching embedding dimension between queries and references.

## Offline Integration

`src/kryptonite/training/speaker_baseline.py::score_trials()` now scores trials in batches through the shared cosine scorer.

Practical effect:

- trial scoring no longer assumes exported embeddings were already normalized correctly;
- score artifacts and verification reports keep the same file contract;
- mean positive / mean negative / score gap remain derived from the shared scorer output.

## HTTP Endpoints

The thin adapter in `src/kryptonite/serve/http.py` now exposes embedding-based JSON endpoints in addition to `/healthz` and `/readyz`.

### `POST /score/pairwise`

Request:

```json
{
  "left": [[1.0, 0.0], [1.0, 1.0]],
  "right": [[1.0, 0.0], [1.0, -1.0]],
  "normalize": true
}
```

Response fields:

- `mode`
- `normalized`
- `embedding_dim`
- `score_count`
- `scores`
- `mean_score`

### `POST /score/one-to-many`

Request:

```json
{
  "queries": [[1.0, 0.0]],
  "references": [[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]],
  "query_ids": ["probe-a"],
  "reference_ids": ["ref-perfect", "ref-close", "ref-opposite"],
  "top_k": 2,
  "normalize": true
}
```

Response fields:

- `scores`: full query-to-reference cosine matrix;
- `top_matches`: sorted top-k references per query.

### `POST /enroll`

Stores an in-memory enrollment embedding by:

1. L2-normalizing each provided embedding;
2. averaging them;
3. L2-normalizing the pooled enrollment vector.

This endpoint is intentionally ephemeral and process-local. It exists to validate the scorer/backend contract before the later full service/runtime work lands.

### `POST /verify`

Scores one or more probe embeddings against a stored enrollment vector and optionally applies a boolean decision threshold.

### `GET /enrollments`

Lists the current in-memory enrollment records.

## Current Scope Limits

- The HTTP adapter still does not embed raw audio. That belongs to the later serving/runtime task.
- Enrollment state is in-memory only and resets on process restart.
- These endpoints are for contract validation and smoke/integration coverage, not final production transport.
