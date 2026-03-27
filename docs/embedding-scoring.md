# Embedding Scoring

Shared L2 normalization and cosine scoring now live in `src/kryptonite/models/scoring.py`.

For the offline cache that now seeds runtime enrollment state, see
[docs/enrollment-embedding-cache.md](./enrollment-embedding-cache.md).

The goal is to keep one contract for:

- offline verification scoring in baseline/training pipelines;
- pairwise and one-to-many score computation;
- thin HTTP scoring endpoints used by the current infer adapter;
- the unified raw-audio `Inferencer` wrapper in `src/kryptonite/serve/inferencer.py`.

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

## Raw-Audio Wrapper

`Inferencer.from_config(...)` is now the shared entrypoint for:

- local Python embedding / enrollment / verification calls over audio paths;
- the FastAPI adapter in `src/kryptonite/serve/http.py`;
- runtime benchmark smoke checks for the demo subset.

Current implementation details:

- runtime backend implementation: `feature_statistics`
- frontend: shared audio loading + Fbank extraction + chunk pooling
- post-processing: mean or mean+std frame pooling, then cosine scoring through the shared scorer
- enrollment state: preload from the offline enrollment cache, then allow process-local overrides

## HTTP Endpoints

The FastAPI adapter in `src/kryptonite/serve/http.py` now exposes embedding-based JSON endpoints plus an OpenAPI schema at `/openapi.json` and interactive docs at `/docs`.

Primary health endpoint:

- `GET /health`

Compatibility aliases:

- `GET /healthz`
- `GET /readyz`

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

The runtime now starts from the offline enrollment cache when one is present and compatible with
the active model bundle metadata. `POST /enroll` remains intentionally process-local so smoke tests
and manual probes can inject or override entries without mutating the checked-in cache artifacts.

`POST /enroll` now also accepts `audio_path` / `audio_paths` instead of precomputed embeddings.

### `POST /verify`

Scores one or more probe embeddings against a stored enrollment vector and optionally applies a boolean decision threshold.

`POST /verify` now also accepts `audio_path` / `audio_paths` to compute runtime embeddings first.

### `POST /embed`

Embeds one or more runtime audio paths through the shared `Inferencer` wrapper and returns:

- per-file post-processing metadata;
- chunk counts;
- final embedding vectors;
- active inferencer backend descriptor.

### `POST /benchmark`

Runs repeated raw-audio embedding iterations over one or more audio paths and returns:

- mean / min / max iteration seconds;
- mean milliseconds per audio file;
- mean chunk count;
- active inferencer backend descriptor.

### `GET /enrollments`

Lists the current in-memory enrollment records.

## Current Scope Limits

- Raw-audio runtime embedding currently uses the shared `feature_statistics` implementation, not a
  true exported ONNX/TensorRT speaker encoder yet.
- Enrollment centroids still load from the offline cache at startup; manual `POST /enroll` calls
  remain process-local and reset on restart.
- The FastAPI layer intentionally stays thin: transport, validation, and OpenAPI live here, while
  inferencer/scoring logic stays in `src/kryptonite/serve/` and adjacent core modules.
