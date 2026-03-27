# Enrollment Embedding Cache

## Goal

Freeze one runtime-ready enrollment bank ahead of API startup so `/verify` can use stable speaker
centroids without recomputing enrollment state inside the serving process.

## What Gets Written

Each cache build writes:

- `enrollment_embeddings.npz`: normalized centroid matrix plus `enrollment_ids`
- `enrollment_metadata.jsonl`
- `enrollment_metadata.parquet`
- `enrollment_summary.json`

The summary is the runtime compatibility anchor. It records:

- source manifest path + SHA-256
- model bundle metadata path + SHA-256
- `compatibility_id` used at startup
- embedding stage / pooling mode / device
- enrollment count, source sample count, and embedding dimension

## Compatibility Contract

`model_bundle/metadata.json` may define `enrollment_cache_compatibility_id`.

- If it is present, the cache summary must carry the same value.
- If it is missing, the runtime falls back to a canonical hash of the whole metadata payload.

That keeps the startup check explicit for real bundles while still giving smoke/demo artifacts a
deterministic fallback.

## Builder

Rebuild the cache with:

```bash
uv run python scripts/build_enrollment_cache.py \
  --config configs/deployment/infer.toml \
  --manifest artifacts/manifests/demo_manifest.jsonl \
  --output-dir artifacts/enrollment-cache
```

The default path uses enrollment rows from the manifest, extracts baseline Fbank/stat embeddings,
averages them per `enrollment_id` (or `speaker_id` when `enrollment_id` is absent), and writes the
runtime cache artifacts.

## Runtime Behavior

`create_http_server(...)` loads `artifacts/enrollment-cache` when present.

- Compatible cache: preloads `/enrollments` and powers `/verify`
- Incompatible cache: startup fails fast
- Missing cache in advisory mode: server still starts, but only manual `POST /enroll` calls can add
  process-local state
