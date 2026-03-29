# Audio Embedding Atlas

## Goal

Build one reusable atlas workflow for projected speaker embeddings so runs can be inspected
visually instead of only through scalar metrics.

The atlas tool accepts:

- a precomputed embeddings matrix in `.npy` or `.npz`
- aligned metadata in `.jsonl`, `.csv`, or `.parquet`

and writes an interactive HTML report plus machine-readable sidecars.

The reproducible command shape is:

```bash
uv run python scripts/build_embedding_atlas.py \
  --embeddings artifacts/eval/run-001/dev_embeddings.npy \
  --metadata artifacts/manifests/dev_manifest.parquet \
  --output-dir artifacts/eval/embedding-atlas/run-001
```

If the repository does not yet have a trained speaker encoder for a target run, there is now an
end-to-end baseline path from manifest to atlas:

```bash
uv run python scripts/build_manifest_embedding_atlas.py \
  --manifest artifacts/manifests/ffsvc2022-surrogate/dev_manifest.jsonl \
  --output-dir artifacts/eval/embedding-atlas/ffsvc2022-surrogate-dev \
  --title "FFSVC 2022 Surrogate Dev Atlas" \
  --device auto \
  --metadata-format parquet
```

That workflow reuses the existing audio loader, chunking policy, and Fbank frontend to build a
simple fixed-size acoustic embedding per utterance:

- chunk utterances with the configured `eval` or `demo` policy
- extract shared log-Mel/Fbank features
- pool frames with `mean` or `mean_std`
- pool chunks with the configured stage pooling rule

The result is a reproducible baseline for visual inspection, not a claim that this equals the final
speaker encoder.

## Output

The command writes:

- `embedding_atlas.html`: interactive 2D map
- `embedding_atlas_points.jsonl`: projected points with metadata and nearest neighbors
- `embedding_atlas_report.json`
- `embedding_atlas_report.md`

The HTML atlas supports:

- search/filter across chosen metadata fields
- point coloring by any metadata field, for example `speaker_id` or `split`
- cosine nearest-neighbor lookup per point
- optional inline audio preview when `audio_path` is present
- optional image preview when metadata carries an image field

## Projection Policy

The built-in projection methods are intentionally lightweight and reproducible:

- `pca`
- `cosine_pca`

`cosine_pca` first L2-normalizes the vectors, which is usually the better default for speaker
embeddings because downstream verification commonly operates in cosine space.

This is not meant to be a full replacement for nonlinear layouts like UMAP. It is the repository
baseline for quick inspection that can run in the locked environment without adding a large extra
stack.

## Scope and Limits

The current repository only has audio-domain data, so the atlas is immediately useful for voice
embeddings. Face-versus-voice correspondence needs a multimodal metadata table with image paths and
aligned ids; the renderer already supports optional image preview, but it cannot invent face data
that is not present in the run artifacts.

The generic atlas still does not compute embeddings by itself. It expects a model run to dump them
first. That separation is deliberate:

- model inference stays in training/eval code
- visualization stays reusable across baselines and experiments

The new manifest-backed CLI is intentionally a thin baseline wrapper around that generic atlas. It
exists so the current dataset bundle can be inspected immediately, before a stronger speaker model
export path lands.
