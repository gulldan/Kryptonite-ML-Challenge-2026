# Cohort Embedding Bank

## Goal

Freeze one reproducible cohort/impostor embedding bank that downstream `AS-norm`, `TAS-norm`, and
threshold-calibration steps can consume without reselecting background utterances ad hoc.

## What Gets Written

Each cohort-bank build writes:

- `cohort_embeddings.npz`: normalized embedding matrix plus `point_ids` and `speaker_ids`
- `cohort_metadata.jsonl`
- `cohort_metadata.parquet`
- `cohort_summary.json`

The summary is the provenance anchor. It records:

- source embedding and metadata paths
- SHA-256 of all source artifacts that shaped the bank
- selection rules and caps
- trial-exclusion behavior
- speaker-disjoint validation manifests and any overlap found
- final counts by speaker / dataset / split / role

## Default Run Integration

The manifest-backed CAM++, CAM++ stage-2, CAM++ stage-3, and ERes2NetV2 runs now assemble a cohort
bank automatically right after the dev embedding export is available.

Default policy:

1. start from the exported dev embedding artifact
2. try to exclude utterances that appear in the active verification trials
3. validate speaker disjointness against the configured train manifest
4. if trial exclusion would empty the bank entirely, keep the pre-exclusion rows and record
   `trial_overlap_fallback_used = true` in `cohort_summary.json`

That fallback is there for tiny smoke/demo runs. On real surrogate or challenge-sized splits, the
expected path is that the bank stays non-empty without reusing evaluation trial utterances.

## Standalone Builder

For existing runs or custom selection, use:

```bash
uv run python scripts/build_cohort_embedding_bank.py \
  --config configs/base.toml \
  --embeddings artifacts/baselines/campp/<run-id>/dev_embeddings.npz \
  --metadata artifacts/baselines/campp/<run-id>/dev_embedding_metadata.parquet \
  --output-dir artifacts/eval/cohort-bank/campp-<run-id> \
  --trials artifacts/manifests/ffsvc2022-surrogate/speaker_disjoint_dev_trials.jsonl \
  --validate-disjoint-speakers-against artifacts/manifests/ffsvc2022-surrogate/train_manifest.jsonl
```

Useful knobs:

- `--include-role`, `--include-split`, `--include-dataset`
- `--min-embeddings-per-speaker`
- `--max-embeddings-per-speaker`
- `--max-embeddings`
- `--no-allow-trial-overlap-fallback`

## Current Scope Limits

- the bank is utterance-level only; speaker centroids can be derived later if a scorer needs them
- selection is deterministic and metadata-driven; there is no random sampling stage
- train-manifest disjointness is validated by speaker id, not by deeper acoustic duplicate search

Leakage/duplicate audits stay the job of the EDA data-integrity reports. The cohort bank only
consumes those contracts and records whether the selected speakers remain outside the configured
train target set.
