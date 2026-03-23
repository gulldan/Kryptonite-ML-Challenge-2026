# FFSVC 2022 Surrogate

## Why This Dataset

For a pre-Dataton pipeline shakeout we want a server-only surrogate that is close to the target task:

- speaker verification
- far-field or device-mismatched audio
- single-channel trials
- official trial files and metadata

`FFSVC 2022` is the closest official benchmark currently available from primary sources. Its dataset page explicitly says:

- the challenge builds on far-field speaker verification;
- the dev set has the same data distribution as the evaluation data;
- the dev/eval artifacts are directly downloadable from the official page;
- the older full `FFSVC2020` main training data now requires manual request, while the public page still exposes only the directly downloadable subset today.

Sources:

- [FFSVC 2022 dataset page](https://ffsvc.github.io/dataset/)
- [FFSVC 2022 site](https://ffsvc.github.io/)

## Important Caveat

This bundle is a surrogate for engineering bring-up, not a challenge-legal training recipe.

The official FFSVC 2022 dataset page says the dev set is for tuning and testing and must not be used for training in the original competition. We are intentionally relaxing that for internal pipeline smoke only, because the Dataton raw data is not yet public and we need to validate:

- ingestion
- metadata parsing
- manifest generation
- speaker-disjoint split logic
- trial generation and scoring
- end-to-end training/evaluation plumbing

Once the real Dataton data is available, the code should swap datasets without changing the pipeline shape.

## What The Plan Downloads

The server-only acquisition plan at `configs/data-acquisition/ffsvc2022-surrogate.toml` downloads:

- FFSVC 2022 dev trials
- FFSVC 2022 dev metadata
- FFSVC 2022 eval trials
- FFSVC 2022 dev WAV archive from Zenodo

It does not automate:

- full `FFSVC2020` train/dev/eval, because the official page now requires email request
- `VoxCeleb`, because the official path is still manual/gated

## Run It

Run this on `gpu-server` from `/mnt/storage/Kryptonite-ML-Challenge-2026`:

```bash
uv run python scripts/acquire_ffsvc2022_surrogate.py --execute
```

Inspect-only mode:

```bash
uv run python scripts/acquire_ffsvc2022_surrogate.py
```

The dataset root is:

```text
/mnt/storage/Kryptonite-ML-Challenge-2026/datasets/ffsvc2022-surrogate
```

## Duplicate Cleanup Policy

`scripts/prepare_ffsvc2022_surrogate.py` now applies a deterministic quarantine policy for the
two confirmed byte-identical upstream duplicate groups found during `KVA-561`:

- `ffsvc22_dev_002177` is quarantined and `ffsvc22_dev_043388` is kept as the canonical row
- `ffsvc22_dev_063743` is quarantined and `ffsvc22_dev_063782` is kept as the canonical row

The policy is intentionally `quarantine`, not silent deletion:

- active `all_manifest.jsonl`, `train_manifest.jsonl`, and `dev_manifest.jsonl` exclude the
  quarantined rows, so downstream baseline and training jobs do not sample them as independent
  examples
- `quarantine_manifest.jsonl` preserves the dropped rows together with the duplicate group id,
  canonical utterance id, and reason, so the upstream bundle issue remains auditable
- `speaker_disjoint_dev_trials.jsonl` is generated from the active held-out `dev` rows only, so a
  quarantined entry cannot leak back into threshold tuning if split parameters change

Residual risk:

- the raw surrogate bundle on disk is unchanged; only the generated manifests apply the cleanup
  policy
- the current policy covers the confirmed byte-identical duplicate groups only; if later audits
  find new duplicate-content clusters, the quarantine list must be extended before using those rows
  in training or evaluation

## Manifest Contract

`scripts/prepare_ffsvc2022_surrogate.py` now emits versioned `kryptonite.manifest.v1` rows for the
active and quarantine manifests. The canonical fields are documented in
`docs/unified-metadata-schema.md`, and the manifests can be checked explicitly with:

```bash
uv run python scripts/validate_manifests.py
```

The prepared bundle now writes, under `artifacts/manifests/ffsvc2022-surrogate/`:

- `all/train/dev/quarantine` manifests in both `.jsonl` and `.csv`
- `official_dev_trials` and `speaker_disjoint_dev_trials` in both `.jsonl` and `.csv`
- `speaker_disjoint_dev_trials_summary.json`
- `speaker_splits.json`
- `speaker_split_summary.json`
- `manifest_inventory.json` with relative artifact paths, row counts, speaker counts, and
  SHA-256 checksums for the generated manifest/list files

Each active manifest row also carries the discovered WAV duration, sample rate, and channel count,
so downstream EDA and preprocessing do not have to re-derive that metadata just to understand the
bundle shape.

`speaker_split_summary.json` is the explicit readiness gate for `KRYP-017`:

- it restates the deterministic train/dev speaker assignment
- it proves the active manifests remain speaker-disjoint
- it checks that `speaker_disjoint_dev_trials.jsonl` is non-empty, touches every held-out dev
  speaker, and, for the real multi-speaker holdout, contains both positive and negative labels
- it embeds the balanced verification-trial summary so downstream tuning can see per-bucket counts
  for duration bucket, domain relation, and channel mismatch

`speaker_disjoint_dev_trials.jsonl` is now a generated balanced verification bundle rather than a
plain dev-only copy of the official FFSVC trials:

- positives and negatives are sampled deterministically from the held-out `dev` manifest
- balancing happens per label across duration bucket (`short`, `medium`, `long`), domain relation
  (`same_domain`, `cross_domain`), and channel relation (`same_channel`, `cross_channel`)
- the row payload includes the derived stratum metadata (`duration_bucket`, `domain_relation`,
  `channel_relation`, `channel_mismatch`) plus left/right speaker/domain/channel fields for audit
- `speaker_disjoint_dev_trials_summary.json` records bucket counts, missing strata, speaker/audio
  coverage, duplicate-pair checks, and label-balance sanity checks

The sampler is configurable from the CLI:

```bash
uv run python scripts/prepare_ffsvc2022_surrogate.py --trials-per-bucket 128
```

If any of those invariants fail, `scripts/prepare_ffsvc2022_surrogate.py` now exits with an error
instead of silently leaving a weak strict-dev split behind.

## Recommended Next Steps

After acquisition:

1. reuse the unified manifest contract for the next real Dataton dataset adapter
2. canonicalize or regenerate the official dev trial file so every row resolves against the active
   manifests
3. reuse the strict-dev split summary as a readiness gate before threshold tuning runs
4. treat this bundle as the engineering stand-in until the Dataton data lands
