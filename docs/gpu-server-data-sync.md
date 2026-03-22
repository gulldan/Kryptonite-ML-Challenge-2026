# GPU Server Data Sync

## Goal

Make the dataset and manifest state on `gpu-server` explicit, reproducible, and auditable instead of assuming that downstream EDA, baseline, and demo tasks can see the right files.

## Command

Run the sync and regenerate the readiness report from the local checkout:

```bash
uv run python scripts/dataset_sync.py --plan configs/data-sync/gpu-server.toml --execute
```

For a non-mutating audit, omit `--execute`:

```bash
uv run python scripts/dataset_sync.py --plan configs/data-sync/gpu-server.toml
```

The plan writes:

- a local report to `artifacts/reports/gpu-server-data-sync.json`
- a remote copy to `/mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/reports/gpu-server-data-sync.json`

## Scope

The current sync plan covers:

- `datasets/`
- `artifacts/manifests/`
- `artifacts/demo-subset/`

Checksums use two strategies:

- `catalog` for raw dataset trees: digest of relative paths plus file sizes
- `sha256` for manifests and demo artifacts: digest of file contents

This keeps large raw-audio syncs inspectable without pretending that a full content hash across every dataset file is cheap or required on each run.

## Current Audit

Audit date: 2026-03-22.

Observed state on `gpu-server` before this task:

- `datasets/` contained only `demo-speaker-recognition` with 6 WAV files
- `artifacts/manifests/` contained only `demo_manifest.jsonl`
- no additional raw dataset layout for baseline or EDA was present under the repo checkout

This means the deploy/demo smoke path had enough data to run, but the repository still did not have full data readiness for dataset profiling, split generation, or baseline training.

After running `scripts/dataset_sync.py --execute`, the generated report at `artifacts/reports/gpu-server-data-sync.json` recorded:

- `datasets/`: 6 files, 192264 bytes, catalog checksum `193e5111f0b8eb0b0049319599e0a126d35ac19f09cbc039ffbbaeb43fb9d3b5`
- `artifacts/manifests/`: 1 file, 1570 bytes, checksum `7262634ec79cf72a8e348f476c86da13fe38b0bbe47b7e04ea3bdb1415d2c96e`
- `artifacts/demo-subset/`: 7 files, 193086 bytes, checksum `1e68e414962c9ceb8f7d431079ea95813376a8bce6d550a67ddc8ce00d618b8a`

The same JSON report was uploaded to `/mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/reports/gpu-server-data-sync.json`.

Path validation on the target machine also passed:

```bash
ssh gpu-server 'cd /mnt/storage/Kryptonite-ML-Challenge-2026 && export PATH="$PWD/.local/bin:$PATH" && export UV_CACHE_DIR="$PWD/.cache/uv" && export XDG_CACHE_HOME="$PWD/.cache" && uv run python scripts/training_env_smoke.py --config configs/deployment/train.toml --require-artifacts'
```

This confirmed that the configured `dataset_root`, `manifests_root`, and `demo_manifest_file` are readable from the active server checkout.

## Resync Procedure

1. Materialize the required raw datasets and manifests in the local checkout under `datasets/` and `artifacts/manifests/`.
2. Run `uv run python scripts/dataset_sync.py --plan configs/data-sync/gpu-server.toml --execute`.
3. Inspect the generated JSON report and confirm the local and remote payload checksums match.
4. On `gpu-server`, run `uv run python scripts/training_env_smoke.py --config configs/deployment/train.toml --require-artifacts` from `/mnt/storage/Kryptonite-ML-Challenge-2026` to verify the configured paths are readable.

## Limitations

- The sync plan only mirrors whatever already exists in the local checkout. It does not fetch competition datasets from object storage, cloud drives, or external archives.
- As of 2026-03-22, the repository-local source tree still contains only the synthetic demo dataset and demo artifacts. Full EDA/baseline readiness remains blocked until real raw datasets and manifests are materialized locally and re-synced.
