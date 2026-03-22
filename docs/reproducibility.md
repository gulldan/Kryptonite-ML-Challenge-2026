# Reproducibility

## Scope

This project uses a lightweight reproducibility layer until a real training loop lands in the repository.

The current hooks cover:

- global seed setup
- deterministic flags for optional backends when available
- `PYTHONHASHSEED` propagation for subprocesses
- checksums for selected config or manifest files
- compact metadata snapshots for smoke checks

## Smoke Check

Use the same command locally and on `gpu-server`:

```bash
uv run python scripts/repro_check.py --config configs/base.toml --self-check
```

The command runs two short subprocess probes against the same config and compares:

- `metadata`
- `probe.random_trace`
- `probe.random_mean`
- `probe.string_hash`

The comparison is intentionally strict right now because the current probe is standard-library only. Once the training stack exists, the same command can be extended to allow tolerance-based comparisons for model metrics.

## Fingerprints

`fingerprint_paths` should point to small, meaningful files such as:

- config files
- manifests
- model metadata

Avoid hashing the entire `datasets/` tree directly. Prefer manifests or curated metadata files instead.
