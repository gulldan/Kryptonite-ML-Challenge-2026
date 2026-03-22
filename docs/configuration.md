# Configuration

## Goal

Keep runtime parameters in versioned config files and keep secret values out of git.

## Files

- `configs/base.toml` contains the versioned default profile
- `configs/schema.json` documents the expected shape
- `.env.example` lists supported secret variables
- `scripts/repro_check.py` performs a reproducibility self-check for a config

## Overrides

Use dotted overrides in `key=value` form:

```bash
uv run python scripts/show_config.py --config configs/base.toml \
  --override runtime.seed=123 \
  --override training.batch_size=32 \
  --override backends.allow_tensorrt=true
```

Supported value forms:

- integers, booleans, floats
- quoted TOML strings
- plain strings, which are treated as raw text

## Secrets

Copy `.env.example` to `.env` on each machine and fill in only the values you need.

The config stores environment variable names, not secret values. At runtime, the loader resolves:

- `WANDB_API_KEY`
- `MLFLOW_TRACKING_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`

By default, the CLI masks secret values when printing them.

## Reproducibility Hooks

The config includes a `reproducibility` section with:

- `deterministic`
- `pythonhashseed`
- `fingerprint_paths`

Run the self-check with:

```bash
uv run python scripts/repro_check.py --config configs/base.toml --self-check
```

This launches two short subprocess runs with the same seed and `PYTHONHASHSEED`, then verifies:

- identical control metadata
- identical fingerprints for tracked inputs
- equal probe outputs from seeded randomness
