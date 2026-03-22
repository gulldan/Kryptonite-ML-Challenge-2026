# Scripts

Keep reproducible command-line entrypoints here.

Scripts should be thin wrappers around reusable code in `src/kryptonite/`, not alternate implementations.

Current entrypoints include:

- `scripts/repro_check.py` for reproducibility smoke validation
- `scripts/show_config.py` for config inspection and overrides
- `scripts/training_env_smoke.py` for training-environment import checks
- `scripts/tracking_smoke.py` for local tracking smoke validation
