# Configs

Keep runtime, training, evaluation, and deployment configuration separate from code.

Current layout:

- `configs/base.toml` for the broad local bootstrap profile
- `configs/deployment/` for thin training and serving smoke profiles
- `configs/training/` for model-family and experiment configs
- `configs/release/` for release and benchmark-oriented presets
- `configs/corruption/` for corruption-specific checks

Deployment profiles carry a `[deployment]` section that pins the expected model bundle and demo subset roots for strict container preflight.

Validate and inspect it with:

```bash
uv run python scripts/show_config.py --config configs/base.toml
uv run python scripts/show_config.py --config configs/base.toml --override runtime.seed=123
uv run python scripts/show_config.py --config configs/deployment/train.toml
uv run python scripts/show_config.py --config configs/deployment/infer.toml
```
