# Configs

Keep runtime, training, evaluation, and deployment configuration separate from code.

Prefer concern-based groupings such as:

- `configs/data/`
- `configs/training/`
- `configs/eval/`
- `configs/serve/`

Current bootstrap config lives in `configs/base.toml`.
Validate and inspect it with:

```bash
uv run python scripts/show_config.py --config configs/base.toml
uv run python scripts/show_config.py --config configs/base.toml --override runtime.seed=123
```
