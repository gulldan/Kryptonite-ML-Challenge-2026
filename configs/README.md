# Configs

Keep runtime, training, evaluation, and deployment configuration separate from code.

Prefer concern-based groupings such as:

- `configs/data/`
- `configs/training/`
- `configs/eval/`
- `configs/serve/`

Current bootstrap config lives in `configs/base.toml`.
Deployment-specific runtime profiles live in `configs/deployment/`.
Deployment profiles also carry a `[deployment]` section that pins the expected model bundle and demo subset roots for strict container preflight.
Release benchmark-pack templates live in `configs/eval/`.
Verification-protocol and other reusable evaluation/report templates also live in `configs/eval/`.
Submission/release bundle templates live in `configs/release/`.

Validate and inspect it with:

```bash
uv run python scripts/show_config.py --config configs/base.toml
uv run python scripts/show_config.py --config configs/base.toml --override runtime.seed=123
uv run python scripts/show_config.py --config configs/deployment/train.toml
uv run python scripts/show_config.py --config configs/deployment/infer.toml
```
