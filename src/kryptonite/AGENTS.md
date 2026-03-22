# AGENTS.md

Local rules for `src/kryptonite/`.

- Keep core domain, data, feature, model, training, evaluation, and serving logic here.
- Respect module boundaries instead of mixing EDA, training, and serving in the same file.
- Prefer explicit typed interfaces and small composable modules.
- Keep notebooks and app entrypoints as consumers of this package, not alternate sources of truth.
