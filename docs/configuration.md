# Configuration

Цель этого документа — дать короткую карту конфигов, а не пересказывать каждую абляцию.
Детальные stage-specific и archive-level документы теперь вынесены из верхнего уровня.

## Canonical files

- `configs/base.toml` — repository-wide defaults
- `configs/schema.json` — shape of the config
- `configs/deployment/*.toml` — runtime and serve profiles
- `configs/training/*.toml` — training recipes and stage configs
- `.env.example` — supported environment variables

## Overrides

Use dotted overrides in `key=value` form:

```bash
uv run python scripts/show_config.py --config configs/base.toml \
  --override runtime.seed=123 \
  --override training.batch_size=32
```

## What to configure first

- runtime/backend intent
- paths to manifests/artifacts
- reproducibility settings
- audio preprocessing policy
- training recipe selection
- tracking backend and secrets

## Backend Selection

Serving configs use requested backend intent and let the service resolve the real runtime path.
Всегда смотрите в `/health` сразу три поля:

- `requested_backend`
- `selected_backend`
- `inferencer.implementation`

## Secrets

Keep secrets in `.env`, not in versioned TOML files.
Common variables:

- `WANDB_API_KEY`
- `MLFLOW_TRACKING_TOKEN`
- `HUGGINGFACE_HUB_TOKEN`

## Audio pipeline

Все normalization/VAD/Fbank/chunking assumptions теперь собраны в одной reference note:

- [reference/audio-pipeline.md](./reference/audio-pipeline.md)

Старые детальные notes по отдельным шагам сохранены в архиве.

## Archive

Если нужен stage-specific или experiment-specific config context, смотрите:

- [archive/configuration.md](./archive/configuration.md)
- [archive/training-environment.md](./archive/training-environment.md)
- [archive/gpu-server-data-sync.md](./archive/gpu-server-data-sync.md)
