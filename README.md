# Kryptonite-ML-Challenge-2026

Speaker verification: обучение моделей, оценка качества (EER, minDCF).

## Установка

```bash
uv sync --dev --group train
```

## Обучить модель

```bash
# CAM++ (512-dim embeddings)
uv run python scripts/run_baseline.py \
  --model campp \
  --config configs/training/campp-baseline.toml \
  --device cuda

# ERes2NetV2 (192-dim embeddings)
uv run python scripts/run_baseline.py \
  --model eres2netv2 \
  --config configs/training/eres2netv2-baseline.toml \
  --device cuda
```

Результат: checkpoint, embeddings, verification report (EER, minDCF, score gap).

### Переопределение параметров через CLI

```bash
uv run python scripts/run_baseline.py \
  --model campp \
  --config configs/training/campp-baseline.toml \
  --device cuda \
  --project-override 'training.batch_size=32' \
  --project-override 'training.max_epochs=50' \
  --project-override 'training.precision="bf16"'
```

## Добавить новую модель

Три файла + конфиг. Подробнее: [docs/training.md](./docs/training.md)

## Pipeline

```
Manifest (JSONL) -> Audio (16kHz) -> Fbank (80-bin) -> Encoder -> Embeddings
                                                                      |
                                          Cosine scoring <- Trial pairs
                                                |
                                          EER / minDCF
```

## Структура

```
src/kryptonite/
  data/          # манифесты, аудио I/O, валидация
  features/      # fbank, chunking
  models/        # campp/, eres2netv2/, scoring
  training/      # baseline pipeline, dataloader, optimizer
  eval/          # verification metrics, reports
scripts/         # CLI entrypoints
configs/         # TOML конфиги
```

## Правила

[AGENTS.md](./AGENTS.md)
