# Kryptonite-ML-Challenge-2026

Speaker verification: обучение моделей, оценка качества, API и demo.

## Установка

```bash
uv sync --dev --group train --group tracking
```

## Обучить модель

```bash
uv run python scripts/run_baseline.py --model campp --config configs/training/campp-baseline.toml --device cuda
uv run python scripts/run_baseline.py --model eres2netv2 --config configs/training/eres2netv2-ffsvc2022-surrogate.toml --device cuda
```

Каждый запуск создаёт checkpoint, эмбеддинги и verification report (EER, minDCF, score gap).

## Добавить новую модель

Четыре файла:
1. Encoder (`src/kryptonite/models/<name>/model.py`) — `nn.Module`, вход `[B, T, 80]` -> выход `[B, emb_dim]`
2. Config loader (`src/kryptonite/training/<name>/config.py`) — ~50 строк
3. Pipeline wrapper (`src/kryptonite/training/<name>/pipeline.py`) — ~35 строк, делегирует в `run_speaker_baseline()`
4. TOML config (`configs/training/<name>-baseline.toml`)

Подробнее: [docs/training.md](./docs/training.md)

## Поднять demo

```bash
docker compose up --build
```

- demo: `http://127.0.0.1:8080/demo`
- health: `http://127.0.0.1:8080/health`

## Структура

```text
src/kryptonite/       # вся логика: модели, training, eval, serving
scripts/              # CLI entrypoints (см. scripts/README.md)
configs/              # TOML конфиги
apps/api/             # FastAPI serving
docs/                 # архитектура, runbooks, contracts
```

## Документация

- [docs/training.md](./docs/training.md) — обучение и добавление моделей
- [docs/code-architecture.md](./docs/code-architecture.md) — карта кода
- [docs/system-architecture-v1.md](./docs/system-architecture-v1.md) — архитектура pipeline
- [docs/configuration.md](./docs/configuration.md) — конфиги и overrides
- [docs/release-runbook.md](./docs/release-runbook.md) — запуск и диагностика runtime

## Правила

[AGENTS.md](./AGENTS.md)
