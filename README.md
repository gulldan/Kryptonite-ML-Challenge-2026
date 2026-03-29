# Kryptonite-ML-Challenge-2026

Репозиторий для speaker verification: обучение, воспроизводимая оценка, API и браузерное demo.

## Быстрый старт

Если задача только поднять рабочий стек и посмотреть результат:

```bash
uv sync --dev --group train --group tracking
docker compose up --build
```

Откройте:

- demo: `http://127.0.0.1:8080/demo`
- health: `http://127.0.0.1:8080/health`
- metrics: `http://127.0.0.1:8080/metrics`

Остановить стек:

```bash
docker compose down -v
```

## Что читать дальше

Если нужно быстро понять репозиторий, идите в таком порядке:

1. [docs/README.md](./docs/README.md) — новая карта документации.
2. [docs/system-architecture-v1.md](./docs/system-architecture-v1.md) — архитектура и границы модулей.
3. [docs/code-architecture.md](./docs/code-architecture.md) — как устроен сам код и где лежит source of truth.
4. [docs/release-runbook.md](./docs/release-runbook.md) — как запускать, проверять и диагностировать runtime.
5. [docs/model-task-contract.md](./docs/model-task-contract.md) — что именно считается задачей и артефактом решения.

Если задача уже конкретная:

- данные и суррогатный датасет: [docs/data.md](./docs/data.md)
- обучение и текущий training path: [docs/training.md](./docs/training.md)
- конфиги и override-механика: [docs/configuration.md](./docs/configuration.md)
- краткая внешняя рамка решения: [docs/model-card.md](./docs/model-card.md)
- подробный audio contract: [docs/reference/audio-pipeline.md](./docs/reference/audio-pipeline.md)

## Что изменилось в документации

Верхний уровень `docs/` теперь содержит только канонические документы для онбординга.
Исторические deep-dive, абляции, closeout-заметки и release-артефакты перенесены в
[docs/archive/README.md](./docs/archive/README.md).

## Правила проекта

- Основные правила лежат в [AGENTS.md](./AGENTS.md).
- Полный индекс текущей документации: [docs/README.md](./docs/README.md).
