# Архитектура решения

Документ описывает верхнеуровневый pipeline без фиксации конкретной финальной модели.
Сейчас это важно: repo уже готовит submit surface, но сам final model freeze ещё не
произошёл.

## Базовый путь

```text
test CSV + audio root
  -> чтение аудио
  -> нормализация / ресемплинг
  -> feature extraction
  -> speaker embeddings
  -> nearest-neighbour search
  -> optional reranking / postprocessing
  -> submission.csv
  -> validator report
```

## Принципы

- обучение, EDA и inference boundary разделены по модулям;
- reusable logic живёт в `src/kryptonite/`, а CLI orchestration — в `scripts/`;
- history и trails вынесены в `research/docs/`;
- финальный organizer-facing path будет собран отдельно в `src/kryptonite/submit/`,
  корневых `scripts/`, `configs/` и `docs/`;
- любой итоговый `submission.csv` обязан проходить отдельную валидацию.

## Границы модулей

| Модуль | Роль |
| --- | --- |
| `src/kryptonite/data/` | Загрузка аудио, manifests, validation. |
| `src/kryptonite/features/` | Feature extraction, chunking, caches. |
| `src/kryptonite/models/` | Model families и scoring interfaces. |
| `src/kryptonite/training/` | Training/fine-tuning pipeline. |
| `src/kryptonite/eda/` | EDA, diagnostics, validation, reranking helpers. |
| `src/kryptonite/eval/` | Метрики и отчёты. |
| `src/kryptonite/runtime/` | Runtime, ONNX, TensorRT boundary. |
| `src/kryptonite/submit/` | Будущий финальный submit pipeline. Пока каркас. |

## Что уже выделено под freeze phase

После выбора финальной модели сюда должен лечь минимальный organizer-facing runtime:

1. `src/kryptonite/submit/` — финальный orchestration code.
2. `scripts/run_submission.py` — канонический CLI `CSV + audio -> submission.csv`.
3. `configs/submission.toml` — финальный runtime-конфиг.
4. `docs/runbook.md`, `docs/model-card.md`, `docs/artifact-manifest.md` — финальный комплект документации.

До этого момента старые model-specific paths в research-коде считаются экспериментальными,
даже если они показывали лучший public score.

## Проверка корректности

```bash
uv run python scripts/validate_submission.py \
  --template-csv datasets/Для\ участников/test_public.csv \
  --submission-csv submission.csv \
  --output-json artifacts/release/submission_validation.json
```

Валидатор проверяет число строк, порядок `filepath`, 10 соседей на строку, повторы,
self-match и индексы вне диапазона.

## Что читать дальше

- [../../README.md](../../README.md)
- [../../TODO.md](../../TODO.md)
- [README.md](./README.md)
- [code-architecture.md](./code-architecture.md)
- [../README.md](../README.md)
- [../../docs/model-task-contract.md](../../docs/model-task-contract.md)
