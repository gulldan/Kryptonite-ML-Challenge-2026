# Архитектура кода

Код организован так, чтобы отделить анализ данных, обучение, инференс и проверку
submission-файлов. Основная логика лежит в `src/kryptonite/`, а `scripts/` содержит
воспроизводимые команды запуска.

## Верхний уровень

```text
.
├── README.md          # как запустить решение и получить submission.csv
├── configs/           # TOML-конфиги обучения и инференса
├── docs/              # отчёт, история экспериментов, runbooks
├── scripts/           # команды запуска
├── src/kryptonite/    # основная Python-логика
├── tests/             # unit/integration/e2e tests
└── artifacts/         # большие локальные результаты, не коммитятся
```

## Основные слои

| Слой | Папка | Ответственность |
| --- | --- | --- |
| Данные | `src/kryptonite/data/` | Чтение аудио, манифесты, проверка схем, нормализация, VAD. |
| EDA и проверка submission | `src/kryptonite/eda/` | Анализ датасета, граф соседей, чрезмерно популярные соседи, псевдометки, проверка `submission.csv`. |
| Признаки | `src/kryptonite/features/` | Fbank, нарезка аудио, официальный CAM++ frontend, кеш признаков. |
| Модели | `src/kryptonite/models/` | CAM++, ERes2NetV2, checkpoint loading, cosine scoring. |
| Обучение | `src/kryptonite/training/` | Общий training pipeline, dataloader, optimizer, fine-tuning, tracking. |
| Оценка | `src/kryptonite/eval/` | Метрики, отчёты, verification/retrieval diagnostics. |
| Инференс и serving | `src/kryptonite/serve/` | ONNX/TensorRT, runtime-адаптеры, API. |

## Поток данных в финальном решении

```text
test CSV + audio
    -> audio loading, 16 kHz mono
    -> официальный CAM++ frontend
    -> CAM++ encoder
    -> векторы голосов
    -> поиск ближайших соседей
    -> graph/class-aware postprocessing
    -> submission.csv
    -> format validator
```

## Почему EDA вынесена отдельно

EDA не смешана с обучением. В `src/kryptonite/eda/` лежат переиспользуемые проверки:

- сравнение train/public аудио;
- проверка локальной валидации против public leaderboard;
- анализ чрезмерно популярных соседей;
- построение кластеров для псевдометок;
- проверка и сравнение submission-файлов.

Это важно для отчёта: выводы EDA можно повторить отдельными командами, а не искать в
ноутбуках.

## Важные CLI-команды

| Команда | Назначение |
| --- | --- |
| `scripts/run_baseline.py` | Обучить базовую CAM++ или ERes2NetV2 модель из TOML-конфига. |
| `scripts/run_campp_finetune.py` | Дообучить CAM++ из существующего checkpoint. |
| `scripts/run_ms41_submission.py` | Единый reproducible runner для текущего финального MS41 submission path. |
| `scripts/run_official_campp_tail.py` | Тонкая CLI-обёртка над `src/kryptonite/eda/official_campp_tail/` для полного инференса CAM++. |
| `scripts/run_class_aware_graph_tail.py` | Финальная слабая class-aware постобработка поверх кеша ближайших соседей. |
| `scripts/run_classifier_first_tail.py` | Получить кеш classifier posterior из векторов голосов. |
| `scripts/validate_submission.py` | Проверить формат итогового `submission.csv`. |
| `scripts/run_eda_profile.py` | Собрать базовый EDA-профиль аудио и CSV. |
| `scripts/build_eda_review_package.py` | Подготовить EDA-пакет для отчёта. |

## Конфиги

Конфиги лежат в `configs/`:

- `configs/training/` - обучение и fine-tuning;
- `configs/release/` - release/TensorRT настройки;
- `configs/base.toml` - базовые пути и runtime-настройки.

Параметры запуска не зашиты в код намертво: важные значения вынесены в TOML или CLI flags.

## Тесты и проверки

В `tests/` есть unit, integration и e2e проверки для:

- схем манифестов;
- аудиопредобработки;
- моделей и checkpoint loading;
- submission validation;
- inference/runtime contracts;
- проверки границ TensorRT/ONNX;
- EDA helpers.

Для финальной сдачи самый важный практический gate - валидатор:

```bash
uv run python scripts/validate_submission.py \
  --template-csv datasets/Для\ участников/test_public.csv \
  --submission-csv submission.csv \
  --output-json artifacts/release/submission_validation.json
```
