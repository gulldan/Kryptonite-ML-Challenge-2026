# Архитектура кода

После cleanup репозиторий разделён по аудитории, а не по типу файлов:

- корень предназначен для финального organizer-facing решения;
- `research/` содержит всё, что относится к исследованию, EDA, benchmark-ам и архиву.

## Верхний уровень

```text
.
├── README.md
├── TODO.md
├── docs/                   # финальные документы
├── configs/                # base/schema и будущий submission config
├── scripts/                # финальные CLI
├── src/kryptonite/
│   ├── submit/             # будущий финальный orchestration package
│   ├── runtime/            # runtime/export/inference boundary
│   └── ...                 # data, features, models, training, eval
├── tests/                  # тесты живого кода
└── research/               # вся исследовательская и архивная зона
```

## Основные слои

| Слой | Папка | Ответственность |
| --- | --- | --- |
| Данные | `src/kryptonite/data/` | Чтение аудио, manifests, schema validation, preprocessing. |
| EDA | `src/kryptonite/eda/` | Profiling, retrieval diagnostics, submission validation, reranking helpers. |
| Признаки | `src/kryptonite/features/` | Fbank, chunking, official CAM++ frontend, feature caching. |
| Модели | `src/kryptonite/models/` | Model families и scoring. |
| Обучение | `src/kryptonite/training/` | Общий training pipeline, optimization, augmentation, model-family wrappers. |
| Оценка | `src/kryptonite/eval/` | Verification/retrieval metrics, отчёты и robustness benchmark. |
| Runtime | `src/kryptonite/runtime/` | ONNX/TensorRT/export/runtime adapters. |
| Будущий submit | `src/kryptonite/submit/` | Канонический путь `CSV + audio -> submission.csv`. Пока каркас. |

## Разделение обязанностей

- `docs/`, `configs/` и `scripts/` в корне — только для финального пути;
- `research/docs/`, `research/scripts/`, `research/configs/`, `research/notebooks/`, `research/archive/` — только для исследований и архива;
- `tests/` в корне — только для живого кода и финального пути;
- `research/tests/` — для research-only и legacy-проверок.

## Базовый dataflow

```text
input CSV + audio root
    -> audio loading and normalization
    -> feature extraction
    -> speaker embeddings
    -> nearest-neighbour search
    -> optional postprocessing / reranking
    -> submission.csv
    -> validator report
```

Этот pipeline намеренно описан без привязки к модели. Конкретная модельная семья и
final postprocessing будут зафиксированы только после отдельного решения.
