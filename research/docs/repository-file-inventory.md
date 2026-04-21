# Реестр файлов репозитория

Документ описывает актуальное дерево после перехода к organizer-first структуре.
Корень предназначен для финального пути, а весь research вынесен в `research/`.

## Что читать первым

| Путь | Зачем нужен |
| --- | --- |
| `README.md` | Краткая точка входа и текущее состояние репозитория. |
| `TODO.md` | Что ещё нужно сделать до выбора финальной модели и финальной упаковки. |
| `docs/README.md` | Что должно войти в финальный комплект документов. |
| `research/README.md` | Где лежат EDA, history, trails, benchmarks и архивы. |
| `research/docs/code-architecture.md` | Карта кода и текущие границы модулей. |

## Корень репозитория

| Путь | Зачем нужен |
| --- | --- |
| `README.md` | Главный organizer-facing entrypoint. |
| `TODO.md` | Рабочий контракт до финальной сдачи. |
| `pyproject.toml` | Python metadata, dependency groups, `ruff`, `ty`, `pytest`. |
| `uv.lock` | Lockfile для `uv`. |
| `artifacts/` | Локальные outputs, кэши и runtime-артефакты. Это рабочая папка, а не архив. |
| `research/` | Вся исследовательская и архивная зона. |

## `configs/`

| Путь | Зачем нужен |
| --- | --- |
| `configs/README.md` | Навигация по финальным и общим конфигам. |
| `configs/base.toml` | Общие локальные настройки. |
| `configs/schema.json` | Схема конфигураций. |
| будущий `configs/submission.toml` | Единый конфиг финального пути после выбора модели. |

## `scripts/`

| Путь | Зачем нужен |
| --- | --- |
| `scripts/README.md` | Навигация по финальным CLI. |
| `scripts/validate_submission.py` | Канонический валидатор `submission.csv`. |
| будущий `scripts/run_submission.py` | Единый runner финального решения. |

## `src/kryptonite/`

| Путь | Зачем нужен |
| --- | --- |
| `src/kryptonite/data/` | Загрузка аудио, manifests, schema validation, preprocessing. |
| `src/kryptonite/eda/` | EDA, retrieval diagnostics, submission validation, reranking helpers. |
| `src/kryptonite/features/` | Feature extraction и caches. |
| `src/kryptonite/models/` | Model families и scoring. |
| `src/kryptonite/training/` | Training orchestration и model-family wrappers. |
| `src/kryptonite/eval/` | Метрики, отчёты и robustness benchmark. |
| `src/kryptonite/runtime/` | Runtime, ONNX, TensorRT и export boundary. |
| `src/kryptonite/submit/` | Будущий финальный orchestration package. Пока каркас. |

## `docs/` и `research/`

| Путь | Зачем нужен |
| --- | --- |
| `docs/README.md` | Навигация по финальным документам. |
| `docs/model-task-contract.md` | Формальный контракт `submission.csv`. |
| `research/README.md` | Главная точка входа в исследовательскую часть. |
| `research/docs/` | EDA, history, trails, internal architecture, archive docs. |
| `research/scripts/` | Все research-only CLI entrypoints. |
| `research/configs/` | Все research-only конфиги. |
| `research/notebooks/` | Все исследовательские notebook entrypoints. |
| `research/archive/` | Organizer baseline, evidence bundles и материалы оргов. |
| `research/tests/` | Research-only и legacy-проверки. |

## `tests/`

| Путь | Зачем нужен |
| --- | --- |
| `tests/README.md` | Навигация по корневым тестам. |
| `tests/unit/` | Проверки живого библиотечного кода и финального пути. |

## Минимальный маршрут чтения сейчас

1. `README.md`
2. `TODO.md`
3. `docs/model-task-contract.md`
4. `research/README.md`
5. `research/docs/challenge-experiment-history.md`
