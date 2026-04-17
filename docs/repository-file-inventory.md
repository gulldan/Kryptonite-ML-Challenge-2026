# Реестр файлов репозитория

Этот документ объясняет, зачем в репозитории нужны основные файлы и папки. Он нужен
для проверяющего, который открывает проект впервые.

`artifacts/` и `datasets/` не расписываются по каждому файлу: это большие локальные
артефакты и внешние данные. Они не являются исходным кодом и не должны коммититься в git.

## Что читать в первую очередь

| Путь | Зачем нужен |
| --- | --- |
| `README.md` | Главная инструкция: что решает проект, как установить зависимости, как получить и проверить `submission.csv`. |
| `docs/challenge-solution-report.md` | Короткий отчёт для жюри по EDA, baseline, экспериментам, инженерным решениям и выводам. |
| `docs/inference-acceleration.md` | Ускорение инференса: команды ONNX/TensorRT, H100-профиль, узкое место frontend. |
| `docs/challenge-experiment-history.md` | Компактная leaderboard-таблица submitted runs, score, файлов отправки и решений. |
| `docs/trails/` | Подробные записи экспериментов, запусков и диагностики. |
| `docs/code-architecture.md` | Короткая карта кода. |
| `docs/release-runbook.md` | Контрольный список финальной сдачи. |

## Верхний уровень

| Путь | Зачем нужен |
| --- | --- |
| `AGENTS.md` | Правила работы в репозитории: инструменты, структура, порядок работы, требования к экспериментам. |
| `CLAUDE.md` | Исторические инструкции для другого агента. Не является основным документом для жюри. |
| `README.md` | Основная точка входа для человека и для воспроизводимости. |
| `pyproject.toml` | Описание Python-проекта, зависимости, настройки `ruff`, `ty`, `pytest`. |
| `uv.lock` | Зафиксированные версии Python-зависимостей для `uv`. |
| `compose.yml`, `compose.gpu.yml` | Docker Compose конфигурации для API и GPU-режима. Для финального `submission.csv` не являются обязательной точкой входа. |

## `apps/`

| Путь | Зачем нужен |
| --- | --- |
| `apps/api/README.md` | Короткое описание API-приложения. |
| `apps/api/AGENTS.md` | Локальные правила для API-папки. |
| `apps/api/main.py` | Тонкая точка входа FastAPI. Основная логика находится в `src/kryptonite/serve/`. |

## `baseline/`

Эта папка хранит исходную и исправленную baseline-ветку. Она нужна как точка сравнения
и как доказательство, что команда понимала стартовое решение организаторов.

| Путь | Зачем нужен |
| --- | --- |
| `baseline/README.md` | Описание baseline-папки. |
| `baseline/requirements.txt` | Исторический список зависимостей baseline. Не является источником зависимостей проекта. Основной инструмент проекта - `uv`. |
| `baseline/train.py` | Обучение baseline-модели. |
| `baseline/test_model.py` | Проверка baseline-модели. |
| `baseline/convert_to_onnx.py` | Экспорт baseline-модели в ONNX. |
| `baseline/inference_onnx.py` | Инференс baseline ONNX. |
| `baseline/calc_metrics.py` | Подсчёт baseline-метрик. |
| `baseline/configs/*.json` | Конфиги baseline-запусков. |
| `baseline/src/dataset.py` | Dataset-код baseline. |
| `baseline/src/ecapa.py` | ECAPA-модель baseline. |
| `baseline/src/mel_frontend.py` | Mel-признаки baseline. |
| `baseline/src/metrics.py` | Метрики baseline. |
| `baseline/src/model.py` | Обёртка baseline-модели. |

## `configs/`

| Путь | Зачем нужен |
| --- | --- |
| `configs/README.md` | Навигация по конфигам. |
| `configs/base.toml` | Базовые пути и общие настройки. |
| `configs/schema.json` | Схема конфигураций. |
| `configs/deployment/*.toml` | Тонкие training/inference smoke-профили для Docker и локального API. |
| `configs/release/*.toml` | Конфиги для TensorRT и release-проверок скорости. |
| `configs/training/campp-*.toml` | CAM++ обучение, fine-tuning, псевдометки, внешние данные и BN-adapter гипотезы. |
| `configs/training/eres2netv2-*.toml` | ERes2NetV2 baseline и fine-tuning ветки. |
| `configs/training/official-3dspeaker-*.toml` | Официальные 3D-Speaker ERes2Net-large ветки. |
| `configs/training/wavlm-*.toml` | WavLM speaker model эксперименты. |
| `configs/training/w2vbert2-*.toml` | W2VBERT2 speaker model эксперименты. |
| `configs/training/teacher-peft.toml` | Teacher/PEFT эксперимент. |

## `scripts/`

Скрипты являются воспроизводимыми командами запуска. Подробная карта лежит в
`scripts/README.md`.

| Группа | Скрипты | Зачем нужны |
| --- | --- | --- |
| Финальный инференс | `run_ms41_submission.py`, `run_official_campp_tail.py`, `run_classifier_first_tail.py`, `run_class_aware_graph_tail.py`, `validate_submission.py` | Получить и проверить `submission.csv`, включая единый MS41 preset runner. |
| Обучение | `run_baseline.py`, `run_campp_finetune.py`, `run_campp_soft_pseudo_finetune.py`, `run_eres2netv2_finetune.py`, `run_hf_xvector_finetune.py`, `run_teacher_peft*.py`, `run_w2vbert2_sv_moonshot.py` | Запустить обучение и fine-tuning разных модельных веток. |
| Псевдометки и внешние данные | `build_*pseudo*`, `build_cnceleb_manifests.py`, `download_external_speaker_datasets.py`, `build_voxblink_v1_audio.py` | Подготовить дополнительные manifests для обучения. |
| Графовые эксперименты | `run_cluster_first_tail.py`, `run_public_graph_community.py`, `run_backbone_fusion_c4_tail.py`, `run_rowwise_tail_router.py`, `run_torch_checkpoint_c4_tail.py` | Проверить разные способы выбора соседей после модели. |
| EDA | `run_eda_*.py`, `build_eda_review_package.py`, `export_eda_csv_pack.py`, `build_validation_cycle_package.py`, `compare_submission_overlap.py` | Подготовить анализ данных, отчёты и сравнения отправок. |
| Экспорт и скорость | `export_campp_onnx.py`, `build_tensorrt_fp16_engine.py`, `benchmark_campp_tensorrt.py`, `materialize_official_campp_frontend_cache.py`, `pack_official_campp_frontend_cache.py`, `profile_campp_*` | Экспорт модели, ускорение и измерение скорости. |
| Служебные проверки | `show_config.py`, `infer_smoke.py`, `training_env_smoke.py`, `repro_check.py`, `validate_manifests.py` | Проверить конфиг, serving/training surface, воспроизводимость и manifests. |

## `src/kryptonite/`

Это основная рабочая логика проекта.

| Путь | Зачем нужен |
| --- | --- |
| `src/kryptonite/config.py` | Общая typed-конфигурация проекта. |
| `src/kryptonite/deployment.py` | Проверки путей и артефактов для запуска. |
| `src/kryptonite/repro.py` | Seed, fingerprints и reproducibility snapshot. |
| `src/kryptonite/tracking.py` | Локальный и внешний трекинг экспериментов. |
| `src/kryptonite/py.typed` | Пометка, что пакет содержит типы. |
| `src/kryptonite/common/` | Небольшие общие утилиты, сейчас в основном парсинг значений. |
| `src/kryptonite/eda/official_campp_tail/` | Разбитая по модулям реализация основного CAM++ inference tail, чтобы `scripts/run_official_campp_tail.py` оставался тонким entrypoint. |

### `src/kryptonite/data/`

| Файлы | Зачем нужны |
| --- | --- |
| `audio_io.py`, `audio_loader.py` | Чтение аудио, ресемплинг, приведение к нужному формату. |
| `schema.py`, `validation.py`, `manifest_artifacts.py`, `participant_manifests.py` | Контракт и проверка manifests. |
| `loudness.py`, `vad.py`, `convolution.py` | Нормализация громкости, VAD, свёртка с импульсными откликами для аугментаций. |

### `src/kryptonite/features/`

| Файлы | Зачем нужны |
| --- | --- |
| `campp_official.py` | Официальный CAM++ frontend, критичный для сильной ветки. |
| `fbank.py`, `chunking.py` | Построение аудиопризнаков и нарезка аудио. |
| `cache.py` | Кеширование признаков. |
| `benchmark.py`, `reporting.py` | Измерения и отчёты по признакам. |

### `src/kryptonite/models/`

| Файлы | Зачем нужны |
| --- | --- |
| `scoring.py` | Нормализация векторов и cosine scoring. |
| `models/campp/` | Реализация CAM++ encoder, слоёв, loss и checkpoint loading. |
| `models/eres2netv2/` | Реализация ERes2NetV2 encoder и загрузка checkpoint. |

### `src/kryptonite/training/`

| Файлы | Зачем нужны |
| --- | --- |
| `baseline_pipeline.py`, `speaker_baseline.py` | Общий training pipeline для speaker models. |
| `baseline_config.py`, `config_helpers.py`, `baseline_reporting.py` | Конфиги и отчёты обучения. |
| `manifest_speaker_data.py`, `production_dataloader.py` | Dataset и dataloader для обучения. |
| `optimization_runtime.py` | Optimizer, scheduler, mixed precision, gradient accumulation. |
| `augmentation_runtime.py`, `augmentation_scheduler.py` | Аудиоаугментации и расписание аугментаций. |
| `trainable_scope.py` | Управление заморозкой частей модели, например BN-only adaptation. |
| `campp/`, `eres2netv2/` | Тонкие model-family wrappers над общим pipeline. |
| `hf_xvector.py` | Hugging Face AudioXVector training branch. |
| `teacher_peft/` | Teacher/PEFT training branch. |

### `src/kryptonite/eda/`

| Файлы | Зачем нужны |
| --- | --- |
| `audio_stats.py`, `speaker_stats.py`, `domain.py`, `dense_audio.py` | Анализ аудио, дикторов и доменных признаков. |
| `manifest.py`, `csv_*`, `review_*` | Загрузка CSV, подготовка EDA-пакетов и отчётов. |
| `retrieval.py`, `dense_gallery.py`, `public_ablation.py`, `validation_cycle.py` | Проверки качества поиска соседей и локальной проверки, похожей на public. |
| `community.py`, `rerank.py`, `classifier_first.py`, `rowwise_tail_router.py`, `fusion.py` | Постобработка соседей, графовые методы, class-aware поправки и объединение результатов. |
| `submission.py` | Валидатор challenge submission. |
| `leaderboard_alignment.py`, `soft_pseudo_stability.py`, `hf_xvector.py` | Диагностика связи локальных метрик, public score и отдельных модельных веток. |

### `src/kryptonite/eval/`

| Файлы | Зачем нужны |
| --- | --- |
| `verification_metrics.py`, `verification_data.py`, `verification_report.py` | Verification-метрики и отчёты. |
| `identification_metrics.py`, `identification_report.py` | Identification-метрики и отчёты. |
| `cohort_bank.py` | Cohort bank для score normalization. |

### `src/kryptonite/serve/`

Эта папка нужна для runtime, API, ONNX и TensorRT. Для финального `submission.csv` она
не является главным входом, но важна для инженерной части и скорости.

| Файлы | Зачем нужны |
| --- | --- |
| `inferencer.py`, `runtime.py`, `inference_backend.py`, `inference_package.py` | Общий inference runtime и выбор backend. |
| `onnx_export.py`, `export_boundary.py` | Экспорт модели и контракт входов-выходов. |
| `tensorrt_engine*.py` | TensorRT FP16 engine, конфиги и runtime. |
| `http.py`, `api_models.py`, `scoring_service.py` | FastAPI endpoints и scoring service. |
| `enrollment_store.py`, `demo.py`, `demo_ui.py` | Demo/API support, не основной путь для challenge submission. |

## `docs/`

| Путь | Зачем нужен |
| --- | --- |
| `docs/README.md` | Навигация по документации. |
| `docs/challenge-solution-report.md` | Короткий отчёт для жюри. |
| `docs/repository-file-inventory.md` | Этот реестр файлов. |
| `docs/challenge-experiment-history.md` | Главный leaderboard-журнал экспериментов. |
| `docs/trails/` | Подробные trail-записи по отдельным экспериментам. |
| `docs/assets/public-lb-score.svg` | График изменения public LB score для README. |
| `docs/code-architecture.md` | Краткая архитектура кода. |
| `docs/system-architecture-v1.md` | Верхнеуровневый путь от аудио до `submission.csv`. |
| `docs/inference-acceleration.md` | Экспорт ONNX/TensorRT для финального MS32, полный замер скорости, H100-профили и вывод об узком месте frontend. |
| `docs/model-task-contract.md` | Формат входа, выхода и валидатор. |
| `docs/model-card.md` | Карточка текущего решения и ограничения. |
| `docs/release-runbook.md` | Финальный контрольный список сдачи. |
| `docs/data.md`, `docs/training.md`, `docs/configuration.md` | Рабочие заметки по данным, обучению и конфигам. |
| `docs/reference/audio-pipeline.md` | Подробный контракт аудио. |

## `deployment/`

| Путь | Зачем нужен |
| --- | --- |
| `deployment/README.md` | Навигация по deployment-файлам. |
| `deployment/docker/*.Dockerfile` | Docker images для обучения и инференса. |

## `tests/`

Тесты нужны, чтобы не ломать существующие контракты при правках.

| Путь | Зачем нужен |
| --- | --- |
| `tests/README.md` | Навигация по тестам. |
| `tests/unit/` | Unit-тесты для данных, признаков, моделей, EDA, обучения, экспорта и валидатора. |

## `notebooks/`, `assets/`, `artifacts/`, `datasets/`

| Путь | Зачем нужен |
| --- | --- |
| `notebooks/` | Место для exploration. Production-логика не должна жить только здесь. |
| `assets/` | Небольшие статичные материалы. Сейчас в основном placeholder. |
| `artifacts/` | Локальные результаты запусков: веса, векторы голосов, файлы отправки, логи, отчёты. Папка игнорируется git. |
| `datasets/` | Локальные данные организаторов и внешние датасеты. Папка игнорируется git. |

## Что можно не читать жюри

Жюри не нужно читать весь код. Минимальный маршрут:

1. `README.md`
2. `docs/challenge-solution-report.md`
3. `docs/release-runbook.md`
4. `docs/inference-acceleration.md`
5. `docs/model-task-contract.md`
6. При вопросах по экспериментам - `docs/challenge-experiment-history.md` и `docs/trails/`
