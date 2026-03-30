# Code Architecture

Карта кода: что где лежит и за что отвечает.

## Верхний уровень

```
src/kryptonite/       бизнес-логика
scripts/              CLI entrypoints (тонкие обёртки)
configs/              TOML конфиги
apps/api/             FastAPI сервер
docs/                 документация
tests/                тесты
```

## Модули `src/kryptonite/`

### config.py (604 строк)
Центральный конфиг проекта. Все настройки runtime, training, features, chunking,
normalization, VAD собраны здесь в typed dataclasses. Загружается из TOML с поддержкой
dotted-key overrides и `.env` файлов.

### data/ -- загрузка данных
| Файл | Строк | Что делает |
|------|-------|------------|
| `audio_io.py` | 146 | Низкоуровневое чтение аудиофайлов (soundfile) |
| `audio_loader.py` | 375 | Загрузка аудио по манифесту: ресемплинг, нормализация, VAD |
| `schema.py` | 395 | ManifestRow -- контракт строки манифеста (speaker_id, audio_path, ...) |
| `validation.py` | 184 | Валидация манифестов по схеме |
| `loudness.py` | 164 | Нормализация громкости (LUFS) |
| `vad.py` | 513 | Voice Activity Detection (Silero VAD) |
| `manifest_artifacts.py` | 249 | Bundles манифестов с checksums |

**Поток:** манифест JSONL -> `audio_loader` (читает файл, ресемплит 16kHz, VAD) -> waveform tensor

### features/ -- извлечение признаков
| Файл | Строк | Что делает |
|------|-------|------------|
| `fbank.py` | 405 | 80-dim log-Mel Fbank: FFT, mel-фильтры, CMVN |
| `chunking.py` | 439 | Нарезка utterance на фиксированные куски (train/eval) |
| `cache.py` | 490 | Кеш признаков на диск |
| `benchmark.py` | 536 | Бенчмарк скорости извлечения |
| `reporting.py` | 331 | Отчёт о паритетности offline/streaming |

**Поток:** waveform -> `chunking` (3s куски) -> `fbank` (80-bin спектрограмма) -> `[frames, 80]`

### models/ -- архитектуры моделей
| Файл | Строк | Что делает |
|------|-------|------------|
| `scoring.py` | 157 | Cosine scoring, L2 нормализация |
| **campp/** | | |
| `model.py` | 200 | CAMPPlusEncoder: dense TDNN блоки -> 512-dim embedding |
| `layers.py` | 288 | Базовые блоки: TDNN, DenseTDNN, StatisticsPool |
| `losses.py` | 91 | ArcMarginLoss, CosineClassifier |
| `checkpoint.py` | 114 | Загрузка/сохранение checkpoint |
| **eres2netv2/** | | |
| `model.py` | 278 | ERes2NetV2Encoder: residual блоки + fusion -> 192-dim embedding |
| `fusion.py` | 31 | Adaptive feature fusion |
| `pooling.py` | 48 | TSTP pooling (temporal statistics) |
| `checkpoint.py` | 100 | Загрузка/сохранение checkpoint |

**Контракт:** `[batch, frames, 80]` -> encoder -> `[batch, emb_dim]`

### training/ -- обучение
| Файл | Строк | Что делает |
|------|-------|------------|
| `baseline_pipeline.py` | 360 | **Главный оркестратор:** train -> embed -> score -> report |
| `speaker_baseline.py` | 628 | Runtime: train_epochs, export_embeddings, score_trials |
| `manifest_speaker_data.py` | 210 | Dataset: загрузка аудио -> fbank -> TrainingExample |
| `production_dataloader.py` | 272 | BalancedSpeakerBatchSampler + DataLoader |
| `optimization_runtime.py` | 344 | Optimizer (AdamW/SGD), scheduler, AMP, grad accumulation |
| `baseline_config.py` | 176 | Dataclasses: BaselineDataConfig, ObjectiveConfig, ... |
| `baseline_reporting.py` | 155 | Markdown отчёт по итогам обучения |
| `config_helpers.py` | 133 | Парсинг TOML -> typed config |
| `environment.py` | 198 | Smoke-check: GPU, torch, imports |
| **campp/** | | |
| `config.py` | 92 | Загрузка TOML -> CAMPPlusBaselineConfig |
| `pipeline.py` | 37 | Создать encoder -> вызвать run_speaker_baseline() |
| `data.py` | 21 | Compatibility re-exports |
| **eres2netv2/** | | |
| `config.py` | 105 | Загрузка TOML -> ERes2NetV2BaselineConfig |
| `pipeline.py` | 37 | Создать encoder -> вызвать run_speaker_baseline() |

**Поток:**
```
run_baseline.py
  -> campp/pipeline.py (или eres2netv2/)
    -> baseline_pipeline.run_speaker_baseline()
      -> production_dataloader (balanced batches)
      -> optimization_runtime (AdamW + cosine LR + AMP)
      -> speaker_baseline.train_epochs()
      -> speaker_baseline.export_dev_embeddings()
      -> speaker_baseline.score_trials()
      -> eval.verification_report (EER, minDCF)
```

### eval/ -- оценка
| Файл | Строк | Что делает |
|------|-------|------------|
| `verification_metrics.py` | 229 | EER, minDCF по threshold sweep |
| `verification_data.py` | 102 | I/O: загрузка scores, trials, metadata (JSONL/Parquet) |
| `verification_report.py` | 566 | Построение отчёта: метрики, гистограммы, ROC/DET, калибровка |
| `cohort_bank.py` | 639 | Сборка когорт-банка для score normalization |

### serve/ -- inference и API
| Файл | Строк | Что делает |
|------|-------|------------|
| `inferencer.py` | 592 | Unified inference: audio -> embedding (PyTorch/ONNX) |
| `http.py` | 571 | FastAPI роуты: /embed, /enroll, /verify, /demo |
| `scoring_service.py` | 284 | Scoring: cosine similarity между enrollment и test |
| `runtime.py` | 656 | Runtime probes, backend resolution, metadata |
| `onnx_export.py` | 485 | Экспорт в ONNX |
| `inference_backend.py` | 215 | Выбор backend (torch/onnx) |
| `inference_package.py` | 200 | Метаданные inference bundle |
| `enrollment_store.py` | 285 | SQLite хранилище enrollment embeddings |
| `demo.py` | 335 | Логика браузерного demo |
| `demo_ui.py` | 93 | Раздача UI статики |
| `api_models.py` | 291 | Pydantic модели запросов |

### Вспомогательные модули
| Файл | Строк | Что делает |
|------|-------|------------|
| `deployment.py` | 188 | Резолв путей, preflight проверки артефактов |
| `repro.py` | 240 | Seed, fingerprints, reproducibility snapshot |
| `tracking.py` | 131 | Эксперимент-трекинг (local/mlflow/wandb) |
| `common/parsing.py` | 58 | Парсинг типов из TOML |

## Scripts

| Скрипт | Что делает |
|--------|------------|
| `run_baseline.py` | Обучить модель: `--model {campp,eres2netv2} --config ...` |
| `run_campp_baseline.py` | Шорткат для CAM++ |
| `run_eres2netv2_baseline.py` | Шорткат для ERes2NetV2 |
| `show_config.py` | Показать итоговый конфиг |
| `training_env_smoke.py` | Проверить окружение |
| `validate_manifests.py` | Валидировать манифесты |

## Configs

| Файл | Что делает |
|------|------------|
| `base.toml` | Базовый конфиг: пути, runtime, features, chunking |
| `training/campp-baseline.toml` | CAM++ обучение |
| `training/eres2netv2-baseline.toml` | ERes2NetV2 обучение |

## Куда добавлять новый код

| Что нужно | Куда |
|-----------|------|
| Новая модель | `models/<name>/`, `training/<name>/`, TOML config |
| Новый формат данных | `data/` |
| Новый тип признаков | `features/` |
| Новая метрика | `eval/` |
| Новый API endpoint | `serve/http.py` |
| CLI скрипт | `scripts/` |
