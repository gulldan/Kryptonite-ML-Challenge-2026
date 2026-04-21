# CAM++ English пакет дообучения

Переносимый training bundle (набор кода и команд для обучения) под первый цикл `CAM++ English VoxCeleb`.

## Что входит

- `prepare_data.py` — готовит split (разбиение) по спикерам и `3D-Speaker`-совместимые CSV.
- `eval_campp.py` — считает baseline (базовые) метрики `Precision@1`, `Precision@5`, `Precision@10`.
- `finetune_campp.py` — дообучает `CAM++` и поддерживает заморозку `backbone` (основной части модели) и аугментации.
- `build_submission.py` — строит `submission.csv` для `test_public.csv`.
- `run_ms42_submission.py` и `ms42_release/` — финальный `CAM++` inference runtime с official frontend, class-aware rerank и graph tail.
- `repair_mlflow.py` — находит и закрывает stale runs (зависшие запуски) в `MLflow`.
- `configs/campp_en_ft.base.yaml` — общий конфиг.
- `configs/campp_en_ft.local.yaml` — локальный конфиг.
- `configs/campp_en_ft.server.yaml` — серверный конфиг.
- `configs/campp_en_ft.soft_noaug.*.yaml` — мягкий `finetune` (дообучение) без аугментаций.
- `configs/campp_en_ft.soft_aug.*.yaml` — тот же мягкий `finetune`, но с аугментациями.

## Окружение через `uv`

Канонический путь:

```bash
uv venv .venv
source .venv/bin/activate
uv sync
uv sync --extra modelscope
```

## Конфиги

Используйте:
- `code/campp/configs/campp_en_ft.local.yaml` — на локальной машине;
- `code/campp/configs/campp_en_ft.server.yaml` — на сервере.

Для следующих экспериментов удобнее брать готовые варианты:
- `code/campp/configs/campp_en_ft.soft_noaug.local.yaml`
- `code/campp/configs/campp_en_ft.soft_aug.local.yaml`
- `code/campp/configs/campp_en_ft.soft_noaug.server.yaml`
- `code/campp/configs/campp_en_ft.soft_aug.server.yaml`

Схема `base + override` (базовый конфиг плюс небольшой конфиг-перекрытие) уже включена через `_base_`.

Главные поля:
- `paths.data_root` — корень распакованного датасета;
- `paths.train_csv` — исходный `train.csv`;
- `paths.audio_header_cache` — parquet (формат колоночного хранения) с длительностями из EDA;
- `paths.experiment_root` — каталог всех артефактов запуска;
- `paths.mlflow_tracking_uri` — адрес `MLflow`; если пусто, используется локальный `mlruns/`;
- `pretrained.weight_path` — локальный путь до английских весов `CAM++`.

Предобученные веса по умолчанию ожидаются в `data/pretrained_models/`, а не внутри `code/`.

## Шаг 1. Подготовить split и manifests

```bash
python code/campp/prepare_data.py --config code/campp/configs/campp_en_ft.local.yaml
```

Что получится:
- `data/campp_runs/.../prepared/train_manifest.csv`
- `data/campp_runs/.../prepared/val_manifest.csv`
- `data/campp_runs/.../prepared/test_manifest.csv`
- `data/campp_runs/.../prepared/speaker_to_index.json`
- `data/campp_runs/.../prepared/split_summary.json`

Split (разбиение) делается по спикерам:
- `80%` train;
- `10%` validation (валидация, отложенная проверка для выбора режима и лучшей модели);
- `10%` internal test (внутренний тест, честная итоговая проверка).

В `validation` и `internal test` идут только спикеры с минимум `11` записями.

## Шаг 2. Оценить базовую английскую модель на validation

```bash
python code/campp/eval_campp.py \
  --config code/campp/configs/campp_en_ft.local.yaml \
  --split validation
```

По умолчанию сравниваются два inference mode (режима извлечения эмбеддингов):
- `single_crop` — один кусок длиной `eval_chunk_sec`;
- `segment_mean` — несколько кусков и усреднение эмбеддингов.

Результаты сохраняются в:
- `runs/pretrained_val_modes/`

## Шаг 3. Оценить базовую английскую модель на internal test

После выбора лучшего режима из `run_summary.json` предыдущего шага:

```bash
python code/campp/eval_campp.py \
  --config code/campp/configs/campp_en_ft.local.yaml \
  --split test \
  --best-mode-from data/campp_runs/campp_en_ft/runs/pretrained_val_modes/run_summary.json
```

Результаты сохраняются в:
- `runs/pretrained_test_bestmode/`

## Шаг 4. Короткий benchmark (замер скорости)

```bash
python code/campp/finetune_campp.py \
  --config code/campp/configs/campp_en_ft.local.yaml \
  --run-name campp_en_benchmark \
  --max-steps 1000
```

Этот запуск нужен, чтобы измерить:
- время шага;
- время эпохи;
- прогноз на `3–5` эпох;
- укладывается ли локальный запуск в `8–9` часов.

## Шаг 5. Первый полноценный finetune (дообучение) без аугментаций

```bash
python code/campp/finetune_campp.py \
  --config code/campp/configs/campp_en_ft.soft_noaug.local.yaml \
  --run-name campp_en_ft_noaug_run001
```

В этом режиме:
- `backbone frozen` (основная часть модели заморожена) на первые `2` эпохи;
- `backbone_lr` (скорость обучения основной части модели) уменьшен;
- `classifier_lr` (скорость обучения классификатора) тоже уменьшен;
- сохраняются `epoch_001.pt`, `epoch_002.pt`, …, `best_p10.pt`, `last.pt`.

## Шаг 5b. Мягкий finetune с аугментациями

```bash
python code/campp/finetune_campp.py \
  --config code/campp/configs/campp_en_ft.soft_aug.local.yaml \
  --run-name campp_en_ft_aug_run001
```

В этом режиме добавлены умеренные аугментации:
- `noise` (шум);
- `reverb` (реверберация, эффект помещения);
- `band_limit` (сужение полосы частот, имитация канала);
- `silence_shift` (паузы и смещение полезного сигнала).

На один пример применяется не более одной аугментации.

## Шаг 6. Оценить лучший checkpoint (сохранённые веса) на internal test

```bash
python code/campp/eval_campp.py \
  --config code/campp/configs/campp_en_ft.local.yaml \
  --split test \
  --checkpoint data/campp_runs/campp_en_ft/runs/campp_en_ft_noaug_run001/checkpoints/best_p10.pt \
  --modes segment_mean \
  --run-name campp_en_ft_noaug_run001_test_eval
```

## Шаг 7. Построить `submission.csv`

Для `pretrained` (предобученных весов) с лучшим режимом из validation:

```bash
python code/campp/build_submission.py \
  --config code/campp/configs/campp_en_ft.local.yaml \
  --best-mode-from data/campp_runs/campp_en_ft/runs/pretrained_val_modes/run_summary.json \
  --csv data/Для\ участников/test_public.csv \
  --topk 10 \
  --run-name submission_pretrained_segment_mean_test_public
```

Для лучшего `checkpoint` (сохранённых весов) после дообучения:

```bash
python code/campp/build_submission.py \
  --config code/campp/configs/campp_en_ft.local.yaml \
  --checkpoint data/campp_runs/campp_en_ft/runs/campp_en_ft_noaug_run001/checkpoints/best_p10.pt \
  --mode segment_mean \
  --csv data/Для\ участников/test_public.csv \
  --topk 10 \
  --run-name submission_ft_segment_mean_test_public
```

Результат лежит в:
- `data/campp_runs/campp_en_ft/submissions/<run-name>/submission.csv`

Формат строго такой:
- колонка `filepath`
- колонка `neighbours`
- порядок строк совпадает с `test_public.csv`
- в `neighbours` ровно `10` индексов без дублей и без self-index (индекса самой записи)

## Финальный CAM++ MS42 release

Для итогового запуска используется отдельный runtime:

```bash
MODEL=campp ./run.sh
```

Прямой запуск без root-entrypoint:

```bash
python code/campp/run_ms42_submission.py \
  --config code/campp/configs/campp_ms42_release.yaml \
  --manifest-csv data/Для\ участников/test_public.csv \
  --template-csv data/Для\ участников/test_public.csv \
  --data-root data/Для\ участников \
  --output-dir data/campp_runs/ms42_release/submissions/CAMPP_MS42_RELEASE \
  --run-id CAMPP_MS42_RELEASE
```

Ожидаемые веса:
- `data/pretrained_models/speaker_verification/campplus/ms31_official_pseudo_filtered_lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt`

TensorRT в этот перенос не включён: packaged backend — `torch`. На `ROCm` PyTorch также использует имя устройства `cuda`.

## Как переносить на сервер

Самый простой путь:

1. Перенести репозиторий `git clone` или архивом без данных.
2. На сервере распаковать датасет в нужный каталог.
3. Отредактировать только `code/campp/configs/campp_en_ft.server.yaml`.
4. Поднять окружение через `uv`.
5. Прогнать те же команды:
   - `prepare_data.py`
   - `eval_campp.py`
   - `finetune_campp.py`

В коде нет жёстких локальных абсолютных путей, поэтому перенос сводится к замене путей в серверном конфиге.

## Что логируется в `MLflow`

Если `paths.mlflow_tracking_uri` указан:
- логирование идёт в удалённый `MLflow`.

Если `paths.mlflow_tracking_uri` пустой:
- используется локальный `file://.../mlruns` внутри `experiment_root`.

Во всех случаях локально сохраняются:
- `metrics.csv`;
- `run_summary.json`;
- `config_resolved.yaml`;
- `history.csv` и `train.log` для обучения;
- manifests и `split_summary.json`.

Чекпоинты в `MLflow` не грузятся. В `MLflow` пишется путь к ним на диске.

## Как почистить зависшие `MLflow` runs

Сначала посмотреть кандидатов:

```bash
python code/campp/repair_mlflow.py \
  --config code/campp/configs/campp_en_ft.soft_noaug.local.yaml
```

Потом закрыть stale runs (зависшие запуски):

```bash
python code/campp/repair_mlflow.py \
  --config code/campp/configs/campp_en_ft.soft_noaug.local.yaml \
  --apply
```
