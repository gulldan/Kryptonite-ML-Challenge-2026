# WavLM пакет

Архивный baseline из исследовательской ветки. Этот каталог сохранён только для
истории и точечного воспроизведения старых экспериментов; текущий organizer-facing
entrypoint `./run.sh` его не использует.

Переносимый пакет для baseline-цикла с `microsoft/wavlm-base-plus-sv`.

Сейчас в пакете есть:
- `prepare_data.py` — готовит тот же speaker-level split (разбиение по спикерам), что и для других baseline-ов;
- `eval_wavlm.py` — считает retrieval-метрики (метрики поиска соседей) на `validation`;
- `build_submission.py` — строит `submission.csv` для `test_public.csv`;
- `common.py`, `retrieval.py` и `configs/*` — общая инфраструктура и конфиги.

## Окружение через `uv`

```bash
uv venv .venv
source .venv/bin/activate
uv sync
```

## Конфиг

Используйте:
- `research/archive/legacy-baselines/wavlm/configs/wavlm_base_plus_sv.local.yaml` —
  основной локальный конфиг.

Главные поля:
- `paths.experiment_root` — папка артефактов baseline-а;
- `paths.pretrained_root` — локальный cache (кэш) Hugging Face модели;
- `pretrained.model_id` — `microsoft/wavlm-base-plus-sv`;
- `evaluation.primary_mode=xvector_full_file` — один `xvector`-эмбеддинг на файл;
  альтернативный режим `xvector_chunk_mean` доступен через конфиг или `--mode`.

## Подготовить split и manifests

```bash
python research/archive/legacy-baselines/wavlm/prepare_data.py \
  --config research/archive/legacy-baselines/wavlm/configs/wavlm_base_plus_sv.local.yaml
```

## Оценить pretrained на validation

```bash
python research/archive/legacy-baselines/wavlm/eval_wavlm.py \
  --config research/archive/legacy-baselines/wavlm/configs/wavlm_base_plus_sv.local.yaml \
  --split validation
```

## Собрать submission

```bash
python research/archive/legacy-baselines/wavlm/build_submission.py \
  --config research/archive/legacy-baselines/wavlm/configs/wavlm_base_plus_sv.local.yaml \
  --csv data/Для\ участников/test_public.csv \
  --topk 10 \
  --run-name submission_pretrained_wavlm_base_plus_sv_test_public
```

Результат лежит в:
- `data/wavlm_runs/wavlm_base_plus_sv/submissions/<run-name>/submission.csv`

## Статус

Этот baseline архивирован и не является частью текущего финального launcher-а.
Для organizer-facing submit path используйте корневой `./run.sh`, который сейчас
поддерживает только `w2v-trt` и `campp-pt`.
