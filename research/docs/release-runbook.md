# Инструкция финальной сдачи

Архивный research-runbook. Он документирует старый `MS41`-ориентированный release path
и хранится для воспроизводимости прошлых экспериментов. Этот файл не означает, что
финальная модель уже выбрана для текущего repo state.

Этот документ описывает практический порядок подготовки `submission.csv` для сдачи.
README остаётся короткой точкой входа, а здесь находятся рабочие команды.

## 1. Проверить окружение

```bash
uv sync --dev --group train --group infer --group eda
```

Минимальные ожидания:

- Python `3.12`;
- `uv` управляет зависимостями;
- CUDA GPU доступен для полного пересчёта;
- команда запускается из корня репозитория.

Финальное обучение проводилось внутри Docker-образа
`nvcr.io/nvidia/pytorch:26.03-py3`. Полный private-инференс можно запускать в том же
образе или в совместимом CUDA-окружении после `uv sync`.

## 2. Проверить данные

Для public:

```text
datasets/Для участников/test_public.csv
datasets/Для участников/test_public/
```

Для private организаторы должны дать такой же CSV-формат: столбец `filepath` и папку
с аудиофайлами.

## 3. Проверить веса и артефакты

Перед быстрым воспроизведением текущего кандидата должны существовать:

```text
artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt
artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z/
```

Для быстрого полного пересчёта из аудио дополнительно нужен TensorRT engine MS32:

```text
artifacts/model-bundle-campp-ms32-onnx/model.plan
```

Если engine ещё не собран, команды экспорта ONNX и сборки TensorRT находятся в
[inference-acceleration.md](./inference-acceleration.md).

Если кеш ближайших соседей и векторы голосов отсутствуют, нужно использовать полный
пересчёт из аудио.

## 4. Получить файл отправки

Короткий путь для текущего финального MS41 preset:

```bash
uv run --group train python scripts/run_ms41_submission.py \
  --config configs/release/ms41-private-full.toml \
  --manifest-csv datasets/Для\ участников/test_private.csv
```

Эта команда запускает тот же трёхступенчатый MS41 pipeline, копирует итоговый
`submission.csv` в корень репозитория, валидирует его и пишет `speed_summary.txt`.

Основной путь:

1. Посчитать или загрузить MS32 векторы голосов и top-200 ближайших соседей.
2. Посчитать кеш classifier posterior.
3. Запустить `scripts/run_class_aware_graph_tail.py`.
4. Получить `submission.csv`.
5. Запустить валидатор.

### Закрытый набор с полного пересчёта из аудио

Это основной сценарий для организаторов. Он не использует готовые embeddings или top-k
cache. На вход нужны private CSV, private аудио и checkpoint модели.

С полного нуля здесь означает: без готовых кешей для private. Веса модели должны быть
переданы вместе с решением.

Перед запуском нужно заменить только две переменные:

- `PRIVATE_CSV` - путь к private CSV;
- `DATA_ROOT` - папка, относительно которой указаны пути из столбца `filepath`.

```bash
PRIVATE_CSV=datasets/Для\ участников/test_private.csv
DATA_ROOT=datasets/Для\ участников

RUN_ID=MS41_private_full
OUT=artifacts/release/private-ms41-full
BASE_RUN="${RUN_ID}_ms32"
BASE_OUT="$OUT/ms32"
CHECKPOINT=artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt

mkdir -p "$OUT" "$BASE_OUT"
SECONDS=0

stage_start=$SECONDS
uv run --group train python scripts/run_official_campp_tail.py \
  --checkpoint-path "$CHECKPOINT" \
  --manifest-csv "$PRIVATE_CSV" \
  --template-csv "$PRIVATE_CSV" \
  --data-root "$DATA_ROOT" \
  --output-dir "$BASE_OUT" \
  --experiment-id "$BASE_RUN" \
  --encoder-backend tensorrt \
  --tensorrt-config configs/release/tensorrt-fp16-ms32.toml \
  --device cuda \
  --search-device cuda \
  --batch-size 64 \
  --frontend-workers 16 \
  --frontend-prefetch 256 \
  --search-batch-size 4096 \
  --top-cache-k 200 \
  --mode segment_mean \
  --eval-chunk-seconds 6.0 \
  --segment-count 3 \
  --pad-mode repeat \
  --force-embeddings \
  --skip-c4
embedding_and_top200_seconds=$((SECONDS - stage_start))

stage_start=$SECONDS
uv run --group train python scripts/run_classifier_first_tail.py \
  --checkpoint-path "$CHECKPOINT" \
  --embeddings-path "$BASE_OUT/embeddings_${BASE_RUN}.npy" \
  --manifest-csv "$PRIVATE_CSV" \
  --output-dir "$OUT" \
  --experiment-id "${RUN_ID}_classcache" \
  --device cuda \
  --class-batch-size 4096 \
  --class-top-k 5 \
  --class-scale 32.0 \
  --force-classifier \
  --class-cache-only
class_cache_seconds=$((SECONDS - stage_start))

stage_start=$SECONDS
uv run --group train python scripts/run_class_aware_graph_tail.py \
  --indices-path "$BASE_OUT/indices_${BASE_RUN}_top200.npy" \
  --scores-path "$BASE_OUT/scores_${BASE_RUN}_top200.npy" \
  --class-indices-path "$OUT/class_indices_${RUN_ID}_classcache_top5.npy" \
  --class-probs-path "$OUT/class_probs_${RUN_ID}_classcache_top5.npy" \
  --manifest-csv "$PRIVATE_CSV" \
  --template-csv "$PRIVATE_CSV" \
  --output-dir "$OUT" \
  --experiment-id "$RUN_ID" \
  --class-overlap-top-k 3 \
  --class-overlap-weight 0.03 \
  --same-top1-bonus 0.01 \
  --same-query-topk-bonus 0.005 \
  --edge-top 10 \
  --reciprocal-top 20 \
  --rank-top 100 \
  --iterations 5 \
  --label-min-size 5 \
  --label-max-size 120 \
  --label-min-candidates 3 \
  --shared-top 20 \
  --shared-min-count 0 \
  --reciprocal-bonus 0.03 \
  --density-penalty 0.02
graph_tail_seconds=$((SECONDS - stage_start))

cp "$OUT/submission_${RUN_ID}.csv" submission.csv

stage_start=$SECONDS
uv run python scripts/validate_submission.py \
  --template-csv "$PRIVATE_CSV" \
  --submission-csv submission.csv \
  --output-json "$OUT/submission_validation.json"
validation_seconds=$((SECONDS - stage_start))

sha256sum submission.csv | tee "$OUT/submission.sha256"

total_seconds=$SECONDS
{
  echo "embedding_and_top200_seconds=$embedding_and_top200_seconds"
  echo "class_cache_seconds=$class_cache_seconds"
  echo "graph_tail_seconds=$graph_tail_seconds"
  echo "validation_seconds=$validation_seconds"
  echo "total_seconds=$total_seconds"
} | tee "$OUT/speed_summary.txt"
```

На выходе должны появиться:

- `submission.csv` в корне репозитория;
- `$OUT/submission_validation.json`;
- `$OUT/submission.sha256`;
- `$OUT/speed_summary.txt`.

### Быстрый открытый путь с готовыми большими файлами

Этот путь оставлен для воспроизведения уже посчитанного public-кандидата. Для private
его использовать нельзя, если нет заранее посчитанных private embeddings и top-k cache.

```bash
RUN_ID=MS41_ms32_classaware_c4_weak_20260415T0530Z
OUT=artifacts/release/final-ms41
MS32_DIR=artifacts/backbone_public/modelscope_campplus_voxceleb_default/ms32_pseudo_ms31_20260414T0551Z
CHECKPOINT=artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt
EMBEDDINGS_PATH="$MS32_DIR/embeddings_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z.npy"
INDICES_PATH="$MS32_DIR/indices_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy"
SCORES_PATH="$MS32_DIR/scores_MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z_top200.npy"

uv run --group train python scripts/run_classifier_first_tail.py \
  --checkpoint-path "$CHECKPOINT" \
  --embeddings-path "$EMBEDDINGS_PATH" \
  --manifest-csv datasets/Для\ участников/test_public.csv \
  --output-dir "$OUT" \
  --experiment-id "${RUN_ID}_classcache" \
  --device cuda \
  --class-batch-size 4096 \
  --class-top-k 5 \
  --class-scale 32.0 \
  --class-cache-only

uv run --group train python scripts/run_class_aware_graph_tail.py \
  --indices-path "$INDICES_PATH" \
  --scores-path "$SCORES_PATH" \
  --class-indices-path "$OUT/class_indices_${RUN_ID}_classcache_top5.npy" \
  --class-probs-path "$OUT/class_probs_${RUN_ID}_classcache_top5.npy" \
  --manifest-csv datasets/Для\ участников/test_public.csv \
  --template-csv datasets/Для\ участников/test_public.csv \
  --output-dir "$OUT" \
  --experiment-id "$RUN_ID" \
  --class-overlap-top-k 3 \
  --class-overlap-weight 0.03 \
  --same-top1-bonus 0.01 \
  --same-query-topk-bonus 0.005 \
  --edge-top 10 \
  --reciprocal-top 20 \
  --rank-top 100 \
  --iterations 5 \
  --label-min-size 5 \
  --label-max-size 120 \
  --label-min-candidates 3 \
  --shared-top 20 \
  --shared-min-count 0 \
  --reciprocal-bonus 0.03 \
  --density-penalty 0.02

cp "$OUT/submission_${RUN_ID}.csv" submission.csv
```

## 5. Проверить формат

```bash
PRIVATE_CSV=datasets/Для\ участников/test_private.csv

uv run python scripts/validate_submission.py \
  --template-csv "$PRIVATE_CSV" \
  --submission-csv submission.csv \
  --output-json artifacts/release/submission_validation.json
```

Перед сдачей должно быть:

- `passed=True`;
- `error_count=0`;
- число строк равно числу строк во входном CSV;
- нет пустых ячеек со списком соседей;
- нет совпадений файла с самим собой;
- нет индексов вне диапазона.

## 6. Зафиксировать результат

После каждой отправки нужно обновить
[challenge-experiment-history.md](./challenge-experiment-history.md) и отдельную запись в
[trails/](./trails/):

- идентификатор запуска;
- путь к `submission.csv`;
- SHA-256;
- статус валидатора;
- public/private score, если он известен;
- решение: принять, отклонить или оставить для будущего объединения моделей или диагностики;
- затем пересобрать график для README:
  `uv run python scripts/render_public_lb_chart.py`.

## 7. Замерить скорость

Для критерия скорости нужен полный замер на той же инфраструктуре, где запускается
baseline организаторов:

```text
старт: команда инференса получает test CSV и аудио
финиш: готов и провалидирован submission.csv
```

В истории уже есть диагностические замеры TensorRT, кешей и повторных запусков.
Краткая карта ускорений и команды ONNX/TensorRT лежат в
[inference-acceleration.md](./inference-acceleration.md). Перед финальной сдачей
нужно записать один свежий end-to-end замер именно для полного выбранного режима.

Private-команда из раздела 4 уже делает такой замер и пишет файл:

```bash
cat artifacts/release/private-ms41-full/speed_summary.txt
```

В этом файле будут:

- время построения векторов голосов и top-200 соседей;
- время расчёта classifier posterior;
- время финальной графовой постобработки;
- время валидации;
- полное время от запуска до готового `submission.csv`.

Именно `total_seconds` из этого файла нужно сравнивать со временем baseline организаторов
на той же машине.

## 8. Типичные проблемы

| Симптом | Что проверить |
| --- | --- |
| `filepath order/content differs from template` | `submission.csv` построен не по тому CSV или строки были отсортированы. |
| Валидатор сообщает, что файл попал в собственные соседи | В списке соседей не удалён индекс самой строки. |
| Индексы вне диапазона | Кеш ближайших соседей не соответствует текущему тестовому CSV. |
| Слишком медленно | Проверить, используется ли кеш признаков и TensorRT; отдельно измерить извлечение признаков и encoder. |
| Public/private команда требует ручных правок | Вынести меняющиеся пути в shell-переменные или CLI flags, не менять код. |

## 9. Что сдавать вместе с репозиторием

- код;
- `README.md`;
- `research/docs/challenge-solution-report.md`;
- `research/docs/inference-acceleration.md`;
- финальный `submission.csv`;
- веса модели и большие артефакты, если организаторы не пересчитывают всё с нуля;
- SHA-256 и validation JSON для итогового файла.
