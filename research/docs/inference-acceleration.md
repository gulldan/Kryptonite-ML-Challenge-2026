# Ускорение инференса

Этот документ - единая страница для жюри по скорости финального решения. Он не
заменяет [release-runbook.md](./release-runbook.md): там лежит полный запуск
`submission.csv`, а здесь объяснены ускорение, профили и команды ONNX/TensorRT.

Обновление от 2026-04-16: после W2V1j текущий лучший public-кандидат в журнале -
`W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1` со score
`0.8344`. Поэтому speed workstream расширен с CAM++-only на сравнение четырёх
семейств: organizer baseline, CAM++, w2v-BERT 2.0 и ERes / WavLM. Старые CAM++
разделы ниже остаются как детальный пример уже подтверждённого TensorRT профиля.

## Что именно ускоряем

Сравнение `SPEED1_family_onnx_tensorrt_comparison_20260416` ускоряет по одному
представителю каждого семейства:

| Семейство | Выбранный источник | Public LB источника | TensorRT boundary |
| --- | --- | ---: | --- |
| Organizer baseline | `datasets/Для участников/baseline.onnx` без модификаций | `0.0779` | не конвертируем; запускаем как reference |
| CAM++ | MS32 encoder + MS41 class-aware C4 tail | `0.7473` | official FBank features -> CAM++ encoder |
| w2v-BERT 2.0 | W2V1j stage3 `teacher_peft` checkpoint | `0.8344` | HF feature extractor tensors -> encoder |
| ERes / WavLM | H9 official 3D-Speaker ERes2Net-large checkpoint | `0.5834` | FBank features -> ERes2Net encoder |

Главный контроль качества: TensorRT-generated `submission.csv` должен пройти
валидатор и быть сравнен с исходным submitted CSV своего семейства. Public LB score
для исходных файлов - внешнее наблюдение, а не локально воспроизводимая метрика.

Состав CAM++ MS41 пути, который уже был подробно профилирован:

1. MS32 CAM++ encoder строит векторы голосов и top-200 ближайших соседей.
2. MS41 добавляет classifier posterior cache, слабую class-aware поправку и графовую
   постобработку.
3. На выходе получается `submission.csv`, который проверяется валидатором.

TensorRT применяется к MS32 CAM++ encoder. Это именно та encoder-ветка, которая
используется в финальном MS41, а не ранний диагностический кандидат.

Для W2V1j и H9 добавлены отдельные generic TensorRT entrypoints:

- `scripts/export_teacher_peft_onnx.py`
- `scripts/export_official_3dspeaker_eres2net_onnx.py`
- `scripts/build_generic_tensorrt_engine.py`
- `scripts/run_teacher_peft_c4_tail.py --encoder-backend tensorrt`
- `scripts/run_official_3dspeaker_eres2net_tail.py --encoder-backend tensorrt`

Выбор моделей, source submissions и команды запуска зафиксированы в
`configs/release/speed-family-comparison.toml` и в trail:
`research/docs/trails/2026-04-16-speed-family-tensorrt-comparison.md`.

## Правило оценки скорости

Для критерия скорости жюри нужен только полный end-to-end замер:

```text
старт: команда получает полный test CSV и папку с аудио
финиш: готов и провалидирован submission.csv
```

Модели и engines считаются подготовленными заранее. Экспорт в ONNX и сборка TensorRT
обязательно документируются как воспроизводимая preparation phase, но не включаются в
число, по которому сравнивается скорость формирования `submission.csv`.

Нельзя подменять этот замер benchmark-ом encoder-а или профилем на `10k` строках.
Профили на `10k` нужны для объяснения bottleneck, но сравнивать с baseline
организаторов нужно `total_seconds` полного запуска из [release-runbook.md](./release-runbook.md).

## SPEED1: полный public submit на remote GPU1

Итоговый артефакт:
`artifacts/speed-family-comparison/speed_results.json`.

График в README:
`research/docs/assets/speed-comparison.svg`.

| Семейство | Prepared path | Full submit | Embedding/frontend | Search | C4 rerank | Source overlap |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Organizer baseline | organizer `baseline.onnx` без модификаций | `254.828s` | `254.828s` | n/a | n/a | n/a |
| CAM++ | MS32 TensorRT + MS41 tail, packed frontend path | `80.486s` | `67.402s` | `0.775s` | `11.690s` | mean@10 `8.595`, top1 `87.28%` |
| w2v-BERT 2.0 | W2V1j stage3 TensorRT encoder | `2562.068s` | `2549.060s` | `0.708s` | `11.653s` | mean@10 `9.880`, top1 `98.94%` |
| ERes / WavLM | H9 ERes2Net TensorRT encoder + parallel FBank workers | `345.641s` | `332.742s` | `0.741s` | `11.532s` | mean@10 `7.345`, top1 `69.12%` |

Все три TensorRT-generated CSV прошли локальный validator. Public LB score для
этих CSV не заявляется: source LB значения относятся к исходным отправленным
файлам, а TensorRT варианты требуют отдельной внешней отправки, особенно для ERes,
где overlap с source submission заметно ниже.

Per-layer TensorRT traces:

- CAM++:
  `artifacts/speed-family-comparison/layer-profiles/campp_ms32_b128_segment6_profile.json`
- W2V1j:
  `artifacts/speed-family-comparison/layer-profiles/w2vbert2_stage3_b1024_crop6_profile.json`
- ERes2Net H9:
  `artifacts/speed-family-comparison/layer-profiles/official_eres2net_h9_b128_chunk10_profile.json`

Полная private-команда в runbook не содержит `--limit-rows` и пишет:

```text
artifacts/release/private-ms41-full/speed_summary.txt
```

Именно `total_seconds` из этого файла надо сравнивать со временем baseline на той же
машине и на том же полном наборе данных.

## Главный вывод профилирования

После TensorRT H100 encoder уже не главный узкий участок для CAM++ и ERes. Почти
весь observed wall-clock первого пересчёта аудио съедает audio frontend,
особенно Kaldi-style FBank/HF feature extraction. GPU encoder быстрый, exact
top-k практически бесплатный.

Практический вывод: следующий выигрыш по скорости надо искать в извлечении признаков,
кешировании, packed feature store и уменьшении padding, а не в одном ручном
super-kernel для encoder.

## Диагностический профиль стадий

Артефакт:
`artifacts/profiles/campp_pipeline_stages_h100_head10000/stage_profile.md`.

Это диагностический профиль на `10k` public rows. Он показывает структуру времени, но
не является финальной оценкой скорости.

| Стадия | Время | Вывод |
| --- | ---: | --- |
| Полный wall-clock | `53.376s` | `187.35` rows/s на `10k` строк |
| Frontend wall | `53.068s` | `99.42%` observed wall |
| Fbank summed CPU | `755.557s` | сумма по 16 worker threads |
| Decode summed CPU | `81.512s` | сумма по worker threads |
| Segment summed CPU | `6.680s` | не bottleneck |
| Encoder padding | `9.530s` | заметный overhead batching/padding |
| TensorRT execute | `5.436s` | только `10.18%` observed wall |
| H2D + D2H | `0.929s` | копии не главный фактор |
| Exact top-k | `0.152s` | практически бесплатно |
| Embedding aggregate | `0.133s` | практически бесплатно |

`Fbank summed CPU` больше wall-clock, потому что это сумма времени по потокам.
Эффективный параллелизм frontend получился около `15.9x` на 16 threads. Значит
стадия уже хорошо распараллелена, но именно она держит весь первый прогон.

## Профиль модели

Артефакт:
`artifacts/profiles/campp_model_layers_h100_b64_t600/campp_model_layer_profile.md`.

PyTorch high-level modules:

| Module | Доля |
| --- | ---: |
| `head` / `FeatureContextModel` | `36.40%` |
| `xvector.block2` | `28.33%` |
| `xvector.block3` | `21.02%` |
| `xvector.block1` | `9.80%` |
| transit/stat/dense/output вместе | около `4.45%` |

Модельное время сидит в 2D feature context head и CAM dense TDNN blocks. Финальный
statistics/dense head почти не влияет на скорость.

TensorRT после lowering сгладил профиль:

| Класс TensorRT layer | Доля layer time |
| --- | ---: |
| reformat/shuffle | `30.08%` |
| fused conv | `29.99%` |
| fused batchnorm | `14.62%` |
| shape/reduce/plugin | `6.44%` |
| pool | `4.58%` |

Самый тяжёлый отдельный TensorRT layer был около `0.229ms` на вызов. AutoKernel
диагностика совпала с этим выводом: покрываемые kernels давали только `12.6%` GPU time,
теоретический потолок был около `1.1x`.

## Экспорт MS32 CAM++ в ONNX

Скрипт: [`scripts/export_campp_onnx.py`](../scripts/export_campp_onnx.py).

```bash
uv run python scripts/export_campp_onnx.py \
  --config configs/base.toml \
  --checkpoint artifacts/baselines/campp-ms31-official-pseudo-filtered-lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt \
  --output-root artifacts/model-bundle-campp-ms32-onnx \
  --model-version campp-ms32-filtered-pseudo-onnx \
  --sample-frame-count 600
```

Ожидаемые артефакты:

- `artifacts/model-bundle-campp-ms32-onnx/model.onnx`
- `artifacts/model-bundle-campp-ms32-onnx/metadata.json`
- `artifacts/model-bundle-campp-ms32-onnx/export_boundary.json`
- `artifacts/model-bundle-campp-ms32-onnx/export_report.md`

## Сборка TensorRT FP16-движка

Скрипт: [`scripts/build_tensorrt_fp16_engine.py`](../scripts/build_tensorrt_fp16_engine.py).
Конфиг: [`configs/release/tensorrt-fp16-ms32.toml`](../configs/release/tensorrt-fp16-ms32.toml).

```bash
uv run python scripts/build_tensorrt_fp16_engine.py \
  --config configs/release/tensorrt-fp16-ms32.toml
```

Ожидаемые артефакты:

- `artifacts/model-bundle-campp-ms32-onnx/model.plan`
- `artifacts/release/ms32-campp/fp16/tensorrt_fp16_engine_report.json`
- `artifacts/release/ms32-campp/fp16/tensorrt_fp16_engine_report.md`

## Полный запуск для замера скорости

Основная команда лежит в [release-runbook.md](./release-runbook.md), раздел
`Закрытый набор с полного пересчёта из аудио`. Важные свойства команды:

- используется полный private/test CSV, без `--limit-rows`;
- входом являются CSV и аудио, а не готовые embeddings;
- encoder backend: `--encoder-backend tensorrt`;
- TensorRT config: `configs/release/tensorrt-fp16-ms32.toml`;
- после инференса запускается `scripts/validate_submission.py`;
- итоговое время пишется в `speed_summary.txt`.

Ключевой фрагмент encoder-стадии:

```bash
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
```

После этого runbook считает classifier posterior, запускает graph tail, копирует
финальный файл в `submission.csv`, валидирует его и записывает stage timings.

## Кеш признаков для повторных запусков

Первый полный прогон из аудио ограничен Fbank/frontend. Для повторных прогонов на том
же CSV можно материализовать точный кеш признаков.

Скрипт:
[`scripts/materialize_official_campp_frontend_cache.py`](../scripts/materialize_official_campp_frontend_cache.py).

```bash
uv run python scripts/materialize_official_campp_frontend_cache.py \
  --manifest-csv "$PRIVATE_CSV" \
  --data-root "$DATA_ROOT" \
  --cache-dir artifacts/cache/campp-official-private \
  --output-root artifacts/profiles/campp_frontend_cache_materialization_private \
  --workers 16 \
  --prefetch 256 \
  --cache-mode readwrite \
  --mode segment_mean \
  --eval-chunk-seconds 6.0 \
  --segment-count 3 \
  --pad-mode repeat
```

Для быстрого чтения кеш можно упаковать:

```bash
uv run python scripts/pack_official_campp_frontend_cache.py \
  --manifest-csv "$PRIVATE_CSV" \
  --data-root "$DATA_ROOT" \
  --cache-dir artifacts/cache/campp-official-private \
  --output-dir artifacts/cache/campp-official-private-pack \
  --mode segment_mean \
  --eval-chunk-seconds 6.0 \
  --segment-count 3 \
  --pad-mode repeat
```

Packed cache полезен для повторных экспериментов с tail-постобработкой, но для критерия
скорости жюри надо явно договориться, считается ли подготовка кеша частью времени.
Без такой договорённости честный замер - полный прогон из аудио.

## Скрипты, которые можно прогнать

| Скрипт | Что проверяет |
| --- | --- |
| [`scripts/export_campp_onnx.py`](../scripts/export_campp_onnx.py) | Экспорт MS32 CAM++ encoder в ONNX. |
| [`scripts/build_tensorrt_fp16_engine.py`](../scripts/build_tensorrt_fp16_engine.py) | Сборка TensorRT FP16 engine и отчёт по parity/speed gates. |
| [`scripts/benchmark_campp_tensorrt.py`](../scripts/benchmark_campp_tensorrt.py) | Encoder-boundary benchmark. Не заменяет полный speed score. |
| [`scripts/profile_campp_pipeline_stages.py`](../scripts/profile_campp_pipeline_stages.py) | Диагностика стадий полного пайплайна. Для финальной оценки запускать без `--limit-rows`. |
| [`scripts/profile_campp_model_layers.py`](../scripts/profile_campp_model_layers.py) | Профиль PyTorch modules, CUDA kernels и TensorRT layers. |
| [`scripts/materialize_official_campp_frontend_cache.py`](../scripts/materialize_official_campp_frontend_cache.py) | Материализация точного frontend cache. |
| [`scripts/pack_official_campp_frontend_cache.py`](../scripts/pack_official_campp_frontend_cache.py) | Упаковка frontend cache в быстрый формат чтения. |
| [`scripts/export_teacher_peft_onnx.py`](../scripts/export_teacher_peft_onnx.py) | Экспорт W2V1j teacher-PEFT encoder в ONNX. |
| [`scripts/export_official_3dspeaker_eres2net_onnx.py`](../scripts/export_official_3dspeaker_eres2net_onnx.py) | Экспорт H9 official ERes2Net encoder в ONNX. |
| [`scripts/build_generic_tensorrt_engine.py`](../scripts/build_generic_tensorrt_engine.py) | TensorRT FP16 build/parity/benchmark для generic ONNX-графов. |
| [`scripts/render_speed_comparison_chart.py`](../scripts/render_speed_comparison_chart.py) | SVG-график end-to-end скорости по семействам. |

## Формулировка для защиты

TensorRT ускорил CAM++ encoder, но end-to-end профиль показал, что после этого главный
предел скорости - официальный Kaldi-style Fbank frontend. Поэтому финальный быстрый путь
использует TensorRT для MS32 encoder, а дальнейшая оптимизация должна идти через полный
замер на всём датасете, кеширование признаков, packing и уменьшение padding.
