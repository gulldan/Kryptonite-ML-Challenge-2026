# Скрипты

Папка содержит воспроизводимые CLI-команды. Бизнес-логика остаётся в
`src/kryptonite/`; скрипты только связывают конфиги, пути и конкретные запуски.

## Финальный инференс и файл отправки

| Скрипт | Для чего нужен |
| --- | --- |
| `run_ms41_submission.py` | Один reproducible entrypoint для текущего финального MS41 пайплайна по preset-конфигу. |
| `run_official_campp_tail.py` | Тонкий CLI над `src/kryptonite/eda/official_campp_tail/`: полный прогон CAM++ от аудио до nearest-neighbour tail. |
| `run_classifier_first_tail.py` | Посчитать top-k posterior классов из checkpoint и готовых векторов голосов. Для финального MS41 используется режим `--class-cache-only`. |
| `run_class_aware_graph_tail.py` | Финальная слабая class-aware поправка и графовая постобработка поверх готового кеша ближайших соседей. |
| `run_rowwise_tail_router.py` | Диагностический выбор правила постобработки для каждой строки. Проверен, но не выбран финальным. |
| `validate_submission.py` | Проверить формат `submission.csv`: строки, порядок, 10 соседей, повторы, совпадения файла с самим собой, диапазон индексов. |
| `materialize_reference_submission.py` | Зафиксировать точную копию reference submission для уже известного fixed public файла. Не заменяет пересчёт private из аудио. |

Основной сценарий запуска описан в [../README.md](../README.md).

## Обучение и дообучение

| Скрипт | Для чего нужен |
| --- | --- |
| `run_baseline.py` | Общий запуск baseline model family из TOML-конфига. |
| `run_campp_baseline.py` | Укороченная точка входа для CAM++ baseline. |
| `run_eres2netv2_baseline.py` | Укороченная точка входа для ERes2NetV2 baseline. |
| `run_campp_finetune.py` | Дообучение CAM++ из checkpoint, включая варианты с замороженными частями encoder. |
| `run_campp_soft_pseudo_finetune.py` | CAM++ fine-tuning на мягких псевдометках. |
| `run_eres2netv2_finetune.py` | Дообучение ERes2NetV2 из checkpoint. |
| `run_hf_xvector_finetune.py` | Дообучение Hugging Face AudioXVector-моделей. |
| `run_teacher_peft.py` / `run_teacher_peft_finetune.py` | PEFT/teacher эксперименты. |
| `run_w2vbert2_sv_moonshot.py` | Обёртка эксперимента W2VBERT2 speaker verification. |

## Псевдометки, внешние данные и графовые эксперименты

| Скрипт | Для чего нужен |
| --- | --- |
| `build_participant_training_manifests.py` | Собрать train/dev manifests из данных участников. |
| `build_pseudo_label_manifests.py` | Собрать manifests с псевдометками из public clusters. |
| `build_weighted_pseudo_label_manifests.py` | Собрать manifests с псевдометками и весами уверенности. |
| `build_strict_consensus_pseudo_manifests.py` | Собрать manifests с псевдометками, подтверждёнными несколькими учителями. |
| `build_multi_teacher_soft_pseudo_manifests.py` | Собрать мягкие псевдометки от нескольких teacher-веток. |
| `build_cnceleb_manifests.py` | Собрать CN-Celeb manifests и mixed train manifests. |
| `download_external_speaker_datasets.py` | Скачать внешние speaker datasets, если они доступны. |
| `run_cluster_first_tail.py` | Графовая постобработка с приоритетом кластеров. |
| `run_public_graph_community.py` | Графовая постобработка по готовым векторам голосов. |
| `run_backbone_fusion_c4_tail.py` | Объединение результатов нескольких моделей. |
| `run_torch_checkpoint_c4_tail.py` | Прогнать PyTorch checkpoint через текущую C4-постобработку. |

## Анализ данных и отчётность

| Скрипт | Для чего нужен |
| --- | --- |
| `run_eda_profile.py` | Собрать базовый EDA-профиль аудио и CSV. |
| `run_eda_baseline_onnx.py` | EDA baseline ONNX на train/local validation. |
| `run_eda_public_onnx.py` | Диагностика public-инференса для baseline ONNX. |
| `run_eda_retrieval_eval.py` | Precision@K по готовым векторам голосов. |
| `build_eda_review_package.py` | Собрать компактный EDA-пакет для отчёта. |
| `export_eda_csv_pack.py` | Экспортировать EDA-пакет только с CSV. |
| `build_validation_cycle_package.py` | Собрать пакет диагностики локальной проверки, похожей на public. |
| `compare_submission_overlap.py` | Сравнить два submission-файла по совпадению соседей. |
| `render_public_lb_chart.py` | Сгенерировать SVG-график public LB score для README из таблицы истории. |

## Экспорт, TensorRT и профилирование

Подробный runbook с командами и результатами H100-профиля:
[../docs/inference-acceleration.md](../docs/inference-acceleration.md).

| Скрипт | Для чего нужен |
| --- | --- |
| `export_campp_onnx.py` | Экспорт CAM++ encoder в ONNX. |
| `export_teacher_peft_onnx.py` | Экспорт W2V/w2v-BERT 2.0 `teacher_peft` encoder в ONNX. |
| `export_official_3dspeaker_eres2net_onnx.py` | Экспорт official 3D-Speaker ERes2Net encoder в ONNX. |
| `build_tensorrt_fp16_engine.py` | Сборка TensorRT FP16 engine. |
| `build_generic_tensorrt_engine.py` | Сборка и проверка TensorRT FP16 engine для ONNX-графов с одним или несколькими входами. |
| `benchmark_campp_tensorrt.py` | Benchmark TensorRT engine. |
| `materialize_official_campp_frontend_cache.py` | Посчитать постоянный кеш CAM++ признаков. |
| `pack_official_campp_frontend_cache.py` | Упаковать кеш признаков в быстрый формат. |
| `profile_campp_pipeline_stages.py` | Замерить время стадий public pipeline. |
| `profile_campp_model_layers.py` | Профилировать CAM++ model/layer breakdown. |
| `profile_tensorrt_engine_layers.py` | Снять per-layer TensorRT профиль через `trtexec --dumpProfile`. |
| `run_speed_family_submit_suite.py` | Последовательно прогнать полное формирование public submit для подготовленных моделей. |
| `collect_speed_family_results.py` | Собрать summary/timing/validator/overlap JSON в общий speed-results файл. |
| `render_speed_comparison_chart.py` | Сгенерировать SVG-график end-to-end скорости по семействам моделей. |

## Служебные команды

| Скрипт | Для чего нужен |
| --- | --- |
| `show_config.py` | Посмотреть итоговый конфиг. |
| `infer_smoke.py` | Быстрая проверка serving/runtime surface и выбранного backend. |
| `training_env_smoke.py` | Быстрая проверка training environment. |
| `repro_check.py` | Проверить воспроизводимость seed/snapshot. |
| `validate_manifests.py` | Проверить manifests по схеме. |
