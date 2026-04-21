# Лог экспериментов

## Как читать записи

- `validation` (валидация, отложенная проверка) — используется для выбора режима и настроек.
- `internal test` (внутренний тест) — честная локальная проверка после выбора режима.
- `leaderboard` (таблица результатов) — публичный результат после отправки сабмита.
- `pretrained` (предобученные веса) — модель без дообучения на нашем датасете.

## Сводка Public Leaderboard

| Дата | Модель | Эксперимент / чекпоинт | `validation Precision@10` | Public `leaderboard` | Комментарий |
| --- | --- | --- | ---: | ---: | --- |
| 2026-04-16 | `CAM++ + weak class-aware rerank` | `MS42_ms34_classaware_c4_weak_local` | — | `0.7471` | лучший подтверждённый `CAM++` результат в текущей ветке |
| 2026-04-16 | `CAM++ + stronger class-aware rerank` | `MS42plus_ms34_classaware_c4_stronger_local` | — | `0.7470` | усиление weak-bonus не улучшило `MS42` |
| 2026-04-16 | `CAM++ + midpoint class-aware rerank` | `MS42mid_ms34_classaware_c4_mid_local` | — | `0.7468` | ослабление weak-bonus тоже оказалось хуже `MS42` |
| 2026-04-16 | `CAM++ + MS42-teacher pseudo refresh + class-aware` | `MS43_campp_ms42teacher_pseudorefresh_classaware_local5080` | — | `0.7438` | новый student проиграл `MS42`, несмотря на сильный local proxy |
| 2026-04-15 | `CAM++ pseudo refresh from MS32` | `MS34_campp_ms32_safe_refresh_local5080` | — | `0.7412` | лучший training-refresh от `MS32` |
| 2026-04-15 | `CAM++ aggressive pseudo refresh from MS32` | `MS35_campp_ms32_aggressive_refresh_local5080` | — | `0.7318` | слишком жёсткий pseudo-filtering ухудшил результат |
| 2026-04-14 | `CAM++ official + pseudo-label self-training` | `MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z` | — | `0.7379` | сильный teacher baseline для поздней `CAM++`-ветки |
| 2026-04-13 | English `ERes2Net pretrained` | `submission_pretrained_eres2net_en_vox_test_public` | `0.9610` | `0.6057` | лучший публичный результат среди `ERes`-ветки |
| 2026-04-12 | `CAM++ English VoxCeleb pretrained` | `submission_pretrained_segment_mean_test_public` | `0.9603` | `0.5695` | сильный базовый `pretrained` |
| 2026-04-14 | `NVIDIA TitaNet-Large pretrained` | `submission_pretrained_titanet_large_en_test_public` | `0.9577` | `0.4646` | рабочий английский baseline через `NeMo`, но слабее `CAM++` и `ERes2Net en` |
| 2026-04-13 | `CAM++ finetuned` | `submission_run006_best_p10_test_public` (`epoch 7`) | `0.9734` | `0.5626` | локально лучше, публично хуже baseline |
| 2026-04-13 | `CAM++ finetuned` | `submission_epoch001_probe_test_public` (`epoch 1`) | `0.9639` | `0.5618` | ранний чекпоинт тоже хуже baseline |
| 2026-04-13 | `CAM++ sample v1` | `epoch 10` | — | `0.5387` | public-значение из сообщения пользователя |
| 2026-04-13 | `CAM++ sample v1` | `epoch 5` | — | `0.5337` | public-значение из сообщения пользователя |
| 2026-04-14 | Chinese `ERes2Net-large 3dspeaker pretrained` | `submission_pretrained_eres2net_large_3dspeaker_test_public` | `0.8778` | `0.3403` | китайский `pretrained` переносится слабо |
| 2026-04-13 | `ERes2NetV2 zh-cn common pretrained` | `submission_pretrained_eres2netv2_official_chunk_mean_test_public` | `0.9392` | `0.2911` | даже после фикса official-style inference публично очень слабо |
| 2026-04-14 | `WavLM Base Plus SV pretrained` | `submission_pretrained_wavlm_base_plus_sv_test_public` | `0.5664` | `0.1297` | формат сабмита перепроверен; вероятна ошибка в текущем inference-протоколе |

## Эксперименты

### 2026-04-12 — CAM++ English VoxCeleb pretrained, public submission

- Эксперимент: `submission_pretrained_segment_mean_test_public`
- Коммит: `aad65e8`
- Модель: `CAM++ English VoxCeleb pretrained`
- Режим инференса (получения эмбеддингов): `segment_mean` (несколько сегментов и усреднение эмбеддингов)
- Локальный split (разбиение): `80/10/10` по спикерам
- `validation Precision@1`: `0.9820`
- `validation Precision@5`: `0.9710`
- `validation Precision@10`: `0.9603`
- `internal test Precision@1`: `0.9803`
- `internal test Precision@5`: `0.9693`
- `internal test Precision@10`: `0.9585`
- Public `leaderboard` (таблица результатов): `0.5695`
- Путь до сабмита: `data/campp_runs/campp_en_ft/submissions/submission_pretrained_segment_mean_test_public/submission.csv`
- Время сборки сабмита: `640.64` сек, примерно `10 минут 41 сек`
- Источник весов: `iic/speech_campplus_sv_en_voxceleb_16k`
- Короткий вывод: базовый английский `CAM++` уже даёт рабочую отправную точку, но разрыв между локальным `internal test` и public `leaderboard` показывает, что без дообучения под домен соревнования качество ещё заметно ограничено.

### 2026-04-13 — CAM++ finetune checkpoint epoch 7, public submission

- Эксперимент: `submission_run006_best_p10_test_public`
- Коммит: `64b6ef1`
- Модель: `CAM++ English VoxCeleb finetuned`
- Источник чекпоинта: `campp_en_ft_noaug_run006_b200_e10_live_mlflow/checkpoints/best_p10.pt`
- Эпоха чекпоинта: `7`
- Режим инференса (получения эмбеддингов): `segment_mean` (несколько сегментов и усреднение эмбеддингов)
- Локальная `validation Precision@1`: `0.9898`
- Локальная `validation Precision@5`: `0.9818`
- Локальная `validation Precision@10`: `0.9734`
- Локальная `validation hit_rate@10` (доля запросов, где в топ-10 нашёлся хотя бы один правильный сосед): `0.9955`
- Локальная `validation nDCG@10` (качество порядка внутри топ-10 с большим весом верхних позиций): `0.9772`
- Public `leaderboard` (таблица результатов): `0.5626`
- Путь до сабмита: `data/campp_runs/campp_en_ft/submissions/submission_run006_best_p10_test_public/submission.csv`
- Время сборки сабмита: `692.47` сек, примерно `11 минут 32 сек`
- Короткий вывод: локальные retrieval-метрики (метрики поиска соседей) заметно выросли, но public score оказался ниже pretrained-бейзлайна, значит текущее дообучение без аугментаций не переносится на public test.

### 2026-04-13 — CAM++ finetune checkpoint epoch 1, public submission

- Эксперимент: `submission_epoch001_probe_test_public`
- Коммит: `64b6ef1`
- Модель: `CAM++ English VoxCeleb finetuned`
- Источник чекпоинта: `campp_en_ft_epoch001_probe_b200_nw10/checkpoints/epoch_001.pt`
- Эпоха чекпоинта: `1`
- Режим инференса (получения эмбеддингов): `segment_mean` (несколько сегментов и усреднение эмбеддингов)
- Локальная `validation Precision@1`: `0.9843`
- Локальная `validation Precision@5`: `0.9738`
- Локальная `validation Precision@10`: `0.9639`
- Локальная `validation hit_rate@10` (доля запросов, где в топ-10 нашёлся хотя бы один правильный сосед): `0.9935`
- Локальная `validation nDCG@10` (качество порядка внутри топ-10 с большим весом верхних позиций): `0.9686`
- Public `leaderboard` (таблица результатов): `0.5618`
- Путь до сабмита: `data/campp_runs/campp_en_ft/submissions/submission_epoch001_probe_test_public/submission.csv`
- Время сборки сабмита: `678.33` сек, примерно `11 минут 18 сек`
- Короткий вывод: даже очень ранний checkpoint (сохранённое состояние обучения) оказался хуже pretrained-бейзлайна, то есть проблема не только в «слишком долгом» дообучении, а в самой текущей схеме адаптации.

### 2026-04-13 — CAM++ soft cycle v4, no augmentations

- Эксперимент: `campp_en_soft_cycle_v4_noaug_run001`
- Коммит: `99ea3bfbd9bb`
- Модель: `CAM++ English VoxCeleb finetuned`
- Режим обучения: `soft cycle v4`, без `augmentations`
- Источник метрик: `data/campp_runs/campp_en_ft/runs/campp_en_soft_cycle_v4_noaug_run001/run_summary.json`
- Лучший чекпоинт: `checkpoints/best_p10.pt`
- Лучшая эпоха по `Precision@10`: `1`
- Локальная `validation Precision@1`: `0.9820`
- Локальная `validation Precision@5`: `0.9710`
- Локальная `validation Precision@10`: `0.9603`
- Локальная `validation hit_rate@10`: `0.9925`
- Локальная `validation nDCG@10`: `0.9653`
- Локальная `validation MRR@10`: `0.9857`
- Время обучения: `3918.26` сек, примерно `1 час 5 минут`
- Короткий вывод: переход на `soft cycle v4` без аугментаций не дал прироста относительно базового pretrained-режима и остаётся заметно слабее лучшего noaug finetune `run006` по локальным retrieval-метрикам.

### 2026-04-13 — CAM++ soft cycle v4, with augmentations

- Эксперимент: `campp_en_soft_cycle_v4_aug_run001`
- Коммит: `99ea3bfbd9bb`
- Модель: `CAM++ English VoxCeleb finetuned`
- Режим обучения: `soft cycle v4`, с `augmentations`
- Источник метрик: `data/campp_runs/campp_en_ft/runs/campp_en_soft_cycle_v4_aug_run001/run_summary.json`
- Лучший чекпоинт: `checkpoints/best_p10.pt`
- Лучшая эпоха по `Precision@10`: `5`
- Локальная `validation Precision@1`: `0.9821`
- Локальная `validation Precision@5`: `0.9709`
- Локальная `validation Precision@10`: `0.9604`
- Локальная `validation hit_rate@10`: `0.9928`
- Локальная `validation nDCG@10`: `0.9654`
- Локальная `validation MRR@10`: `0.9858`
- Время обучения: `11154.17` сек, примерно `3 часа 6 минут`
- Короткий вывод: добавление аугментаций в `soft cycle v4` почти не изменило локальные метрики относительно noaug-варианта, но лучший чекпоинт сместился на более позднюю эпоху.

## Наблюдения и гипотезы после первых finetune-запусков

- `pretrained` (предобученные веса без дообучения) пока остаётся лучшей отправной точкой: `0.5695` против `0.5626` и `0.5618` у двух дообученных вариантов.
- Рост локальных retrieval-метрик (метрик поиска соседей) не переносится на public `leaderboard` (публичную таблицу результатов), значит текущая `validation` (валидация, отложенная проверка) не отражает сложность public test.
- Деградация видна уже на `epoch 1`, значит проблема не только в переобучении (слишком долгом обучении), а в самой схеме адаптации.
- Дообучение без `augmentations` (аугментаций, искусственных искажений) плохо согласуется с EDA (разведочным анализом данных): в public test больше тишины, каналовых искажений и более тяжёлый акустический сдвиг.
- Для `campp_en_soft_cycle_v4_noaug_run001` значения `validation Precision@1/5/10` (`0.9820` / `0.9710` / `0.9603`) совпадают с самой первой baseline-записью `submission_pretrained_segment_mean_test_public` один в один по этим метрикам.
- Для `campp_en_soft_cycle_v4_aug_run001` в проверенных файлах зафиксированы локальные `validation`-метрики и артефакт public submission, но значение public `leaderboard` `0.5501` в `experiment_log.md` и связанных логах не записано, поэтому здесь оно отдельно не утверждается без явной записи результата.
- По доступным записям гипотеза об улучшении `soft cycle v4` не подтверждается: noaug не дал отличий от первого baseline по ключевым `validation`-метрикам, а aug по локальным метрикам тоже не показал заметного прироста.
- Текущий `finetune` (дообучение) слишком агрессивно меняет `embedding space` (пространство голосовых векторов), которое уже было хорошо настроено на `VoxCeleb`.
- Эта старая локальная `CAM++`-ветка больше не является основной: позже рабочая ветка была перенесена в соседний `Kryptonite-ML-Challenge-2026`, где использовались `official_campp`, filtered pseudo-label self-training и weak class-aware rerank. Ниже зафиксирована именно эта поздняя ветка как актуальная линия `CAM++`.
- Следующие приоритетные гипотезы:
  - уменьшить `backbone lr` (скорость обучения основной части модели) минимум в `5–10` раз;
  - заморозить `backbone` (основную часть модели) на первые `1–2` эпохи и учить только верх;
  - добавить аугментации под реальные искажения из EDA: шум, реверберация, сужение полосы частот, паузы;
  - проверить более жёсткий локальный split (разбиение), который ближе к public test по сложности.

### 2026-04-13 — CAM++ sample v1, public leaderboard update

- Эксперимент: `campp_en_ft_soft_noaug_sample_v1_10ep_run002`
- Источник значений public `leaderboard`: сообщение пользователя
- Public `leaderboard` для `epoch 5`: `0.5337`
- Public `leaderboard` для `epoch 10`: `0.5387`
- Короткий вывод: у первого эксперимента `epoch 10` пока лучше `epoch 5` на `0.0050` по public score.

## 2026-04-14–2026-04-16 — Поздняя CAM++ ветка в `Kryptonite-ML-Challenge-2026`

Важно:

- Эти эксперименты выполнялись не в старом локальном пайплайне `kriptio_tembr/code/campp`, а
  в соседнем репозитории `Kryptonite-ML-Challenge-2026`.
- Я переношу их сюда специально, потому что именно они стали актуальной `CAM++`-веткой и
  дали все сильные public-результаты `0.74+`.
- Главное отличие этой ветки от старых локальных finetune-запусков:
  `official_campp` frontend, fixed `6s` crops, `vad=none`, transductive public tails,
  filtered pseudo-label self-training и потом weak class-aware rerank.

### 2026-04-14 — `MS32` как новый базовый `CAM++` teacher

- Эксперимент: `MS32_campp_ms31_filtered_pseudo_lowlr_public_c4_20260414T0551Z`
- Репозиторий: `Kryptonite-ML-Challenge-2026`
- Модель: `CAM++ official frontend + pseudo-label self-training`
- Гипотеза:
  если перейти на exact `official_campp` и аккуратно доучить модель с filtered pseudo labels,
  можно резко улучшить перенос на public test относительно старого локального finetune.
- Что было предпринято:
  взяли сильный `MS31` teacher, построили public pseudo labels, оставили conservative
  low-LR self-training, fixed `6s`, `vad=none`.
- Public `leaderboard`: `0.7379`
- Короткий вывод:
  это был первый по-настоящему сильный `CAM++` teacher в новой ветке и именно от него
  потом пошли все локальные продолжения `MS34`, `MS35` и class-aware линия `MS42*`.

### 2026-04-15 — `MS34` safe refresh от `MS32`

- Эксперимент: `MS34_campp_ms32_safe_refresh_local5080`
- Гипотеза:
  `MS32` уже сильный, значит следующий безопасный прирост должен прийти не от новой
  архитектуры, а от более чистого pseudo-pool поверх того же teacher.
- Что было предпринято:
  оставили exact `MS32` checkpoint и ту же cluster-first graph family, но ужали selector:
  cluster sizes `[10, 60]`, `min_top1_score=0.57`, `min_top1_margin=0.02`,
  `max_indegree_quantile=0.90`, `max_rows_per_cluster=48`.
- Pseudo pool:
  `41506` pseudo rows, `2953` pseudo clusters.
- Local proxy:
  `c4_top1_score_mean=0.7295`, `c4_top10_mean_score_mean=0.6666`,
  `label_used_share=0.9043`, `c4_indegree_max_10=51`.
- Public `leaderboard`: `0.7412`
- Решение:
  эксперимент принят.
  `MS34` стал новым лучшим trainable `CAM++` checkpoint и базой для class-aware ветки.

### 2026-04-15 — `MS35` aggressive pseudo refresh от `MS32`

- Эксперимент: `MS35_campp_ms32_aggressive_refresh_local5080`
- Гипотеза:
  если `MS34` вырос из-за cleaner pseudo labels, возможно, ещё более строгий graph и
  selector дадут ещё больший прирост.
- Что было предпринято:
  дополнительно ужали graph (`cluster_max_size=120`, `shared_min_count=5`,
  `cluster_min_candidates=4`, `split_edge_top=10`, `label_size_penalty=0.25`) и сам
  selector (`[12, 48]`, `min_top1_score=0.60`, `min_top1_margin=0.03`,
  `max_indegree_quantile=0.85`, `max_rows_per_cluster=32`).
- Pseudo pool:
  `17247` pseudo rows, `2183` pseudo clusters.
- Local proxy:
  `c4_top1_score_mean=0.7158`, `c4_top10_mean_score_mean=0.6523`,
  `label_used_share=0.8931`, `c4_indegree_max_10=56`.
- Public `leaderboard`: `0.7318`
- Решение:
  эксперимент отклонён.
  Слишком жёсткая фильтрация уже отрезала полезный сигнал вместе с шумом.

### 2026-04-15 — `MS41` local parity как подготовка к class-aware ветке

- Эксперимент: `MS41_local_parity_ms32`
- Гипотеза:
  прежде чем переносить class-aware rerank на `MS34`, нужно локально проверить, что
  `MS41`-style weak class-aware policy воспроизводится на уже имеющихся `MS32` артефактах.
- Что было предпринято:
  построили classifier top-5 cache от `MS32`, переиспользовали `MS32` public `top200`
  cache и прогнали weak class-aware rerank с теми же весами, которые позже стали `MS42`.
- Решение:
  kept as calibration.
  Этот шаг был не ради public submit, а ради уверенности, что class-aware код и local
  recipe ведут себя согласованно перед переносом на более сильный `MS34`.

### 2026-04-15 — `MS42` weak class-aware rerank на базе `MS34`

- Эксперимент: `MS42_ms34_classaware_c4_weak_local`
- Гипотеза:
  раз `MS34` уже улучшил trainable checkpoint, самый дешёвый следующий gain должен прийти
  из слабого classifier-aware сигнала поверх того же public `top200`, без нового train.
- Что было предпринято:
  оставили graph как в conservative C4-like tail
  (`edge_top=10`, `reciprocal_top=20`, `rank_top=100`, `iterations=5`,
  `label_max_size=120`, `density_penalty=0.02`)
  и добавили weak class-aware bonus:
  `class_overlap_weight=0.03`, `same_top1_bonus=0.01`,
  `same_query_topk_bonus=0.005`.
- Local proxy:
  `top1_score_mean=0.7609`, `top10_mean_score_mean=0.6963`,
  `label_used_share=0.9116`, `indegree_max_10=55`,
  `adjusted_top1_changed_share=0.0784`.
- Public `leaderboard`: `0.7471`
- Решение:
  эксперимент принят.
  Это текущий лучший подтверждённый `CAM++`-результат в ветке.

### 2026-04-19 — перенос `MS42` runtime и public materialization в эту репу

- Эксперимент: `CAMPP_ROCM_PUBLIC_20260419`
- Коммит: pending до фиксации текущих правок
- Модель: `CAM++ official frontend + MS42 weak class-aware rerank`
- Источник весов: `data/pretrained_models/speaker_verification/campplus/ms31_official_pseudo_filtered_lowlr/20260414T055357Z-f1f2fa87143a/campp_encoder.pt`
- Режим инференса: `torch`, `segment_mean`, `6.0s`, `3` сегмента, `vad=none`, weak class-aware graph tail
- Локальная проверка формата: `134697` строк, `passed=True`, `errors=0`
- Время сборки на локальном `ROCm` smoke/full run: `1423.608` сек
- SHA256 `submission.csv`: `cb36f66bbffdea19a53cfcbf3fd00d5a42481356595fa476212c3dc056df455a`
- Public `leaderboard`: `0.7471` для принятого `MS42_ms34_classaware_c4_weak_local`; текущая запись фиксирует перенос runtime и materialized public CSV, а не новую отправку на платформу
- Путь до сабмита: `data/campp_runs/ms42_release/submissions/imported_rocm_public_20260419/submission.csv`
- Короткий вывод: финальный launcher в этой репе должен использовать `CAM++` как primary path; `WavLM` остаётся только fallback-веткой.

### 2026-04-15 — `MS42b` lighter class-aware variant

- Эксперимент: `MS42b_ms34_classaware_c4_lighter_local`
- Гипотеза:
  более мягкие weak-bonus веса могут сохранить большую часть gain от `MS42`, но снизить
  hubness.
- Что было предпринято:
  ослабили веса до
  `class_overlap_weight=0.02`, `same_top1_bonus=0.008`,
  `same_query_topk_bonus=0.004`.
- Local proxy:
  `top1_score_mean=0.7519`, `top10_mean_score_mean=0.6878`,
  `label_used_share=0.9078`, `indegree_max_10=53`.
- Решение:
  оставлен только как local hedge.
  Ветка стала чище, но уступила `MS42` по качеству retrieval.

### 2026-04-15 — `MS42c` ultralight class-aware variant

- Эксперимент: `MS42c_ms34_classaware_c4_ultralight_local`
- Гипотеза:
  возможно, почти весь gain даёт совсем маленький class-aware сигнал, а более слабый
  rerank ещё сильнее улучшит стабильность.
- Что было предпринято:
  ослабили веса до
  `class_overlap_weight=0.015`, `same_top1_bonus=0.005`,
  `same_query_topk_bonus=0.0025`.
- Local proxy:
  `top1_score_mean=0.7451`, `top10_mean_score_mean=0.6813`,
  `label_used_share=0.9058`, `indegree_max_10=51`.
- Решение:
  эксперимент отклонён.
  Он подтвердил, что слишком слабый class-aware сигнал возвращает значимую часть gains
  обратно.

### 2026-04-15 — `MS42-mid` midpoint между `MS42` и `MS42b`

- Эксперимент: `MS42mid_ms34_classaware_c4_mid_local`
- Гипотеза:
  optimum может лежать между `MS42` и `MS42b`, а midpoint даст почти ту же силу при
  немного лучшей hubness.
- Что было предпринято:
  midpoint-веса:
  `class_overlap_weight=0.025`, `same_top1_bonus=0.009`,
  `same_query_topk_bonus=0.0045`.
- Local proxy:
  `top1_score_mean=0.7564`, `top10_mean_score_mean=0.6920`,
  `label_used_share=0.9086`, `indegree_max_10=54`.
- Public `leaderboard`: `0.7468`
- Решение:
  эксперимент отклонён.
  Ослабление weak-bonus в сторону lighter-вариантов оказалось неверным направлением.

### 2026-04-16 — `MS42+` stronger class-aware variant

- Эксперимент: `MS42plus_ms34_classaware_c4_stronger_local`
- Гипотеза:
  если `MS42` уже хороший, возможно, ещё более сильный weak-bonus даст ещё небольшой
  прирост.
- Что было предпринято:
  усилили веса до
  `class_overlap_weight=0.035`, `same_top1_bonus=0.012`,
  `same_query_topk_bonus=0.006`.
- Local proxy:
  `top1_score_mean=0.7667`, `top10_mean_score_mean=0.7018`,
  `label_used_share=0.9142`, `indegree_max_10=60`.
- Public `leaderboard`: `0.7470`
- Решение:
  эксперимент отклонён.
  Несмотря на лучшие local proxy, public score оказался чуть хуже `MS42`, а hubness
  выросла сильнее.

### 2026-04-16 — `MS42g` graph-lite-tight

- Эксперимент: `MS42g_ms34_classaware_c4_graph_lite_tight_local`
- Гипотеза:
  после того как sweep по class-bonus упёрся в потолок, следующая дешёвая ось — не
  менять class-aware веса, а чуть поджать graph tail.
- Что было предпринято:
  оставили exact веса `MS42`, но изменили graph:
  `label_max_size=100` и `density_penalty=0.025`.
- Local proxy:
  `top1_score_mean=0.7603`, `top10_mean_score_mean=0.6958`,
  `label_used_share=0.9091`, `indegree_max_10=48`.
- Решение:
  оставлен как более консервативный hedge.
  Hubness стала лучше, но это не выглядит как явная замена `MS42`.

### 2026-04-16 — `MS43` teacher pseudo refresh от `MS42`

- Эксперимент:
  `MS43_campp_ms42teacher_pseudorefresh_plain_local5080` +
  `MS43_campp_ms42teacher_pseudorefresh_classaware_local5080`
- Гипотеза:
  если `MS42` уже лучший full policy, тогда следующий сильный шаг — не ещё один rerank
  sweep, а новый student: построить pseudo labels от `MS42` teacher signal, стартовать
  от `MS34` checkpoint и потом снова прогнать plain/class-aware tails.
- Что было предпринято:
  1. `MS43a` — экспортировали adjusted teacher `top200` cache из `MS42`;
  2. `MS43b` — пересобрали cluster-first pseudo graph;
  3. собрали pseudo pool с proven `MS34`-style selector:
     `46816` pseudo rows, `3129` pseudo clusters;
  4. обучили новый student `4` эпохи от `MS34` в `official_campp`, fixed `6s`,
     `vad=none`, с `VoxBlink2-surrogate-lite` аугментациями;
  5. посчитали plain C4 и class-aware rerank от нового чекпоинта.
- Train outcome:
  финальная эпоха `mean_loss=1.463829`, `accuracy=0.970679`.
- Plain local proxy:
  `c4_top1_score_mean=0.7376`, `c4_top10_mean_score_mean=0.6757`,
  `label_used_share=0.9114`, `c4_indegree_max_10=58`.
- Class-aware local proxy:
  `top1_score_mean=0.7716`, `top10_mean_score_mean=0.7082`,
  `label_used_share=0.9166`, `indegree_max_10=66`.
- Public `leaderboard`:
  `0.7438` для class-aware варианта (`2026-04-16 09:55`).
- Решение:
  эксперимент отклонён.
  Несмотря на сильный local proxy, новый student проиграл `MS42`, значит проблема была
  не только в rerank, а в самом новом чекпоинте. После этого логично было вернуться к
  `MS42` как к лучшему рабочему кандидату.

### 2026-04-13 — ERes2NetV2 vanilla pretrained, internal evaluation

- Эксперимент: `pretrained_val_modes` / `pretrained_test_bestmode`
- Коммит: `1ba4ad687f3e`
- Модель: `ERes2NetV2 common pretrained`
- Источник весов: `iic/speech_eres2netv2_sv_zh-cn_16k-common`
- `validation Precision@1`: `0.9837`
- `validation Precision@5`: `0.9629`
- `validation Precision@10`: `0.9379`
- Лучший режим на `validation`: `segment_mean`
- `internal test Precision@1`: `0.9822`
- `internal test Precision@5`: `0.9592`
- `internal test Precision@10`: `0.9320`
- Путь до сабмита: `data/eres2netv2_runs/eres2netv2_common/submissions/submission_pretrained_eres2netv2_test_public/submission.csv`
- Время сборки сабмита: `1991.91` сек, примерно `33 минуты 12 сек`
- Короткий вывод: локально vanilla `ERes2NetV2` выглядит сильным, поэтому очень низкий public score нельзя объяснить тем, что модель «совсем не работает».

### 2026-04-13 — ERes2NetV2 public anomaly, root-cause hypothesis

- Источник public значения: сообщение пользователя
- Зафиксированная аномалия: public `leaderboard` около `0.29` для vanilla `ERes2NetV2`
- Основная гипотеза: в baseline использовался `CAM++`-подобный inference protocol (протокол инференса, способ резки аудио и усреднения эмбеддингов), а не official `3D-Speaker` режим для `ERes2NetV2`
- Что именно расходилось:
  - локальная проверка шла через `segment_mean` с порогом длинных файлов и усреднением по нескольким сегментам;
  - official `3D-Speaker` для `ERes2NetV2` режет файл на последовательные чанки по `10` секунд и усредняет эмбеддинги по всем чанкам;
  - batch size (размер батча) в evaluation (валидации, отложенной проверке) был завязан на training batch size (размер батча в обучении), что тоже делало inference нестабильным для воспроизведения.
- Локальная проверка гипотезы:
  - после добавления `official_chunk_mean` эмбеддинги на двух реальных файлах совпали с manual official-style извлечением почти идеально;
  - косинусное сходство (сходство направлений векторов) было `~1.0` для обоих файлов;
  - значит основная проблема была именно в предыдущем протоколе инференса, а не в загрузке весов или сборке модели.
- Статус на момент гипотезы: требовался повторный запуск vanilla `ERes2NetV2` уже в `official_chunk_mean` режиме и повторная отправка сабмита.

### 2026-04-13 — ERes2NetV2 vanilla pretrained, official_chunk_mean public submission

- Эксперимент: `pretrained_validation_official_chunk_mean_v2fix` + `submission_pretrained_eres2netv2_official_chunk_mean_test_public`
- Коммит: `affd8edd51c4`
- Модель: `ERes2NetV2 common pretrained`
- Источник весов: `iic/speech_eres2netv2_sv_zh-cn_16k-common`
- Режим инференса (получения эмбеддингов): `official_chunk_mean` (последовательные чанки по `10` секунд, ограничение длины до `90` секунд, усреднение эмбеддингов по всем чанкам)
- `validation Precision@1`: `0.9842`
- `validation Precision@5`: `0.9636`
- `validation Precision@10`: `0.9392`
- `validation hit_rate@10` (доля запросов, где в топ-10 нашёлся хотя бы один правильный сосед): `0.9937`
- `validation nDCG@10` (качество порядка внутри топ-10 с большим весом верхних позиций): `0.9500`
- `validation MRR@10` (средний обратный ранг первого правильного соседа): `0.9874`
- Public `leaderboard` (таблица результатов): `0.2911`
- Путь до сабмита: `data/eres2netv2_runs/eres2netv2_common/submissions/submission_pretrained_eres2netv2_official_chunk_mean_test_public/submission.csv`
- Время сборки сабмита: `2288.60` сек, примерно `38 минут 09 сек`
- Короткий вывод: даже после перехода на более правильный official-style inference (официальный режим инференса) public score остался очень низким, поэтому текущая проблема уже не похожа на баг в формате `submission.csv`.

### 2026-04-13 — English ERes2Net vanilla pretrained, official_chunk_mean public submission

- Эксперимент: `pretrained_validation_eres2net_en_vox` + `submission_pretrained_eres2net_en_vox_test_public`
- Коммит: `affd8edd51c4`
- Модель: `English ERes2Net pretrained`
- Источник весов: `iic/speech_eres2net_sv_en_voxceleb_16k`
- Режим инференса (получения эмбеддингов): `official_chunk_mean` (последовательные чанки по `10` секунд, ограничение длины до `90` секунд, усреднение эмбеддингов по всем чанкам)
- `validation Precision@1`: `0.9826`
- `validation Precision@5`: `0.9713`
- `validation Precision@10`: `0.9610`
- `validation hit_rate@10` (доля запросов, где в топ-10 нашёлся хотя бы один правильный сосед): `0.9927`
- `validation nDCG@10` (качество порядка внутри топ-10 с большим весом верхних позиций): `0.9659`
- `validation MRR@10` (средний обратный ранг первого правильного соседа): `0.9861`
- Public `leaderboard` (таблица результатов): `0.6057`
- Источник public значения: сообщение пользователя
- Путь до сабмита: `data/eres2net_runs/eres2net_en_vox/submissions/submission_pretrained_eres2net_en_vox_test_public/submission.csv`
- Время сборки сабмита: `1368.91` сек, примерно `22 минуты 49 сек`
- Короткий вывод: английский `ERes2Net` оказался резко сильнее `ERes2NetV2 zh-cn common` на public test, несмотря на меньший размер модели, что подтверждает важность совпадения домена предобучения.

### 2026-04-14 — Chinese ERes2Net-large 3dspeaker pretrained, official_chunk_mean public submission

- Эксперимент: `pretrained_validation_eres2net_large_3dspeaker` + `submission_pretrained_eres2net_large_3dspeaker_test_public`
- Коммит: `a058642dc211`
- Модель: `Chinese ERes2Net-large 3dspeaker pretrained`
- Источник весов: `iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k`
- Режим инференса (получения эмбеддингов): `official_chunk_mean` (последовательные чанки по `10` секунд, ограничение длины до `90` секунд, усреднение эмбеддингов по всем чанкам)
- `validation Precision@1`: `0.9494`
- `validation Precision@5`: `0.9119`
- `validation Precision@10`: `0.8778`
- `validation hit_rate@10` (доля запросов, где в топ-10 нашёлся хотя бы один правильный сосед): `0.9783`
- `validation nDCG@10` (качество порядка внутри топ-10 с большим весом верхних позиций): `0.8940`
- `validation MRR@10` (средний обратный ранг первого правильного соседа): `0.9592`
- Public `leaderboard` (таблица результатов): `0.3403`
- Источник public значения: сообщение пользователя
- Путь до сабмита: `data/eres2net_runs/eres2net_large_3dspeaker/submissions/submission_pretrained_eres2net_large_3dspeaker_test_public/submission.csv`
- Время сборки сабмита: `2591.30` сек, примерно `43 минуты 11 сек`
- Короткий вывод: ещё один purely Chinese pretrained (чисто китайский предобученный вариант) дал слабый public результат. На текущем соревновании китайские `pretrained` пока системно уступают английскому `VoxCeleb`-варианту, даже если локальная `validation` выглядит приемлемо.

### 2026-04-14 — NVIDIA TitaNet-Large pretrained, official batch inference

- Эксперимент: `pretrained_validation_titanet_large_en` + `submission_pretrained_titanet_large_en_test_public`
- Коммит: `47e8ec0085d1`
- Модель: `NVIDIA TitaNet-Large pretrained`
- Источник весов: `nvidia/speakerverification_en_titanet_large`
- Режим инференса (получения эмбеддингов): `official_batch_inference` через `NeMo EncDecSpeakerLabelModel`
- `validation Precision@1`: `0.9824`
- `validation Precision@5`: `0.9698`
- `validation Precision@10`: `0.9577`
- `validation hit_rate@10` (доля запросов, где в топ-10 нашёлся хотя бы один правильный сосед): `0.9927`
- `validation nDCG@10` (качество порядка внутри топ-10 с большим весом верхних позиций): `0.9633`
- `validation MRR@10` (средний обратный ранг первого правильного соседа): `0.9860`
- `parity-check min_cos`: `0.9927`
- `parity-check max_abs_diff`: `0.0074`
- Подобранный `evaluation.batch_size`: `24`
- Пиковая GPU-память на probe: `10134.4 MB`
- Public `leaderboard` (таблица результатов): `0.4646`
- Источник public значения: сообщение пользователя
- Путь до сабмита: `data/titanet_runs/titanet_large_en/submissions/submission_pretrained_titanet_large_en_test_public/submission.csv`
- Время сборки сабмита: `314.52` сек, примерно `5 минут 15 сек`
- Короткий вывод: `TitaNet-Large` корректно посчитался через официальный `NeMo` batch path, без расхождения с reference-проверкой на малом наборе. По скорости и загрузке GPU этот baseline получился заметно эффективнее `WavLM`.

### 2026-04-14 — WavLM Base Plus SV pretrained, public anomaly

- Эксперимент: `pretrained_validation_wavlm_base_plus_sv` + `submission_pretrained_wavlm_base_plus_sv_test_public`
- Коммит: `a0586423670b`
- Модель: `WavLM Base Plus SV pretrained`
- Источник весов: `microsoft/wavlm-base-plus-sv`
- Режим инференса (получения эмбеддингов): `xvector_chunk_mean` (последовательные чанки по `10` секунд с усреднением `xvector`-эмбеддингов)
- `validation Precision@1`: `0.6554`
- `validation Precision@5`: `0.6019`
- `validation Precision@10`: `0.5664`
- `validation hit_rate@10` (доля запросов, где в топ-10 нашёлся хотя бы один правильный сосед): `0.7986`
- `validation nDCG@10` (качество порядка внутри топ-10 с большим весом верхних позиций): `0.5849`
- `validation MRR@10` (средний обратный ранг первого правильного соседа): `0.7006`
- Public `leaderboard` (таблица результатов): `0.1297`
- Источник public значения: сообщение пользователя
- Путь до сабмита: `data/wavlm_runs/wavlm_base_plus_sv/submissions/submission_pretrained_wavlm_base_plus_sv_test_public/submission.csv`
- Время сборки сабмита: `2262.06` сек, примерно `37 минут 42 сек`
- Перепроверка сабмита: число строк, порядок `filepath`, `top-10` и диапазон индексов совпадают с `test_public.csv`, явной ошибки в формате `submission.csv` не найдено.
- Текущая гипотеза: проблема не в writer-е сабмита, а в самом текущем `WavLM` inference protocol. Этот baseline нужно пересчитать ещё раз с более близким к official `speakerverification` режимом.
