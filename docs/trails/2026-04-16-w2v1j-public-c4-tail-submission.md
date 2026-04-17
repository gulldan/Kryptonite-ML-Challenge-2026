# 2026-04-16 — W2V1j: финальный stage3 checkpoint и сборка public submission

## Что это за модель

- Это moonshot-ветка на базе `facebook/w2v-bert-2.0` для speaker verification.
- В репозитории она оформлена как `teacher_peft`-пайплайн:
  - большой multilingual backbone `w2v-BERT 2.0`
  - поверх него speaker head
  - multi-frame aggregation (MFA)
  - layer adapters
  - ArcMargin-классификатор
- Цель ветки: отдельно проверить тяжёлый large-PTM рецепт против основных CAM++ /
  ERes2Net направлений.

## Какие стадии были в этой ветке

У этой модели было три последовательные стадии.

1. `stage1` — LoRA + adapters + MFA на замороженном backbone
   - конфиг: `configs/training/w2vbert2-mfa-lora-stage1.toml`
   - смысл:
     сначала не трогать весь backbone целиком, а адаптировать его более дешёвым способом
     через LoRA, layer adapters и MFA
   - сохранённый полезный checkpoint, который потом был переиспользован:
     `artifacts/baselines/w2vbert2-mfa-lora-stage1/20260415T103816Z-b2c32e479771/teacher_peft`

2. `stage2` — joint full fine-tuning после merge LoRA
   - конфиг: `configs/training/w2vbert2-mfa-joint-ft-stage2.toml`
   - смысл:
     взять stage1 checkpoint, влить LoRA в backbone, разморозить энкодер и
     аккуратно дотюнить всю модель
   - итоговый stage2 run:
     `artifacts/baselines/w2vbert2-mfa-joint-ft-stage2/20260415T175740Z-24273cd620a9`
   - ключевые локальные метрики stage2:
     - `score_gap = 0.694516`
     - `eer = 0.015162`
     - `min_dcf = 0.102414`

3. `stage3` — LMFT с более длинными кропами
   - конфиг: `configs/training/w2vbert2-mfa-lmft-stage3.toml`
   - смысл:
     финальная доадаптация уже после stage2, с фиксированными `6s` train crops и более
     жёстким ArcMargin
   - финальный stage3 run:
     `artifacts/baselines/w2vbert2-mfa-lmft-stage3/20260416T073610Z-787cea6b9872`
   - финальный checkpoint модели:
     `artifacts/baselines/w2vbert2-mfa-lmft-stage3/20260416T073610Z-787cea6b9872/teacher_peft`
   - ключевые локальные метрики stage3:
     - `score_gap = 0.691787`
     - `eer = 0.014806`
     - `min_dcf = 0.093501`

## Что было нестандартного по orchestration

- Stage1 не пересчитывали заново, когда стало понятно, что хороший checkpoint уже есть.
- Stage2 тоже не гоняли повторно после того, как он завершился и сохранил валидный
  checkpoint.
- Для stage3 пришлось чинить совместимость checkpoint metadata:
  первый resume-попытка `W2V1i...` упала, потому что stage2 checkpoint фактически уже был
  сохранён как full model, а metadata ещё говорила `peft_adapter`.
- После исправления metadata был запущен нормальный resume-run:
  `W2V1j_w2vbert2_mfa_lora_lmft_resume_s3_fixfmt_20260416T073554Z`

Связанные wrapper-артефакты:

- wrapper log:
  `artifacts/logs/W2V1j_w2vbert2_mfa_lora_lmft_resume_s3_fixfmt_20260416T073554Z.log`
- wrapper summary:
  `artifacts/reports/w2vbert2/W2V1j_w2vbert2_mfa_lora_lmft_resume_s3_fixfmt_20260416T073554Z_summary.json`

## Как из stage3 делали public submission

- Отдельный шаг после завершения обучения:
  финальный stage3 checkpoint использовали для извлечения public embeddings и затем
  прогоняли через graph-tail с exact top-k + C4 label propagation.
- Для этого был добавлен отдельный entrypoint:
  `scripts/run_teacher_peft_c4_tail.py`

Базовая схема этого шага:

1. загрузить `teacher_peft` checkpoint
2. прочитать public manifest
3. сделать deterministic eval crops
4. получить embeddings
5. посчитать `exact_topk`
6. прогнать `label_propagation_rerank`
7. записать `submission.csv`
8. прогнать локальный validator

## Почему public-tail переписывали

Изначально public-tail был слишком медленным:

- один главный процесс последовательно делал
  `load waveform -> crop -> HF feature extractor -> encoder`
- `GPU1` ждала CPU
- wall-clock был слишком плохой для полного public прогона

Поэтому extraction переписали на более эффективную схему:

- загрузка аудио и кропы ушли в `DataLoader` workers
- HF feature extractor переехал в worker-side collator
- в главный процесс прилетали уже готовые батчи
- перенос на CUDA делался через pinned memory
- агрегация кропов в строку делалась через `np.add.at(...)`

Новые ручки, добавленные в CLI:

- `--row-batch-size`
- `--num-workers`
- `--prefetch-factor`
- `--pin-memory/--no-pin-memory`

## Как тюнили быстрый public-tail на remote

Сначала были короткие smoke-прогоны на `GPU1`:

- `bs512` fast path:
  `embedding_s = 46.620837` на `1024` строках
- `bs768` fast path:
  `embedding_s = 34.262989`
- `bs1024` fast path:
  `embedding_s = 34.165135`

Но на полном датасете проявилась скрытая проблема:

- конфигурация `num_workers=12`, `prefetch_factor=4` оказалась плохой
- `DataLoader` prefetch измеряется батчами, а не строками
- стартовая очередь стала слишком жирной, из-за чего full-run стартовал слишком тяжело

После этого был сделан более честный medium-smoke на `2048` строках:

- `batch_size = 1024`
- `num_workers = 4`
- `prefetch_factor = 1`

Именно этот профиль оказался лучшим для полного public-run.

## Финальный public-tail run

- run id:
  `W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1`
- log:
  `artifacts/logs/W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1.log`
- output dir:
  `artifacts/backbone_public/teacher_peft/W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1`

Финальные тайминги:

- `embedding_s = 2448.668842`
- `search_s = 0.691275`
- `rerank_s = 11.364635`

То есть extraction занял около `40.8` минут, что уже было заметно лучше исходной
медленной версии.

## Итог public-tail run

Локальные диагностические метрики:

- `top10_mean_score_mean = 0.7455544471740723`
- `top1_score_mean = 0.8007782101631165`
- `label_used_share = 0.9076074448577177`
- `indegree_gini_10 = 0.3264392829340306`
- `indegree_max_10 = 49`

Результат локального validator:

- `validator_passed = true`
- `submission_row_count = 134697`
- `template_row_count = 134697`
- `error_count = 0`

Финальные файлы этого шага:

- summary:
  `artifacts/backbone_public/teacher_peft/W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1/W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1_summary.json`
- validation:
  `artifacts/backbone_public/teacher_peft/W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1/submission_W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1_validation.json`
- submission:
  `artifacts/backbone_public/teacher_peft/W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1/submission_W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1.csv`
- release copy:
  `artifacts/submissions/W2V1j_teacher_peft_public_c4_20260416T131620Z_fast_bs1024_w4p1_submission.csv`

SHA-256 release copy:

- `688e10555ef307807c28dd454439d1ca2f477b67bda7a5b0869144d8f3c5853a`

Результат ручной загрузки на Public LB:

- `public_lb = 0.8344`
- это внешний результат leaderboard, локально не воспроизводится
- относительно предыдущего лучшего зафиксированного submitted run `MS41` с `0.7473`
  прирост составил `+0.0871`
- относительно organizer baseline `0.0779` прирост составил `+0.7565`

## Что важно понимать по этой ветке

- Это не one-shot обучение, а цепочка из трёх стадий.
- Рабочая финальная модель для дальнейшей локальной работы — это именно stage3
  checkpoint:
  `artifacts/baselines/w2vbert2-mfa-lmft-stage3/20260416T073610Z-787cea6b9872/teacher_peft`
- Public submission — это уже отдельный downstream inference/tail шаг поверх stage3.
- То есть:
  - **модель** = финальный stage3 checkpoint
  - **submission** = результат применения public-tail к этой модели

## Решение

- Финальный stage3 checkpoint считается сохранённой рабочей версией модели.
- Финальный public-tail run подтвердился внешним Public LB score `0.8344`.
- На текущем public leaderboard это лучший зафиксированный результат в репозитории.
