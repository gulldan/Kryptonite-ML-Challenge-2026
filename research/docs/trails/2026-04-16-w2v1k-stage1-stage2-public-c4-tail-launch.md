# 2026-04-16 — W2V1k/W2V1l stage1/stage2 public C4-tail metric launch

## Цель

- После внешнего public-успеха финального `W2V1j` stage3 submission нужно отдельно
  прогнать тот же public-tail для checkpoint после `stage1` и после `stage2`.
- Это нужно, чтобы понять, где именно в цепочке `stage1 -> stage2 -> stage3` произошёл
  основной прирост public-качества:
  - уже на frozen-backbone LoRA/adapters `stage1`,
  - после joint full fine-tuning `stage2`,
  - или только после финального LMFT `stage3`.

## Чекпоинты

- `stage1` checkpoint:
  `artifacts/baselines/w2vbert2-mfa-lora-stage1/20260415T103816Z-b2c32e479771/teacher_peft`
- `stage2` checkpoint:
  `artifacts/baselines/w2vbert2-mfa-joint-ft-stage2/20260415T175740Z-24273cd620a9/teacher_peft`
- `stage3` reference for comparison:
  `artifacts/baselines/w2vbert2-mfa-lmft-stage3/20260416T073610Z-787cea6b9872/teacher_peft`

## Общий evaluation protocol

Обе абляции используют тот же inference/tail recipe, что и зафиксированный лучший
`W2V1j` public run:

- entrypoint:
  `scripts/run_teacher_peft_c4_tail.py`
- manifest:
  `artifacts/eda/participants_public_baseline/test_public_manifest.csv`
- template csv:
  `datasets/Для участников/test_public.csv`
- extraction profile:
  - `batch_size=1024`
  - `num_workers=4`
  - `prefetch_factor=1`
  - pinned memory
- retrieval / rerank:
  - `exact_topk`
  - `top_cache_k=300`
  - same C4 label propagation defaults as `W2V1j`

Это сохраняет сопоставимость: различается только checkpoint стадии, а не downstream
tail.

## Run ids и артефакты

### Stage1 public-tail

- run id:
  `W2V1k_stage1_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1`
- output dir:
  `artifacts/backbone_public/teacher_peft/W2V1k_stage1_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1`
- log:
  `artifacts/logs/W2V1k_stage1_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1.log`

### Stage2 public-tail

- run id:
  `W2V1l_stage2_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1`
- output dir:
  `artifacts/backbone_public/teacher_peft/W2V1l_stage2_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1`
- log:
  `artifacts/logs/W2V1l_stage2_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1.log`

## GPU scheduling

- Во время запроса `remote` уже показывал занятую `GPU0` с высокой утилизацией и около
  `78.5 GiB` памяти, поэтому в этот запуск не добавлялся второй тяжёлый job на `GPU0`.
- Чтобы не мешать существующей задаче, `stage1` и `stage2` public-tail абляции
  запускаются последовательно на `GPU1`.

## Первый launch и технический фикс

- Самый первый запуск `W2V1k` упал до extraction progress.
- Причина была не в checkpoint и не в модели: public manifest
  `artifacts/eda/participants_public_baseline/test_public_manifest.csv` содержал
  `resolved_path` вида `<repo-root>/...`, то есть путь от
  другого GPU-host.
- На `remote` контейнеру нужен путь относительно собственного dataset-root
  `datasets/Для участников/...`.
- Для этого `scripts/run_teacher_peft_c4_tail.py` был доработан:
  - добавлен `--data-root`
  - если `resolved_path` не существует локально, путь восстанавливается как
    `data_root / filepath`
- После синка обновлённого скрипта на `remote` очередь была перезапущена с теми же
  run id и с параметром:
  `--data-root 'datasets/Для участников'`

## Текущий live status

- После relaunch `stage1` дошёл до реального extraction:
  - `extract rows=341/134697`
  - `pct=0.3`
  - `rows_per_s=9.75`
  - `elapsed_s=35.0`
- В этот момент `GPU1` уже показывала около `68.7 GiB` памяти и ненулевую утилизацию,
  то есть прогон перешёл из стадии cold-start в рабочий extraction.

## Ожидаемый результат

После завершения обоих прогонов нужно зафиксировать:

- `embedding_s`, `search_s`, `rerank_s`
- `top10_mean_score_mean`
- `top1_score_mean`
- `label_used_share`
- `indegree_gini_10`
- `indegree_max_10`
- validator status
- относительное сравнение `stage1 vs stage2 vs stage3`

## Статус на момент записи

- `stage1` ablation: completed
- `stage2` ablation: completed on `GPU1`

## Stage1 completed result

Финальный output `stage1`:

- output dir:
  `artifacts/backbone_public/teacher_peft/W2V1k_stage1_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1`
- release submission copy:
  `artifacts/submissions/W2V1k_stage1_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1_submission.csv`
- summary json:
  `artifacts/backbone_public/teacher_peft/W2V1k_stage1_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1/W2V1k_stage1_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1_summary.json`

Локальные метрики `stage1`:

- `embedding_s = 2539.024440`
- `search_s = 0.643123`
- `rerank_s = 11.733930`
- `top10_mean_score_mean = 0.7498600483`
- `top1_score_mean = 0.8053842783`
- `label_used_share = 0.9125592998`
- `indegree_gini_10 = 0.2969486793`
- `indegree_max_10 = 41`
- `validator_passed = true`

Внешний Public LB результат `stage1`:

- `public_lb = 0.8062`
- против organizer baseline `0.0779`: `+0.7283`
- против `MS41 = 0.7473`: `+0.0589`
- против финального `W2V1j stage3 = 0.8344`: `-0.0282`

Интерпретация:

- Даже checkpoint после `stage1` уже очень силён: он уверенно проходит выше старой
  CAM++ safe-ветки.
- Но `stage3` всё ещё даёт заметный выигрыш, значит последующие стадии не просто
  стабилизируют обучение, а реально улучшают hidden/public transfer.

## Stage2 completed result

Финальный output `stage2`:

- output dir:
  `artifacts/backbone_public/teacher_peft/W2V1l_stage2_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1`
- remote submission:
  `artifacts/backbone_public/teacher_peft/W2V1l_stage2_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1/submission_W2V1l_stage2_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1.csv`
- local submission copy:
  `artifacts/submissions/W2V1l_stage2_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1_submission.csv`
- local summary copy:
  `artifacts/submissions/W2V1l_stage2_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1_summary.json`
- local validator copy:
  `artifacts/submissions/W2V1l_stage2_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1_validation.json`
- log:
  `artifacts/logs/W2V1l_stage2_teacher_peft_public_c4_20260416T144453Z_fast_bs1024_w4p1.log`

Локальные метрики `stage2`:

- `embedding_s = 2684.535089`
- `search_s = 0.700455`
- `rerank_s = 11.773768`
- `top10_mean_score_mean = 0.7443845272`
- `top1_score_mean = 0.8002873063`
- `label_used_share = 0.9161302776`
- `indegree_gini_10 = 0.3045464260`
- `indegree_max_10 = 42`
- `validator_passed = true`
- SHA-256 локальной submission-копии:
  `fce58c1006301d3dfafbb4b9eba2e1b6b2b7b412c57853625f58ae7f89f80d7e`

Внешний Public LB результат `stage2`:

- `public_lb = 0.8222`
- против organizer baseline `0.0779`: `+0.7443`
- против `MS41 = 0.7473`: `+0.0749`
- против `W2V1k stage1 = 0.8062`: `+0.0160`
- против финального `W2V1j stage3 = 0.8344`: `-0.0122`

Относительно уже известных стадий:

- против `stage1` по локальному proxy:
  - `top10_mean_score_mean`: `0.7444` против `0.7499`
  - `top1_score_mean`: `0.8003` против `0.8054`
- против финального `stage3` по локальному proxy:
  - `top10_mean_score_mean`: `0.7444` против `0.7456`
  - `top1_score_mean`: `0.8003` против `0.8008`

Интерпретация:

- Локальный C4-tail proxy занижал реальный выигрыш `stage2`: по локальным метрикам он
  выглядел почти равным `stage1`, но на public leaderboard дал явный прирост `+0.0160`.
- На внешнем public уже видна чистая лестница качества:
  `stage1 (0.8062) < stage2 (0.8222) < stage3 (0.8344)`.
- Значит joint full fine-tuning (`stage2`) уже возвращает большую часть финального
  прироста, а финальный LMFT (`stage3`) добавляет ещё небольшой, но реальный последний
  шаг `+0.0122`.
