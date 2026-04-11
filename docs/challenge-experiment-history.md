# Challenge Experiment History

История решений и результатов для презентации и последующего анализа. Все public
значения ниже считаются внешними: локально public labels недоступны, поэтому
leaderboard score нельзя пересчитать без платформы.

## Метрика

Платформа считает `Precision@10` для retrieval по скрытым `speaker_id`.

Для каждой записи из `test_public.csv` сабмит содержит 10 индексов ближайших
соседей. Сосед считается правильным, если его скрытый `speaker_id` совпадает со
скрытым `speaker_id` query-записи. Финальный score - среднее значение по всем
query:

```text
precision@10_i = correct_neighbors_i / 10
public_score = mean(precision@10_i)
```

Интерпретация результата `0.1024`: в среднем среди 10 отправленных соседей
примерно `1.024` соседа имеют того же диктора.

## Leaderboard History

| Date | Experiment | Main changes | Local validation | Public LB | Delta vs organizer baseline | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-04-11 | Organizer baseline | Исходный baseline от организаторов. Используется как внешняя стартовая точка. | Not comparable | `0.0779` | baseline | Зафиксировать как внешний reference. |
| 2026-04-11 | `baseline_fixed_participants` | Исправлена crop-логика; train использует random crop, val/test deterministic crop. Добавлен speaker-disjoint split, seed control, `train_ratio=0.98`, deterministic val/test, ONNX export только embeddings, opset `20`, public inference через center crop и exact FAISS top-10. | speaker-disjoint val `precision@10 = 0.9174` | `0.1024` | `+0.0245` absolute, около `+31.5%` relative | Изменения полезны для public, но локальная val сильно завышена и не коррелирует с public. Следующий приоритет - public-like validation. |
| 2026-04-11 | `B2_raw_3crop` | Inference-only ablation поверх `baseline_fixed_v1`: deterministic 3-crop, L2 per crop, mean pooling, exact top-10. | dense shifted `0.4729`, honest shifted v2 `0.2117` | `0.1098` | `+0.0074` vs baseline_fixed | Multi-crop даёт реальный public gain, но без trim/rerank уступает B4/B7. |
| 2026-04-11 | `B4_trim_3crop` | Conservative trim + deterministic 3-crop, exact top-10. | dense shifted `0.5042`, honest shifted v2 `0.2308` | `0.1150` | `+0.0126` vs baseline_fixed | Trim полезен именно под public-like shift; B4 становится чистым preprocessing control. |
| 2026-04-11 | `B7_trim_3crop_reciprocal_top50` | B4 embeddings + reciprocal top-50 rerank with top-20 reciprocal bonus. | dense shifted `0.5122`, honest shifted v2 `0.2331` | `0.1206` | `+0.0182` vs baseline_fixed, `+0.0427` vs organizer baseline | Лучший submitted run; reciprocal rerank снижает public hubness и даёт лучший public score. |
| 2026-04-11 | `B8_trim_3crop_reciprocal_local_scaling` | B7 + density penalty based on mean top-20 candidate density. | honest shifted v2 `0.2320`; public Gini@10 `0.3385`, max in-degree `56` | `0.1223` | `+0.0199` vs baseline_fixed, `+0.0444` vs organizer baseline | Новый лучший submitted run; local scaling подтверждён public, несмотря на небольшой локальный miss относительно B7. |

## What Changed In `baseline_fixed_participants`

- Crop policy:
  - было: train получал фиксированный начальный кусок, val/test получали случайный кусок;
  - стало: train получает random crop, val/test получают deterministic center crop.
- Split:
  - было: random row split;
  - стало: speaker-disjoint split с контролем `min_val_utts >= 11`.
- Train/val ratio:
  - было: `split_ratio=0.2` фактически давал train 20%, val 80%;
  - стало: `train_ratio=0.98`, train `659804` rows, val `13473` rows.
- Reproducibility:
  - добавлены seed control и deterministic public inference.
- Export/inference:
  - ONNX теперь экспортирует только embeddings, без classifier logits;
  - ONNX opset поднят до `20`;
  - submission собран через cosine-equivalent L2-normalized embeddings и exact FAISS top-10.

## Impact

- Public score вырос с `0.0779` до `0.1024`.
- Абсолютный прирост: `+0.0245`.
- Относительный прирост: примерно `+31.5%`.
- В терминах соседей: `0.1024` означает примерно `137930` правильных neighbor-slots
  из `1346970` проверяемых на public (`134697 * 10`).

## Key Interpretation

Локальный `precision@10 = 0.9174` нельзя использовать как прямой прогноз public
leaderboard. Разрыв с public `0.1024` показывает, что текущий speaker-disjoint
train-derived validation остается сильно оптимистичным: он измеряет качество
внутри train-domain, а public, вероятно, отличается по домену, условиям записи,
обработке, шумам, каналам или распределению дикторов.

Практический вывод для следующих экспериментов: сначала строить `public_like_val`,
который ближе к public по размеру пула, длительностям, clipping/silence buckets,
domain buckets и сложности retrieval. Улучшения модели нужно сравнивать уже на
такой валидации, иначе локальные `0.90+` могут не переноситься на leaderboard.

## Current Public Submission Artifact

- Submission: `artifacts/baseline_fixed_participants/submission_center_opset20.csv`
- Validation report:
  `artifacts/baseline_fixed_participants/submission_center_opset20_validation.json`
- ONNX: `artifacts/baseline_fixed_participants/model_embeddings.onnx`
- Public embeddings:
  `artifacts/baseline_fixed_participants/test_public_emb_center_opset20.npy`

## Public Ablation Cycle

- Best submitted run: `B8_trim_3crop_reciprocal_local_scaling`, public LB `0.1223`.
- Public gain vs `baseline_fixed_v1`: `+0.0199` absolute, about `+19.4%` relative.
- Public gain vs organizer baseline: `+0.0444` absolute, about `+57.0%` relative.
- Submitted public ranking: B0 `0.1024` < B2 `0.1098` < B4 `0.1150` < B7 `0.1206` < B8 `0.1223`.
- Spearman rank correlation on B0/B2/B4/B7, plus B8 for honest shifted v2:
  - smoke val: `1.0`
  - dense gallery val: `0.8`
  - dense shifted val: `1.0`
  - honest dense shifted v2: `0.9`
- Public B7 hubness @10 improved vs preprocessing-only runs, and B8 reduces hubness
  further:
  - B2 Gini@10 `0.5010`, max in-degree `149`
  - B4 Gini@10 `0.5139`, max in-degree `174`
  - B7 Gini@10 `0.4056`, max in-degree `87`
  - B8 Gini@10 `0.3385`, max in-degree `56`
- Honest `dense_shifted_v2` now uses one synthetic channel condition per file, shared
  by all crops. Under that stricter protocol B7 remains the best local selector:
  B7 `0.2331`, B8 `0.2320`, B9 `0.2323`, B10 `0.2308`. Public still prefers B8,
  so hubness reduction is a useful signal even when local P@10 is slightly lower.

Artifacts:

- `artifacts/eda/public_ablation_cycle.zip`
- `artifacts/eda/validation_cycle_package.zip`
- `artifacts/eda/baseline_fixed_dense_shifted_v2_honest.zip`
- `artifacts/eda/next_cycle_review_package.zip`
