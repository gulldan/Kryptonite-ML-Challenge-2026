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
| 2026-04-11 | `C4_b8_labelprop_mutual10` | B4 embeddings + B8 reciprocal/local-density ranking + deterministic weighted label propagation on mutual top-10 graph; top-10 prefers same propagated label when label size and candidate count are sane. | honest shifted v2 `0.2413`; validator passed; public Gini@10 `0.3504`, max in-degree `74` | `0.1249` | `+0.0225` vs baseline_fixed, `+0.0470` vs organizer baseline | Новый лучший submitted run. Graph/community postprocess даёт реальный public gain, но скачок пока маленький; следующий шаг - проверить C5 или идти в stronger backbone + graph branch. |
| 2026-04-12 | `P2_campp_h800_b1024_public_c4` | CAM++ trained from scratch on participant split, batch `1024` on H800, then same C4-tail graph/retrieval postprocess on public. | dense shifted v2 `P@10 = 0.3552`, `top1 = 0.7722`; validator passed | `0.1753` | `+0.0504` vs C4, `+0.0974` vs organizer baseline | Backbone switch is confirmed useful, but CAM++ is not the main candidate because ERes2NetV2 is much stronger on public. Keep as fusion candidate. |
| 2026-04-12 | `P1_eres2netv2_h800_b128_public_c4` | ERes2NetV2 trained from scratch on participant split, batch `128` on H800, then same C4-tail graph/retrieval postprocess on public. | dense shifted v2 interrupted for public generation; train final acc `0.9940`, dev score gap `0.5689`; validator passed | `0.2410` | `+0.1161` vs C4, `+0.1631` vs organizer baseline | New production candidate. This is the first large public jump; prioritize ERes2NetV2 graph tuning/fusion over more baseline-tail polishing. |
| 2026-04-12 | `F1_eres075_cam025_rankscore_public_c4` | Rank/robust-score fusion of ERes2NetV2 and CAM++ public top-200 graphs, weights `0.75/0.25`, then same C4 labelprop mutual10 tail. | validator passed; source top-200 overlap p50 `66`, p95 `100`; label used share `0.6345`; Gini@10 `0.4611` | `0.2305` | `-0.0105` vs P1, `+0.1056` vs C4 | Rejected as production candidate. Fusion added noisy CAM++ neighbor evidence and hurt public score; keep P1 as safe branch. |
| 2026-04-12 | `G6_p1_clusterfirst_mutual20_shared4_penalty020_top300` | P1 ERes2NetV2 embeddings + top-300 mutual graph, shared-neighbor filter, size-penalized cluster-first retrieval. | validator passed; `cluster_used_share=0.7552`, p99 cluster size `84.66`, max cluster `521`, Gini@10 `0.3460` | `0.2369` | `-0.0041` vs P1, `+0.1120` vs C4 | Rejected as production candidate. Cluster-first graph changed many neighbors but did not beat C4 on public; next step is a more orthogonal pretrained encoder. |
| 2026-04-12 | `organizer_baseline_e20_earlystop_epoch10_center` | Original organizer ECAPA baseline trained up to epoch `10` with validation early stopping/scheduler guard, then center-crop public inference with exact FAISS top-10. | speaker-disjoint val `precision@10 = 0.928308`; validator passed | `0.1046` | `+0.0267` vs organizer baseline, `+0.0022` vs baseline_fixed, `-0.0203` vs C4 | Rejected/dead-end. Longer guarded training improves local validation but barely moves public and remains far below graph/backbone branches; do not spend more cycles on this baseline family except as a diagnostic control. |
| 2026-04-12 | `P3_eres2netv2_g6_pseudo_ft_public_c4` | P1 ERes2NetV2 initialized pseudo-label self-training on original train + filtered G6 public clusters, then same public C4 tail. | validator passed; `top10_mean_score_mean=0.6116`; `label_used_share=0.8297`; Gini@10 `0.2810`, max in-degree `44`; mean overlap vs P1 `4.99/10` | `0.2861` | `+0.0451` vs P1, `+0.2082` vs organizer baseline | New public best. Pseudo-label self-training is confirmed useful; use P3 as the new safe branch while E1 WavLM-domain continues. |
| 2026-04-12 | `E1_wavlm_domain_ft_public_c4` | `microsoft/wavlm-base-plus-sv` fine-tuned on train with mixed crops, bandlimit/silence/gain/noise/peak augmentations, ArcMargin head, then same public C4 tail. | validator passed; `top10_mean_score_mean=0.8124`; `label_used_share=0.7778`; Gini@10 `0.2649`, max in-degree `47`; train epoch 3 loss `5.4627`, acc `0.6662` | `0.2833` | `-0.0028` vs P3, `+0.0423` vs P1, `+0.2054` vs organizer baseline | Direct WavLM is slightly below P3, so it does not replace the safe branch. Keep as strong orthogonal candidate for later P3+WavLM fusion because public score is close while representation family is different. |
| 2026-04-12 | `H1_wavlm_base_plus_sv_pretrained_public_c4` | Direct pretrained `microsoft/wavlm-base-plus-sv` embeddings with the same C4 graph tail, no participant/domain fine-tuning. | validator passed; `top10_mean_score_mean=0.9582`; `label_used_share=0.7476`; Gini@10 `0.2324`, max in-degree `34`; overlap vs P1 mean `0.309/10` | `0.1228` | `+0.0449` vs organizer baseline, `-0.0021` vs C4, `-0.1633` vs P3 | Rejected as a direct branch. Raw pretrained WavLM has clean geometry but does not align to hidden public speakers enough; domain fine-tuning is essential. |
| 2026-04-13 | `H6_wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4` | Fresh official-HF raw pretrained `microsoft/wavlm-base-plus-sv` probe: `AutoFeatureExtractor` + `AutoModelForAudioXVector`, no silence trim, 3 evenly spaced 6s crops, then C4 graph tail. | validator passed; `top10_mean_score_mean=0.9590`; `label_used_share=0.7489`; Gini@10 `0.2321`, max in-degree `34`; overlap vs H1 mean `2.970/10`, top1 equal `21.93%`. | pending | pending | Public probe candidate requested to rule out trim/frontend mismatch in raw WavLM. Expectation remains low because H1 scored only `0.1228`, but H6 is a materially different no-trim official-HF ranking. |
| 2026-04-12 | `MS1_modelscope_campplus_voxceleb_default` | User-provided default ModelScope `iic/speech_campplus_sv_en_voxceleb_16k` submission without challenge fine-tuning. | Local validator passed for `artifacts/backbone_public/campp/default_model_submission.csv`; Gini@10 `0.4917`, max in-degree `214`. | `0.5695` | `+0.4916` vs organizer baseline, `+0.2834` vs P3 | New best observed branch. Treat the exact ModelScope frontend/inference policy as the safe branch; local reimplementation does not yet reproduce the neighbor ranking. |
| 2026-04-12 | `MS2_modelscope_campplus_voxceleb_default_public_exact/c4` | Converted ModelScope CAM++ VoxCeleb checkpoint into local `CAMPPlusEncoder` by key remap only, then ran local 3x6s trim public inference and exact/C4 tails. | exact validator passed; C4 validator passed; exact overlap vs MS1 `2.459/10`, top1 match `15.86%`; C4 overlap vs MS1 `2.510/10`, top1 match `15.41%`. | not submitted | pending | Diagnostic/rejected as a reproduction of MS1. Same checkpoint weights are not enough; frontend/crop/trim/official ModelScope inference differs materially from our local path. |
| 2026-04-12 | `MS3_modelscope_campplus_voxceleb_default_notrim_1crop` | Same converted ModelScope checkpoint, but public inference uses `--no-trim`, one 6s center crop, exact and C4 tails. | exact validator passed; C4 validator passed; exact overlap vs MS1 `2.075/10`, top1 match `11.92%`; C4 overlap vs MS1 `2.190/10`, top1 match `11.90%`. | not submitted | pending | Rejected frontend ablation. Removing trim and using a single crop does not reproduce MS1; the remaining gap points to official ModelScope fbank/full-utterance/frontend behavior. |
| 2026-04-12 | `MS4_official_repo_reproduction_pretrained_segment_mean` | Ran `RustamOper05/kryptonite_tembr_research` `code/campp/build_submission.py` through `gh` clone with official 3D-Speaker CAM++ and `torchaudio.compliance.kaldi.fbank` segment-mean frontend. | validator passed; overlap vs MS1 `9.961/10`, top1 match `99.53%`, same neighbor set share `96.13%`; official embeddings row-wise cosine vs MS2 local fbank only `0.6114` mean. | not submitted | inherits MS1 | Confirms the gap is embeddings/frontend, not submission formatting. Use official repo frontend as the parity source. |
| 2026-04-12 | `MS5_official_campp_runner_cached_embeddings` | Added repo-local official CAM++ runner and loaded cached MS4 official embeddings, then generated exact and C4 submissions through this repository's shared submission/rerank code. | exact validator passed, `top10_mean_score_mean=0.6765`, Gini@10 `0.4917`, max in-degree `214`; C4 validator passed, `top10_mean_score_mean=0.6662`, Gini@10 `0.3587`, max in-degree `83`. | not submitted | pending | Runner parity/control. The repository can now reproduce the official-frontend branch without relying on the temporary GitHub clone. |
| 2026-04-12 | `MS6_official_campp_c1_component_candidate` | Official MS4 embeddings + B8 reciprocal/local-density ranking + mutual-20 component postprocess. | validator passed; overlap vs MS1 exact `7.968/10`, top1 equal `76.3%`, same neighbor set share `13.57%`; `top10_mean_score_mean=0.6720`, Gini@10 `0.3371`, max in-degree `81`. | pending | pending | Fast public probe candidate to test whether hubness reduction improves the strong ModelScope branch. First file to submit if trying to beat `0.5695`; exact MS1 remains the safe known score. |
| 2026-04-12 | `MS7_campp_from_scratch_official_frontend_recalc` | Recomputed the arm11 CAM++ participant checkpoint with the new official CAM++ frontend runner. Source checkpoint was copied from arm11 run `20260411T200858Z-757aa9406317`; training summary says `provenance_initialization=from_scratch`. | exact validator passed, `top10_mean_score_mean=0.8367`, Gini@10 `0.5443`, max in-degree `268`; C4 validator passed, `top10_mean_score_mean=0.8287`, Gini@10 `0.3266`, max in-degree `59`; C4 overlap vs MS1 default only `1.521/10`, so this is a materially new ranking. | `0.2597` | `+0.1818` vs organizer baseline, `-0.3098` vs MS1 default | Rejected as a replacement for ModelScope default. Corrected frontend improves the experiment validity, but this arm11 checkpoint is still far below MS1 `0.5695`; do not spend more public submissions on this from-scratch checkpoint except maybe fusion diagnostics. |
| 2026-04-13 | `H7c_eres2net_large_3dspeaker_pretrained_public_c4_b128_bf16` | Clean pretrained official 3D-Speaker ERes2Net-large probe using `iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k` weights, official 3D-Speaker ERes2Net architecture/FBank/chunking, and soundfile FLAC decode workaround for broken arm11 torchaudio decoder. | running on arm11 GPU1; smoke extraction passed on 3 files; batch `32` and batch `80` fp32 attempts were stopped as too slow; active run uses batch `128` with bf16 autocast, log `artifacts/logs/H7c_eres2net_large_3dspeaker_pretrained_public_c4_b128_bf16_20260413T0418Z.log`. | pending | pending | Active public candidate. This tests whether clean pretrained 3D-Speaker ERes2Net-large has transfer behavior closer to ModelScope CAM++ default than our from-scratch ERes2NetV2 branch. |

Current public best:

- User-reported external best:
  `MS1_modelscope_campplus_voxceleb_default`, public LB `0.5695`
  using ModelScope `iic/speech_campplus_sv_en_voxceleb_16k`.
- Artifact:
  `artifacts/backbone_public/campp/default_model_submission.csv`

- Fast public probe candidate:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/official_graph_20260412T_candidate/submission_C1_b8_mutual20_component.csv`
  (`MS6_official_campp_c1_component_candidate`, validator passed, public score pending).

- Fast corrected arm11 CAM++ candidate:
  `artifacts/backbone_public/campp/submission_MS7_new_code_campp_from_scratch_official_frontend_c4.csv`
  (`MS7_campp_from_scratch_official_frontend_recalc`, validator passed, public LB `0.2597`; rejected as a replacement for MS1).

- Best repo-local scored artifact:
  `P3_eres2netv2_g6_pseudo_ft_public_c4`, public LB `0.2861`
- Artifact:
  `artifacts/backbone_public/eres2netv2_g6_pseudo_ft/20260412T100738Z-6b686847f5d8/submission_P3_eres2netv2_g6_pseudo_ft_public_c4.csv`
- Latest orthogonal candidate: `E1_wavlm_domain_ft_public_c4`, public LB `0.2833`
  with artifact
  `artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z/submission_E1_wavlm_domain_ft_public_c4.csv`.

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

- Best submitted inference-only ablation before graph postprocess:
  `B8_trim_3crop_reciprocal_local_scaling`, public LB `0.1223`.
- B8 public gain vs `baseline_fixed_v1`: `+0.0199` absolute, about `+19.4%` relative.
- B8 public gain vs organizer baseline: `+0.0444` absolute, about `+57.0%` relative.
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

## Graph / Community Postprocess Cycle

Hypothesis: because public retrieval is transductive over the full test pool and each
speaker has at least `K+1` utterances, graph/community structure should add signal beyond
pairwise cosine ranking.

Implementation:

- Reusable code: `src/kryptonite/eda/community.py`
- Public runner: `scripts/run_public_graph_community.py`
- Input embeddings: `artifacts/eda/public_ablation_cycle/embeddings_B4_trim_3crop.npy`
- Output directory: `artifacts/eda/public_graph_community/`

Runs checked:

| Date | Experiment | Local honest shifted v2 | Public LB | Validator | Decision |
| --- | --- | ---: | ---: | --- | --- |
| 2026-04-11 | `C4_b8_labelprop_mutual10` | `0.2413` | `0.1249` | passed | Keep as current best. |
| 2026-04-11 | `C5_b8_labelprop_mutual10_shared2` | `0.2395` | not submitted | passed | Next public candidate if another submission is available. |
| 2026-04-11 | `C6_b8_labelprop_mutual15` | `0.2387` | not submitted | passed | Submit only after C5 or if broader graph looks necessary. |
| 2026-04-11 | `C1/C2/C3` connected-components variants | effectively B8 fallback | not submitted | passed | Rejected/diagnostic: mutual-kNN connected components collapse into giant components, so component guard falls back instead of giving useful communities. |

Current best public submission artifact:

- `artifacts/eda/public_graph_community/submission_C4_b8_labelprop_mutual10.csv`

Graph-cycle lesson:

- Label propagation over a mutual top-10 graph improves public from B8 `0.1223`
  to C4 `0.1249`.
- Connected-component community detection is too brittle on this graph because it forms
  giant components; deterministic weighted label propagation is a better first graph
  postprocess for this backbone.
- The public gain is real but small, so graph postprocess is useful as a safe layer,
  while the larger gap to leaderboard leaders still points to a stronger encoder/backbone
  plus graph/community inference.

## Backbone Transition Prep

Date: 2026-04-11

Hypothesis: the current baseline encoder is saturated; the next meaningful gain should
come from a stronger backbone evaluated through the already-confirmed C4 postprocess
tail, rather than from more tuning of the old encoder.

Prepared code/config:

- Participant training manifest builder:
  `scripts/build_participant_training_manifests.py`
- Reusable manifest conversion logic:
  `src/kryptonite/data/participant_manifests.py`
- First ERes2NetV2 candidate config:
  `configs/training/eres2netv2-participants-candidate.toml`
- Generic checkpoint-to-C4-tail runner for CAM++/ERes2NetV2 checkpoints:
  `scripts/run_torch_checkpoint_c4_tail.py`

Generated manifests:

- `artifacts/manifests/participants_fixed/train_manifest.jsonl`: `659804` rows
- `artifacts/manifests/participants_fixed/dev_manifest.jsonl`: `13473` rows
- `artifacts/manifests/participants_fixed/manifest_inventory.json`

Checks:

- `scripts/validate_manifests.py --manifests-root artifacts/manifests/participants_fixed --strict`
  passed: `673277` valid rows, `0` invalid rows.
- `pytest tests/unit/test_eda.py -q` passed: `6 passed`.
- `ruff check` passed for touched Python files.

Important note:

- A CPU full-data smoke command was accidentally started without `max_train_rows`; it was
  stopped and produced no usable experiment result. This is recorded as a process lesson:
  do not smoke-test full participant configs on CPU without explicit row limits.

Next intended run:

```bash
uv run --group train python scripts/run_baseline.py \
  --model eres2netv2 \
  --config configs/training/eres2netv2-participants-candidate.toml \
  --device cuda
```

After checkpoint training, evaluate through C4 tail:

```bash
uv run --group train python scripts/run_torch_checkpoint_c4_tail.py \
  --model eres2netv2 \
  --checkpoint-path artifacts/baselines/eres2netv2-participants/<run-id> \
  --manifest-csv artifacts/eda/baseline_fixed_dense_shifted_v2_honest/dense_gallery_manifest.csv \
  --output-dir artifacts/backbone_eval/eres2netv2-candidate/<run-id> \
  --experiment-id E1_eres2netv2_c4_dense_shifted_v2 \
  --shift-mode v2
```

Decision gate:

- Public submission should wait until the new backbone beats current C4 by at least
  `+0.008...+0.010` on honest dense shifted v2 after the full C4 tail, unless it is
  explicitly being tested as a fusion candidate.

## ERes2NetV2 Backbone Training Run

Date: 2026-04-11

Hypothesis: a from-scratch ERes2NetV2 trained on the participant speaker split should
produce stronger embeddings than the saturated organizer-style baseline encoder, and
must be evaluated through the existing C4 tail before any public submission.

Failed launch:

- Experiment id: `eres2netv2_participants_20260411_220149`
- Command/config: `configs/training/eres2netv2-participants-candidate.toml`, default
  `batch_size=64`, `eval_batch_size=64`, `device=cuda`.
- Log: `artifacts/logs/eres2netv2_participants_20260411_220149.log`
- Result: rejected/failed. CUDA OOM during the first training forward pass on RTX 4090;
  the process was stopped and no checkpoint or metric was produced.

Active launch:

- Experiment id: `eres2netv2_participants_b32_20260411_220247`
- Command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --group train python scripts/run_baseline.py \
  --model eres2netv2 \
  --config configs/training/eres2netv2-participants-candidate.toml \
  --project-override training.batch_size=32 \
  --project-override training.eval_batch_size=32 \
  --device cuda
```

- PID file: `artifacts/logs/eres2netv2_participants_b32_20260411_220247.pid`
- Log: `artifacts/logs/eres2netv2_participants_b32_20260411_220247.log`
- Status at launch: running on CUDA, GPU utilization reached `100%`, memory around
  `20.9 GiB`; no initial OOM observed.
- Decision: keep this as the first real ERes2NetV2 training attempt. After the checkpoint
  is written, evaluate it with `scripts/run_torch_checkpoint_c4_tail.py` on honest
  dense shifted v2 before any public leaderboard submission.

## ERes2NetV2 Remote H800 Training Setup

Date: 2026-04-11

Goal: move the ERes2NetV2 backbone training from the local RTX 4090 machine to `arm11`,
where the challenge dataset is mounted in the same relative `datasets/` layout and the
Docker container exposes two NVIDIA H800 PCIe GPUs.

Remote paths:

- Host repository path: `/data/rnd/jupyter/kleshchenok/audio/embbedings`
- Container repository path: `/jupyter/kleshchenok/audio/embbedings`
- Container: `MK_RND`

Preparation checks:

- Repository copied to the remote host while excluding local `.venv`, caches,
  `datasets/`, and large transient artifacts.
- Existing remote `datasets/` directory preserved.
- Required training/evaluation artifacts copied:
  - `artifacts/manifests/participants_fixed/`
  - `artifacts/baseline_fixed_participants/train_split.csv`
  - `artifacts/baseline_fixed_participants/val_split.csv`
  - `artifacts/eda/baseline_fixed_dense_shifted_v2_honest/`
  - `artifacts/eda/participants_audio6/file_stats.parquet`
- Container environment synced with:

```bash
uv sync --dev --group train
```

Validation:

- `scripts/validate_manifests.py --manifests-root artifacts/manifests/participants_fixed --strict`
  passed inside the container: `673277` valid rows, `2` manifests, `0` invalid rows.
- Training imports passed inside the container for `run_speaker_baseline` and
  `ERes2NetV2Encoder`.

Remote launch:

- Experiment id: `eres2netv2_h800_b64_20260411_200341`
- PID file: `artifacts/logs/eres2netv2_h800_b64_20260411_200341.pid`
- Log: `artifacts/logs/eres2netv2_h800_b64_20260411_200341.log`
- Latest run pointer: `artifacts/logs/latest_eres2netv2_h800_b64.txt`

```bash
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_baseline.py \
  --model eres2netv2 \
  --config configs/training/eres2netv2-participants-candidate.toml \
  --device cuda
```

Decision: use the default H800 batch settings from the config (`batch_size=64`,
`eval_batch_size=64`) for the first remote launch because H800 has enough memory and
the local OOM was specific to the 24 GiB RTX 4090.

Status after launch: running inside `MK_RND` on GPU0. Initial remote check showed process
PID `248490`, H800 memory around `35 GiB`, and GPU utilization around `93%`; no startup
OOM observed.

Follow-up correction:

- The first remote launch was stopped intentionally because the training loop did not emit
  per-epoch/per-step progress to stdout, leaving the log empty during long epochs.
- Added stdout progress logging in `run_classification_batches`: epoch start, first batch,
  periodic batch progress, loss, accuracy, examples/sec, and elapsed seconds.
- Because `batch_size=64` used only about `35 GiB` on H800, the next ERes2NetV2 launch
  should use `training.batch_size=128` and `training.eval_batch_size=128` to improve
  throughput and make better use of the 80 GiB card.

Parallel GPU plan:

- GPU0: `ERes2NetV2` participants candidate, H800 batch/eval batch `128`.
- GPU1: `CAM++` participants candidate, H800 batch/eval batch `128` if it fits; fall back
  to `96` or `64` only on OOM.
- These are independent backbone hypotheses. The current code path is not DDP-enabled, so
  using both cards as separate jobs is safer and gives faster hypothesis coverage than
  forcing multi-GPU training into the existing single-process pipeline.

Active H800 launches after logging fix:

- `eres2netv2_h800_b128_20260411_200735`
  - GPU: `0`
  - Overrides: `training.batch_size=128`, `training.eval_batch_size=128`
  - Log: `artifacts/logs/eres2netv2_h800_b128_20260411_200735.log`
  - Initial status: alive, epoch `1/10`, `5155` batches/epoch, GPU memory about
    `69-70 GiB`.
- `campp_h800_b128_20260411_200735`
  - GPU: `1`
  - Overrides: `training.batch_size=128`, `training.eval_batch_size=128`
  - Status: stopped intentionally after startup because it used only about `9 GiB`; this
    did not make good use of the H800 card.
- `campp_h800_b1024_20260411_200846`
  - GPU: `1`
  - Overrides: `training.batch_size=1024`, `training.eval_batch_size=1024`
  - Log: `artifacts/logs/campp_h800_b1024_20260411_200846.log`
  - Initial status: alive, epoch `1/10`, `645` batches/epoch, GPU memory about
    `62-63 GiB`.

Decision: keep `ERes2NetV2 batch128` and `CAM++ batch1024` running as the current
parallel backbone training hypotheses. If a run OOMs during training or eval, restart
only that run with the next lower batch (`ERes2NetV2`: `112` or `96`; `CAM++`: `768`
or `512`).

## Backbone Training Completion And C4-Tail Eval Prep

Date: 2026-04-12

Training results on `arm11`:

- `campp_h800_b1024_20260411_200846`
  - Output root: `artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317`
  - Final train loss: `0.816177`
  - Final train accuracy: `0.987965`
  - Dev score gap: `0.522243`
  - Status: completed successfully, checkpoint/report written.
- `eres2netv2_h800_b128_20260411_200735`
  - Output root: `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee`
  - Final train loss: `0.964109`
  - Final train accuracy: `0.994018`
  - Dev score gap: `0.568871`
  - Status: training, checkpoint, embeddings, and score summary completed; final report
    generation was still running at check time.

Training reproducibility parameters:

- Shared environment and data contract:
  - Remote host: `arm11`
  - Docker container: `MK_RND`
  - Host repository path: `/data/rnd/jupyter/kleshchenok/audio/embbedings`
  - Container repository path: `/jupyter/kleshchenok/audio/embbedings`
  - Container setup command: `uv sync --dev --group train`
  - Base config: `configs/base.toml`
  - Code state recorded after run: base commit
    `012d87f2ea6a37cccccf327ff22bcdb43e139131` plus uncommitted training/logging
    changes in `scripts/run_torch_checkpoint_c4_tail.py`,
    `src/kryptonite/training/optimization_runtime.py`, and
    `src/kryptonite/training/speaker_baseline.py`.
  - Seed: `42`; deterministic mode: `true`; `PYTHONHASHSEED=42` contract from
    `configs/base.toml`.
  - Train manifest:
    `artifacts/manifests/participants_fixed/train_manifest.jsonl`
  - Dev manifest:
    `artifacts/manifests/participants_fixed/dev_manifest.jsonl`
  - Split/provenance: participant train split from `baseline_fixed`; no VoxBlink2 data
    or checkpoint; both backbones initialized `from_scratch`.
  - Audio/features: mono `16 kHz`, 80-bin log-fbank, 25 ms frame, 10 ms shift, FFT `512`,
    Hann window, `cmvn_mode="none"`.
  - Training crops: one random crop per utterance, crop seconds sampled per batch from
    `2.0` to `6.0`; short utterances use repeat padding through the chunking pipeline.
  - Eval/dev embedding crops during training report: max/full chunk `6.0s`, chunk
    overlap `1.5s`, embedding pooling `mean`.
  - VAD/trim during training dataloading: `vad.mode="none"`; no waveform trim in the
    training loader.
  - Augmentation during these runs: no waveform augmentation was active in
    `ManifestSpeakerDataset`; `silence_augmentation.enabled=false`. The base
    `augmentation_scheduler` config existed but was not consumed by this production
    dataloader path.
  - Sampler: `BalancedSpeakerBatchSampler`; batches select balanced speakers and one
    utterance request per selected speaker where possible.
  - Loss/classifier: cosine classifier plus `ArcMarginLoss`.
  - Precision: `bf16` autocast on CUDA.
  - Epochs: `10`.
  - Dataloader workers: `6`, `prefetch_factor=4`, persistent workers enabled.

- `eres2netv2_h800_b128_20260411_200735` exact training run:
  - Config: `configs/training/eres2netv2-participants-candidate.toml`
  - GPU: `CUDA_VISIBLE_DEVICES=0`
  - CLI overrides: `training.batch_size=128`, `training.eval_batch_size=128`
  - Effective batch size: `128`; effective eval batch size: `128`.
  - Model config: `embedding_size=192`, `m_channels=64`, `base_width=26`, `scale=2`,
    `expansion=2`, `num_blocks=[3,4,6,3]`, `pooling_func="TSTP"`,
    `two_embedding_layers=false`.
  - Objective: `classifier_blocks=0`, `classifier_hidden_dim=192`, `scale=32.0`,
    `margin=0.3`, `easy_margin=false`.
  - Optimizer/scheduler: `sgd`, cosine schedule, `learning_rate=0.12`,
    `min_learning_rate=0.00005`, `momentum=0.9`, `weight_decay=0.0001`,
    `warmup_epochs=2`, `gradient_accumulation_steps=1`, `grad_clip_norm=5.0`.
  - Train log: `artifacts/logs/eres2netv2_h800_b128_20260411_200735.log`
  - Output root:
    `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee`
  - Checkpoint:
    `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt`
  - Training command:

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_baseline.py \
  --model eres2netv2 \
  --config configs/training/eres2netv2-participants-candidate.toml \
  --project-override training.batch_size=128 \
  --project-override training.eval_batch_size=128 \
  --device cuda
```

  - Public C4-tail command:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --group train python scripts/run_torch_checkpoint_c4_tail.py \
  --model eres2netv2 \
  --checkpoint-path artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee \
  --experiment-id P1_eres2netv2_h800_b128_public_c4 \
  --shift-mode none \
  --device cuda \
  --search-device cuda \
  --batch-size 1024
```

  - Public C4-tail effective defaults not shown in the command:
    `precision=bf16`, `search_batch_size=2048`, `top_cache_k=100`,
    `crop_seconds=6.0`, `n_crops=3`, `trim=true`, `edge_top=10`,
    `shared_min_count=0`, `seed=42`.
  - Public LB: `0.2410`.

- `campp_h800_b1024_20260411_200846` exact training run:
  - Config: `configs/training/campp-participants-candidate.toml`
  - GPU: `CUDA_VISIBLE_DEVICES=1`
  - CLI overrides: `training.batch_size=1024`, `training.eval_batch_size=1024`
  - Effective batch size: `1024`; effective eval batch size: `1024`.
  - Model config: `embedding_size=512`, `growth_rate=32`, `bottleneck_scale=4`,
    `init_channels=128`, `head_channels=32`, `head_res_blocks=[2,2]`,
    `block_layers=[12,24,16]`, `block_kernel_sizes=[3,3,3]`,
    `block_dilations=[1,2,2]`, `memory_efficient=true`.
  - Objective: `classifier_blocks=0`, `classifier_hidden_dim=512`, `scale=32.0`,
    `margin=0.2`, `easy_margin=false`.
  - Optimizer/scheduler: `adamw`, cosine schedule, `learning_rate=0.0015`,
    `min_learning_rate=0.00005`, `momentum=0.9`, `weight_decay=0.0001`,
    `warmup_epochs=1`, `gradient_accumulation_steps=1`, `grad_clip_norm=5.0`.
  - Train log: `artifacts/logs/campp_h800_b1024_20260411_200846.log`
  - Output root:
    `artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317`
  - Checkpoint:
    `artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317/campp_encoder.pt`
  - Training command:

```bash
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
uv run --group train python scripts/run_baseline.py \
  --model campp \
  --config configs/training/campp-participants-candidate.toml \
  --project-override training.batch_size=1024 \
  --project-override training.eval_batch_size=1024 \
  --device cuda
```

  - Public C4-tail command:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --group train python scripts/run_torch_checkpoint_c4_tail.py \
  --model campp \
  --checkpoint-path artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317/campp_encoder.pt \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/campp/20260411T200858Z-757aa9406317 \
  --experiment-id P2_campp_h800_b1024_public_c4 \
  --shift-mode none \
  --device cuda \
  --search-device cuda \
  --batch-size 1024
```

  - Public C4-tail effective defaults not shown in the command:
    `precision=bf16`, `search_batch_size=2048`, `top_cache_k=100`,
    `crop_seconds=6.0`, `n_crops=3`, `trim=true`, `edge_top=10`,
    `shared_min_count=0`, `seed=42`.
  - Public LB: `0.1753`.

Important remote-path fix:

- `dense_gallery_manifest.csv` contained local absolute paths under
  `/mnt/storage/Kryptonite-ML-Challenge-2026`.
- For `arm11`, wrote
  `artifacts/eda/baseline_fixed_dense_shifted_v2_honest/dense_gallery_manifest.remote.csv`
  with `resolved_path` rewritten to `/jupyter/kleshchenok/audio/embbedings`.

Next evaluation:

- Run C4-tail dense shifted v2 eval first for CAM++ on free GPU1.
- Run C4-tail dense shifted v2 eval for ERes2NetV2 after the final report process releases
  GPU0, or run it on GPU1 if GPU0 remains occupied by reporting.

Active C4-tail evaluation:

- Experiment id: `E2_campp_h800_b1024_c4_dense_shifted_v2`
- Run id: `campp_c4_dense_shifted_v2_20260412_061705`
- Checkpoint:
  `artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317/campp_encoder.pt`
- Manifest:
  `artifacts/eda/baseline_fixed_dense_shifted_v2_honest/dense_gallery_manifest.remote.csv`
- Output dir:
  `artifacts/backbone_eval/campp-candidate/20260411T200858Z-757aa9406317`
- Log: `artifacts/logs/campp_c4_dense_shifted_v2_20260412_061705.log`
- Status at launch: alive on GPU1, manifest rows `38500`, embedding extraction started.

Result:

- `E2_campp_h800_b1024_c4_dense_shifted_v2`
  - `p10 = 0.3552`
  - `top1_accuracy = 0.7721818181818182`
  - `top1_score_mean = 0.7131559252738953`
  - `top10_mean_score_mean = 0.5882105231285095`
  - `embedding_s = 907.987723`
  - `search_s = 0.169722`
  - `rerank_s = 2.48215`
  - Summary:
    `artifacts/backbone_eval/campp-candidate/20260411T200858Z-757aa9406317/E2_campp_h800_b1024_c4_dense_shifted_v2_summary.json`

Active follow-up evaluation:

- Experiment id: `E3_eres2netv2_h800_b128_c4_dense_shifted_v2`
- Run id: `eres2netv2_c4_dense_shifted_v2_20260412_063505`
- Checkpoint:
  `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt`
- Manifest:
  `artifacts/eda/baseline_fixed_dense_shifted_v2_honest/dense_gallery_manifest.remote.csv`
- Output dir:
  `artifacts/backbone_eval/eres2netv2-candidate/20260411T200748Z-15ced4a6d3ee`
- Status at launch: GPU0 was free; C4-tail evaluation started on GPU0.

Public leaderboard candidate generation:

- Hypothesis: trained participant backbones should be tested on public LB through the same
  C4-tail graph/retrieval postprocess, because dense shifted local `P@10` is not enough
  to decide whether the new embeddings transfer to the public pool.
- ERes2NetV2 dense-val eval was stopped early to prioritize public submission generation.
- Public manifest:
  `artifacts/eda/backbone_public/test_public_manifest.remote.csv`
  (`134697` rows, `resolved_path` rewritten for the `arm11` container).
- `P1_eres2netv2_h800_b128_public_c4`
  - Checkpoint:
    `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt`
  - Output dir:
    `artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee`
  - Log: `artifacts/logs/eres2netv2_public_c4_20260412_063759.log`
  - Status at launch: alive on GPU0, public extraction started.
  - Follow-up: stopped at about `10%` extraction because running both public extraction
    processes in parallel was CPU/IO-bound (`~21 rows/s` each), delaying the first
    leaderboard candidate. ERes2NetV2 public generation should be restarted after the
    CAM++ public submission is produced.
  - Follow-up 2: restarted immediately after confirming the server has enough CPU/disk
    capacity and the user prefers both public candidates to run at once. New run id:
    `eres2netv2_public_c4_restart_20260412_065600`.
  - Result artifact:
    `artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/submission_P1_eres2netv2_h800_b128_public_c4.csv`
  - Validator: passed, `134697` rows, `k=10`, `0` errors.
  - Runtime summary: `embedding_s=4785.937889`, `search_s=0.642428`,
    `rerank_s=9.132642`.
  - Public confidence summary: `top1_score_mean=0.7809972167015076`,
    `top10_mean_score_mean=0.7344830632209778`.
  - Public LB: `0.2410`.
- `P2_campp_h800_b1024_public_c4`
  - Checkpoint:
    `artifacts/baselines/campp-participants/20260411T200858Z-757aa9406317/campp_encoder.pt`
  - Output dir:
    `artifacts/backbone_public/campp/20260411T200858Z-757aa9406317`
  - Log: `artifacts/logs/campp_public_c4_20260412_063800.log`
  - Status at launch: alive on GPU1, public extraction started.
  - Follow-up: kept running as the first public candidate because CAM++ already has
    `P@10=0.3552` on dense shifted v2.
  - Result artifact:
    `artifacts/backbone_public/campp/20260411T200858Z-757aa9406317/submission_P2_campp_h800_b1024_public_c4.csv`
  - Validator: passed, `134697` rows, `k=10`, `0` errors.
  - Runtime summary: `embedding_s=5150.635756`, `search_s=0.791136`,
    `rerank_s=11.580039`.
  - Public confidence summary: `top1_score_mean=0.7773728966712952`,
    `top10_mean_score_mean=0.7309617400169373`.
  - Public LB: `0.1753`.

Conclusion:

- The public leaderboard confirms the backbone switch is a real improvement over the
  previous best C4 baseline (`0.1249`).
- `ERes2NetV2 + C4-tail` is the new production candidate: `0.2410`, a `+0.1161`
  absolute gain over `C4_b8_labelprop_mutual10`.
- `CAM++ + C4-tail` improved over C4 too (`0.1753`) but underperformed ERes2NetV2 by
  `0.0657` absolute on public despite a strong dense shifted local `P@10=0.3552`.
- The dense shifted local result did not rank CAM++ and ERes2NetV2 correctly because
  ERes2NetV2 dense-val was interrupted before completion. Finish ERes2NetV2 dense-val
  and add public LB as the authoritative rank check for this cycle.

## ERes2NetV2 + CAM++ Fusion Public Candidate

Date: 2026-04-12

Hypothesis:

- `ERes2NetV2 + C4-tail` is now the safe branch at public LB `0.2410`, but CAM++ may
  still carry complementary neighbor evidence despite being weaker alone (`0.1753`).
- Fuse the public top-neighbor graphs from both trained backbones, then reuse the same
  C4-style label propagation tail. This tests whether backbone complementarity improves
  the transductive speaker-community graph without retraining.

Implementation:

- New reusable helper:
  `src/kryptonite/eda/fusion.py`
- New CLI:
  `scripts/run_backbone_fusion_c4_tail.py`
- Fusion method: exact top-200 from each embedding space, row-wise union, rank/robust-score
  fusion, then top-100 fused cache into existing `LabelPropagationConfig`.
- Fusion config:
  - `left_name=eres2netv2`, `left_weight=0.75`
  - `right_name=campp`, `right_weight=0.25`
  - `source_top_k=200`
  - `top_cache_k=100`
  - `rank_weight=1.0`
  - `score_z_weight=0.15`
- C4-tail config:
  - `edge_top=10`
  - `reciprocal_top=20`
  - `rank_top=100`
  - `iterations=5`
  - `label_min_size=5`
  - `label_max_size=120`
  - `label_min_candidates=3`
  - `shared_min_count=0`
  - `reciprocal_bonus=0.03`
  - `density_penalty=0.02`

Remote command:

```bash
cd /jupyter/kleshchenok/audio/embbedings
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_backbone_fusion_c4_tail.py \
  --left-embeddings artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/embeddings_P1_eres2netv2_h800_b128_public_c4.npy \
  --right-embeddings artifacts/backbone_public/campp/20260411T200858Z-757aa9406317/embeddings_P2_campp_h800_b1024_public_c4.npy \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/fusion/20260412T085023Z \
  --experiment-id F1_eres075_cam025_rankscore_public_c4 \
  --left-weight 0.75 \
  --right-weight 0.25 \
  --source-top-k 200 \
  --top-cache-k 100 \
  --search-device cuda \
  --search-batch-size 2048
```

Result before public LB:

- Experiment id: `F1_eres075_cam025_rankscore_public_c4`
- Submission:
  `artifacts/backbone_public/fusion/20260412T085023Z/submission_F1_eres075_cam025_rankscore_public_c4.csv`
- Summary:
  `artifacts/backbone_public/fusion/20260412T085023Z/F1_eres075_cam025_rankscore_public_c4_summary.json`
- Validator: passed, `134697` rows, `k=10`, `0` errors.
- Runtime: `left_search_s=2.422092`, `right_search_s=0.809668`,
  `fusion_elapsed_s=60.584358`, `rerank_s=9.081155`.
- Fusion overlap: source top-200 overlap `p50=66`, `p95=100`; enough disagreement exists
  for fusion to be a real hypothesis, not a duplicate of ERes2NetV2.
- Public graph diagnostics: `label_used_share=0.63449`, `label_size_max=267`,
  `indegree_gini_10=0.46110`, `indegree_max_10=153`.

Decision:

- Public LB: `0.2305`.
- The run is below `P1_eres2netv2_h800_b128_public_c4 = 0.2410` by `0.0105`, so it is
  rejected as the production candidate.
- Interpretation: naive rank/robust-score fusion injected too much noisy CAM++ neighbor
  evidence into the ERes2NetV2 graph. The top-200 overlap confirmed complementarity, but
  that complementarity was not clean enough under the current fusion weights and C4 tail.
- Keep `P1_eres2netv2_h800_b128_public_c4` as the safe branch.
- Do not run a broad blind fusion sweep before improving calibration. Any later fusion
  should be gated by local/public rank checks and should try more conservative CAM++
  influence, for example CAM++ only as a tie-breaker or mutual-confirmation bonus.

## Backbone Undertraining Diagnostic Check

Date: 2026-04-12

Question:

- Are `P1_eres2netv2_h800_b128_public_c4` and `P2_campp_h800_b1024_public_c4`
  undertrained because both were trained for only `10` epochs?
- Do we log enough loss/accuracy data to answer this?

Artifacts:

- `artifacts/backbone_training_diagnostics/20260412T_undertraining_check/training_curves.csv`
- `artifacts/backbone_training_diagnostics/20260412T_undertraining_check/training_diagnostics_summary.json`
- Source logs copied from `arm11`:
  - `artifacts/tracking/20260411T200748Z-15ced4a6d3ee/metrics.jsonl`
  - `artifacts/tracking/20260411T200858Z-757aa9406317/metrics.jsonl`

What is logged now:

- Per-epoch train loss, train accuracy, and learning rate are logged in
  `training_summary.json` and `artifacts/tracking/<run_id>/metrics.jsonl`.
- Final dev/retrieval metrics are logged after training: `eer`, `min_dcf`,
  `score_gap`, `rank_1_accuracy`, `rank_5_accuracy`, and `rank_10_accuracy`.
- Missing for a strong undertraining/overtraining decision: per-epoch dev metrics and
  per-epoch checkpoints. Current runs only validate the final epoch.

Observed curves:

| Model | Epoch 1 loss / acc | Epoch 10 loss / acc | Last-epoch loss delta | Last-epoch acc delta | Final LR | Final dev summary |
| --- | --- | --- | --- | --- | --- | --- |
| `ERes2NetV2` | `14.8727` / `0.2344` | `0.9641` / `0.9940` | `-0.1736` | `+0.0020` | `0.00005` | `rank1=0.9682`, `rank10=0.9888`, `eer=0.0300`, `score_gap=0.5689` |
| `CAM++` | `9.3825` / `0.3815` | `0.8162` / `0.9880` | `-0.0306` | `+0.0010` | `0.00005` | `rank1=0.9467`, `rank10=0.9803`, `eer=0.0443`, `score_gap=0.5222` |

Interpretation:

- The models are not grossly undertrained in the simple sense: train accuracy is already
  very high, and final dev metrics are strong.
- `ERes2NetV2` still reduces train loss at epoch 10, but the cosine scheduler has already
  decayed to `5e-5`; simply appending 4 epochs with the same ended schedule is unlikely
  to be a high-upside move.
- If testing longer training, use a new controlled recipe, not "continue blindly":
  `15-20` epochs from scratch or resume with an explicit low-LR fine-tune schedule, plus
  selected-epoch dev evaluation/checkpoints.
- Because public improved massively from backbone switch (`0.1249 -> 0.2410`) while
  local final dev is already high, the next likely bottleneck is not basic convergence.
  It is domain-shift-aware recipe and stronger/pretrained encoder work.

Decision:

- Do not spend the next slot on "same run plus 4 epochs" without extra validation.
- Add future logging requirement: any training-duration hypothesis must persist
  per-epoch or selected-epoch dev metrics and checkpoint paths, so the best epoch and
  overfit/underfit behavior can be recovered.
- A reasonable controlled follow-up is `ERes2NetV2` `15-20` epochs with a schedule designed
  for that length, plus per-epoch or every-2-epoch C4-tail dev evaluation.

## P1 Cluster-First Graph Tail Cycle

Date: 2026-04-12

Hypothesis:

- The public task is transductive over the full `test_public` pool, and each hidden
  speaker should have at least `K+1` utterances. `P1_eres2netv2_h800_b128_public_c4`
  is the safe branch, but the C4 tail is still only a local label-propagation preference
  over top-10 mutual edges.
- A broader mutual-kNN graph with shared-neighbor filtering and cluster-first retrieval
  may recover speaker communities better than C4, provided oversized graph components
  are controlled.

Implementation:

- New reusable code: `src/kryptonite/eda/community.py`
  - `ClusterFirstConfig`
  - `cluster_first_rerank()`
  - weighted mutual edge builder with shared-neighbor counts
  - optional label-size penalty to prevent broad graph label collapse
  - cluster assignment export for pseudo-label/self-training follow-up
- New CLI: `scripts/run_cluster_first_tail.py`
- Source embeddings:
  `artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/embeddings_P1_eres2netv2_h800_b128_public_c4.npy`
- Manifest:
  `artifacts/eda/backbone_public/test_public_manifest.remote.csv`
- Top-k cache generated once on `arm11`:
  `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g1/indices_G1_p1_clusterfirst_mutual50_shared2_top300_top300.npy`
  and matching `scores_*.npy`.

Remote runs:

| Run | Config summary | Validator | Key graph diagnostics | Artifact | Decision |
| --- | --- | --- | --- | --- | --- |
| `G1_p1_clusterfirst_mutual50_shared2_top300` | top-300 cache, `edge_top=50`, `reciprocal_top=100`, `shared_min_count=2`, no size penalty | passed | `cluster_used_share=0.0260`, p99 cluster size `679.26`, max cluster `22352`, Gini@10 `0.3600` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g1/` | Rejected/diagnostic: graph collapsed into giant communities and mostly fell back. |
| `G2_p1_clusterfirst_mutual20_shared4_top300` | cached top-300, `edge_top=20`, `reciprocal_top=50`, `shared_min_count=4`, no size penalty | passed | `cluster_used_share=0.3176`, p99 `110.84`, max `8964`, Gini@10 `0.3519` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g2_shared4/` | Diagnostic: stricter graph helps, but oversized labels remain too large. |
| `G3_p1_clusterfirst_mutual20_shared4_penalty_top300` | G2 + `self_weight=1.0`, `label_size_penalty=0.35` | passed | `cluster_used_share=0.0018`, p99 `1.0`, max `55`, Gini@10 `0.3348` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g3_penalty/` | Rejected/diagnostic: penalty is too strong and destroys communities into singletons. |
| `G4_p1_clusterfirst_mutual20_shared4_penalty010_top300` | G2 + `label_size_penalty=0.10` | passed | `cluster_used_share=0.5965`, p50 same-cluster candidates `5`, p99 `101`, max `1602`, Gini@10 `0.3519` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g4_penalty/` | Kept as conservative cluster-first candidate. |
| `G5_p1_clusterfirst_mutual20_shared4_penalty015_top300` | G2 + `label_size_penalty=0.15` | passed | `cluster_used_share=0.6985`, p50 same-cluster candidates `7`, p99 `93.77`, max `1020`, Gini@10 `0.3503`, max in-degree `64` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g5_penalty/` | Kept as balanced high-upside candidate. |
| `G6_p1_clusterfirst_mutual20_shared4_penalty020_top300` | G2 + `label_size_penalty=0.20` | passed | `cluster_used_share=0.7552`, p50 same-cluster candidates `7`, p99 `84.66`, max `521`, Gini@10 `0.3460`, max in-degree `65` | `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g6_penalty/` | Submitted; public LB `0.2369`; rejected as production candidate. |

Logged reruns for candidate artifacts:

- `G5` log:
  `artifacts/logs/20260412T093233Z_p1_clusterfirst_g5_penalty.log`
- `G6` log:
  `artifacts/logs/20260412T093233Z_p1_clusterfirst_g6_penalty.log`
- Process note: `G2`-`G4` were foreground diagnostics and their terminal logs were not
  persisted; this was corrected by rerunning the retained candidates `G5` and `G6` with
  `tee` logs.

Representative retained command (`G6`):

```bash
cd /jupyter/kleshchenok/audio/embbedings
PYTHONUNBUFFERED=1 uv run --group train python scripts/run_cluster_first_tail.py \
  --indices-path artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g1/indices_G1_p1_clusterfirst_mutual50_shared2_top300_top300.npy \
  --scores-path artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g1/scores_G1_p1_clusterfirst_mutual50_shared2_top300_top300.npy \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g6_penalty \
  --experiment-id G6_p1_clusterfirst_mutual20_shared4_penalty020_top300 \
  --top-cache-k 300 \
  --edge-top 20 \
  --reciprocal-top 50 \
  --rank-top 300 \
  --iterations 8 \
  --cluster-min-size 5 \
  --cluster-max-size 160 \
  --cluster-min-candidates 3 \
  --shared-top 50 \
  --shared-min-count 4 \
  --split-edge-top 8 \
  --self-weight 0.0 \
  --label-size-penalty 0.20
```

Comparison to safe `P1+C4`:

- `P1+C4` remains the only scored safe branch: public LB `0.2410`.
- `P1+C4` submission hubness from its CSV: Gini@10 `0.3510`, max in-degree `69`.
- `G5`/`G6` are structurally different but not wild: mean top-10 neighbor overlap with
  `P1+C4` is `6.32` for `G5` and `6.25` for `G6`; p50 overlap is `7` for both.
- `G6` reduces Gini@10 to `0.3460` and raises cluster usage to `0.7552`, but mean
  submitted-neighbor cosine falls from `P1+C4` `0.73448` to `0.72667`. This is exactly
  the intended high-upside tradeoff: trust community structure more than local pairwise
  score.

Decision:

- Submitted `G6` as the high-upside public slot:
  `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g6_penalty/submission_G6_p1_clusterfirst_mutual20_shared4_penalty020_top300.csv`
- Public LB: `0.2369`.
- Decision: rejected as production candidate because it is below
  `P1_eres2netv2_h800_b128_public_c4 = 0.2410` by `0.0041`.
- Keep `P1_eres2netv2_h800_b128_public_c4` as the production fallback.
- `G5` remains available as a more conservative unsubmitted diagnostic:
  `artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g5_penalty/submission_G5_p1_clusterfirst_mutual20_shared4_penalty015_top300.csv`
- Use exported cluster assignment CSVs from `G5`/`G6` as the first pseudo-label pool for
  the next self-training experiment, after filtering oversized clusters and low-neighbor
  support rows.
- Next direction: stop spending public slots on graph-only variants over P1. The graph
  tail is useful but the public miss confirms the next likely bottleneck is representation
  quality, so move to an orthogonal pretrained encoder before fusion or self-training.

## 2026-04-12 — Parallel P1 Self-Training and WavLM Pretrained Fine-Tune

Context:

- `G6_p1_clusterfirst_mutual20_shared4_penalty020_top300` scored `0.2369` on public,
  below safe `P1_eres2netv2_h800_b128_public_c4 = 0.2410`.
- Decision: stop graph-only public variants over P1 and use G6 clusters only as a
  pseudo-label pool while testing an orthogonal pretrained representation.

Pseudo-label manifest:

```bash
cd /jupyter/kleshchenok/audio/embbedings
uv run --group train python scripts/build_pseudo_label_manifests.py \
  --clusters-csv artifacts/backbone_public/cluster_first/20260412T093233Z_p1_clusterfirst_g6_penalty/clusters_G6_p1_clusterfirst_mutual20_shared4_penalty020_top300.csv \
  --public-manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --base-train-manifest artifacts/manifests/participants_fixed/train_manifest.jsonl \
  --output-dir artifacts/manifests/pseudo_g6 \
  --experiment-id g6 \
  --min-cluster-size 8 \
  --max-cluster-size 80
```

Manifest result:

- Pseudo rows: `84280`
- Pseudo clusters: `4791`
- Mixed train rows: `744084`
- Pseudo manifest:
  `artifacts/manifests/pseudo_g6/g6_pseudo_manifest.jsonl`
- Mixed manifest:
  `artifacts/manifests/pseudo_g6/g6_mixed_train_manifest.jsonl`

Parallel remote launches:

| Run id | GPU | Hypothesis | Config / command | Status |
| --- | --- | --- | --- | --- |
| `eres2netv2_g6_pseudo_ft_20260412T100724Z` | `CUDA_VISIBLE_DEVICES=0` | P1 self-training: initialize from safe ERes2NetV2 P1 checkpoint, train on original participant train plus filtered G6 pseudo clusters, then recluster/re-run C4 tail after checkpoint is ready. | `configs/training/eres2netv2-g6-pseudo-finetune.toml`; init checkpoint `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt`; `training.batch_size=128`, `max_epochs=3`, mixed crop `2s..6s`, bf16, SGD cosine LR `0.01`, AAM margin `0.2`; log `artifacts/logs/eres2netv2_g6_pseudo_ft_20260412T100724Z.log` | Training completed. Final epoch loss `0.949111`, train acc `0.973660`, LR `1e-5`. Checkpoint `artifacts/baselines/eres2netv2-g6-pseudo-finetune/20260412T100738Z-6b686847f5d8/eres2netv2_encoder.pt`; training summary same dir. The post-train finalizer was manually stopped after checkpoint/dev embeddings because it started generating all-pairs dev verification trials from `13473` dev rows; that local score is not needed for public C4 tail and would waste time/memory. |
| `wavlm_base_plus_sv_ft_20260412T101216Z` | `CUDA_VISIBLE_DEVICES=1` | Orthogonal pretrained encoder: fine-tune `microsoft/wavlm-base-plus-sv` on participant train with raw waveform crops and ArcMargin classifier. This tests a representation family different from CAM++/ERes2NetV2 before fusion. | `configs/training/wavlm-base-plus-sv-participants-finetune.toml`; model provenance Hugging Face `microsoft/wavlm-base-plus-sv`; original participant train manifest only; `batch_size=8`, `steps_per_epoch=2000`, `max_epochs=1`, 4s crops, bf16, AdamW, model LR `2e-5`, classifier LR `1e-3`, feature encoder frozen, gradient checkpointing enabled; log `artifacts/logs/wavlm_base_plus_sv_ft_20260412T101216Z.log` | Completed. `16000` examples, `214.31s`, loss `14.2402`, train acc `0.0000`, embedding size `512`, speaker count `10848`. Saved model dir `artifacts/baselines/wavlm-base-plus-sv-participants-finetune/wavlm_base_plus_sv_ft_20260412T101216Z/hf_model`; checkpoint `hf_xvector_finetune.pt`; metrics `artifacts/tracking/wavlm_base_plus_sv_ft_20260412T101216Z/metrics.jsonl`. Diagnostic note: exact train acc is not useful yet because this was a very short 10.8k-class adaptation run; public tail/fusion decides usefulness. |

Public inference launched from WavLM fine-tuned model:

```bash
cd /jupyter/kleshchenok/audio/embbedings
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id artifacts/baselines/wavlm-base-plus-sv-participants-finetune/wavlm_base_plus_sv_ft_20260412T101216Z/hf_model \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_ft_public_c4_20260412T101651Z \
  --experiment-id H2_wavlm_base_plus_sv_ft_public_c4 \
  --batch-size 80 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --top-cache-k 200 \
  --search-batch-size 2048 \
  --search-device cuda
```

Current H2 status:

- Run id: `wavlm_base_plus_sv_ft_public_c4_20260412T101651Z`
- Log: `artifacts/logs/wavlm_base_plus_sv_ft_public_c4_20260412T101651Z.log`
- Output dir:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_ft_public_c4_20260412T101651Z/`
- Completed. Validator passed; submission:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_ft_public_c4_20260412T101651Z/submission_H2_wavlm_base_plus_sv_ft_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_ft_public_c4_20260412T101651Z/H2_wavlm_base_plus_sv_ft_public_c4_summary.json`
- Key diagnostics: `top1_score_mean=0.99985`, `top10_mean_score_mean=0.99985`,
  `indegree_gini_10=0.2457`, max in-degree `35`, label used share `0.8531`.
- Random 2000-row pairwise cosine sample from H2 embeddings: mean `0.9954`, p50
  `0.9968`, p99 `0.9997`.
- Submission overlap with safe P1: mean `0.005/10`, p50 `0`, zero-overlap share
  `0.9951`.

H2 decision:

- Do not submit H2 directly. The validator passed, but the embedding space is nearly
  collapsed after short fine-tuning; the almost-zero overlap with P1 is not useful
  diversity by itself.
- Keep H2 artifact only as a diagnostic showing that naive short WavLM fine-tuning with
  a large 10.8k-class ArcMargin head is unsafe.

Follow-up launch on freed GPU1:

```bash
cd /jupyter/kleshchenok/audio/embbedings
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id microsoft/wavlm-base-plus-sv \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_public_c4_20260412T104526Z \
  --experiment-id H1_wavlm_base_plus_sv_pretrained_public_c4 \
  --batch-size 80 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --top-cache-k 200 \
  --search-batch-size 2048 \
  --search-device cuda
```

- Run id: `wavlm_base_plus_sv_pretrained_public_c4_20260412T104526Z`
- Log: `artifacts/logs/wavlm_base_plus_sv_pretrained_public_c4_20260412T104526Z.log`
- Purpose: separate pretrained WavLM representation quality from the failed short
  fine-tune. If H1 avoids the H2 cosine-collapse pattern, use it for `P1 + WavLM`
  fusion; otherwise move to a different pretrained family.
- Completed. Validator passed; submission:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_public_c4_20260412T104526Z/submission_H1_wavlm_base_plus_sv_pretrained_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_public_c4_20260412T104526Z/H1_wavlm_base_plus_sv_pretrained_public_c4_summary.json`
- Key diagnostics: `top1_score_mean=0.9640`, `top10_mean_score_mean=0.9582`,
  `label_count=20360`, `label_used_share=0.7476`, `indegree_gini_10=0.2324`,
  max in-degree `34`.
- Random 2000-row pairwise cosine sample: mean `0.6314`, p50 `0.6453`, p99
  `0.9257`. Unlike H2, H1 is not collapsed.
- Submission overlap with safe P1: mean `0.309/10`, p50 `0`, zero-overlap share
  `0.8183`. This is real orthogonality, but direct public quality is low.
- Public LB score: `0.1228`.

H1 decision:

- Rejected as a direct branch. H1 is only slightly above B8 `0.1223` and below C4
  `0.1249`, despite a very high `top10_mean_score_mean=0.9582`.
- Interpretation: generic pretrained WavLM geometry is not enough by itself; the useful
  WavLM result is the domain-adapted E1 branch (`0.2833`), not raw pretrained inference.
- Keep H1 only as a diagnostic/fusion ingredient, and gate any H1 fusion against P3/E1
  rather than spending further direct public slots on raw pretrained WavLM variants.

NeMo/TitaNet note:

- Attempted import check for TitaNet path: `import nemo.collections.asr` failed with
  `ModuleNotFoundError: No module named 'nemo'`.
- Do not install the NeMo stack mid-run while GPU0 training is active; use a
  Transformers-compatible pretrained speaker model first, then revisit TitaNet setup if
  the current queue is exhausted.

Follow-up launch on GPU1:

```bash
cd /jupyter/kleshchenok/audio/embbedings
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id anton-l/wav2vec2-base-superb-sv \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_public_c4_20260412T111914Z \
  --experiment-id H3_wav2vec2_base_superb_sv_pretrained_public_c4 \
  --batch-size 80 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --top-cache-k 200 \
  --search-batch-size 2048 \
  --search-device cuda
```

- Run id: `wav2vec2_superb_sv_pretrained_public_c4_20260412T111914Z`
- Log: `artifacts/logs/wav2vec2_superb_sv_pretrained_public_c4_20260412T111914Z.log`
- Purpose: second orthogonal pretrained encoder using the same reproducible HF xvector
  tail, without NeMo dependency risk.
- Completed. Validator passed; submission:
  `artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_public_c4_20260412T111914Z/submission_H3_wav2vec2_base_superb_sv_pretrained_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_public_c4_20260412T111914Z/H3_wav2vec2_base_superb_sv_pretrained_public_c4_summary.json`
- Key diagnostics: `top1_score_mean=0.9725`, `top10_mean_score_mean=0.9695`,
  `label_count=24383`, `label_used_share=0.6948`, `indegree_gini_10=0.2418`,
  max in-degree `34`.
- Random 2000-row pairwise cosine sample: mean `0.7974`, p50 `0.8111`, p99
  `0.9508`. H3 is not as collapsed as H2, but it is more compressed than H1.
- Submission overlap with safe P1: mean `0.163/10`, p50 `0`, zero-overlap share
  `0.8811`.
- Decision: do not spend a public LB slot on H3 direct. Keep as diagnostic/fusion
  material only.

Local artifact sync note:

- The standard repo sync excludes `artifacts/`, so lightweight H1/H2/H3 summary,
  validation, and submission files were copied back manually. Large `.npy` embedding
  arrays remain remote-only unless needed locally.

Heavier GPU1 utilization launch:

```bash
cd /jupyter/kleshchenok/audio/embbedings
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id microsoft/wavlm-base-plus-sv \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_20260412T114046Z \
  --experiment-id H4_wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4 \
  --batch-size 128 \
  --crop-seconds 8.0 \
  --n-crops 5 \
  --top-cache-k 300 \
  --search-batch-size 4096 \
  --search-device cuda
```

- Run id: `wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_20260412T114046Z`
- Log: `artifacts/logs/wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_20260412T114046Z.log`
- Purpose: use more of GPU1 safely and test whether longer/more crops improve the
  strongest orthogonal pretrained candidate H1 before public LB spend.
- Monitor at 2026-04-12 11:57 UTC: extraction at `40%`; GPU1 used ~`22.3 GiB`, but
  util only ~`31%`, suggesting decode/IO/CPU-loop bottleneck despite larger batch.
- Completed. Validator passed; submission:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_20260412T114046Z/submission_H4_wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_20260412T114046Z/H4_wavlm_base_plus_sv_pretrained_8s5c_b128_public_c4_summary.json`
- Diagnostics: `top1_score_mean=0.9635`, `top10_mean_score_mean=0.9576`,
  `label_used_share=0.7413`, random pairwise cosine mean `0.6347`, p50 `0.6491`,
  p99 `0.9245`.
- Overlap with safe P1: mean `0.313/10`, p50 `0`, zero-overlap share `0.8186`.
- Overlap with H1: mean `2.785/10`, p50 `3`.
- Decision: H4 does not materially change the H1 picture. Keep as a slightly heavier
  WavLM fusion candidate; do not submit direct.

Additional concurrent GPU1 launch:

```bash
cd /jupyter/kleshchenok/audio/embbedings
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id anton-l/wav2vec2-base-superb-sv \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_8s5c_b128_public_c4_20260412T115810Z \
  --experiment-id H5_wav2vec2_base_superb_sv_pretrained_8s5c_b128_public_c4 \
  --batch-size 128 \
  --crop-seconds 8.0 \
  --n-crops 5 \
  --top-cache-k 300 \
  --search-batch-size 4096 \
  --search-device cuda
```

- Run id: `wav2vec2_superb_sv_pretrained_8s5c_b128_public_c4_20260412T115810Z`
- Log: `artifacts/logs/wav2vec2_superb_sv_pretrained_8s5c_b128_public_c4_20260412T115810Z.log`
- Purpose: keep GPU1 closer to full utilization while still below the requested
  no-OOM VRAM target. At launch with H4 concurrent: GPU1 ~`44.7 GiB`, 100% util.
- Completed. Validator passed; submission:
  `artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_8s5c_b128_public_c4_20260412T115810Z/submission_H5_wav2vec2_base_superb_sv_pretrained_8s5c_b128_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wav2vec2_superb_sv_pretrained_8s5c_b128_public_c4_20260412T115810Z/H5_wav2vec2_base_superb_sv_pretrained_8s5c_b128_public_c4_summary.json`
- Diagnostics: `top1_score_mean=0.9724`, `top10_mean_score_mean=0.9695`,
  `label_used_share=0.6805`, `indegree_gini_10=0.2457`, max in-degree `34`,
  same-label candidates p50 `5`, p95 `19`.
- Public LB decision: do not submit H4/H5 direct. H5 mostly confirms the earlier H3
  compressed Wav2Vec2 geometry and is not a safer direct candidate than P1.

E1 domain WavLM fine-tune launch:

- Hypothesis: the useful next jump should come from pretrained-first WavLM adaptation,
  not from direct pretrained inference or another graph-only P1 variant. The previous
  short WavLM fine-tune (`H2`) collapsed because LR/run length were too aggressive and
  too underdeveloped for a 10.8k-class ArcMargin head.
- Code/config changes:
  - `src/kryptonite/training/hf_xvector.py` now supports train-only mixed crop lengths
    plus domain augmentations before HF feature extraction: random bandlimit, leading /
    trailing silence, mild peak limiting, random gain, and light Gaussian noise.
  - Config:
    `configs/training/wavlm-base-plus-sv-e1-domain-finetune.toml`
  - Checks before sync: `ruff check`, `ty check`, and `pytest tests/unit/test_eda.py -q`
    passed for the touched WavLM training path.
- Remote sync: repo synced to `arm11` with the standard rsync command excluding
  `.venv/`, `.cache/`, `datasets/`, and general `artifacts/`.
- Run id: `wavlm_e1_domain_ft_20260412T130254Z`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- Log: `artifacts/logs/wavlm_e1_domain_ft_20260412T130254Z.log`
- PID file: `artifacts/logs/wavlm_e1_domain_ft_20260412T130254Z.pid`
- Output root:
  `artifacts/baselines/wavlm-base-plus-sv-e1-domain-finetune/wavlm_e1_domain_ft_20260412T130254Z/`
- Metrics path:
  `artifacts/tracking/wavlm_e1_domain_ft_20260412T130254Z/metrics.jsonl`
- Exact launch command:

```bash
cd /jupyter/kleshchenok/audio/embbedings
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
uv run --group train python scripts/run_hf_xvector_finetune.py \
  --config configs/training/wavlm-base-plus-sv-e1-domain-finetune.toml \
  --run-id wavlm_e1_domain_ft_20260412T130254Z \
  --device cuda
```

- Training recipe: Hugging Face `microsoft/wavlm-base-plus-sv`, original participant
  train manifest `artifacts/manifests/participants_fixed/train_manifest.jsonl`, seed
  `42`, `batch_size=96`, `steps_per_epoch=8000`, `max_epochs=3`, mixed crops `2s..6s`,
  bf16, AdamW, model LR `3e-6`, classifier LR `5e-4`, cosine schedule, warmup `1000`,
  feature encoder frozen, gradient checkpointing enabled, ArcMargin scale `32`, margin
  `0.2`.
- Augmentation recipe: bandlimit p `0.45`, edge silence p `0.45` with leading up to
  `0.8s` and trailing up to `1.4s`, peak limiter p `0.20`, gain p `0.25` in
  `[-3, +2] dB`, Gaussian noise p `0.15` with SNR `18..35 dB`.
- First monitor: epoch `1/3`, step `200/8000`, loss `16.7273`, train acc `0.000156`,
  LR `6e-7`, throughput `162.8` examples/s; GPU1 ~`18.7 GiB`, 93-98% util. VRAM is
  below the allowed 80% cap, but compute utilization is already high, so the first E1
  pass is kept running rather than restarted for a larger batch.
- Epoch 1 completed: `768000` examples, `8000` steps, train loss `10.5764`, train acc
  `0.2428`, epoch seconds `4770.59`, LR `2.3968e-6`. Metrics persisted in
  `artifacts/tracking/wavlm_e1_domain_ft_20260412T130254Z/metrics.jsonl` and
  `training_summary.json`.
- Follow-up monitor shortly after epoch 2 start: step `200/8000`, loss `7.2987`, train
  acc `0.5120`, throughput ~`165` examples/s, GPU1 ~`19.2 GiB`, util ~`91%`.
- Epoch 2 completed: `768000` examples, `8000` steps, train loss `6.4776`, train acc
  `0.5751`, epoch seconds `4484.56`, LR `9.1941e-7`. Metrics persisted in
  `artifacts/tracking/wavlm_e1_domain_ft_20260412T130254Z/metrics.jsonl` and
  `training_summary.json`.
- Follow-up monitor shortly after epoch 3 start: step `500/8000`, loss `5.6425`, train
  acc `0.6495`, throughput ~`169` examples/s, GPU1 ~`19.4 GiB`, util `100%`.
- Epoch 3 completed: `768000` examples, `8000` steps, train loss `5.4627`, train acc
  `0.6662`, epoch seconds `4479.97`, LR `1.5e-7`. Metrics persisted in
  `artifacts/tracking/wavlm_e1_domain_ft_20260412T130254Z/metrics.jsonl` and
  `training_summary.json`.
- Saved model:
  `artifacts/baselines/wavlm-base-plus-sv-e1-domain-finetune/wavlm_e1_domain_ft_20260412T130254Z/hf_model/`
- Saved checkpoint:
  `artifacts/baselines/wavlm-base-plus-sv-e1-domain-finetune/wavlm_e1_domain_ft_20260412T130254Z/hf_xvector_finetune.pt`
- Decision: training completed cleanly and GPU1 was freed. Per operator instruction,
  launch exactly one public C4 tail on GPU1, then stop further GPU work until the public
  LB result is known.

E1 domain WavLM public C4 tail:

- Hypothesis: after domain fine-tuning, the WavLM speaker geometry may become useful
  enough for a direct public C4 submission and later P3+WavLM fusion if public/local
  diagnostics are promising.
- Run id: `wavlm_e1_domain_ft_public_c4_20260412T165411Z`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- Log:
  `artifacts/logs/wavlm_e1_domain_ft_public_c4_20260412T165411Z.log`
- PID file:
  `artifacts/logs/wavlm_e1_domain_ft_public_c4_20260412T165411Z.pid`
- Model dir:
  `artifacts/baselines/wavlm-base-plus-sv-e1-domain-finetune/wavlm_e1_domain_ft_20260412T130254Z/hf_model/`
- Output dir:
  `artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z/`
- Exact launch command:

```bash
cd /jupyter/kleshchenok/audio/embbedings
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 \
uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id artifacts/baselines/wavlm-base-plus-sv-e1-domain-finetune/wavlm_e1_domain_ft_20260412T130254Z/hf_model \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv "datasets/Для участников/test_public.csv" \
  --output-dir artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z \
  --experiment-id E1_wavlm_domain_ft_public_c4 \
  --batch-size 128 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --top-cache-k 300 \
  --search-batch-size 4096 \
  --search-device cuda
```

- Initial monitor: model loaded, extraction started on `134697` public rows; GPU1
  ~`15.9 GiB`, util `97%`. GPU0 remains occupied by an external process and is not used.
- Completed. Extraction took `1232.73s`; exact top-k search took `0.74s`; C4-style label
  propagation/rerank took `9.77s`. Validator passed with `134697/134697` rows, `K=10`,
  and `0` errors.
- Submission:
  `artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z/submission_E1_wavlm_domain_ft_public_c4.csv`
- Summary:
  `artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z/E1_wavlm_domain_ft_public_c4_summary.json`
- Validation:
  `artifacts/backbone_public/hf_xvector/wavlm_e1_domain_ft_public_c4_20260412T130254Z/submission_E1_wavlm_domain_ft_public_c4_validation.json`
- Diagnostics: `top1_score_mean=0.8552`, `top10_mean_score_mean=0.8124`,
  `label_used_share=0.7778`, Gini@10 `0.2649`, max in-degree `47`, same-label
  candidates p50 `6`, p95 `29`, reciprocal share `0.1660`.
- Lightweight CSV/JSON artifacts were synced back locally for public LB upload.
- Public LB score: `0.2833`.
- Decision: direct E1 is slightly below the current safe P3 score `0.2861` by `0.0028`,
  so it does not replace P3. The score is close enough, and the backbone is orthogonal
  enough, to keep E1 for a later P3+WavLM fusion branch. Per operator instruction,
  stop here: do not start fusion, another tail, or another training job now. GPU1 was
  free after the run; GPU0 was not used.

P3 public tail from P1 pseudo-fine-tuned checkpoint:

- Hypothesis: if G6 pseudo clusters contain enough clean same-speaker structure, the
  P1 initialized encoder fine-tuned on mixed real+pseudo labels should improve public C4
  retrieval or at least become a useful fusion candidate.
- Source checkpoint:
  `artifacts/baselines/eres2netv2-g6-pseudo-finetune/20260412T100738Z-6b686847f5d8/eres2netv2_encoder.pt`
- Run id: `p3_eres2netv2_g6_pseudo_ft_public_c4_20260412T132640Z`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- Log:
  `artifacts/logs/p3_eres2netv2_g6_pseudo_ft_public_c4_20260412T132640Z.log`
- PID file:
  `artifacts/logs/p3_eres2netv2_g6_pseudo_ft_public_c4_20260412T132640Z.pid`
- Output dir:
  `artifacts/backbone_public/eres2netv2_g6_pseudo_ft/20260412T100738Z-6b686847f5d8/`
- Exact launch command:

```bash
cd /jupyter/kleshchenok/audio/embbedings
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_torch_checkpoint_c4_tail.py \
  --model eres2netv2 \
  --checkpoint-path artifacts/baselines/eres2netv2-g6-pseudo-finetune/20260412T100738Z-6b686847f5d8/eres2netv2_encoder.pt \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/eres2netv2_g6_pseudo_ft/20260412T100738Z-6b686847f5d8 \
  --experiment-id P3_eres2netv2_g6_pseudo_ft_public_c4 \
  --shift-mode none \
  --device cuda \
  --search-device cuda \
  --precision bf16 \
  --batch-size 256 \
  --search-batch-size 4096 \
  --top-cache-k 200 \
  --crop-seconds 6.0 \
  --n-crops 3
```

- First monitor: started embedding extraction on `134697` public rows; GPU0 ~`21.9 GiB`,
  util ~`64%`; validator/submission pending.
- Completed. Validator passed; submission:
  `artifacts/backbone_public/eres2netv2_g6_pseudo_ft/20260412T100738Z-6b686847f5d8/submission_P3_eres2netv2_g6_pseudo_ft_public_c4.csv`
- Summary:
  `artifacts/backbone_public/eres2netv2_g6_pseudo_ft/20260412T100738Z-6b686847f5d8/P3_eres2netv2_g6_pseudo_ft_public_c4_summary.json`
- Runtime: embedding extraction `2819.30s`, top-k search `0.77s`, C4 rerank `11.27s`.
- Retrieval diagnostics: `top1_score_mean=0.6901`, `top10_mean_score_mean=0.6116`,
  label count `12103`, label used share `0.8297`, reciprocal share `0.1836`,
  same-label candidates p50 `8`, p95 `26`.
- Hubness/overlap vs safe P1:
  - P3 `Gini@10=0.2810`, max in-degree `44`;
  - safe P1 `Gini@10=0.3510`, max in-degree `69`;
  - mean overlap vs P1 `4.99/10`, p50 `5`, p10 `2`, p90 `8`;
  - top1 same vs P1 `0.4890`, zero-overlap share `0.0134`.
- Public LB: `0.2861`.
- Decision: new public best and new safe branch. The lower cosine score scale was not a
  blocker; lower hubness plus the pseudo-label adapted encoder transferred to public.
  Pseudo-label self-training over filtered G6 clusters is now confirmed as a productive
  direction.
- Lightweight P3 summary, validation, and submission CSV were synced back locally under
  the same `artifacts/backbone_public/eres2netv2_g6_pseudo_ft/...` path; large embedding
  `.npy` remains remote-only.

Aborted follow-up:

- After the P3 public gain was known, a cheap P3+P1 rank-score fusion sweep
  (`F2/F3/F4`) was briefly launched on GPU0 with
  `scripts/run_backbone_fusion_c4_tail.py`.
- The user then requested GPU0 be left free for their own work. The fusion sweep was
  stopped before producing a completed summary/submission, and GPU0 was confirmed free
  (`0 MiB`, `0% util`). No decision should be drawn from the partial fusion output.

## 2026-04-12 — Local P1 Classifier-First Diagnostic

Hypothesis:

- The P1 ERes2NetV2 classifier head may contain useful train-speaker posterior signal
  that the public C4 path discarded by using only embeddings.
- If public is close to a closed-set/transductive speaker assignment problem, a
  class-aware retrieval pass should expose that by grouping public clips through
  classifier top-k classes before fallback embedding retrieval.

Implementation:

- Added reusable class-aware rerank logic in `src/kryptonite/eda/classifier_first.py`.
- Added CLI entrypoint `scripts/run_classifier_first_tail.py`.
- Copied only the required P1 artifacts from `arm11` to local:
  - checkpoint:
    `artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt`
  - public embeddings:
    `artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/embeddings_P1_eres2netv2_h800_b128_public_c4.npy`
- Ran locally on RTX 4090 using cached P1 embeddings, not audio extraction.

Command:

```bash
uv run --group train python scripts/run_classifier_first_tail.py \
  --checkpoint-path artifacts/baselines/eres2netv2-participants/20260411T200748Z-15ced4a6d3ee/eres2netv2_encoder.pt \
  --embeddings-path artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/embeddings_P1_eres2netv2_h800_b128_public_c4.npy \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv datasets/Для\ участников/test_public.csv \
  --output-dir artifacts/backbone_public/classifier_first/20260412T_local_h1 \
  --experiment-id H1_p1_classifier_first_top1bucket \
  --indices-path artifacts/backbone_public/classifier_first/20260412T_local_h1/indices_H1_p1_classifier_first_top1bucket_top100.npy \
  --scores-path artifacts/backbone_public/classifier_first/20260412T_local_h1/scores_H1_p1_classifier_first_top1bucket_top100.npy \
  --top-cache-k 100 \
  --class-batch-size 4096 \
  --search-batch-size 2048
```

Local result:

- Validator: passed.
- Submission:
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/submission_H1_p1_classifier_first_top1bucket.csv`
- Summary:
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/H1_p1_classifier_first_top1bucket_summary.json`
- Classifier classes in checkpoint: `10848`; `speaker_to_index_count=10848`.
- `class_used_share=0.9913`.
- Top1 class bucket stats: `7398` non-empty classes, p50 size `3`, p95 `40.15`,
  p99 `167`, max `26811`.
- Hubness: `Gini@10=0.5953`, max in-degree `474`.
- Mean submitted score fell to `0.5605` from P1+C4 `0.7345`.
- Overlap with safe P1+C4 submission: mean `2.35/10`, p50 `2`, p10 `0`, p90 `5`;
  top1 neighbor unchanged only `19.92%` of rows.

Decision:

- Keep as diagnostic; do not spend a public slot on this raw hard class-first candidate
  unless a more conservative variant first reduces hubness.
- The classifier posterior is not immediately a clean closed-set assignment signal:
  the largest predicted class bucket contains `26811` public rows, which is incompatible
  with the expected ~12 rows per true speaker and likely indicates domain/miscalibration
  or overconfident common-class collapse.
- Next safe use of logits is as a soft edge feature inside P1 graph/community or a
  conservative rerank within existing P1 top-100, not as a hard top1 bucket assignment.

## 2026-04-12 — Local Conservative Logits and Class-Aware Graph Diagnostics

Context:

- `H1_p1_classifier_first_top1bucket` rejected hard bucket assignment because it moved
  too far away from safe P1+C4 and created severe hubness.
- Follow-up question: are P1 classifier logits harmful in general, or only harmful when
  used as hard global buckets?

Implementation:

- Extended `src/kryptonite/eda/classifier_first.py` with:
  - `bucket_backfill=False` support for conservative top-k-only logits rerank;
  - `class_adjusted_topk()` for soft posterior-overlap score adjustment on an existing
    embedding top-k cache.
- Added `scripts/run_class_aware_graph_tail.py` for class-aware score adjustment before
  C4-style label propagation.
- All runs below reused cached local P1 public artifacts and did not perform audio
  extraction.

Inputs:

- Safe P1 submission:
  `artifacts/backbone_public/eres2netv2/20260411T200748Z-15ced4a6d3ee/submission_P1_eres2netv2_h800_b128_public_c4.csv`
- P1 top-100 cache:
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/indices_H1_p1_classifier_first_top1bucket_top100.npy`
  and
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/scores_H1_p1_classifier_first_top1bucket_top100.npy`
- P1 classifier top-5 cache:
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/class_indices_H1_p1_classifier_first_top1bucket_top5.npy`
  and
  `artifacts/backbone_public/classifier_first/20260412T_local_h1/class_probs_H1_p1_classifier_first_top1bucket_top5.npy`

Runs:

| Experiment | Local method | Validator | Gini@10 | Max in-degree | Mean overlap vs P1+C4 | Top1 same vs P1+C4 | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `H2_p1_logits_top100_conservative` | Conservative logits rerank restricted to P1 top-100, `bucket_backfill=false`, min class candidates `10`, class top-3. | passed | `0.5362` | `245` | `4.88/10` | `0.5337` | Rejected as public candidate. It is safer than hard H1 but still creates too much hubness and changes too much of safe P1. |
| `H3_p1_classaware_c4` | Class-aware score adjustment on P1 top-100 with posterior overlap weight `0.08`, same-top1 `0.02`, same-query-top3 `0.01`, then C4 labelprop. | passed | `0.3830` | `97` | `6.96/10` | `0.6637` | Kept as diagnostic. It proves logits are usable as soft graph weights, but hubness is still above P1. |
| `H3b_p1_classaware_c4_weak` | Weaker class-aware score adjustment on P1 top-100 with posterior overlap weight `0.03`, same-top1 `0.01`, same-query-top3 `0.005`, then C4 labelprop. | passed | `0.3619` | `71` | `7.89/10` | `0.7874` | Best local continuation. It is close to P1 hubness (`0.3510`, max `69`) while perturbing enough neighbors to be a plausible public slot if a low-risk logits test is desired. |

Artifacts:

- H2 submission:
  `artifacts/backbone_public/classifier_first/20260412T_local_h2_conservative/submission_H2_p1_logits_top100_conservative.csv`
- H2 summary:
  `artifacts/backbone_public/classifier_first/20260412T_local_h2_conservative/H2_p1_logits_top100_conservative_summary.json`
- H3 submission:
  `artifacts/backbone_public/class_aware_graph/20260412T_local_h3/submission_H3_p1_classaware_c4.csv`
- H3 summary:
  `artifacts/backbone_public/class_aware_graph/20260412T_local_h3/H3_p1_classaware_c4_summary.json`
- H3b submission:
  `artifacts/backbone_public/class_aware_graph/20260412T_local_h3b_weak/submission_H3b_p1_classaware_c4_weak.csv`
- H3b summary:
  `artifacts/backbone_public/class_aware_graph/20260412T_local_h3b_weak/H3b_p1_classaware_c4_weak_summary.json`
- Comparison JSON:
  `artifacts/backbone_public/classifier_first/20260412T_local_h2_h3_comparison.json`

Decision:

- Reject direct logits reranking (`H2`) for public: it still behaves like an assignment
  shortcut and increases hubness too much.
- Keep `H3b_p1_classaware_c4_weak` as the only reasonable local logits candidate. It
  supports the narrower conclusion: P1 classifier logits can be useful only as weak
  graph-edge evidence, not as hard class assignment.
- If using a public slot for a logits hypothesis, submit H3b rather than H1/H2/H3.
  Otherwise wait for the orthogonal WavLM branch and use the same weak class-aware graph
  idea only after checking complementarity.

Current decision:

- New safe public candidate: `P3_eres2netv2_g6_pseudo_ft_public_c4 = 0.2861`.
- Pseudo-label self-training is confirmed: P3 improves over safe P1 by `+0.0451`
  absolute on public. Next ERes2NetV2-side work should start from P3 artifacts, not P1:
  conservative P1+P3 fusion, P3 graph-tail tuning, and P3+orthogonal-pretrained fusion.
- Do not submit direct H1/H2/H3/H4/H5 pretrained HF xvector outputs. H2 collapsed after
  short naive fine-tuning; H1/H4 are useful orthogonal WavLM fusion candidates; H3/H5 are
  compressed Wav2Vec2 diagnostics.
- Let `wavlm_e1_domain_ft_20260412T130254Z` run as the main pretrained-first branch.
  After it finishes, run public C4 tail from its saved `hf_model/`, check pairwise cosine
  collapse, P1 overlap, Gini@10, max in-degree, and then test `P1 + WavLM-E1` fusion if
  the geometry is not collapsed.

## 2026-04-12 — Local ERes2NetV2 20-Epoch Guarded Baseline

Hypothesis:

- A longer ERes2NetV2 participant baseline may improve the P1 backbone if the run is
  allowed up to 20 epochs but guarded against obvious train-side overfitting.
- This is not a replacement for public-like validation. The local guard is intentionally
  conservative: full dev retrieval after every epoch is too expensive for a local RTX
  4090 launch, so the run uses train-loss patience and a train-accuracy ceiling, then
  restores the best train-loss state before checkpoint/export.

Implementation:

- Added generic baseline early stopping fields under `[optimization]`:
  `early_stopping_enabled`, `early_stopping_monitor`, `early_stopping_min_delta`,
  `early_stopping_patience_epochs`, `early_stopping_min_epochs`,
  `early_stopping_restore_best`, and `early_stopping_stop_train_accuracy`.
- Updated the shared baseline training loop to snapshot the best model/classifier state,
  restore it before writing the final checkpoint, and persist early-stopping metadata in
  `training_summary.json`.
- Added config:
  `configs/training/eres2netv2-participants-20epoch-local-guarded.toml`.

Config summary:

- Model family: `ERes2NetV2`, from scratch, 192-dim embedding.
- Train manifest:
  `artifacts/manifests/participants_fixed/train_manifest.jsonl` (`659804` rows).
- Dev manifest:
  `artifacts/manifests/participants_fixed/dev_manifest.jsonl` (`13473` rows).
- Seed: inherited from `configs/base.toml`, `42`.
- Device: local RTX 4090, `CUDA_VISIBLE_DEVICES=0`, `bf16`.
- Batch size: `32`; gradient accumulation steps: `2`; effective batch size: `64`;
  eval batch size: `64`.
- Max epochs: `20`.
- Optimizer/scheduler: SGD, LR `0.12`, min LR `0.00005`, momentum `0.9`,
  weight decay `0.0001`, cosine scheduler, warmup `2`, grad clip `5.0`.
- Crop/preprocessing: train crop `2.0-6.0s`, one crop, eval chunks `6.0s` with
  `1.5s` overlap, mean pooling, VAD disabled.
- Loss/objective: ArcMargin, scale `32.0`, margin `0.3`, classifier hidden dim `192`.
- Early stopping: monitor `train_loss`, min delta `0.0005`, patience `3`, min epochs
  `8`, restore best state, stop immediately after `train_accuracy >= 0.995`.

Local checks before launch:

- Config load check passed and resolved to `max_epochs=20`,
  `early_stopping_enabled=true`, monitor `train_loss`, stop accuracy `0.995`.
- `uv run ruff check` passed on the touched training files and new early-stopping test.
- `uvx ty check` passed on the touched training modules.
- New focused test passed:
  `uv run pytest tests/unit/test_eres2netv2_baseline.py::test_eres2netv2_baseline_early_stopping_records_and_restores_best -q`.
- Existing full `tests/unit/test_eres2netv2_baseline.py` still has a pre-existing failure:
  the smoke test expects `slice_dashboard_path` and `error_analysis_*` fields on
  `WrittenVerificationEvaluationReport`, but the current report dataclass does not expose
  those attributes. This was not changed as part of the 20-epoch guarded run.

Launch:

- First local foreground validation with batch size `64` reached `epoch=1/20` but failed
  with CUDA OOM on the RTX 4090: the process used about `20.27 GiB`, with only
  `188.94 MiB` free, while trying to allocate another `134 MiB`.
- Config was adjusted to `training.batch_size=32` and
  `gradient_accumulation_steps=2` to preserve effective batch `64`.

```bash
RUN_ID=eres2netv2_e20_guarded_local_20260412T104013Z
mkdir -p artifacts/logs
setsid -f bash -lc 'cd /mnt/storage/Kryptonite-ML-Challenge-2026 && PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 uv run --group train python scripts/run_eres2netv2_baseline.py --config configs/training/eres2netv2-participants-20epoch-local-guarded.toml --device cuda --output json >> artifacts/logs/eres2netv2_e20_guarded_local_20260412T104013Z.log 2>&1'
pgrep -n -f "run_eres2netv2_baseline.py --config configs/training/eres2netv2-participants-20epoch-local-guarded.toml" \
  > artifacts/logs/${RUN_ID}.pid
printf '%s\n' "${RUN_ID}" > artifacts/logs/latest_eres2netv2_e20_guarded_local
```

Superseded failed detached attempt:

```bash
RUN_ID=eres2netv2_e20_guarded_local_20260412T104013Z
mkdir -p artifacts/logs
nohup bash -lc 'cd /mnt/storage/Kryptonite-ML-Challenge-2026 && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --group train python scripts/run_eres2netv2_baseline.py --config configs/training/eres2netv2-participants-20epoch-local-guarded.toml --device cuda --output json' \
  > artifacts/logs/${RUN_ID}.log 2>&1 &
echo $! > artifacts/logs/${RUN_ID}.pid
printf '%s\n' "${RUN_ID}" > artifacts/logs/latest_eres2netv2_e20_guarded_local
```

Aborted/mis-targeted artifacts:

- Run wrapper id: `eres2netv2_e20_guarded_local_20260412T104013Z`.
- Log path: `artifacts/logs/eres2netv2_e20_guarded_local_20260412T104013Z.log`.
- PID path: `artifacts/logs/eres2netv2_e20_guarded_local_20260412T104013Z.pid`.
- Output root prefix:
  `artifacts/baselines/eres2netv2-participants-20epoch-local-guarded/`.
- Tracking experiment: `eres2netv2-participants-20epoch-local-guarded`.
- No detached ERes2NetV2 run is active. A foreground validation attempt created the run
  root `artifacts/baselines/eres2netv2-participants-20epoch-local-guarded/20260412T104117Z-7fa4c946c8ec`
  before failing with CUDA OOM.

Decision:

- Aborted as the wrong target after clarification: the requested run is the original
  organizer baseline, not ERes2NetV2.
- Do not use this ERes2NetV2 section as an active experiment record.

## 2026-04-12 — Local Organizer Baseline 20-Epoch Early-Stopping Run

Hypothesis:

- The first organizer baseline may benefit from allowing up to 20 epochs if training is
  guarded by validation retrieval early stopping instead of always stopping at the
  original short schedule.
- Unlike the aborted ERes2NetV2 guard above, this run monitors the organizer baseline's
  speaker-disjoint validation `precision@10`, so it is a real overfitting guard.

Implementation:

- Updated `baseline/train.py` to support:
  - best checkpoint by configurable validation metric;
  - early stopping by validation metric with `min_delta`, `patience`, and `min_epochs`;
  - optional train-accuracy stop threshold;
  - optional `ReduceLROnPlateau`;
  - optional gradient clipping;
  - per-epoch `metrics.jsonl`;
  - final `training_summary.json`.
- Added config:
  `baseline/configs/participants_baseline_fixed_20epoch_earlystop.json`.
- Added `pandas==2.2.2` to the `train` dependency group because organizer
  `baseline/train.py` imports pandas and the repo-local `uv` environment did not have it.
- Added `faiss-cpu==1.13.2` to the `train` dependency group because organizer
  `baseline/src/metrics.py` imports `faiss` for validation retrieval metrics.

Config summary:

- Model family: original organizer ECAPA-style baseline in `baseline/src/model.py`.
- Train CSV: `datasets/Для участников/train.csv`.
- Split policy: organizer fixed speaker-disjoint split, `train_ratio=0.98`,
  `min_val_utts=11`, seed `2026`.
- Output dir: `artifacts/baseline_fixed_participants_e20_earlystop`.
- Device: local RTX 4090, `CUDA_VISIBLE_DEVICES=0`.
- Max epochs: `20`.
- Batch size: `256`.
- Optimizer: AdamW, LR `0.001`, weight decay `0.00001`.
- Audio policy: train chunk `3.0s`, validation chunk `6.0s`, sample rate `16000`.
- Validation metric: `precision@10`.
- Early stopping: metric `precision@10`, mode `max`, min delta `0.0005`, patience `3`,
  min epochs `6`, restore best checkpoint.
- Scheduler: ReduceLROnPlateau, factor `0.5`, patience `1`, threshold `0.0005`,
  min LR `0.00001`.
- Grad clip: `5.0`.

Local checks before launch:

- Config load check passed: `epochs=20`, `early_stopping_enabled=true`,
  `early_stopping_metric=precision@10`, scheduler `reduce_on_plateau`.
- `uv run ruff check baseline/train.py tests/unit/test_organizer_baseline_fixed.py`
  passed.
- `uv run pytest tests/unit/test_organizer_baseline_fixed.py -q` passed:
  `4 passed`.
- `PYTHONPATH=baseline uvx ty check baseline/train.py` passed.

Launch:

- First detached launch failed before training with
  `ModuleNotFoundError: No module named 'faiss'`. This was an environment gap against
  `baseline/requirements.txt`, not a model/training result. Added `faiss-cpu==1.13.2`
  and relaunched.

```bash
RUN_ID=organizer_baseline_e20_earlystop_local_20260412T104626Z
mkdir -p artifacts/logs
setsid -f bash -lc 'cd /mnt/storage/Kryptonite-ML-Challenge-2026 && PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --project /mnt/storage/Kryptonite-ML-Challenge-2026 --group train bash -lc "cd baseline && python train.py --config configs/participants_baseline_fixed_20epoch_earlystop.json" >> artifacts/logs/organizer_baseline_e20_earlystop_local_20260412T104626Z.log 2>&1'
pgrep -n -f "train.py --config configs/participants_baseline_fixed_20epoch_earlystop.json" \
  > artifacts/logs/${RUN_ID}.pid
printf '%s\n' "${RUN_ID}" > artifacts/logs/latest_organizer_baseline_e20_earlystop_local
```

Artifacts:

- Run wrapper id: `organizer_baseline_e20_earlystop_local_20260412T104626Z`.
- Training process status: stopped manually at user request after epoch `10`
  completed and while epoch `11` was in progress.
- Log path:
  `artifacts/logs/organizer_baseline_e20_earlystop_local_20260412T104626Z.log`.
- PID path:
  `artifacts/logs/organizer_baseline_e20_earlystop_local_20260412T104626Z.pid`.
- Model checkpoint:
  `artifacts/baseline_fixed_participants_e20_earlystop/model.pt`.
- Metrics JSONL:
  `artifacts/baseline_fixed_participants_e20_earlystop/metrics.jsonl`.
- Training summary, written after the manual interruption:
  `artifacts/baseline_fixed_participants_e20_earlystop/training_summary.json`.
- ONNX:
  `artifacts/baseline_fixed_participants_e20_earlystop/model_embeddings_epoch10_center_opset20.onnx`.
- Public embeddings:
  `artifacts/baseline_fixed_participants_e20_earlystop/test_public_emb_epoch10_center_opset20.npy`.
- Public submission:
  `artifacts/baseline_fixed_participants_e20_earlystop/submission_epoch10_center_opset20.csv`.
- Upload-friendly copy with the same SHA256:
  `artifacts/baseline_fixed_participants_e20_earlystop/submission.csv`.
- Submission validation:
  `artifacts/baseline_fixed_participants_e20_earlystop/submission_epoch10_center_opset20_validation.json`.
- Upload-copy validation:
  `artifacts/baseline_fixed_participants_e20_earlystop/submission_validation.json`.
- Public inference log:
  `artifacts/logs/organizer_baseline_e20_epoch10_public_center_infer_20260412.log`.

Monitor snapshot:

- 2026-04-12 10:48 UTC: run active on local RTX 4090, epoch `1/20`, around
  `64/2577` training batches processed, GPU utilization `100%`, memory used about
  `13.7 GiB`.
- 2026-04-12 11:42 UTC: run still active on local RTX 4090, training epoch `10/20`
  around the halfway point. `metrics.jsonl` has `9` completed epochs. Best validation
  `precision@10` is epoch `7`: `0.922089` with train loss `0.211063` and train accuracy
  `0.945035`. Epoch `8` dropped to `0.918600`; epoch `9` recovered only to `0.921361`,
  which is not an improvement under `min_delta=0.0005`. Early-stopping bad epochs are
  now `2`; `ReduceLROnPlateau` lowered LR from `0.001` to `0.0005`. If epoch `10` does
  not exceed `0.922589`, the configured patience should stop the run after epoch `10`
  and keep the epoch-7 checkpoint.
- 2026-04-12 11:52 UTC: user requested stopping training and generating a public
  submission to measure Public LB. The training process was terminated during epoch
  `11/20` before epoch-11 validation/checkpointing. The last completed validation was
  epoch `10`, which became the best checkpoint:
  - epoch `10`;
  - validation `precision@10 = 0.928308`;
  - train loss `0.069959`;
  - train accuracy `0.980654`;
  - learning rate `0.0005`.
  The run summary was written manually from
  `artifacts/baseline_fixed_participants_e20_earlystop/metrics.jsonl` because the
  normal train-script finalizer does not execute after manual termination.

Stop command:

```bash
RUN_ID=$(cat artifacts/logs/latest_organizer_baseline_e20_earlystop_local)
PID=$(cat artifacts/logs/${RUN_ID}.pid)
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
kill -TERM -- -"$PGID"
```

Public submission generation:

```bash
cd /mnt/storage/Kryptonite-ML-Challenge-2026/baseline
uv run --project /mnt/storage/Kryptonite-ML-Challenge-2026 --group train \
  python convert_to_onnx.py \
    --config configs/participants_baseline_fixed_20epoch_earlystop.json \
    --pt ../artifacts/baseline_fixed_participants_e20_earlystop/model.pt \
    --out ../artifacts/baseline_fixed_participants_e20_earlystop/model_embeddings_epoch10_center_opset20.onnx \
    --opset 20

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
uv run --project /mnt/storage/Kryptonite-ML-Challenge-2026 --group train \
  python inference_onnx.py \
    --onnx_path ../artifacts/baseline_fixed_participants_e20_earlystop/model_embeddings_epoch10_center_opset20.onnx \
    --csv "../datasets/Для участников/test_public.csv" \
    --data_base_dir "../datasets/Для участников" \
    --output_emb ../artifacts/baseline_fixed_participants_e20_earlystop/test_public_emb_epoch10_center_opset20.npy \
    --output_indices ../artifacts/baseline_fixed_participants_e20_earlystop/submission_epoch10_center_opset20.csv \
    --batch_size 64 \
    --num_workers 8 \
    --sample_rate 16000 \
    --chunk_seconds 6.0 \
    --num_crops 1 \
    --device cuda
```

Public inference result:

- ONNX Runtime provider: `CUDAExecutionProvider`.
- Public rows embedded: `134697`.
- Embedding dimension: `192`.
- Inference plus FAISS indexing wall time: `135.241s`.
- Submission SHA256:
  `a6e2428590e909d132f84deb3535cfe874a36ee58c9f73493e45b477afb3896a`.
  The same hash applies to
  `artifacts/baseline_fixed_participants_e20_earlystop/submission.csv`.
- ONNX SHA256:
  `fd66fac9bdab090bfa050727d11e0ba763166cc6ba6eb074f53c10a7269f0ca2`.
- Embeddings SHA256:
  `d9b0de9495c792fdf15a1faaf181edf4bcc88e708b9e28fd8f41f387cbd73398`.

Submission validation:

```bash
uv run python scripts/validate_submission.py \
  --template-csv "datasets/Для участников/test_public.csv" \
  --submission-csv artifacts/baseline_fixed_participants_e20_earlystop/submission_epoch10_center_opset20.csv \
  --output-json artifacts/baseline_fixed_participants_e20_earlystop/submission_epoch10_center_opset20_validation.json \
  --k 10
```

- Validator status: `passed=True`, `errors=0`.
- Public LB score from external upload: `0.1046`.
- Public LB deltas:
  - `+0.0267` vs original organizer baseline `0.0779`;
  - `+0.0022` vs `baseline_fixed_participants` `0.1024`;
  - `-0.0203` vs C4 graph branch `0.1249`;
  - `-0.1364` vs current P1 ERes2NetV2 best `0.2410`.

Decision:

- Rejected as a dead-end branch. The gap between local validation `0.928308` and public
  LB `0.1046` confirms that longer guarded training of the original organizer baseline
  mostly over-optimizes the local split/domain and does not solve the public retrieval
  failure.
- Do not continue this baseline family as a production path. Keep only as a diagnostic
  control for sanity checks against `baseline_fixed_participants`.

## 2026-04-12 - Track 2 Pseudo-Label Augmentation Fine-Tune

Experiment id: `P4_eres2netv2_g6_pseudo_track2_aug_e70`

Hypothesis:

- The current public best is `P3_eres2netv2_g6_pseudo_ft_public_c4` with public LB
  `0.2861`, so pseudo-label self-training is confirmed useful. The next orthogonal
  hypothesis is to keep the P3 pseudo-label branch and reduce the measured public-domain
  gap with a production training augmentation package.
- A1 must-have augmentation: real noise/music/babble-style noise bank, RIR convolution,
  and speed perturbation.
- A2 test-matching augmentation: channel/codec-like band limiting and quantization,
  random EQ, far-field attenuation, trailing silence, inserted pauses, and random
  VAD-drop. This targets the EDA finding that public test has less high-frequency energy
  and more pauses.

Code/config changes:

- Added production scheduler/runtime wiring in `ManifestSpeakerDataset` and
  `build_production_train_dataloader`, so scheduled waveform augmentations are applied
  before random crop and fbank extraction.
- Added direct raw MUSAN and RIRS_NOISES fallbacks to the augmentation runtime. Training
  only registers noise/RIR candidates whose audio files exist, so arm11 downloads are now
  used directly instead of relying on prebuilt artifact banks.
- Fixed `scripts/download_datasets.py` so downloads executed with `cwd=datasets` write
  archives as `musan.tar.gz` / `rirs_noises.zip`, not `datasets/<archive>` from inside the
  `datasets` directory.
- Switched MUSAN primary download to the full Hugging Face LFS zip mirror
  `thusinh1969/musan` after OpenSLR/trmal stalled at only tens of kilobytes. Switched
  RIRS_NOISES primary download to the Hugging Face zip mirror `EaseZh/rirs_noises`, with
  OpenSLR zip URLs retained as fallback.
- Updated RIRS_NOISES discovery to accept both `datasets/rirs_noises/RIRS_NOISES` and
  the actual extracted `datasets/RIRS_NOISES` layout; `scripts/download_datasets.py` now
  treats `datasets/RIRS_NOISES` as the downloaded directory.
- Optimized the production dataloader/scheduler for the large raw RIR catalog: sampler
  batches are yielded lazily instead of materializing a full epoch before batch 1, and
  augmentation candidate pools/cumulative weights are cached by family and severity.
- Added `configs/training/eres2netv2-g6-pseudo-track2-augment.toml`.
- The config explicitly opts into speed perturbation with
  `augmentation_scheduler.family_weights.speed=0.75`; older configs without this field
  keep speed disabled.

Training configuration:

- Config path: `configs/training/eres2netv2-g6-pseudo-track2-augment.toml`.
- Model family: ERes2NetV2, same architecture as P3, embedding size `192`.
- Initialization/provenance:
  `artifacts/baselines/eres2netv2-g6-pseudo-finetune/20260412T100738Z-6b686847f5d8/eres2netv2_encoder.pt`.
- Train manifest:
  `artifacts/manifests/pseudo_g6/g6_mixed_train_manifest.jsonl`.
- Dev manifest:
  `artifacts/manifests/participants_fixed/dev_manifest.jsonl`.
- Dataset/split version: original participant train plus filtered G6 public pseudo
  clusters; dev is `participants_fixed`.
- Seed: `42`.
- Batch size: `128`; eval batch size: `128`.
- Precision: `bf16`.
- Optimizer/scheduler: SGD momentum `0.9`, weight decay `0.00005`, cosine LR,
  initial LR `0.003`, min LR `0.000003`, warmup epochs `3`.
- Objective/loss: ArcMargin classifier, scale `32.0`, margin `0.2`, easy margin `false`.
- Crop/preprocessing: random train crop `2.0..6.0s`, one crop, eval chunks `6.0s`
  with `1.5s` overlap, VAD disabled.
- Augmentation policy: warmup `2` epochs, ramp `8` epochs, max `3` augmentations per
  sample, clean/light/medium/heavy ramps from `0.55/0.35/0.10/0.00` to
  `0.20/0.30/0.30/0.20`; family weights noise `1.20`, reverb `1.00`, distance `0.95`,
  codec `1.10`, silence `0.85`, speed `0.75`.
- Early stopping: `max_epochs=70` as an upper cap, `early_stopping_enabled=true`,
  monitor `train_loss`, `min_delta=0.0005`, patience `8`, min epochs `12`,
  restore best state `true`, train-accuracy hard stop `0.9975`.
- GPU/device assignment: `arm11`, container `MK_RND`, `CUDA_VISIBLE_DEVICES=0`.
- Container/environment: `/jupyter/kleshchenok/audio/embbedings`, `uv sync --dev --group train`.
- Local validation design: final pipeline dev scoring after training. Per-epoch dev is
  not currently part of this production loop; early stopping uses persisted per-epoch
  training metrics, and the final checkpoint is scored on `participants_fixed` dev.

Pre-launch scheduler smoke:

- Local command:

```bash
uv run python - <<'PY'
from pathlib import Path
from kryptonite.training.eres2netv2 import load_eres2netv2_baseline_config
from kryptonite.training.augmentation_scheduler import build_augmentation_scheduler_report
cfg = load_eres2netv2_baseline_config(
    config_path=Path("configs/training/eres2netv2-g6-pseudo-track2-augment.toml")
)
report = build_augmentation_scheduler_report(
    project_root=Path("."),
    scheduler_config=cfg.project.augmentation_scheduler,
    silence_config=cfg.project.silence_augmentation,
    total_epochs=cfg.project.training.max_epochs,
    samples_per_epoch=256,
    seed=cfg.project.runtime.seed,
)
print(report.catalog.candidate_counts_by_family)
print(report.summary.missing_families)
PY
```

- Candidate counts after direct raw MUSAN/RIRS fallback: noise `2859`, reverb `60437`,
  distance `3`, codec `7`, silence `3`,
  speed `4`.
- Missing families: none.

Remote launch records:

- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T144522Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T144522Z.log`.
- Result: failed before training. The download wrapper attempted to save to
  `datasets/musan.tar.gz` while running with `cwd=datasets`, so `wget` exited with
  `datasets/musan.tar.gz: No such file or directory`. No GPU training started.
- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T145307Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T145307Z.log`.
- Result: failed before training for the same downloader path issue because the remote
  script had not yet received the fixed `scripts/download_datasets.py`. No GPU training
  started.
- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T145511Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T145511Z.log`.
- Result: stopped before training. It used the fixed path handling, but the OpenSLR/trmal
  MUSAN transfer stalled at roughly `70K`; no GPU training started.
- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T150004Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T150004Z.log`.
- Result: downloaded and extracted MUSAN and RIRS_NOISES successfully, then failed at the
  augmentation smoke before training because RIRS extracted to `datasets/RIRS_NOISES`
  while the first runtime lookup only checked `datasets/rirs_noises/RIRS_NOISES`.
- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T151236Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T151236Z.log`.
- Result: stopped before batch 1 after the RIRS path fix. The train process entered epoch
  1 but spent the startup window materializing all `5814` batches and repeatedly filtering
  the large RIR catalog. This exposed a scheduler/dataloader performance bug; no useful
  training metrics were produced.
- Remote run id: `eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z`.
- Remote log path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z.log`.
- Remote pid path:
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z.pid`.
- Latest pointer:
  `artifacts/logs/latest_eres2netv2_g6_pseudo_track2_aug_e70`.
- Output root:
  `artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/<tracking-run-id>/`.

Launch command:

```bash
ssh arm11 'docker exec -i MK_RND bash' <<'REMOTE'
set -euo pipefail
cd /jupyter/kleshchenok/audio/embbedings
RUN_ID=eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z
mkdir -p artifacts/logs datasets
printf '%s\n' "$RUN_ID" > artifacts/logs/latest_eres2netv2_g6_pseudo_track2_aug_e70
cat > "/tmp/${RUN_ID}.sh" <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
cd /jupyter/kleshchenok/audio/embbedings
RUN_ID=eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z
uv sync --dev --group train
uv run python scripts/download_datasets.py --dataset musan
uv run python scripts/download_datasets.py --dataset rirs-noises
uv run python - <<'PY'
from pathlib import Path
from kryptonite.training.eres2netv2 import load_eres2netv2_baseline_config
from kryptonite.training.augmentation_runtime import TrainingAugmentationRuntime
cfg = load_eres2netv2_baseline_config(
    config_path=Path("configs/training/eres2netv2-g6-pseudo-track2-augment.toml")
)
runtime = TrainingAugmentationRuntime.from_project_config(
    project_root=Path("."),
    scheduler_config=cfg.project.augmentation_scheduler,
    silence_config=cfg.project.silence_augmentation,
    total_epochs=cfg.project.training.max_epochs,
)
counts = runtime.catalog.candidate_counts_by_family
print(counts, flush=True)
required = {"noise", "reverb", "codec", "silence", "speed"}
missing = sorted(required.difference(counts))
if missing:
    raise SystemExit(f"missing augmentation families: {missing}")
PY
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_eres2netv2_finetune.py \
  --config configs/training/eres2netv2-g6-pseudo-track2-augment.toml \
  --init-checkpoint artifacts/baselines/eres2netv2-g6-pseudo-finetune/20260412T100738Z-6b686847f5d8/eres2netv2_encoder.pt \
  --device cuda \
  --output json
JOB
chmod +x "/tmp/${RUN_ID}.sh"
nohup "/tmp/${RUN_ID}.sh" > "artifacts/logs/${RUN_ID}.log" 2>&1 &
echo $! > "artifacts/logs/${RUN_ID}.pid"
REMOTE
```

Status:

- `eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z` launched detached on arm11
  GPU0. Downloads are complete and reused: `datasets/musan` and `datasets/RIRS_NOISES`.
  Remote smoke counts: noise `2016`, reverb `60417`, distance `3`, codec `7`,
  silence `3`, speed `4`; missing families none. As of `2026-04-12T15:18:04Z`, training
  reached `epoch=1/70 batch=1/5814`, batch-1 loss `17.890003`, accuracy `0.000000`,
  throughput `33.8` examples/s, and GPU0 was active at roughly `74905/81559 MiB` and
  `100%` utilization.
- Hourly monitor started with pid path
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z.hourly_monitor.pid`
  and monitor log
  `artifacts/logs/eres2netv2_g6_pseudo_track2_aug_e70_20260412T151705Z.hourly_monitor.log`.
  First monitor snapshot at `2026-04-12T17:10:08Z`: training was running, epoch 1
  completed with train loss `10.206978` and accuracy `0.416613`; epoch 2 had reached
  `batch=4640/5814` (`79.8%`), train loss `3.319962`, accuracy `0.878155`, throughput
  about `201.7` examples/s, GPU0 `76789/81559 MiB`, `100%` utilization.
- Final training outcome checked on `2026-04-13T03:53:47Z`: early stopping fired at
  epoch `12` with reason `patience_exhausted`; best epoch was `4`, best train loss
  `2.519065`, and `restore_best=true` restored the best checkpoint before writing.
  Final epoch 12 train loss was `2.965722` and train accuracy `0.873003`.
- Training artifacts:
  checkpoint
  `artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/20260412T151747Z-b522b570a1b7/eres2netv2_encoder.pt`;
  training summary
  `artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/20260412T151747Z-b522b570a1b7/training_summary.json`;
  dev embeddings
  `artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/20260412T151747Z-b522b570a1b7/dev_embeddings.npz`.
- The training wrapper was manually stopped after checkpoint/summary/dev embeddings were
  written because the generic baseline pipeline moved into all-pairs dev trial generation:
  this config had `trials_manifest=""`, and `participants_fixed` dev has `13473` rows.
  That scorer path would materialize an impractically large in-memory trial list and is
  not the intended public-submission evaluation path for this branch. No `score_summary`
  or verification report was produced by this run.
- GPU0 was freed after stopping the post-train scorer tail; `nvidia-smi` showed
  `0 MiB` and `0%` utilization on both GPUs.

P4 public C4 tail launch:

- Purpose: create a leaderboard submission from the Track 2 augmentation checkpoint using
  the same public C4 tail as the current P3 best (`top-cache-k=200`, `3` crops,
  `6.0s`, no synthetic shift), so the LB comparison is direct.
- Run id: `p4_eres2netv2_track2_aug_public_c4_20260413T035400Z`.
- GPU: `CUDA_VISIBLE_DEVICES=0`.
- Log:
  `artifacts/logs/p4_eres2netv2_track2_aug_public_c4_20260413T035400Z.log`.
- PID file:
  `artifacts/logs/p4_eres2netv2_track2_aug_public_c4_20260413T035400Z.pid`.
- Checkpoint:
  `artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/20260412T151747Z-b522b570a1b7/eres2netv2_encoder.pt`.
- Output dir:
  `artifacts/backbone_public/eres2netv2_track2_aug/20260412T151747Z-b522b570a1b7/`.
- Exact launch command:

```bash
cd /jupyter/kleshchenok/audio/embbedings
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
uv run --group train python scripts/run_torch_checkpoint_c4_tail.py \
  --model eres2netv2 \
  --checkpoint-path artifacts/baselines/eres2netv2-g6-pseudo-track2-augment/20260412T151747Z-b522b570a1b7/eres2netv2_encoder.pt \
  --manifest-csv artifacts/eda/backbone_public/test_public_manifest.remote.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/eres2netv2_track2_aug/20260412T151747Z-b522b570a1b7 \
  --experiment-id P4_eres2netv2_g6_pseudo_track2_aug_public_c4 \
  --shift-mode none \
  --device cuda \
  --search-device cuda \
  --precision bf16 \
  --batch-size 256 \
  --search-batch-size 4096 \
  --top-cache-k 200 \
  --crop-seconds 6.0 \
  --n-crops 3
```

- Initial monitor: embedding extraction started on `134697` public rows; GPU0 used
  about `21945/81559 MiB` with `100%` utilization.

## 2026-04-12 — Public Gap Diagnostic: Order Leak And Validation Mismatch

Context:

- User reported that the public leaderboard has reached about `0.71`, while current best
  local branch `P3_eres2netv2_g6_pseudo_ft_public_c4` is only `0.2861`.
- Goal: check whether the gap is caused by a submission/id bug, a public CSV ordering
  leak, or a deeper representation/validation problem.

Checks:

- Submission format/id path:
  - All scored public submissions checked locally have validator pass status:
    `134697/134697` rows, `K=10`, no duplicate/self/out-of-range neighbor indices.
  - `test_public.csv` is exactly `test_public/000000.flac` through
    `test_public/134696.flac` in row order, so neighbor integers are row indices.
- Train-order leak:
  - `train.csv` is fully grouped by speaker in row order: `11053` consecutive speaker
    runs for `11053` speaker ids.
  - On train order alone, simple neighbor heuristics would score very high:
    forward `i+1..i+10` gives `P@10 ~= 0.912`; symmetric `i±1..i±5` gives
    `P@10 ~= 0.951`.
- Public-order leak probe:
  - Generated diagnostic public submissions under `artifacts/diagnostics/order_leak/`:
    `submission_order_forward10.csv`, `submission_order_backward10.csv`, and
    `submission_order_symmetric5.csv`.
  - All three pass `scripts/validate_submission.py`.
  - Public audio statistics do not show row adjacency structure: lag-1 correlations for
    duration/RMS/peak/silence/rolloff/centroid are approximately `0`.
  - P1 public embeddings also do not show row adjacency structure: lag-1/2/5/10 cosine
    distributions are effectively the same as random pairs. Example P1 lag-1 mean
    `0.1949`, random mean `0.1967`; lag-1 p50 `0.1459`, random p50 `0.1474`.
  - Existing P1/P3/E1 submissions almost never select nearby row indices:
    P3 has only `0.000154` of neighbor slots within `±10` and `0.000756` within `±50`.

Interpretation:

- There is no evidence of a simple public row-order leak, despite train being grouped.
  The order-only submissions are kept as optional cheap LB probes only because they pass
  validation and directly test the hypothesis.
- No local evidence points to a submission row/id bug.
- Follow-up public probe: direct pretrained H1 WavLM scored only `0.1228`, so raw
  off-the-shelf WavLM embeddings also do not explain the `0.71` leaderboard gap.
- Follow-up user report: default ModelScope CAM++ VoxCeleb model
  `iic/speech_campplus_sv_en_voxceleb_16k` scored `0.5695` without challenge
  fine-tuning. This sharply points to backbone/provenance rather than submission format
  as the main gap.
- The main gap is more likely that the current representation is too weak for hidden
  public clustering and current local validation is not a faithful public proxy. The
  strongest immediate direction is domain-adapted/pretrained speaker-recognition
  backbones or stronger pseudo-label/fusion over public embeddings, not raw pretrained
  inference and not more trust in train-derived speaker-disjoint validation.

## 2026-04-12 — Local Submission Audit After ModelScope CAM++ Result

Context:

- User reported `0.5695` for default ModelScope CAM++ VoxCeleb and asked to run local
  submission checks because the issue might be how `submission.csv` is built.

Local checks:

- Searched local workspace for ModelScope/CAM++ submission artifacts. No
  ModelScope-named submission artifact is present locally yet.
- Audited all local public submission CSVs under `artifacts/`:
  `34` true submission files found, `34/34` passed `validate_submission()` against
  `datasets/Для участников/test_public.csv`.
- Audit report:
  `artifacts/diagnostics/submission_audit/all_public_submission_validation.json`.
- Writer/validator smoke:
  - `write_submission()` writes `neighbours` as one CSV field, e.g.
    `test_public/000000.flac,"1,2,3,4,5,6,7,8,9,10"`.
  - Validator rejects duplicate and self-match rows.
  - Smoke report:
    `artifacts/diagnostics/submission_audit/writer_validator_smoke.json`.
- Local metric semantics:
  - On grouped `train.csv`, manual order-neighbor `P@10` equals baseline
    `precision_at_k_from_indices()` result:
    manual `0.9512352271056341`, metric `0.951235227105634`.
  - Parser roundtrip shape for a local submission subset: `[1000, 10]`.
  - Smoke report:
    `artifacts/diagnostics/submission_audit/local_metric_semantics_smoke.json`.
- Key command checks:
  - `uv run pytest tests/unit/test_eda.py::test_submission_validator_checks_paths_and_neighbours tests/unit/test_organizer_baseline_fixed.py::test_calc_metrics_validates_template_order_and_index_bounds tests/unit/test_embedding_scoring.py -q`
    passed: `6 passed`.
  - A broader pytest command including `tests/unit/test_submission_bundle.py` failed
    during collection because `kryptonite.serve` does not currently export
    `build_submission_bundle`; this is unrelated to public submission CSV creation.

Conclusion:

- Local evidence does not support a generic submission-format or off-by-one bug in our
  public CSV writer/validator/metric path.
- The ModelScope CAM++ result should be treated as a backbone quality/provenance signal.
  Next action is to bring the ModelScope submission/embeddings into the repo, validate
  the exact CSV locally, then use those embeddings for C4/cluster/fusion/pseudo-label
  runs.

## 2026-04-12 — ModelScope CAM++ Local Reproduction And CSV Comparison

Context:

- User provided the scored default ModelScope CAM++ submission:
  `artifacts/backbone_public/campp/default_model_submission.csv`, public LB `0.5695`.
- Goal: run the same ModelScope CAM++ VoxCeleb checkpoint locally through the current repo
  inference path and compare the produced ranking against the scored CSV.

Checkpoint preparation:

- Source ModelScope checkpoint:
  `artifacts/modelscope_cache/iic/speech_campplus_sv_en_voxceleb_16k/campplus_voxceleb.bin`.
- Local converted checkpoint:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt`.
- Conversion was a key remap only; tensor values were unchanged. The ModelScope
  `state_dict` and local `CAMPPlusEncoder` both contain `937` keys with matching tensor
  shapes after remap.
- Conversion report:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/conversion_summary.json`.

Local run:

```bash
uv run python scripts/run_torch_checkpoint_c4_tail.py \
  --model campp \
  --checkpoint-path artifacts/backbone_public/modelscope_campplus_voxceleb_default/converted/modelscope_campplus_voxceleb_encoder.pt \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms2_c4 \
  --experiment-id MS2_modelscope_campplus_voxceleb_default_public_c4 \
  --device cuda \
  --search-device cuda \
  --precision bf16 \
  --batch-size 512 \
  --search-batch-size 2048 \
  --top-cache-k 100 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --trim \
  --shift-mode none
```

Outputs:

- C4 submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms2_c4/submission_MS2_modelscope_campplus_voxceleb_default_public_c4.csv`
- Exact-only submission from the same local embeddings:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms2_c4/submission_MS2_modelscope_campplus_voxceleb_default_public_exact.csv`
- Comparison report:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms2_c4/default_vs_local_comparison.json`

Validation and runtime:

- MS1 scored CSV validation: passed, `0` errors.
- MS2 local exact validation: passed.
- MS2 local C4 validation: passed.
- Local extraction runtime on RTX 4090: `625.99` seconds for `134697` rows.
- Search runtime: `0.97` seconds. C4 rerank runtime: `7.01` seconds.

Comparison against the scored MS1 CSV:

| Compared files | First-row neighbors | Top-1 match | Mean overlap@10 | Median overlap@10 | Same ordered row share |
| --- | --- | ---: | ---: | ---: | ---: |
| MS1 scored vs MS2 local exact | MS1 `1437,24932,75809,37108,39021,124530,117542,76244,8574,90474`; local exact `37815,29484,39021,99711,21757,134289,12635,74352,76244,59815` | `15.86%` | `2.459/10` | `2` | `0.0%` |
| MS1 scored vs MS2 local C4 | MS1 same as above; local C4 `39021,134289,74352,59815,46821,37815,99711,29484,21757,76244` | `15.41%` | `2.510/10` | `2` | `0.0%` |
| MS2 local exact vs MS2 local C4 | local exact vs local C4 | `69.10%` | `6.610/10` | `7` | `0.21%` |

Additional frontend ablation:

- Run id: `MS3_modelscope_campplus_voxceleb_default_public_notrim_1crop_c4`.
- Command delta vs MS2: `--no-trim --n-crops 1 --crop-seconds 6.0`.
- Output directory:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms3_notrim_1crop`.
- Comparison report:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms3_notrim_1crop/default_vs_ms3_notrim_1crop_comparison.json`.

| Compared files | First-row neighbors | Top-1 match | Mean overlap@10 | Median overlap@10 | Same ordered row share |
| --- | --- | ---: | ---: | ---: | ---: |
| MS1 scored vs MS3 no-trim exact | MS1 same as above; MS3 exact `21757,59815,37815,76599,126291,125785,27220,803,18136,1436` | `11.92%` | `2.075/10` | `1` | `0.0%` |
| MS1 scored vs MS3 no-trim C4 | MS1 same as above; MS3 C4 `21757,30197,18136,36063,13519,113047,76599,37815,27220,99711` | `11.90%` | `2.190/10` | `1` | `0.0%` |

Hubness:

- MS1 scored CSV: Gini@10 `0.4917`, max in-degree `214`.
- MS2 local exact: Gini@10 `0.4928`, max in-degree `159`.
- MS2 local C4: Gini@10 `0.3397`, max in-degree `64`.

Embedding diagnostics:

- Local MS2 embeddings path:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms2_c4/embeddings_MS2_modelscope_campplus_voxceleb_default_public_c4.npy`.
- Local MS3 embeddings path:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/20260412T_local_ms3_notrim_1crop/embeddings_MS3_modelscope_campplus_voxceleb_default_public_notrim_1crop_c4.npy`.
- Both local embedding arrays are `float32`, shape `[134697, 512]`, contain `0` NaNs,
  and are L2-normalized with norm p50 `1.0`.
- Row-wise cosine between MS2 and MS3 embeddings: mean `0.9469`, p50 `0.9690`, p05
  `0.8440`, p95 `1.0`. This means the local trim/crop ablation changes embeddings
  moderately but does not explain the much larger ranking mismatch against MS1.
- Embedding diagnostic report:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/local_embedding_diagnostics.json`.

Conclusion:

- The scored MS1 CSV and the local MS2 run are format-compatible and share the same
  checkpoint weights, but their neighbor rankings differ materially.
- This rules out a generic submission writer problem for this comparison. MS3 also shows
  that simply disabling trim and switching to one center crop does not recover the scored
  ranking.
- The gap is in the inference/frontend policy: official/default ModelScope preprocessing,
  fbank details, segment/full-utterance behavior, or normalization differs from the repo's
  current local fbank path.
- The scored MS1 CSV should remain the safe public artifact until the exact ModelScope
  inference path is reproduced locally. MS2 is useful only as a diagnostic and should not
  be submitted as a replacement without further frontend parity work.

## 2026-04-12 — Official GitHub CAM++ Reproduction

Context:

- User shared the exact code repo used for the `0.5695` submission:
  `https://github.com/RustamOper05/kryptonite_tembr_research`.
- Access was verified through `gh`; repo is private and default branch is `master`.
- Clone path for inspection:
  `/tmp/kryptonite_tembr_research`.

Relevant code path:

- Submission builder:
  `/tmp/kryptonite_tembr_research/code/campp/build_submission.py`.
- Retrieval/frontend:
  `/tmp/kryptonite_tembr_research/code/campp/retrieval.py`.
- Model construction:
  `/tmp/kryptonite_tembr_research/code/campp/common.py`.
- Config:
  `/tmp/kryptonite_tembr_research/code/campp/configs/campp_en_ft.base.yaml`.

Important differences from this repository's local MS2/MS3 path:

- The GitHub repo uses official 3D-Speaker code at commit
  `065629c313eaf1a01c65c640c46d77e61e9607b4`:
  `speakerlab.models.campplus.DTDNN.CAMPPlus`.
- It extracts features with `torchaudio.compliance.kaldi.fbank(..., dither=0.0)` and
  applies utterance cepstral mean normalization:
  `features - features.mean(dim=0, keepdim=True)`.
- Its `segment_mean` policy uses 6s segments, repeats short clips, and for files longer
  than 6s averages 3 evenly spaced segment embeddings. It does not use this repository's
  conservative silence trim.
- This repository's failed reproduction used local `FbankExtractor` plus local
  `CAMPPlusEncoder`, so the same checkpoint weights saw materially different features.

Reproduction command:

```bash
PYTHONPATH=/tmp/kryptonite_tembr_research/code/campp \
uv run --with requests --with pandas --with pyarrow --with PyYAML --with soundfile --with scipy --with tqdm \
  python /tmp/kryptonite_tembr_research/code/campp/build_submission.py \
  --config /tmp/kryptonite_tembr_research/code/campp/configs/campp_en_ft.base.yaml \
  --mode segment_mean \
  --csv '/tmp/kryptonite_tembr_research/data/Для участников/test_public.csv' \
  --topk 10 \
  --run-name reproduction_pretrained_segment_mean_test_public \
  --save-embeddings
```

Copied artifacts:

- Reproduced submission:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/official_repo_reproduction_20260412T2250/submission.csv`.
- Official-repo embeddings:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/official_repo_reproduction_20260412T2250/embeddings.npy`.
- Comparison report:
  `artifacts/backbone_public/modelscope_campplus_voxceleb_default/official_repo_reproduction_20260412T2250/default_vs_official_repo_reproduction.json`.

Results:

- Reproduction validator: passed, `0` errors.
- Runtime: embedding `666.13s`, search `0.89s`, total `667.02s` on local RTX 4090.
- Reproduced first row exactly matches MS1:
  `1437,24932,75809,37108,39021,124530,117542,76244,8574,90474`.
- MS1 scored CSV vs official-repo reproduction:
  - top1 match `99.53%`;
  - ordered cell equality `96.19%`;
  - exact same ordered row share `81.22%`;
  - same neighbor set share `96.13%`;
  - mean overlap@10 `9.961/10`;
  - median overlap@10 `10`;
  - rows with full overlap@10: `129489/134697`.
- The small non-identical tail is consistent with dependency/runtime top-k tie or minor
  numeric differences; it is not a semantic mismatch.

Embedding comparison:

- Official-repo embeddings are `float32`, shape `[134697, 512]`, contain `0` NaNs, and
  are not stored L2-normalized: norm p50 `19.092`, min `10.697`, max `41.330`.
- Retrieval normalizes them inside `topk_indices_from_embeddings()`.
- Row-wise cosine between official-repo embeddings and local MS2 embeddings:
  mean `0.6114`, p50 `0.6315`, p05 `0.3067`, p95 `0.8499`.
- Row-wise cosine between official-repo embeddings and local MS3 embeddings:
  mean `0.6129`, p50 `0.6318`, p05 `0.3197`, p95 `0.8420`.

Conclusion:

- The `0.5695` result is reproducible from the GitHub code path.
- The issue is not submission formation. The issue is that this repository's local CAM++
  inference path produces different embeddings from the official 3D-Speaker/ModelScope
  frontend.
- For the ModelScope CAM++ branch, future work should import or faithfully reproduce the
  official frontend/model path before applying C4, fusion, pseudo-labeling, or fine-tuning.

## 2026-04-13 — Official Pretrained WavLM and ERes2Net-Large Probes

H6 WavLM official-HF no-trim public C4:

- Hypothesis: the earlier raw WavLM public score `0.1228` may have been hurt by this
  repository's silence trim policy rather than the pretrained model itself.
- Model: Hugging Face `microsoft/wavlm-base-plus-sv`; official Transformers class
  `AutoModelForAudioXVector`; feature frontend `AutoFeatureExtractor`.
- Command:

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 uv run --group train python scripts/run_hf_xvector_tail.py \
  --model-id microsoft/wavlm-base-plus-sv \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv 'datasets/Для участников/test_public.csv' \
  --output-dir artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4_20260413T \
  --experiment-id H6_wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4 \
  --batch-size 80 \
  --crop-seconds 6.0 \
  --n-crops 3 \
  --no-trim \
  --top-cache-k 200 \
  --search-batch-size 2048 \
  --search-device cuda
```

- Completed locally on RTX 4090. Embedding runtime `1257.17s`; search `0.999s`; rerank
  `6.579s`.
- Validator passed, `0` errors, `134697` rows.
- Submission:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4_20260413T/submission_H6_wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4.csv`.
- Short copy for upload:
  `artifacts/backbone_public/hf_xvector/submission_H6_wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4.csv`.
- Summary:
  `artifacts/backbone_public/hf_xvector/wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4_20260413T/H6_wavlm_base_plus_sv_pretrained_official_notrim_6s3c_public_c4_summary.json`.
- Diagnostics: `top1_score_mean=0.9646`, `top10_mean_score_mean=0.9590`,
  `label_used_share=0.7489`, Gini@10 `0.2321`, max in-degree `34`.
- Comparison against H1 trim run: mean overlap@10 `2.970`, median `3`, top1 equal
  `21.93%`. H6 is a materially different no-trim ranking, but public expectation remains
  low until LB is checked because H1 scored only `0.1228`.

H7 official 3D-Speaker ERes2Net-large public C4 launch:

- Hypothesis: a clean pretrained 3D-Speaker ERes2Net-large may transfer better than the
  from-scratch ERes2NetV2/CAM++ participant checkpoints and should be tested analogously
  to the successful ModelScope CAM++ default branch.
- Official source: 3D-Speaker repository, ERes2Net-large model id
  `iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k`; 3D-Speaker README lists
  ERes2Net-large as `22.46M` parameters and the official model id.
- arm11 preparation: cloned 3D-Speaker to `/tmp/3D-Speaker`; ModelScope downloaded
  `eres2net_large_model.ckpt` under
  `/root/.cache/modelscope/hub/models/iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/`.
- Decoder note: official `infer_sv_batch.py` uses `torchaudio.load`, but the arm11
  environment's torchaudio requires TorchCodec/FFmpeg libraries that are unavailable for
  FLAC decode. Smoke showed `soundfile` can read the same FLAC paths. Added
  `scripts/run_official_3dspeaker_eres2net_tail.py`, which keeps official 3D-Speaker
  ERes2Net architecture, weights, FBank, 10s circular chunking, and mean segment pooling,
  replacing only the broken audio decoder with `soundfile`.
- Smoke: 3-file extraction passed on arm11 GPU1 with the new runner; C4 smoke failed only
  because top-cache was intentionally too small for the full label-propagation config.
- Initial remote run id: `H7_eres2net_large_3dspeaker_pretrained_public_c4_20260413T0400Z` (stopped before 5% because batch `32` was too slow).
- Active remote run id: `H7b_eres2net_large_3dspeaker_pretrained_public_c4_b80_20260413T0408Z`.
- GPU assignment: `CUDA_VISIBLE_DEVICES=1` on `arm11` inside container `MK_RND`.
- Log:
  `artifacts/logs/H7b_eres2net_large_3dspeaker_pretrained_public_c4_b80_20260413T0408Z.log`.
- Output directory:
  `artifacts/backbone_public/official_3dspeaker_eres2net_large_20260413T/full_b80/`.
- Command:

```bash
cd /jupyter/kleshchenok/audio/embbedings
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1 uv run --with scipy \
  python scripts/run_official_3dspeaker_eres2net_tail.py \
  --checkpoint-path artifacts/modelscope_cache/official_3dspeaker/speech_eres2net_large_sv_zh-cn_3dspeaker_16k/eres2net_large_model.ckpt \
  --speakerlab-root /tmp/3D-Speaker \
  --manifest-csv artifacts/eda/participants_public_baseline/test_public_manifest.csv \
  --template-csv artifacts/links/participants_dataset/test_public.csv \
  --data-root artifacts/links/participants_dataset \
  --output-dir artifacts/backbone_public/official_3dspeaker_eres2net_large_20260413T/full_b80 \
  --experiment-id H7b_eres2net_large_3dspeaker_pretrained_public_c4_b80_20260413T0408Z \
  --device cuda \
  --search-device cuda \
  --batch-size 80 \
  --search-batch-size 2048 \
  --top-cache-k 200
```

- Status at launch: running; active batch-80 log line processed `1/134697` rows, GPU1 memory around `48.6 GiB` with `100%` utilization. Public score pending.
