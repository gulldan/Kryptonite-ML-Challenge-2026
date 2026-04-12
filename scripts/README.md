# Scripts

Тонкие CLI-обёртки над `src/kryptonite/`.

## Training

- `run_baseline.py` — обучить любую модель: `--model {campp,eres2netv2} --config ...`
- `run_eres2netv2_finetune.py` — дообучить ERes2NetV2 из существующего checkpoint
- `run_hf_xvector_finetune.py` — дообучить Hugging Face AudioXVector speaker model
- `run_campp_baseline.py` — CAM++ baseline
- `run_eres2netv2_baseline.py` — ERes2NetV2 baseline
- `run_campp_stage2_training.py` — CAM++ stage 2 (heavy augmentation)
- `run_campp_stage3_training.py` — CAM++ stage 3 (large-margin fine-tuning)
- `run_campp_sweep_shortlist.py` — sweep + ranking кандидатов stage 3
- `run_campp_model_selection.py` — выбор финального кандидата

## Evaluation

- `evaluate_verification_scores.py` — EER / minDCF из score файлов
- `calibrate_verification_thresholds.py` — калибровка порогов
- `build_verification_protocol.py` — snapshot verification protocol
- `build_final_benchmark_pack.py` — release benchmark pack
- `run_torch_checkpoint_c4_tail.py` — прогнать CAM++/ERes2NetV2 checkpoint через текущий C4 tail

## Data preparation

- `build_participant_training_manifests.py` — собрать training/dev manifests из participant split CSV
- `build_pseudo_label_manifests.py` — собрать pseudo-label manifests из public cluster assignments
- `acquire_ffsvc2022_surrogate.py` — скачать FFSVC2022 surrogate
- `prepare_ffsvc2022_surrogate.py` — собрать манифесты и сплиты
- `normalize_audio_dataset.py` — нормализация аудио (16 kHz mono)
- `validate_manifests.py` — валидация манифестов по схеме
- `build_noise_bank.py` — банк шумов для аугментации
- `build_rir_bank.py` — банк RIR для реверберации
- `build_codec_bank.py` — банк кодеков
- `build_far_field_bank.py` — банк far-field пресетов
- `build_corrupted_dev_suites.py` — corrupted dev evaluation suites
- `generate_demo_artifacts.py` — демо-данные для smoke тестов

## Export and serving

- `export_campp_onnx.py` — экспорт CAM++ в ONNX
- `export_boundary_report.py` — encoder input/output контракт
- `build_enrollment_cache.py` — enrollment cache для serving
- `build_tensorrt_fp16_engine.py` — TensorRT FP16 engine
- `build_submission_bundle.py` — финальный submission bundle

## Diagnostics

- `run_eda_profile.py` — offline EDA profile for participant data into Parquet/CSV/JSON
- `export_eda_csv_pack.py` — export a CSV-only analysis pack from EDA artifacts
- `run_eda_retrieval_eval.py` — local retrieval P@K from precomputed embeddings
- `run_public_graph_community.py` — собрать public graph/community submissions C1-C6 из cached embeddings
- `run_cluster_first_tail.py` — cluster-first transductive graph tail over cached embeddings/top-k
- `run_classifier_first_tail.py` — class-aware retrieval from cached embeddings and a trained classifier head
- `run_class_aware_graph_tail.py` — class-aware score adjustment before C4-style graph tail
- `run_hf_xvector_tail.py` — Hugging Face AudioXVector pretrained embeddings + graph tail
- `validate_submission.py` — validate challenge `submission.csv` format and row order
- `training_env_smoke.py` — проверка training environment
- `production_dataloader_smoke.py` — проверка dataloader
- `infer_smoke.py` — проверка inference runtime
- `show_config.py` — инспекция конфигов
- `repro_check.py` — проверка воспроизводимости

## Doc generators

- `build_system_architecture.py` — обновить system-architecture-v1.md
- `build_model_task_contract.py` — обновить model-task-contract
