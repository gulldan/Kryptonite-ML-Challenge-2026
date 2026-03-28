# Scripts

Keep reproducible command-line entrypoints here.

Scripts should be thin wrappers around reusable code in `src/kryptonite/`, not alternate implementations.

Current entrypoints include:

- `scripts/dataset_inventory_report.py` for the repository-level dataset-source policy report and local materialization audit
- `scripts/dataset_leakage_report.py` for reproducible duplicate/leakage/split-integrity audits from manifests
- `scripts/data_issues_backlog_report.py` for turning profile/leakage/audio-quality EDA into an actionable cleanup backlog
- `scripts/acquire_ffsvc2022_surrogate.py` for server-only FFSVC 2022 surrogate data acquisition
- `scripts/dataset_profile_report.py` for reproducible dataset profile JSON/Markdown reports from manifests
- `scripts/dataset_sync.py` for reproducible dataset/manifests sync and gpu-server readiness reporting
- `scripts/generate_demo_artifacts.py` for reproducible mini-demo dataset/manifests/model bundle generation, including CSV sidecars and checksum inventory
- `scripts/loudness_normalization_report.py` for comparing loader-time bounded RMS normalization against the raw waveform path
- `scripts/silence_augmentation_report.py` for waveform-level ablations of boundary padding, inserted pauses, and pause-ratio perturbation
- `scripts/augmentation_scheduler_report.py` for epoch-by-epoch curriculum coverage over clean/light/medium/heavy corruption mixes backed by the assembled banks
- `scripts/build_noise_bank.py` for assembling approved additive-noise corpora into one normalized noise bank with manifest and report artifacts
- `scripts/build_rir_bank.py` for assembling approved room impulse responses into a normalized RIR bank plus reusable room-simulation configs
- `scripts/build_codec_bank.py` for rendering deterministic FFmpeg-based codec/channel presets into preview audio plus manifest/report artifacts
- `scripts/build_far_field_bank.py` for rendering deterministic near/mid/far distance presets into preview audio, kernel controls, and manifest/report artifacts
- `scripts/build_corrupted_dev_suites.py` for freezing deterministic `dev_snr` / `dev_reverb` / `dev_codec` / `dev_distance` / `dev_channel` / `dev_silence` evaluation bundles from one clean dev manifest
- `scripts/build_embedding_atlas.py` for projecting precomputed embeddings into an interactive HTML atlas with nearest neighbors and optional media preview
- `scripts/build_cohort_embedding_bank.py` for freezing one normalized cohort/impostor embedding bank from exported embeddings plus metadata with explicit trial-exclusion and speaker-disjoint provenance
- `scripts/build_enrollment_cache.py` for precomputing one runtime-ready enrollment centroid cache from an enrollment manifest plus model-bundle compatibility metadata
- `scripts/build_manifest_embedding_atlas.py` for exporting baseline manifest-backed Fbank/stat embeddings and immediately rendering an interactive atlas from them
- `scripts/export_boundary_report.py` for writing the machine-readable `encoder_input -> embedding` contract plus a Markdown handoff for later ONNX/TensorRT work
- `scripts/export_campp_onnx.py` for exporting the frozen CAM++ checkpoint family into a real encoder-only ONNX model bundle with checker and ONNX Runtime smoke validation
- `scripts/build_verification_protocol.py` for freezing the repo-native clean dev bundles plus production-like corrupted suites into one auditable verification-protocol snapshot
- `scripts/evaluate_verification_scores.py` for computing EER/minDCF and writing the full offline verification report, with optional `AS-norm` score normalization backed by exported eval embeddings and a frozen cohort bank
- `scripts/calibrate_verification_thresholds.py` for generating the full offline verification report plus named `balanced/min_dcf/demo/production` thresholds and optional slice-aware calibration bundles
- `scripts/build_tas_norm_experiment.py` for running the repo-native `TAS-norm` go/no-go experiment on top of raw scores, `AS-norm`, and frozen cohort statistics
- `scripts/run_campp_baseline.py` for training the repo-native CAM++ baseline and writing checkpoints, dev embeddings, trials, and cosine scores
- `scripts/run_campp_sweep_shortlist.py` for running the bounded CAM++ stage-3 shortlist and ranking candidates on clean + corrupted dev suites
- `scripts/run_campp_model_selection.py` for selecting the final CAM++ stage-3 candidate from a shortlist report and evaluating optional checkpoint averages
- `scripts/run_eres2netv2_baseline.py` for training the repo-native ERes2NetV2 baseline and writing checkpoints, dev embeddings, trials, and cosine scores
- `scripts/production_dataloader_smoke.py` for inspecting the balanced/resumable production train dataloader against a real train manifest
- `scripts/feature_cache_report.py` for reproducible feature-cache materialization plus CPU/GPU benchmark and policy reports
- `scripts/normalize_audio_dataset.py` for deterministic 16 kHz mono normalization, manifest rewriting, and quarantine reporting
- `scripts/prepare_ffsvc2022_surrogate.py` for building manifests, quarantine lists, trials, speaker-disjoint splits, and checksum inventory from the surrogate bundle
- `scripts/vad_trimming_report.py` for comparing `none`, `light`, and `aggressive` loader-time trimming on a manifest-backed dev split
- `scripts/validate_manifests.py` for enforcing the versioned unified manifest schema on data manifests
- `scripts/infer_smoke.py` for inference-runtime and API startup smoke validation
- `scripts/build_triton_model_repository.py` for packaging the current encoder-boundary ONNX or TensorRT artifact into a Triton model repository with generated `config.pbtxt` and sample infer request
- `scripts/triton_infer_smoke.py` for probing a running Triton server through `/v2/models/<name>/infer` with the generated sample request
- `scripts/build_backend_benchmark_report.py` for freezing one config-driven PyTorch vs ONNX Runtime vs TensorRT benchmark report with latency graphs for batch=1 and batched workloads
- `scripts/inference_stress_report.py` for release-oriented stress validation across deterministic corrupted/extreme-duration inputs, malformed requests, and batch bursts
- `scripts/build_final_family_decision.py` for rendering the config-driven ADR that freezes the next export-target family and rejected alternatives
- `scripts/build_final_benchmark_pack.py` for building one self-contained release pack that copies frozen quality/stress/config artifacts and computes pairwise candidate comparisons
- `scripts/build_submission_bundle.py` for packaging the final handoff bundle with configs, model artifacts, docs, demo assets, checksums, and an optional `.tar.gz` archive
- `scripts/repro_check.py` for reproducibility smoke validation
- `scripts/show_config.py` for config inspection and overrides
- `scripts/training_env_smoke.py` for training-environment import checks
- `scripts/tracking_smoke.py` for local tracking smoke validation
