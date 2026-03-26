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
- `scripts/build_manifest_embedding_atlas.py` for exporting baseline manifest-backed Fbank/stat embeddings and immediately rendering an interactive atlas from them
- `scripts/evaluate_verification_scores.py` for computing EER/minDCF and writing the full offline verification report, with optional `AS-norm` score normalization backed by exported eval embeddings and a frozen cohort bank
- `scripts/calibrate_verification_thresholds.py` for generating the full offline verification report plus named `balanced/min_dcf/demo/production` thresholds and optional slice-aware calibration bundles
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
- `scripts/repro_check.py` for reproducibility smoke validation
- `scripts/show_config.py` for config inspection and overrides
- `scripts/training_env_smoke.py` for training-environment import checks
- `scripts/tracking_smoke.py` for local tracking smoke validation
