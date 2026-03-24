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
- `scripts/normalize_audio_dataset.py` for deterministic 16 kHz mono normalization, manifest rewriting, and quarantine reporting
- `scripts/prepare_ffsvc2022_surrogate.py` for building manifests, quarantine lists, trials, speaker-disjoint splits, and checksum inventory from the surrogate bundle
- `scripts/vad_trimming_report.py` for comparing `none`, `light`, and `aggressive` loader-time trimming on a manifest-backed dev split
- `scripts/validate_manifests.py` for enforcing the versioned unified manifest schema on data manifests
- `scripts/infer_smoke.py` for inference-runtime and API startup smoke validation
- `scripts/repro_check.py` for reproducibility smoke validation
- `scripts/show_config.py` for config inspection and overrides
- `scripts/training_env_smoke.py` for training-environment import checks
- `scripts/tracking_smoke.py` for local tracking smoke validation
