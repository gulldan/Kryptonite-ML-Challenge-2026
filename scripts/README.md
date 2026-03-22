# Scripts

Keep reproducible command-line entrypoints here.

Scripts should be thin wrappers around reusable code in `src/kryptonite/`, not alternate implementations.

Current entrypoints include:

- `scripts/acquire_ffsvc2022_surrogate.py` for server-only FFSVC 2022 surrogate data acquisition
- `scripts/dataset_profile_report.py` for reproducible dataset profile JSON/Markdown reports from manifests
- `scripts/dataset_sync.py` for reproducible dataset/manifests sync and gpu-server readiness reporting
- `scripts/generate_demo_artifacts.py` for reproducible mini-demo dataset/manifests/model bundle generation
- `scripts/prepare_ffsvc2022_surrogate.py` for building manifests, trials, and speaker-disjoint splits from the surrogate bundle
- `scripts/infer_smoke.py` for inference-runtime and API startup smoke validation
- `scripts/repro_check.py` for reproducibility smoke validation
- `scripts/show_config.py` for config inspection and overrides
- `scripts/training_env_smoke.py` for training-environment import checks
- `scripts/tracking_smoke.py` for local tracking smoke validation
