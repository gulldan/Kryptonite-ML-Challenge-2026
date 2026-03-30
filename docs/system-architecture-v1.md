# System Architecture v1

`KVA-482` фиксирует верхнеуровневую архитектуру репозитория: где проходит канонический audio path,
какие модули считаются source of truth, и какой runtime сейчас реально обслуживает demo/API.

## Decision

- primary path:
  `16 kHz mono -> optional loudness/VAD -> chunking -> 80-bin log-Mel/Fbank -> encoder_input -> embedding -> cosine scoring -> thresholded decision`
- canonical serving adapter: FastAPI поверх `Inferencer`
- current checked-in runtime implementation: `feature_statistics`
- export boundary mode: `encoder_only`
- top-level repo split: reusable logic в `src/kryptonite/`, thin entrypoints в `apps/`

Этот документ теперь играет роль канонической архитектурной карты. Исторические deep-dive и старые узкоспециализированные заметки лежат в [archive/README.md](./archive/README.md).

## Pipeline Diagram

```text
raw audio
  -> shared loader
  -> optional loudness normalization
  -> optional VAD trimming
  -> utterance chunking
  -> 80-bin log-Mel / Fbank frontend
  -> embedding backend
  -> enrollment / cosine scoring
  -> threshold calibration
  -> HTTP + demo runtime surfaces
```

## Module Boundaries

- `src/kryptonite/data/`: ingestion, manifests, audio loading, normalization, VAD, metadata contracts.
- `src/kryptonite/features/`: chunking, Fbank extraction, feature cache helpers.
- `src/kryptonite/models/`: model families and shared cosine scoring utilities.
- `src/kryptonite/training/`: generic baseline pipeline (`run_speaker_baseline`), model-specific wrappers, staged training, optimization runtime.
- `src/kryptonite/eval/`: verification protocol, reports, calibration, score normalization.
- `src/kryptonite/serve/`: inferencer, enrollment state, HTTP surface, telemetry, deployment helpers.
- `apps/api/`: thin serving adapter.
- `apps/web/`: demo UI.

## Interfaces

- `audio_load_request`:
  one typed contract for sample rate, channels, normalization and VAD policy.
- `fbank_frontend_request`:
  one feature contract for training, offline eval and runtime.
- `encoder_export_boundary`:
  exported model starts at `encoder_input` and ends at `embedding`.
- `cosine_scoring_api`:
  shared scoring helpers for offline reports and live runtime.
- `inferencer_runtime_api`:
  common local-Python and HTTP serving surface.
- `http_transport_surface`:
  `/health`, `/metrics`, `/embed`, `/enroll`, `/verify`, `/score/*`, `/demo/api/*`.

## Logging Points

- `local_training_tracker`:
  local-first experiment metadata and artifacts under `artifacts/tracking/`.
- `verification_reports`:
  offline evaluation artifacts such as `verification_eval_report.json` and `verification_threshold_calibration.json`.
- `serve_json_logs_and_metrics`:
  structured runtime logs plus Prometheus metrics.
- `health_runtime_metadata`:
  `requested_backend`, `selected_backend`, model-bundle metadata, enrollment-cache status.

## What to read with this document

- [model-task-contract.md](./model-task-contract.md) — task semantics and required handoff artifacts.
- [code-architecture.md](./code-architecture.md) — how the repository code is split into stable module boundaries.
- [release-runbook.md](./release-runbook.md) — operational launch and troubleshooting path.
- [training.md](./training.md) — active training path and current family decision.
- [reference/audio-pipeline.md](./reference/audio-pipeline.md) — preprocessing and frontend contract.
- [archive/system-architecture-v1.md](./archive/system-architecture-v1.md) — previous verbose architecture note.

## Rebuild

```bash
uv run python scripts/build_system_architecture.py --config configs/base.toml
```
