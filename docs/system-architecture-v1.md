# System Architecture v1

`KVA-482` фиксирует одну repo-native архитектурную схему для Kryptonite speaker
stack: где заканчивается общий waveform frontend, где начинается encoder
boundary, какие модули владеют scoring/eval/serve, и в каких точках живут
tracking/telemetry.

## Decision

- primary path:
  `16 kHz mono -> optional loudness/VAD -> chunking -> 80-bin log-Mel/Fbank -> encoder_input -> compact encoder or feature_statistics fallback -> embedding -> cosine scoring -> optional AS-norm/TAS-norm offline -> threshold/calibration -> local/HTTP serve`
- export boundary mode: `encoder_only`
- current checked-in runtime backend: `feature_statistics`
- current tracking backend: `local`
- transport rule: FastAPI остаётся thin adapter поверх `Inferencer`

Эта заметка фиксирует системный уровень, а не только один doc по inference или
один doc по evaluation. Цель в том, чтобы следующий этап видел единый pipeline,
список модулей, интерфейсы, logging points и место для export/serve без reverse
engineering по десяткам файлов.

## Pipeline Diagram

```text
16 kHz mono audio
  -> optional bounded loudness normalization
  -> optional VAD trimming
  -> utterance chunking
  -> 80-bin log-Mel/Fbank
  -> encoder_input (BTF float32)
  -> compact encoder or feature_statistics runtime fallback
  -> embedding (BE float32)
  -> cosine scoring
  -> optional AS-norm/TAS-norm offline
  -> threshold calibration
  -> local Python / HTTP serve surfaces
```

## Pipeline Stages

### 1. Raw Audio Ingest

- owner modules:
  `src/kryptonite/data/audio_loader.py`,
  `src/kryptonite/data/audio_io.py`,
  `src/kryptonite/data/loudness.py`,
  `src/kryptonite/data/vad.py`,
  `src/kryptonite/data/normalization/`
- input contract: audio path, manifest row, or HTTP/local request payload
- output contract: normalized waveform plus trim/loudness metadata
- key rule: training и serving обязаны использовать один `AudioLoadRequest`

### 2. Feature Frontend

- owner modules:
  `src/kryptonite/features/chunking.py`,
  `src/kryptonite/features/fbank.py`,
  `src/kryptonite/features/cache.py`
- input contract: normalized waveform batches and chunking config
- output contract: `encoder_input (BTF, float32; batch=dynamic, frames=dynamic, mel_bins=80)`
- key rule: chunking/Fbank остаются runtime-owned и не входят в export graph

### 3. Encoder Runtime

- owner modules:
  `src/kryptonite/models/campp/model.py`,
  `src/kryptonite/models/eres2netv2/model.py`,
  `src/kryptonite/serve/inference_backend.py`
- input contract: `encoder_input`
- output contract: `embedding`
- key rule: архитектура уже готова к compact encoder boundary, но checked-in
  runtime пока честно живёт на `feature_statistics`

### 4. Enrollment And Cosine Scoring

- owner modules:
  `src/kryptonite/models/scoring.py`,
  `src/kryptonite/serve/scoring_service.py`,
  `src/kryptonite/serve/enrollment_cache.py`
- input contract: embedding batches or enrollment/probe matrices
- output contract: pooled enrollment centroids, pairwise scores, one-to-many scores
- key rule: cosine scorer единый для offline и runtime

### 5. Score Normalization

- owner modules:
  `src/kryptonite/eval/score_normalization.py`,
  `src/kryptonite/eval/as_norm.py`,
  `src/kryptonite/eval/tas_norm.py`,
  `src/kryptonite/eval/tas_norm_experiment.py`
- input contract: raw verification scores, embedding exports, metadata, cohort bank
- output contract: normalized scores and experiment summaries
- key rule: AS-norm/TAS-norm сейчас offline-only и не входят в live HTTP path

### 6. Evaluation And Calibration

- owner modules:
  `src/kryptonite/eval/verification_protocol.py`,
  `src/kryptonite/eval/verification_report.py`,
  `src/kryptonite/eval/verification_threshold_calibration.py`,
  `src/kryptonite/eval/verification_error_analysis/`
- input contract: trial manifests, score rows, slice metadata, candidate artifacts
- output contract: verification reports, slice breakdowns, threshold bundles
- key rule: verification остаётся first-class task mode

### 7. Serve And Transport

- owner modules:
  `src/kryptonite/serve/inferencer.py`,
  `src/kryptonite/serve/http.py`,
  `src/kryptonite/serve/api_models.py`,
  `src/kryptonite/serve/telemetry.py`
- input contract: audio paths or embeddings over local Python and JSON HTTP
- output contract: health metadata, embeddings, enrollments, scores, demo decisions
- key rule: transport thin, runtime fat

## Module Boundaries

### `data_contracts`

- responsibility: dataset manifests, audio ingestion, normalization policy, VAD,
  verification-trial materialization
- public entrypoints:
  `kryptonite.data.AudioLoadRequest`,
  `kryptonite.data.load_audio`,
  `kryptonite.data.load_manifest_audio`

### `feature_frontend`

- responsibility: chunking, Fbank extraction, feature cache/report helpers
- public entrypoints:
  `kryptonite.features.UtteranceChunkingRequest`,
  `kryptonite.features.FbankExtractionRequest`,
  `kryptonite.features.FbankExtractor`

### `model_family`

- responsibility: CAM++, ERes2NetV2 and shared cosine scoring helpers
- public entrypoints:
  `kryptonite.models.campp`,
  `kryptonite.models.eres2netv2`,
  `kryptonite.models.cosine_score_pairs`

### `training_recipes`

- responsibility: manifest-backed baseline training, augmentation runtime,
  optimization runtime, stage-specific pipelines
- public entrypoints:
  `kryptonite.training.speaker_baseline`,
  `kryptonite.training.campp.pipeline`,
  `kryptonite.training.eres2netv2.pipeline`

### `evaluation_suite`

- responsibility: verification protocol, score normalization, threshold
  calibration, release-oriented reports
- public entrypoints:
  `kryptonite.eval.verification_protocol`,
  `kryptonite.eval.verification_report`,
  `kryptonite.eval.verification_threshold_calibration`

### `serving_runtime`

- responsibility: unified inferencer, enrollment/scoring state, JSON HTTP
  transport, demo flows, runtime observability
- public entrypoints:
  `kryptonite.serve.Inferencer`,
  `kryptonite.serve.http.create_http_app`,
  `kryptonite.serve.telemetry.ServiceTelemetry`

### `export_and_deployment`

- responsibility: export boundary metadata, backend selection contract,
  model-bundle packaging, Triton/deployment handoff
- public entrypoints:
  `build_export_boundary_contract`,
  `build_inference_package_contract`,
  `build_triton_model_repository`

## Interfaces

### `audio_load_request`

- kind: `typed_python_contract`
- producer: `src/kryptonite/data/audio_loader.py`
- consumers:
  `src/kryptonite/training/speaker_baseline.py`,
  `src/kryptonite/serve/inferencer.py`
- role: одна точка правды для sample-rate/channel/normalization/VAD policy

### `fbank_frontend_request`

- kind: `typed_python_contract`
- producer:
  `src/kryptonite/features/chunking.py` +
  `src/kryptonite/features/fbank.py`
- consumers:
  `src/kryptonite/training/speaker_baseline.py`,
  `src/kryptonite/serve/inference_backend.py`
- role: единый frontend перед encoder boundary

### `encoder_export_boundary`

- kind: `machine_readable_contract`
- producer: `src/kryptonite/serve/export_boundary.py`
- consumers:
  `src/kryptonite/serve/inferencer.py`,
  `src/kryptonite/serve/triton_repository.py`,
  future ONNX/TensorRT tasks
- role:
  `encoder_input (BTF, float32; batch=dynamic, frames=dynamic, mel_bins=80) -> embedding (BE, float32; batch=dynamic, embedding_dim=160)`

### `cosine_scoring_api`

- kind: `shared_python_api`
- producer: `src/kryptonite/models/scoring.py`
- consumers:
  `src/kryptonite/training/speaker_baseline.py`,
  `src/kryptonite/serve/scoring_service.py`,
  `src/kryptonite/eval/score_normalization.py`
- role: pairwise and one-to-many scoring plus enrollment centroid pooling

### `inferencer_runtime_api`

- kind: `shared_python_api`
- producer: `src/kryptonite/serve/inferencer.py`
- consumers: HTTP adapter, deploy/demo smoke checks, local Python callers
- role: общий runtime surface для embed/enroll/verify/benchmark

### `http_transport_surface`

- kind: `json_http_api`
- producer: `src/kryptonite/serve/http.py`
- consumers: browser demo, integration smoke tests, external callers
- role:
  `/health`, `/metrics`, `/embed`, `/enroll`, `/verify`, `/score/*`, demo endpoints

## Logging Points

### `local_training_tracker`

- owner: `src/kryptonite/tracking.py`
- channel: `artifact_tracking`
- sinks: `artifacts/tracking/<run_id>/`
- payload:
  `run.json`, `params.json`, `metrics.jsonl`, `artifacts.json`

### `verification_reports`

- owner: `src/kryptonite/eval/verification_report.py`
- channel: `structured_artifacts`
- sinks: `artifacts/**` and curated `docs/`
- payload:
  `verification_eval_report.json`, `verification_threshold_calibration.json`, slice breakdowns

### `serve_json_logs_and_metrics`

- owner: `src/kryptonite/serve/telemetry.py`
- channel: `runtime_observability`
- sinks: stdout JSON logs and `/metrics`
- payload:
  `service`, `backend`, `implementation`, `model_version`, `latency_ms`,
  `audio_count`, `total_chunk_count`

### `health_runtime_metadata`

- owner: `src/kryptonite/serve/inferencer.py`
- channel: `runtime_metadata`
- sinks: `/health`, `/healthz`, `/readyz`
- payload:
  selected backend, model bundle metadata, export boundary summary, telemetry summary,
  enrollment-cache summary

## Export And Serve Placement

- exported subgraph: `encoder_input -> encoder forward -> embedding`
- runtime-owned pre-engine steps:
  `decode/resample/loudness/VAD/chunking/Fbank`
- runtime-owned post-engine steps:
  `pool_chunk_embeddings`, `average_enrollment_embeddings`,
  `normalize_for_scoring`, `cosine scoring`
- runtime entrypoint:
  `src/kryptonite/serve/inferencer.py::Inferencer.from_config`
- HTTP entrypoint:
  `src/kryptonite/serve/http.py::create_http_app`
- backend fallback chain: `tensorrt -> onnxruntime -> torch`
- current validated backends in checked-in defaults:
  `torch=true`, `onnxruntime=false`, `tensorrt=false`

## Expected Artifacts

- `docs/system-architecture-v1.md`
- `artifacts/system-architecture/system_architecture.json`
- `artifacts/system-architecture/system_architecture.md`
- `artifacts/export-boundary/export_boundary.json`
- `artifacts/model-task-contract/model_task_contract.json`

## Limitations

- checked-in runtime path всё ещё доказывает архитектурную целостность через
  `feature_statistics`, а не через финальный exported encoder
- AS-norm/TAS-norm не productized для HTTP runtime
- waveform frontend intentionally outside ONNX/TensorRT graph
- tracking local-first; внешние tracking adapters пока не first-class

## Supporting References

- `docs/model-task-contract.md`
- `docs/audio-loader.md`
- `docs/audio-fbank-extraction.md`
- `docs/embedding-scoring.md`
- `docs/evaluation-package.md`
- `docs/threshold-calibration.md`
- `docs/export-boundary.md`
- `docs/unified-inference-wrapper.md`
- `docs/inference-observability.md`
- `docs/tracking.md`

## Rebuild

```bash
uv run python scripts/build_system_architecture.py --config configs/base.toml
```

Команда пересобирает локальные артефакты:

- `artifacts/system-architecture/system_architecture.json`
- `artifacts/system-architecture/system_architecture.md`
