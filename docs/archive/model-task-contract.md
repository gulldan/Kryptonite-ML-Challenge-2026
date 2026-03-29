# Model Task Contract

`KVA-480` фиксирует один явный ADR для репозитория: базовая задача системы —
speaker verification, а closed-set и open-set identification остаются
совместимыми режимами поверх того же encoder/scoring контракта, но не являются
основным продуктовым интерфейсом на текущем этапе.

## Decision

- primary task mode: `verification`
- compatibility modes: `closed-set identification`, `open-set identification`
- canonical decision path: `enroll/test -> embeddings -> cosine score -> threshold -> decision`

Репозиторий должен в первую очередь оптимизировать, измерять и документировать
именно verification-сценарий. Любая совместимость с identification должна
сохраняться через общий embedding space и cosine scoring, а не через отдельный
семейство моделей или альтернативный preprocessing path.

## Canonical Workflow

1. Enrollment audio проходит общий audio contract: `16 kHz`, mono, нормализация,
   optional VAD, chunking и Fbank extraction.
2. Encoder получает `encoder_input` и возвращает `embedding`.
3. Enrollment embeddings нормализуются, усредняются и образуют speaker
   reference centroid.
4. Probe audio проходит тот же frontend, затем scoring считает cosine score
   против enrollment centroid.
5. Threshold profile превращает score в verification decision.

## Input And Output Contract

### Raw Audio Input

- target audio: `16000 Hz`, `1` channel, `wav` / `PCM16`
- loudness mode: `none` в `configs/base.toml`
- VAD mode: `none` в `configs/base.toml`
- training crop window: `1.0-4.0 s`
- eval chunk / overlap: `4.0 s` / `1.0 s`
- demo chunk / overlap: `4.0 s` / `1.0 s`

### Encoder Boundary

- export boundary: `encoder_only`
- encoder input: `encoder_input (BTF, float32; batch=dynamic, frames=dynamic, mel_bins=80)`
- encoder output: `embedding (BE, float32; batch=dynamic, embedding_dim=160)`
- scoring metric: `cosine_similarity`
- enrollment pooling: `l2_normalize_each_embedding -> average_embeddings -> l2_normalize_enrollment_centroid`

Эта граница должна оставаться согласованной с
[docs/export-boundary.md](./export-boundary.md), чтобы ONNX/TensorRT и runtime
не расходились по смыслу входов и выходов.

## Task Modes

### Verification

- status: `primary_implemented`
- request unit: `enrollment_id + probe audio or embeddings`
- score output: один cosine score на сравнение enrollment/probe
- decision output: accept/reject после выбора threshold profile
- metric focus: `EER`, `minDCF`, slice breakdown, error analysis

Это единственный режим, для которого в репозитории уже есть полный
data/eval/serve контракт: trial builder, offline evaluation package, threshold
calibration, raw-audio inferencer и HTTP-адаптер.

### Closed-Set Identification

- status: `compatible_not_first_class`
- request unit: `probe audio + finite enrolled speaker gallery`
- score output: ranked cosine score list по gallery
- decision output: top-1 / top-k predicted speaker id

Режим совместим с текущим encoder, потому что shared scorer уже умеет
one-to-many ranking. Но отдельного repo-native отчёта, набора метрик и serving
surface для него пока нет.

### Open-Set Identification

- status: `compatible_not_first_class`
- request unit: `probe audio + finite gallery + reject threshold`
- score output: ranked cosine score list plus best-match score
- decision output: predicted speaker id or `unknown`

Этот режим должен использовать тот же encoder и тот же thresholding contract,
что и verification, а не независимый классификатор.

## Trial Types

### `verification_pair`

- mode: `verification`
- support level: `implemented`
- evaluation unit: `left_audio x right_audio`
- required fields:
  `left_audio`, `right_audio`, `label`, `left_speaker_id`, `right_speaker_id`
- label space: `positive`, `negative`
- slice fields: `duration_bucket`, `domain_relation`, `channel_relation`
- source of truth: `src/kryptonite/data/verification_trials.py`

### `closed_set_gallery_probe`

- mode: `closed-set identification`
- support level: `planned`
- evaluation unit: `probe audio against finite candidate gallery`
- required fields: `probe_audio`, `candidate_enrollment_ids`, `expected_speaker_id`

### `open_set_gallery_probe`

- mode: `open-set identification`
- support level: `planned`
- evaluation unit: `probe audio against finite gallery plus reject option`
- required fields:
  `probe_audio`, `candidate_enrollment_ids`, `expected_speaker_id_or_unknown`

## Expected Artifacts

- `docs/model-task-contract.md`
  Человеко-читаемый ADR с формулировкой задачи, trial types, ожидаемыми
  артефактами и ограничениями.
- `artifacts/model-task-contract/model_task_contract.json`
  Machine-readable snapshot контракта для автоматизации и последующих этапов.
- `artifacts/model-task-contract/model_task_contract.md`
  Generated markdown snapshot того же контракта.
- `artifacts/**/verification_eval_report.json`
  Базовый quality artifact для primary verification task.
- `artifacts/**/verification_threshold_calibration.json`
  Threshold bundle, который переводит cosine scores в demo/production decisions.
- `artifacts/export-boundary/export_boundary.json`
  Runtime/export handoff contract для будущего learned encoder path.

## Limitations

- Checked-in runtime всё ещё использует `feature_statistics`, поэтому рабочий
  demo/runtime flow доказывает в первую очередь корректность контракта, а не
  финальное качество обученного speaker encoder.
- Closed-set и open-set identification пока не имеют first-class report builder
  и serving API, поэтому считаются совместимыми режимами, а не полноценным
  deliverable.
- Verification thresholds всегда завязаны на конкретный score distribution и
  candidate bundle; нельзя переносить их между моделями без повторной
  калибровки.
- Enrollment cache и runtime state валидны только для совместимого model-bundle
  metadata и export boundary.

## Supporting References

- [docs/embedding-scoring.md](./embedding-scoring.md)
- [docs/evaluation-package.md](./evaluation-package.md)
- [docs/export-boundary.md](./export-boundary.md)
- [docs/model-card.md](./model-card.md)
- [docs/unified-inference-wrapper.md](./unified-inference-wrapper.md)

## Rebuild

```bash
uv run python scripts/build_model_task_contract.py --config configs/base.toml
```

Команда пересобирает локальные артефакты:

- `artifacts/model-task-contract/model_task_contract.json`
- `artifacts/model-task-contract/model_task_contract.md`
