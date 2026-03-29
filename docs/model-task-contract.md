# Model Task Contract

`KVA-480` фиксирует каноническую постановку задачи для этого репозитория.
Этот документ теперь короткий и operational: он нужен, чтобы новый человек быстро понял,
что именно считается основной задачей и какой набор артефактов подтверждает решение.

## Decision

- primary task mode: `verification`
- compatibility modes: `closed-set identification`, `open-set identification`
- canonical runtime surface: audio -> embedding -> cosine score -> thresholded decision
- checked-in runtime backend today: `feature_statistics`

## Canonical Workflow

1. Прочитать или получить raw audio.
2. Привести его к общему audio contract.
3. Построить embedding через текущий runtime backend.
4. Выполнить enrollment/scoring через общий cosine scorer.
5. Применить candidate-specific threshold bundle.
6. Зафиксировать offline eval artifacts и runtime smoke artifacts.

## Input And Output Contract

- raw audio target: `16 kHz`, `mono`, `PCM16 WAV`
- loudness and VAD are explicit config-driven stages, а не скрытые defaults
- chunking и Fbank являются общей частью training/eval/runtime frontend
- encoder boundary: `encoder_input -> embedding`
- serving output: score, threshold context, backend/runtime metadata, optional decision

## Task Modes

### Verification

Базовый режим репозитория: pairwise speaker verification и threshold-based decision.

### Closed-set identification

Совместимый режим поверх общего embedding space, но не основная report surface.

### Open-set identification

Планируемый совместимый режим с `unknown`/reject semantics поверх того же пространства эмбеддингов.

## Trial Types

- `verification_pair`
- `closed_set_gallery_probe`
- `open_set_gallery_probe`

## Expected Artifacts

- checked-in contract doc: `docs/model-task-contract.md`
- machine-readable snapshot: `artifacts/model-task-contract/model_task_contract.json`
- markdown snapshot: `artifacts/model-task-contract/model_task_contract.md`
- offline quality bundle: `artifacts/**/verification_eval_report.json`
- threshold bundle: `artifacts/**/verification_threshold_calibration.json`
- export boundary contract: `artifacts/export-boundary/export_boundary.json`

## Limitations

- live runtime green status пока доказывает shape/integration, а не финальное learned-model качество
- identification modes остаются совместимыми режимами, но не канонической release surface
- thresholds валидны только вместе с candidate-specific score distribution
- enrollment cache привязан к активному model bundle

## Supporting References

- [system-architecture-v1.md](./system-architecture-v1.md)
- [release-runbook.md](./release-runbook.md)
- [model-card.md](./model-card.md)
- [archive/model-task-contract.md](./archive/model-task-contract.md) — предыдущая подробная версия
- [archive/evaluation-package.md](./archive/evaluation-package.md)
- [archive/threshold-calibration.md](./archive/threshold-calibration.md)

## Rebuild

```bash
uv run python scripts/build_model_task_contract.py --config configs/base.toml
```
