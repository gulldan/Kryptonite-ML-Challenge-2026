# Code Architecture

Этот документ объясняет не исследовательскую историю проекта, а текущую карту кода:
где лежит source of truth, куда добавлять новую логику и какие модули считать thin entrypoints.

## Top-level rule

- `apps/` содержит тонкие surfaces и transport.
- `src/kryptonite/` содержит бизнес-логику, typed config loaders, training/eval/runtime code.
- `configs/` содержит runtime, training и evaluation конфиги.
- `scripts/` содержит воспроизводимые CLI entrypoints поверх `src/kryptonite/`.
- `docs/` объясняет архитектуру, runbooks и model/release contracts.

## Canonical module map

- `src/kryptonite/common/`
  - low-level shared helpers, которые не должны жить по копии в `training/`, `eval/` и `serve/`
  - сейчас здесь общий parsing layer для typed config loaders
- `src/kryptonite/data/`
  - manifests, metadata contracts, audio loading, corruption-bank inputs, silence/audio policy
- `src/kryptonite/features/`
  - chunking, fbank extraction, feature-oriented transforms
- `src/kryptonite/models/`
  - model families, encoder configs, scoring primitives
- `src/kryptonite/training/`
  - baseline recipes, augmentation runtime/scheduler, family-specific training configs
- `src/kryptonite/eval/`
  - verification protocol, benchmark packs, calibration, score-normalization, offline reports
- `src/kryptonite/serve/`
  - inferencer, enrollment state, HTTP runtime, backend wrappers, runtime-facing reports

## How to read the repository

Если вы пытаетесь понять репозиторий по коду, идите в таком порядке:

1. `src/kryptonite/config.py` — глобальный project/runtime config contract.
2. `src/kryptonite/data/` и `src/kryptonite/features/` — shared audio path.
3. `src/kryptonite/models/` — encoder/scoring layer.
4. `src/kryptonite/training/` — recipes и train-time orchestration.
5. `src/kryptonite/eval/` — offline verification и release benchmarks.
6. `src/kryptonite/serve/` — live runtime, HTTP surface, enrollment and backend selection.
7. `apps/api/` и `apps/web/` — thin user-facing entrypoints.

## Stable boundaries

- Новый data/audio contract добавляется в `data/` или `features/`, а не в `training/` и не в `serve/`.
- Новый training recipe добавляется в `training/`, а scripts только вызывают его.
- Новый offline report/benchmark добавляется в `eval/`.
- Новый live runtime adapter добавляется в `serve/`.
- Shared parsing/path/report helpers сначала ищутся в `common/`, а не копируются локально.

## Current duplication policy

- Typed config parsing не должен дублироваться между `training/`, `eval/` и `serve/`.
- Shared business rules, например silence augmentation scaling, должны жить в одном policy module.
- `__init__.py` пакетов считаются compatibility barrels, а не местом, где описывается архитектура.

## Practical entrypoints

- training:
  - `scripts/run_campp_baseline.py`
  - `scripts/run_campp_stage2_training.py`
  - `scripts/run_campp_stage3_training.py`
- evaluation:
  - `scripts/build_final_benchmark_pack.py`
  - `scripts/run_verification_protocol_snapshot.py`
  - `scripts/run_backend_benchmark.py`
- runtime:
  - `apps/api/`
  - `scripts/run_web_demo.py`

## Where to put new code

- Если код нужен более чем одному subsystem, сначала рассматривайте `src/kryptonite/common/`.
- Если код нужен training и offline eval одновременно, не дублируйте; поднимайте его в shared domain module.
- Если файл начинает расползаться по нескольким ответственностям, делите его на package/module pair, а не на второй giant file рядом.
