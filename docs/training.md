# Training

Этот документ описывает не всю историю экспериментов, а текущий рабочий training path.

## Current primary path

- базовый student path: `CAM++`
- staged flow: baseline -> stage 2 -> stage 3 -> shortlist/model selection
- export-critical family decision уже принята и сохранена в архиве:
  [archive/final-family-decision.md](./archive/final-family-decision.md)
- `ERes2NetV2` остается comparison baseline, а teacher branch — stretch path

## Bring-up commands

Локальная машина:

```bash
uv sync --dev --group train --group tracking
uv run python scripts/training_env_smoke.py --config configs/deployment/train.toml
```

GPU сервер:

```bash
uv sync --dev --group train --group tracking
uv run python scripts/training_env_smoke.py --config configs/deployment/train.toml --require-gpu
```

## Recommended reading order

1. Понять архитектуру: [system-architecture-v1.md](./system-architecture-v1.md)
2. Понять data position: [data.md](./data.md)
3. Проверить config surface: [configuration.md](./configuration.md)
4. Только потом открывать stage-specific archived docs

## Archive map for training details

- baseline and family docs:
  - [archive/campp-baseline.md](./archive/campp-baseline.md)
  - [archive/eres2netv2-baseline.md](./archive/eres2netv2-baseline.md)
- staged CAM++ path:
  - [archive/campp-stage2-training.md](./archive/campp-stage2-training.md)
  - [archive/campp-stage3-training.md](./archive/campp-stage3-training.md)
  - [archive/campp-hyperparameter-sweep-shortlist.md](./archive/campp-hyperparameter-sweep-shortlist.md)
  - [archive/campp-model-selection.md](./archive/campp-model-selection.md)
- teacher and distillation:
  - [archive/teacher-peft.md](./archive/teacher-peft.md)
  - [archive/teacher-student-robust-dev.md](./archive/teacher-student-robust-dev.md)
  - [archive/campp-distillation.md](./archive/campp-distillation.md)
- planning/history:
  - [archive/experiment-matrix-v1.md](./archive/experiment-matrix-v1.md)
  - [archive/release-postmortem.md](./archive/release-postmortem.md)

## Rule of thumb

Если вы onboarding’итесь в репозиторий, не начинайте со stage-2, shortlist или postmortem.
Сначала поймите canonical stack, data position и runtime contract.
