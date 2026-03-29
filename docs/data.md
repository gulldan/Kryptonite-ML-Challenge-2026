# Data

Этот документ отвечает на один вопрос: на каких данных репозиторий реально работает сейчас,
что считать source of truth для manifests, и где проходят policy-ограничения.

## Current repository position

- реальный Dataton dataset пока не считается materialized source of truth внутри checkout
- текущий engineering surrogate: `FFSVC 2022`
- manifests и metadata contracts важнее отдельных ad hoc notebook assumptions
- `datasets/` остается локальным и git-ignored

## What is canonical

- machine-readable source inventory:
  `configs/data-inventory/allowed-sources.toml`
- challenge-policy matrix:
  `configs/data-inventory/dataton-rules-matrix.toml`
- manifest row shape:
  `kryptonite.manifest.v1`
- reproducible outputs:
  generated manifests and reports under `artifacts/`

## Practical reading order

1. Сначала поймите общую позицию по данным здесь.
2. Если нужен policy detail, откройте архивные документы:
   - [archive/dataset-inventory.md](./archive/dataset-inventory.md)
   - [archive/dataton-rules-matrix.md](./archive/dataton-rules-matrix.md)
3. Если нужен текущий surrogate workflow:
   - [archive/ffsvc2022-surrogate.md](./archive/ffsvc2022-surrogate.md)
4. Если нужен schema-level detail:
   - [archive/unified-metadata-schema.md](./archive/unified-metadata-schema.md)

## Working rules

- не считать demo subset доказательством качества модели
- не считать наличие raw datasets в `datasets/` частью git contract
- не смешивать data-policy decisions с training notes
- не выводить preprocessing assumptions из notebooks, если они не зафиксированы в config и reusable code

## Related docs

- [training.md](./training.md)
- [configuration.md](./configuration.md)
- [reference/audio-pipeline.md](./reference/audio-pipeline.md)
