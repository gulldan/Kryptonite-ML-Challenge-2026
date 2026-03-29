# Docs

Эта папка больше не является плоским каталогом из десятков равноправных заметок.
Наверху оставлены только документы, которые реально нужны для онбординга и повседневной работы.

## Начать здесь

- [system-architecture-v1.md](./system-architecture-v1.md) — как устроен pipeline и где проходят модульные границы.
- [code-architecture.md](./code-architecture.md) — как читать сам код, куда класть новую логику и где source of truth.
- [release-runbook.md](./release-runbook.md) — как поднять demo/API, проверить health и что смотреть при сбоях.
- [model-task-contract.md](./model-task-contract.md) — какая задача считается канонической и какие артефакты обязательны.
- [model-card.md](./model-card.md) — короткая внешняя рамка решения и его ограничения.

## По задачам

- [data.md](./data.md) — откуда берутся данные, какой surrogate используется сейчас и где смотреть правила/ограничения.
- [training.md](./training.md) — как поднимать training environment и какой путь обучения считать основным.
- [configuration.md](./configuration.md) — карта конфигов, overrides и secrets.

## Reference

- [reference/audio-pipeline.md](./reference/audio-pipeline.md) — единый audio contract: normalization, VAD, Fbank, chunking, corruption policy.

## Archive

Все детальные deep-dive, release notes, benchmark-записки, stage-specific training docs и исторические ADR перенесены в
[archive/README.md](./archive/README.md).
