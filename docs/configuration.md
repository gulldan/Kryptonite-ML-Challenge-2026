# Конфиги

Конфиги нужны, чтобы важные параметры не были зашиты в коде. Основной формат -
TOML.

## Главные файлы

| Путь | Назначение |
| --- | --- |
| `configs/base.toml` | Общие настройки проекта. |
| `configs/schema.json` | Схема конфигов. |
| `configs/training/*.toml` | Конфиги обучения и fine-tuning. |
| `configs/release/*.toml` | Конфиги release-проверок и TensorRT. |

## Как посмотреть итоговый конфиг

```bash
uv run python scripts/show_config.py --config configs/base.toml
```

## Как проверить схему

```bash
uv run python scripts/validate_config_schema.py configs/base.toml
```

## Как переопределить параметр

Некоторые команды поддерживают переопределения через CLI. Пример:

```bash
uv run python scripts/run_baseline.py \
  --model campp \
  --config configs/training/campp-baseline.toml \
  --device cuda \
  --project-override 'training.batch_size=32'
```

## Что обычно настраивается

- пути к train/dev/test manifests;
- путь к output-папке в `artifacts/`;
- размер batch;
- число эпох;
- learning rate;
- precision;
- политика нарезки аудио;
- аугментации;
- путь к checkpoint.

## Правила

- Не добавлять новые форматы конфигов без необходимости.
- Не хранить секреты в TOML.
- Не хранить локальные абсолютные пути в коммитах, если они не являются частью
  зафиксированного remote workflow.
- Для новых важных экспериментов добавлять отдельный TOML-конфиг, а не менять код.

## Связанные документы

- [training.md](./training.md)
- [release-runbook.md](./release-runbook.md)
- [repository-file-inventory.md](./repository-file-inventory.md)
