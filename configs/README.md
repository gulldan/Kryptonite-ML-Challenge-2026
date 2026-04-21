# Конфиги

Папка содержит только общие и будущие финальные конфиги organizer-facing пути.

Что лежит здесь сейчас:

- `base.toml` — общий локальный профиль с базовыми путями и настройками;
- `schema.json` — схема для проверки конфигурации;
- будущий `submission.toml` — единый финальный конфиг, который появится после выбора модели.

Что не должно лежать здесь:

- исследовательские training/release/smoke-конфиги;
- конфиги benchmark-пайплайнов;
- промежуточные или абляционные профили.

Вся исследовательская конфигурация перенесена в `research/configs/`.

Просмотр итогового конфига:

```bash
uv run python research/scripts/show_config.py --config configs/base.toml
uv run python research/scripts/show_config.py --config configs/base.toml --override runtime.seed=123
```
