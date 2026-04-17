# Архитектура решения

Документ описывает верхнеуровневую архитектуру решения: как из тестового
CSV и аудио получается `submission.csv`.

## Основной путь

```text
test CSV + аудио
  -> чтение аудио
  -> 16 kHz mono
  -> официальный CAM++ frontend
  -> CAM++ encoder
  -> векторы голосов
  -> поиск ближайших соседей
  -> графовая и class-aware постобработка
  -> submission.csv
  -> отчёт валидатора
```

## Главные принципы

- Обучение, EDA и инференс разделены по разным модулям.
- Важные параметры вынесены в TOML-конфиги или CLI flags.
- Большие веса и кеши лежат в `artifacts/`, а не в git.
- Каждый важный эксперимент записывается в `docs/challenge-experiment-history.md` и
  отдельный `docs/trails/<slug>.md`.
- Итоговый `submission.csv` всегда проверяется отдельным валидатором.

## Границы модулей

| Модуль | Роль |
| --- | --- |
| `src/kryptonite/data/` | Загрузка аудио, манифесты, валидация данных. |
| `src/kryptonite/features/` | Fbank, нарезка аудио, официальный CAM++ frontend, кеш признаков. |
| `src/kryptonite/models/` | CAM++/ERes2NetV2 encoder и cosine scoring. |
| `src/kryptonite/training/` | Обучение, fine-tuning, dataloaders, optimizer, tracking. |
| `src/kryptonite/eda/` | EDA, анализ графа соседей, псевдометки, проверка submission. |
| `src/kryptonite/eval/` | Метрики и отчёты. |
| `src/kryptonite/serve/` | ONNX/TensorRT/runtime-адаптеры. |
| `scripts/` | Воспроизводимые команды запуска. |

## Финальный инференс

Финальный кандидат MS41 состоит из двух частей:

1. Сильная CAM++ ветка MS32 строит векторы голосов и top-200 ближайших соседей.
2. MS41 добавляет слабую class-aware поправку и графовую постобработку, после чего
   выбирает top-10 соседей для `submission.csv`.

Почему это сделано в два шага:

- первый шаг долгий, потому что читает аудио и строит векторы голосов;
- второй шаг быстрый и позволяет проверять разные политики выбора соседей;
- такой раздельный пайплайн проще валидировать и сравнивать.

## Проверка корректности

Итоговый файл проверяется командой:

```bash
uv run python scripts/validate_submission.py \
  --template-csv datasets/Для\ участников/test_public.csv \
  --submission-csv submission.csv \
  --output-json artifacts/release/submission_validation.json
```

Валидатор проверяет число строк, порядок `filepath`, 10 соседей на строку, повторы,
совпадение файла с самим собой и индексы вне диапазона.

## Что читать дальше

- [../README.md](../README.md) - точные команды запуска.
- [challenge-solution-report.md](./challenge-solution-report.md) - короткое объяснение решения.
- [code-architecture.md](./code-architecture.md) - карта кода.
- [challenge-experiment-history.md](./challenge-experiment-history.md) - leaderboard-таблица
  экспериментов.
- [trails/](./trails/) - подробные экспериментальные записи.
