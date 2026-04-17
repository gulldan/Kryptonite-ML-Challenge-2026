# Документация

## С чего начать

1. [challenge-solution-report.md](./challenge-solution-report.md) -- самодостаточный отчёт для жюри
2. [release-runbook.md](./release-runbook.md) -- как получить и проверить `submission.csv`
3. [inference-acceleration.md](./inference-acceleration.md) -- ускорение инференса,
   команды ONNX/TensorRT и профиль узких мест
4. [repository-file-inventory.md](./repository-file-inventory.md) -- зачем нужны файлы и папки
5. [challenge-experiment-history.md](./challenge-experiment-history.md) -- таблица submitted public LB
   runs
6. [trails/](./trails/) -- подробные записи экспериментов
7. [../artifacts/release-bundles/ms41-final-evidence-v1/submission_bundle.md](../artifacts/release-bundles/ms41-final-evidence-v1/submission_bundle.md) -- зафиксированный финальный evidence bundle
8. [code-architecture.md](./code-architecture.md) -- карта кода: что где лежит
9. [training.md](./training.md) -- обучение и добавление моделей
10. [configuration.md](./configuration.md) -- конфиги и переопределения параметров

## Архитектура

- [code-architecture.md](./code-architecture.md) -- модули, файлы, потоки данных
- [system-architecture-v1.md](./system-architecture-v1.md) -- пайплайн и границы модулей

## Материалы для жюри

- [challenge-solution-report.md](./challenge-solution-report.md) -- EDA, baseline, эксперименты,
  инженерные решения, выводы
- [repository-file-inventory.md](./repository-file-inventory.md) -- реестр файлов
- [challenge-experiment-history.md](./challenge-experiment-history.md) -- leaderboard-таблица
  запусков, оценок, путей к файлам и решений
- [trails/](./trails/) -- подробные экспериментальные записи и диагностика
- [../artifacts/release-bundles/ms41-final-evidence-v1/submission_bundle.md](../artifacts/release-bundles/ms41-final-evidence-v1/submission_bundle.md) -- собранный пакет доказательств по финальному кандидату
- [inference-acceleration.md](./inference-acceleration.md) -- отдельный материал по
  скорости: TensorRT, узкое место frontend, кеши, команды воспроизведения

## Данные и модели

- [data.md](./data.md) -- данные и манифесты
- [model-task-contract.md](./model-task-contract.md) -- задача и формат `submission.csv`
- [model-card.md](./model-card.md) -- рамка решения
- [challenge-experiment-history.md](./challenge-experiment-history.md) -- public LB и решения для
  презентации
- [trails/](./trails/) -- подробности по отдельным гипотезам

## Операции

- [release-runbook.md](./release-runbook.md) -- контрольный список финальной сдачи
- [inference-acceleration.md](./inference-acceleration.md) -- экспорт ONNX/TensorRT,
  замеры и команды профилирования
- [reference/audio-pipeline.md](./reference/audio-pipeline.md) -- контракт аудио
