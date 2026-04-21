# Скрипты бенчмарка устойчивости

Папка содержит CLI-обёртки для исследовательского бенчмарка устойчивости.

Логика бенчмарка находится в `src/kryptonite/eval/robustness/`, а в этой папке
лежат команды запуска.

Основные точки входа:

- `uv run python research/scripts/robustness/run_benchmark.py`
- `uv run python research/scripts/robustness/extract_embeddings.py`
- `uv run python research/scripts/robustness/render_benchmark_audit.py`

Сгенерированные рантайм-артефакты и отчёты должны оставаться вне git.
