"""Formatting helpers for audio-quality reports."""

from __future__ import annotations

from .models import HistogramBucket


def format_counts(counts: dict[str, int], *, limit: int | None = None) -> str:
    if not counts:
        return "-"
    items = list(counts.items())
    if limit is not None:
        visible_items = items[:limit]
        remaining_count = len(items) - len(visible_items)
    else:
        visible_items = items
        remaining_count = 0
    rendered = ", ".join(f"{name}={count}" for name, count in visible_items)
    if remaining_count > 0:
        rendered = f"{rendered}, +{remaining_count} more"
    return rendered


def format_duration(total_seconds: float) -> str:
    return f"{total_seconds:.2f} s"


def format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f} s"


def format_dbfs(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f} dBFS"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |" for row in rows
    ]
    return "\n".join([header_row, separator_row, *body_rows])


def render_histogram(histogram: list[HistogramBucket]) -> str:
    counts = {bucket.label: bucket.count for bucket in histogram if bucket.count > 0}
    if not counts:
        return "_No data available._"
    return render_text_chart(counts)


def render_text_chart(counts: dict[str, int]) -> str:
    if not counts:
        return "_No data available._"

    max_count = max(counts.values())
    lines = ["```text"]
    for label, count in counts.items():
        bar_width = 0 if max_count == 0 else max(1, round((count / max_count) * 28))
        lines.append(f"{label:<12} {'#' * bar_width:<28} {count}")
    lines.append("```")
    return "\n".join(lines)


def escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")
