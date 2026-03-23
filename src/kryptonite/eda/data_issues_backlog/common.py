"""Shared formatting and utility helpers for the cleanup backlog."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def sorted_counts(counts: Counter[str] | dict[str, int]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts)}


def format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def format_value(value: object) -> str:
    if isinstance(value, dict):
        integer_counts = {str(key): int(val) for key, val in value.items() if isinstance(val, int)}
        if integer_counts:
            return format_counts(integer_counts)
        return json.dumps(value, sort_keys=True)
    if isinstance(value, list):
        return "[" + ", ".join(format_value(item) for item in value[:5]) + "]"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def merge_warnings(*warning_sets: list[str]) -> list[str]:
    merged: list[str] = []
    for warnings in warning_sets:
        for warning in warnings:
            if warning not in merged:
                merged.append(warning)
    return merged


def utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
