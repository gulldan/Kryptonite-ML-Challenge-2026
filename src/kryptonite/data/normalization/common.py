"""Shared helpers for normalized manifest bundles."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from kryptonite.data.schema import normalize_manifest_entry

from .constants import DATA_MANIFEST_PRIORITY
from .models import SourceManifestTable


def coerce_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unsupported float value type: {type(value)!r}")


def relative_to_project(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def source_manifest_sort_key(table: SourceManifestTable) -> tuple[int, str]:
    try:
        return (DATA_MANIFEST_PRIORITY.index(table.name), table.name)
    except ValueError:
        return (len(DATA_MANIFEST_PRIORITY), table.name)


def read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object JSONL rows in {path}:{line_number}")
        rows.append(payload)
    return rows


def row_identity_key(row: Mapping[str, object]) -> str:
    normalized = normalize_manifest_entry(row)
    parts = (
        coerce_str(normalized.get("dataset")) or "unknown-dataset",
        coerce_str(normalized.get("split")) or "unknown-split",
        coerce_str(normalized.get("speaker_id")) or "unknown-speaker",
        coerce_str(normalized.get("utterance_id"))
        or coerce_str(normalized.get("audio_path"))
        or json.dumps(dict(row), sort_keys=True),
    )
    return "|".join(parts)


def deduplicate_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    deduplicated: dict[str, dict[str, object]] = {}
    for row in rows:
        deduplicated.setdefault(row_identity_key(row), dict(row))
    return list(deduplicated.values())


def detect_dataset_name(source_tables: Sequence[SourceManifestTable]) -> str:
    for table in source_tables:
        for row in table.rows:
            dataset_name = coerce_str(normalize_manifest_entry(row).get("dataset"))
            if dataset_name is not None:
                return dataset_name
    return source_tables[0].path.parent.name
