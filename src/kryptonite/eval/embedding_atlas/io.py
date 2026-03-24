"""I/O helpers for embedding-atlas sources."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def load_embedding_matrix(
    path: Path | str,
    *,
    embeddings_key: str = "embeddings",
    ids_key: str | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    source_path = Path(path)
    suffix = source_path.suffix.lower()

    if suffix == ".npy":
        embeddings = np.load(source_path)
        point_ids = None
    elif suffix == ".npz":
        payload = np.load(source_path)
        if embeddings_key not in payload:
            available = ", ".join(sorted(payload.files))
            raise ValueError(
                f"Embeddings key {embeddings_key!r} is missing from {source_path}; "
                f"available keys: {available}"
            )
        embeddings = payload[embeddings_key]
        if ids_key is not None:
            if ids_key not in payload:
                available = ", ".join(sorted(payload.files))
                raise ValueError(
                    f"IDs key {ids_key!r} is missing from {source_path}; "
                    f"available keys: {available}"
                )
            point_ids = [str(value) for value in payload[ids_key].tolist()]
        else:
            point_ids = None
    else:
        raise ValueError(f"Unsupported embeddings format for {source_path}; expected .npy or .npz.")

    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D embeddings matrix in {source_path}, got shape {matrix.shape}."
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Embeddings matrix in {source_path} must be non-empty.")
    if point_ids is not None and len(point_ids) != matrix.shape[0]:
        raise ValueError(
            f"Embeddings IDs length mismatch for {source_path}: "
            f"{len(point_ids)} ids for {matrix.shape[0]} rows."
        )
    return matrix, point_ids


def load_metadata_rows(path: Path | str) -> list[dict[str, object]]:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, object]] = []
        for line_number, line in enumerate(source_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Metadata line {line_number} in {source_path} must be a JSON object."
                )
            rows.append(dict(payload))
        return rows
    if suffix == ".csv":
        with source_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
    raise ValueError(f"Unsupported metadata format for {source_path}; expected .jsonl or .csv.")


def align_metadata_rows(
    *,
    metadata_rows: list[dict[str, object]],
    point_id_field: str,
    point_ids: list[str] | None,
    expected_count: int,
) -> list[dict[str, object]]:
    if point_ids is None:
        if len(metadata_rows) != expected_count:
            raise ValueError(
                f"Metadata row count mismatch: expected {expected_count}, got {len(metadata_rows)}."
            )
        return metadata_rows

    indexed_rows: dict[str, dict[str, object]] = {}
    for row in metadata_rows:
        point_id = _coerce_string(row.get(point_id_field))
        if point_id is None:
            raise ValueError(f"Metadata rows must define non-empty {point_id_field!r}.")
        if point_id in indexed_rows:
            raise ValueError(
                f"Duplicate metadata point id {point_id!r} for field {point_id_field!r}."
            )
        indexed_rows[point_id] = row

    aligned: list[dict[str, object]] = []
    missing = [point_id for point_id in point_ids if point_id not in indexed_rows]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Metadata is missing {len(missing)} ids referenced by the embeddings table. "
            f"Examples: {preview}"
        )
    for point_id in point_ids:
        aligned.append(indexed_rows[point_id])
    return aligned


def stringify_metadata_fields(row: dict[str, object]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for key, value in row.items():
        if value is None:
            continue
        if isinstance(value, float):
            payload[key] = format(value, ".6g")
        elif isinstance(value, (str, int, bool)):
            payload[key] = str(value)
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, str)):
            payload[key] = json.dumps(list(value), ensure_ascii=True)
        else:
            payload[key] = json.dumps(value, ensure_ascii=True, sort_keys=True)
    return payload


def _coerce_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "align_metadata_rows",
    "load_embedding_matrix",
    "load_metadata_rows",
    "stringify_metadata_fields",
]
