"""I/O helpers for verification score, trial, and metadata artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl


def load_verification_score_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load JSONL score rows that contain at least `label` and `score`."""

    return _load_jsonl_rows(path)


def load_verification_trial_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load JSONL verification trial rows."""

    return _load_jsonl_rows(path)


def load_verification_metadata_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load embedding-export metadata from JSONL or Parquet."""

    metadata_path = Path(path)
    suffix = metadata_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl_rows(metadata_path)
    if suffix == ".parquet":
        frame = pl.read_parquet(metadata_path)
        rows = frame.to_dicts()
        if not rows:
            raise ValueError(f"No metadata rows found in {metadata_path}")
        return rows
    raise ValueError(
        f"Unsupported metadata format for {metadata_path}; expected .jsonl or .parquet."
    )


def build_trial_item_index(metadata_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index metadata rows by the identifiers commonly used inside trial files."""

    index: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        for candidate in _metadata_candidate_keys(row):
            index.setdefault(candidate, row)
    return index


def resolve_trial_side_identifier(row: dict[str, Any], side: str) -> str | None:
    """Resolve the left/right identifier used to join a trial row to metadata."""

    for key in (f"{side}_id", f"{side}_audio"):
        value = row.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _load_jsonl_rows(path: Path | str) -> list[dict[str, Any]]:
    source_path = Path(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(source_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object JSONL rows in {source_path}:{line_number}")
        rows.append(payload)
    if not rows:
        raise ValueError(f"No rows found in {source_path}")
    return rows


def _metadata_candidate_keys(row: dict[str, Any]) -> tuple[str, ...]:
    candidates: list[str] = []
    for key in ("trial_item_id", "utterance_id", "audio_path"):
        value = row.get(key)
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized:
            continue
        candidates.append(normalized)
        if key == "audio_path":
            candidates.append(Path(normalized).name)
    return tuple(dict.fromkeys(candidates))


__all__ = [
    "build_trial_item_index",
    "load_verification_metadata_rows",
    "load_verification_score_rows",
    "load_verification_trial_rows",
    "resolve_trial_side_identifier",
]
