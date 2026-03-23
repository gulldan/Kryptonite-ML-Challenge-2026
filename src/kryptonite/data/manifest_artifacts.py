"""Helpers for reproducible manifest bundles, CSV sidecars, and checksum inventories."""

from __future__ import annotations

import csv
import hashlib
import json
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from kryptonite.deployment import resolve_project_path

PREFERRED_FIELD_ORDER: Final[tuple[str, ...]] = (
    "schema_version",
    "record_type",
    "dataset",
    "source_dataset",
    "speaker_id",
    "utterance_id",
    "session_id",
    "split",
    "role",
    "language",
    "device",
    "channel",
    "snr_db",
    "rir_id",
    "duration_seconds",
    "sample_rate_hz",
    "num_channels",
    "audio_path",
    "label",
    "left_audio",
    "right_audio",
)


@dataclass(frozen=True, slots=True)
class WavAudioMetadata:
    duration_seconds: float
    sample_rate_hz: int
    num_channels: int


@dataclass(frozen=True, slots=True)
class TabularArtifact:
    name: str
    kind: str
    jsonl_path: str
    csv_path: str
    row_count: int
    jsonl_sha256: str
    csv_sha256: str
    speaker_count: int | None = None
    audio_path_count: int | None = None
    total_duration_seconds: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "jsonl_path": self.jsonl_path,
            "csv_path": self.csv_path,
            "row_count": self.row_count,
            "jsonl_sha256": self.jsonl_sha256,
            "csv_sha256": self.csv_sha256,
        }
        if self.speaker_count is not None:
            payload["speaker_count"] = self.speaker_count
        if self.audio_path_count is not None:
            payload["audio_path_count"] = self.audio_path_count
        if self.total_duration_seconds is not None:
            payload["total_duration_seconds"] = self.total_duration_seconds
        return payload


@dataclass(frozen=True, slots=True)
class FileArtifact:
    name: str
    kind: str
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
        }


def inspect_wav_audio_file(path: Path) -> WavAudioMetadata:
    with wave.open(str(path), "rb") as handle:
        sample_rate_hz = handle.getframerate()
        num_channels = handle.getnchannels()
        frame_count = handle.getnframes()

    if sample_rate_hz <= 0:
        raise ValueError(f"WAV file has non-positive sample rate: {path}")

    return WavAudioMetadata(
        duration_seconds=round(frame_count / sample_rate_hz, 6),
        sample_rate_hz=sample_rate_hz,
        num_channels=num_channels,
    )


def write_tabular_artifact(
    *,
    name: str,
    kind: str,
    rows: Sequence[Mapping[str, object]],
    jsonl_path: Path,
    project_root: str,
    field_order: Sequence[str] | None = None,
) -> TabularArtifact:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows))

    csv_path = jsonl_path.with_suffix(".csv")
    fieldnames = _resolve_fieldnames(rows=rows, field_order=field_order)
    with csv_path.open("w", newline="") as handle:
        if fieldnames:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field_name: _stringify_csv_value(row.get(field_name))
                        for field_name in fieldnames
                    }
                )

    row_count = len(rows)
    speaker_ids = {
        str(row["speaker_id"])
        for row in rows
        if row.get("speaker_id") is not None and str(row["speaker_id"]).strip()
    }
    audio_paths = {
        str(row["audio_path"])
        for row in rows
        if row.get("audio_path") is not None and str(row["audio_path"]).strip()
    }
    durations = [
        duration
        for row in rows
        if (duration := _coerce_float(row.get("duration_seconds"))) is not None
    ]

    return TabularArtifact(
        name=name,
        kind=kind,
        jsonl_path=_relative_to_project(jsonl_path, project_root),
        csv_path=_relative_to_project(csv_path, project_root),
        row_count=row_count,
        jsonl_sha256=_compute_sha256(jsonl_path),
        csv_sha256=_compute_sha256(csv_path),
        speaker_count=len(speaker_ids) if speaker_ids else None,
        audio_path_count=len(audio_paths) if audio_paths else None,
        total_duration_seconds=round(sum(durations), 6) if durations else None,
    )


def build_file_artifact(
    *,
    name: str,
    kind: str,
    path: Path,
    project_root: str,
) -> FileArtifact:
    return FileArtifact(
        name=name,
        kind=kind,
        path=_relative_to_project(path, project_root),
        sha256=_compute_sha256(path),
    )


def write_manifest_inventory(
    *,
    dataset: str,
    inventory_path: Path,
    project_root: str,
    manifest_tables: Sequence[TabularArtifact],
    auxiliary_tables: Sequence[TabularArtifact] = (),
    auxiliary_files: Sequence[FileArtifact] = (),
) -> str:
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "output_root": _relative_to_project(inventory_path.parent, project_root),
        "manifest_tables": [artifact.to_dict() for artifact in manifest_tables],
        "auxiliary_tables": [artifact.to_dict() for artifact in auxiliary_tables],
        "auxiliary_files": [artifact.to_dict() for artifact in auxiliary_files],
    }
    inventory_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _relative_to_project(inventory_path, project_root)


def _resolve_fieldnames(
    *,
    rows: Sequence[Mapping[str, object]],
    field_order: Sequence[str] | None,
) -> list[str]:
    preferred = list(field_order or PREFERRED_FIELD_ORDER)
    discovered = {key for row in rows for key in row}
    ordered = [field_name for field_name in preferred if field_name in discovered]
    remaining = sorted(discovered - set(ordered))
    if ordered or remaining:
        return [*ordered, *remaining]
    return list(preferred)


def _stringify_csv_value(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unsupported duration value type for manifest export: {type(value)!r}")


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_project(path: Path, project_root: str) -> str:
    root = resolve_project_path(project_root, ".")
    return str(path.resolve().relative_to(root))
