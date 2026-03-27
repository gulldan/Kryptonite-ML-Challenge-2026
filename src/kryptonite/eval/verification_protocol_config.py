"""Typed config loader for the internal verification-protocol snapshot."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

DEFAULT_PROTOCOL_REQUIRED_SLICE_FIELDS: tuple[str, ...] = (
    "duration_bucket",
    "noise_slice",
    "rt60_slice",
    "codec_slice",
    "channel_slice",
    "distance_slice",
    "silence_ratio_bucket",
    "silence_slice",
)


@dataclass(frozen=True, slots=True)
class VerificationProtocolCleanSetConfig:
    bundle_id: str
    stage: str
    description: str
    trial_manifest_path: str
    metadata_manifest_path: str
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.bundle_id.strip():
            raise ValueError("clean_sets.bundle_id must not be empty.")
        if not self.stage.strip():
            raise ValueError("clean_sets.stage must not be empty.")
        if not self.description.strip():
            raise ValueError("clean_sets.description must not be empty.")
        if not self.trial_manifest_path.strip():
            raise ValueError("clean_sets.trial_manifest_path must not be empty.")
        if not self.metadata_manifest_path.strip():
            raise ValueError("clean_sets.metadata_manifest_path must not be empty.")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True, slots=True)
class VerificationProtocolConfig:
    title: str
    ticket_id: str
    protocol_id: str
    summary: str
    output_root: str
    required_slice_fields: tuple[str, ...]
    clean_sets: tuple[VerificationProtocolCleanSetConfig, ...]
    corrupted_suite_catalog_path: str | None
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must not be empty.")
        if not self.ticket_id.strip():
            raise ValueError("ticket_id must not be empty.")
        if not self.protocol_id.strip():
            raise ValueError("protocol_id must not be empty.")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty.")
        if not self.clean_sets:
            raise ValueError("At least one [[clean_sets]] entry is required.")
        if not self.required_slice_fields:
            raise ValueError("required_slice_fields must not be empty.")

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "ticket_id": self.ticket_id,
            "protocol_id": self.protocol_id,
            "summary": self.summary,
            "output_root": self.output_root,
            "required_slice_fields": list(self.required_slice_fields),
            "clean_sets": [item.to_dict() for item in self.clean_sets],
            "corrupted_suite_catalog_path": self.corrupted_suite_catalog_path,
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
        }


def load_verification_protocol_config(
    *,
    config_path: Path | str,
) -> VerificationProtocolConfig:
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Verification protocol config must be a TOML table.")

    protocol_id = str(raw.get("protocol_id", "")).strip()
    output_root = str(raw.get("output_root", "")).strip() or (
        f"artifacts/eval/{protocol_id or 'verification-protocol'}"
    )
    clean_sets_raw = raw.get("clean_sets")
    if not isinstance(clean_sets_raw, list):
        raise ValueError("[[clean_sets]] entries must be provided as a TOML array-of-tables.")

    return VerificationProtocolConfig(
        title=str(raw.get("title", "")).strip(),
        ticket_id=str(raw.get("ticket_id", "")).strip(),
        protocol_id=protocol_id,
        summary=str(raw.get("summary", "")).strip(),
        output_root=output_root,
        required_slice_fields=tuple(
            _coerce_string_list(
                raw.get("required_slice_fields", list(DEFAULT_PROTOCOL_REQUIRED_SLICE_FIELDS)),
                "required_slice_fields",
            )
        ),
        clean_sets=tuple(_load_clean_set(item) for item in clean_sets_raw),
        corrupted_suite_catalog_path=_coerce_optional_string(
            raw.get("corrupted_suite_catalog_path")
        ),
        validation_commands=tuple(
            _coerce_string_list(raw.get("validation_commands", []), "validation_commands")
        ),
        notes=tuple(_coerce_string_list(raw.get("notes", []), "notes")),
    )


def _load_clean_set(raw: object) -> VerificationProtocolCleanSetConfig:
    if not isinstance(raw, dict):
        raise ValueError("Each [[clean_sets]] entry must be a TOML table.")
    entry = cast(dict[str, object], raw)
    return VerificationProtocolCleanSetConfig(
        bundle_id=str(entry.get("bundle_id", "")).strip(),
        stage=str(entry.get("stage", "")).strip(),
        description=str(entry.get("description", "")).strip(),
        trial_manifest_path=str(entry.get("trial_manifest_path", "")).strip(),
        metadata_manifest_path=str(entry.get("metadata_manifest_path", "")).strip(),
        notes=tuple(_coerce_string_list(entry.get("notes", []), "clean_sets.notes")),
    )


def _coerce_string_list(raw: object, field_name: str) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be an array of strings.")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(f"{field_name}[{index}] must be a string.")
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"{field_name}[{index}] must not be empty.")
        values.append(stripped)
    return values


def _coerce_optional_string(raw: object) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("Configured path values must be strings when provided.")
    stripped = raw.strip()
    return stripped or None


__all__ = [
    "DEFAULT_PROTOCOL_REQUIRED_SLICE_FIELDS",
    "VerificationProtocolCleanSetConfig",
    "VerificationProtocolConfig",
    "load_verification_protocol_config",
]
