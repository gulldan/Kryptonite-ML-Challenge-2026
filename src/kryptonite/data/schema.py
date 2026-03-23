"""Unified manifest schema helpers for manifests-backed audio datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MANIFEST_SCHEMA_VERSION = "kryptonite.manifest.v1"
MANIFEST_RECORD_TYPE = "utterance"


@dataclass(frozen=True, slots=True)
class ManifestValidationIssue:
    field: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "code": self.code,
            "message": self.message,
        }


class ManifestValidationError(ValueError):
    """Raised when a manifest row cannot be validated against the schema contract."""

    def __init__(
        self,
        *,
        issues: Sequence[ManifestValidationIssue],
        manifest_path: str | None = None,
        line_number: int | None = None,
    ) -> None:
        self.issues = list(issues)
        self.manifest_path = manifest_path
        self.line_number = line_number

        location_parts = [
            part for part in (manifest_path, _format_line_number(line_number)) if part
        ]
        location = " ".join(location_parts) if location_parts else "manifest row"
        details = "; ".join(f"{issue.field}: {issue.message}" for issue in self.issues)
        super().__init__(f"Invalid {location}: {details}")


@dataclass(frozen=True, slots=True)
class ManifestRow:
    dataset: str
    source_dataset: str
    speaker_id: str
    audio_path: str
    utterance_id: str | None = None
    session_id: str | None = None
    split: str | None = None
    role: str | None = None
    language: str | None = None
    device: str | None = None
    channel: str | None = None
    snr_db: float | None = None
    rir_id: str | None = None
    duration_seconds: float | None = None
    sample_rate_hz: int | None = None
    num_channels: int | None = None
    schema_version: str = MANIFEST_SCHEMA_VERSION
    record_type: str = MANIFEST_RECORD_TYPE

    @classmethod
    def from_mapping(
        cls,
        entry: Mapping[str, object],
        *,
        allow_legacy_aliases: bool = True,
        require_schema_version: bool = True,
        manifest_path: str | None = None,
        line_number: int | None = None,
    ) -> ManifestRow:
        issues = validate_manifest_entry(
            entry,
            allow_legacy_aliases=allow_legacy_aliases,
            require_schema_version=require_schema_version,
        )
        if issues:
            raise ManifestValidationError(
                issues=issues,
                manifest_path=manifest_path,
                line_number=line_number,
            )

        normalized = normalize_manifest_entry(entry, allow_legacy_aliases=allow_legacy_aliases)
        return cls(
            dataset=str(normalized["dataset"]),
            source_dataset=str(normalized["source_dataset"]),
            speaker_id=str(normalized["speaker_id"]),
            audio_path=str(normalized["audio_path"]),
            utterance_id=_coerce_string(normalized.get("utterance_id")),
            session_id=_coerce_string(normalized.get("session_id")),
            split=_coerce_string(normalized.get("split")),
            role=_coerce_string(normalized.get("role")),
            language=_coerce_string(normalized.get("language")),
            device=_coerce_string(normalized.get("device")),
            channel=_coerce_string(normalized.get("channel")),
            snr_db=_coerce_float(normalized.get("snr_db")),
            rir_id=_coerce_string(normalized.get("rir_id")),
            duration_seconds=_coerce_float(normalized.get("duration_seconds")),
            sample_rate_hz=_coerce_positive_int(normalized.get("sample_rate_hz")),
            num_channels=_coerce_positive_int(normalized.get("num_channels")),
            schema_version=str(normalized["schema_version"]),
            record_type=str(normalized["record_type"]),
        )

    def to_dict(self, *, extra_fields: Mapping[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "record_type": self.record_type,
            "dataset": self.dataset,
            "source_dataset": self.source_dataset,
            "speaker_id": self.speaker_id,
            "audio_path": self.audio_path,
        }
        for field_name in (
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
        ):
            value = getattr(self, field_name)
            if value is not None:
                payload[field_name] = value

        if extra_fields is not None:
            conflicting_keys = sorted(set(payload) & set(extra_fields))
            if conflicting_keys:
                joined = ", ".join(conflicting_keys)
                raise ValueError(
                    f"Extra fields must not override canonical manifest fields: {joined}"
                )
            payload.update(extra_fields)
        return payload


def normalize_manifest_entry(
    entry: Mapping[str, object],
    *,
    allow_legacy_aliases: bool = True,
) -> dict[str, object]:
    normalized: dict[str, object] = dict(entry)

    dataset = _coerce_string(entry.get("dataset")) or _coerce_string(entry.get("source_dataset"))
    source_dataset = _coerce_string(entry.get("source_dataset")) or dataset
    speaker_id = _coerce_string(entry.get("speaker_id"))
    session_id = _canonical_session_id(
        entry=entry, speaker_id=speaker_id, allow_legacy_aliases=allow_legacy_aliases
    )
    num_channels = _coerce_positive_int(entry.get("num_channels"))
    if num_channels is None and allow_legacy_aliases:
        num_channels = _coerce_positive_int(entry.get("channels"))
    channel = _coerce_string(entry.get("channel")) or _infer_channel_label(num_channels)
    snr_db = _coerce_float(entry.get("snr_db"))
    if snr_db is None and allow_legacy_aliases:
        snr_db = _coerce_float(entry.get("snr"))

    normalized["schema_version"] = (
        _coerce_string(entry.get("schema_version")) or MANIFEST_SCHEMA_VERSION
    )
    normalized["record_type"] = _coerce_string(entry.get("record_type")) or MANIFEST_RECORD_TYPE
    if dataset is not None:
        normalized["dataset"] = dataset
    if source_dataset is not None:
        normalized["source_dataset"] = source_dataset
    if speaker_id is not None:
        normalized["speaker_id"] = speaker_id
    if session_id is not None:
        normalized["session_id"] = session_id
    if num_channels is not None:
        normalized["num_channels"] = num_channels
    if channel is not None:
        normalized["channel"] = channel
    if snr_db is not None:
        normalized["snr_db"] = snr_db
    return normalized


def validate_manifest_entry(
    entry: Mapping[str, object],
    *,
    allow_legacy_aliases: bool = True,
    require_schema_version: bool = True,
) -> list[ManifestValidationIssue]:
    normalized = normalize_manifest_entry(entry, allow_legacy_aliases=allow_legacy_aliases)
    issues: list[ManifestValidationIssue] = []

    raw_schema_version = _coerce_string(entry.get("schema_version"))
    if require_schema_version and raw_schema_version is None:
        issues.append(
            ManifestValidationIssue(
                field="schema_version",
                code="missing_field",
                message=f"must be set to {MANIFEST_SCHEMA_VERSION!r}",
            )
        )
    elif raw_schema_version is not None and raw_schema_version != MANIFEST_SCHEMA_VERSION:
        issues.append(
            ManifestValidationIssue(
                field="schema_version",
                code="invalid_value",
                message=f"expected {MANIFEST_SCHEMA_VERSION!r}, got {raw_schema_version!r}",
            )
        )

    raw_record_type = _coerce_string(entry.get("record_type"))
    if require_schema_version and raw_record_type is None:
        issues.append(
            ManifestValidationIssue(
                field="record_type",
                code="missing_field",
                message=f"must be set to {MANIFEST_RECORD_TYPE!r}",
            )
        )
    record_type = _coerce_string(normalized.get("record_type"))
    if record_type != MANIFEST_RECORD_TYPE:
        issues.append(
            ManifestValidationIssue(
                field="record_type",
                code="invalid_value",
                message=f"expected {MANIFEST_RECORD_TYPE!r}, got {record_type!r}",
            )
        )

    for field_name in ("dataset", "speaker_id", "audio_path"):
        if _coerce_string(normalized.get(field_name)) is None:
            issues.append(
                ManifestValidationIssue(
                    field=field_name,
                    code="missing_field",
                    message="must be a non-empty string",
                )
            )

    source_dataset = _coerce_string(entry.get("source_dataset"))
    if require_schema_version and source_dataset is None:
        issues.append(
            ManifestValidationIssue(
                field="source_dataset",
                code="missing_field",
                message="must be a non-empty string",
            )
        )
    elif source_dataset is not None and _coerce_string(normalized.get("source_dataset")) is None:
        issues.append(
            ManifestValidationIssue(
                field="source_dataset",
                code="invalid_type",
                message="must be a non-empty string",
            )
        )

    _append_positive_float_issue(issues, normalized, "duration_seconds")
    _append_positive_float_issue(issues, normalized, "snr_db")
    _append_positive_int_issue(issues, normalized, "sample_rate_hz")
    _append_positive_int_issue(issues, normalized, "num_channels")

    for field_name in (
        "utterance_id",
        "session_id",
        "split",
        "role",
        "language",
        "device",
        "channel",
        "rir_id",
    ):
        if field_name in normalized and normalized[field_name] is not None:
            if _coerce_string(normalized[field_name]) is None:
                issues.append(
                    ManifestValidationIssue(
                        field=field_name,
                        code="invalid_type",
                        message="must be a non-empty string when present",
                    )
                )

    return issues


def _append_positive_float_issue(
    issues: list[ManifestValidationIssue],
    entry: Mapping[str, object],
    field_name: str,
) -> None:
    if field_name not in entry or entry[field_name] is None:
        return
    value = _coerce_float(entry[field_name])
    if value is None or value <= 0.0:
        issues.append(
            ManifestValidationIssue(
                field=field_name,
                code="invalid_value",
                message="must be a positive number when present",
            )
        )


def _append_positive_int_issue(
    issues: list[ManifestValidationIssue],
    entry: Mapping[str, object],
    field_name: str,
) -> None:
    if field_name not in entry or entry[field_name] is None:
        return
    value = _coerce_positive_int(entry[field_name])
    if value is None:
        issues.append(
            ManifestValidationIssue(
                field=field_name,
                code="invalid_value",
                message="must be a positive integer when present",
            )
        )


def _canonical_session_id(
    *,
    entry: Mapping[str, object],
    speaker_id: str | None,
    allow_legacy_aliases: bool,
) -> str | None:
    session_id = _coerce_string(entry.get("session_id"))
    if session_id is not None:
        if speaker_id and ":" not in session_id:
            return f"{speaker_id}:{session_id}"
        return session_id

    if not allow_legacy_aliases:
        return None

    session_index = _coerce_string(entry.get("session_index"))
    if session_index is None:
        return None
    if speaker_id is not None:
        return f"{speaker_id}:{session_index}"
    return session_index


def _infer_channel_label(num_channels: int | None) -> str | None:
    if num_channels == 1:
        return "mono"
    if num_channels == 2:
        return "stereo"
    return None


def _coerce_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, str)):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None


def _format_line_number(line_number: int | None) -> str | None:
    if line_number is None:
        return None
    return f"line {line_number}"
