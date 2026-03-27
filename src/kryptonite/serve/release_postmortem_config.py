"""Typed config loader for release postmortem and backlog reports."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUPPORTED_RELEASE_POSTMORTEM_OUTCOMES = frozenset({"worked", "missed", "risk"})
SUPPORTED_RELEASE_POSTMORTEM_DISPOSITIONS = frozenset({"next_iteration", "de_scoped", "monitor"})
SUPPORTED_RELEASE_POSTMORTEM_PRIORITIES = frozenset({"P0", "P1", "P2", "P3"})


def normalize_release_postmortem_outcome(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_RELEASE_POSTMORTEM_OUTCOMES:
        raise ValueError(
            "finding outcome must be one of "
            f"{sorted(SUPPORTED_RELEASE_POSTMORTEM_OUTCOMES)}, got {value!r}."
        )
    return normalized


def normalize_release_postmortem_disposition(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_RELEASE_POSTMORTEM_DISPOSITIONS:
        raise ValueError(
            "backlog disposition must be one of "
            f"{sorted(SUPPORTED_RELEASE_POSTMORTEM_DISPOSITIONS)}, got {value!r}."
        )
    return normalized


def normalize_release_postmortem_priority(value: str) -> str:
    normalized = value.strip().upper()
    if normalized not in SUPPORTED_RELEASE_POSTMORTEM_PRIORITIES:
        raise ValueError(
            "backlog priority must be one of "
            f"{sorted(SUPPORTED_RELEASE_POSTMORTEM_PRIORITIES)}, got {value!r}."
        )
    return normalized


@dataclass(frozen=True, slots=True)
class ReleasePostmortemEvidenceConfig:
    label: str
    kind: str
    path: str
    description: str

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("evidence.label must not be empty.")
        if not self.kind.strip():
            raise ValueError("evidence.kind must not be empty.")
        if not self.path.strip():
            raise ValueError("evidence.path must not be empty.")
        if not self.description.strip():
            raise ValueError("evidence.description must not be empty.")


@dataclass(frozen=True, slots=True)
class ReleasePostmortemFindingConfig:
    area: str
    outcome: str
    title: str
    detail: str
    evidence_labels: tuple[str, ...]
    related_issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.area.strip():
            raise ValueError("finding.area must not be empty.")
        object.__setattr__(
            self,
            "outcome",
            normalize_release_postmortem_outcome(self.outcome),
        )
        if not self.title.strip():
            raise ValueError("finding.title must not be empty.")
        if not self.detail.strip():
            raise ValueError("finding.detail must not be empty.")


@dataclass(frozen=True, slots=True)
class ReleaseBacklogItemConfig:
    title: str
    priority: str
    disposition: str
    area: str
    rationale: str
    related_issue: str | None
    dependencies: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("backlog_item.title must not be empty.")
        object.__setattr__(
            self,
            "priority",
            normalize_release_postmortem_priority(self.priority),
        )
        object.__setattr__(
            self,
            "disposition",
            normalize_release_postmortem_disposition(self.disposition),
        )
        if not self.area.strip():
            raise ValueError("backlog_item.area must not be empty.")
        if not self.rationale.strip():
            raise ValueError("backlog_item.rationale must not be empty.")


@dataclass(frozen=True, slots=True)
class ReleasePostmortemConfig:
    title: str
    release_id: str
    release_tag: str | None
    summary: str
    output_root: str
    evidence: tuple[ReleasePostmortemEvidenceConfig, ...]
    findings: tuple[ReleasePostmortemFindingConfig, ...]
    backlog_items: tuple[ReleaseBacklogItemConfig, ...]
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("title must not be empty.")
        if not self.release_id.strip():
            raise ValueError("release_id must not be empty.")
        if not self.output_root.strip():
            raise ValueError("output_root must not be empty.")
        if not self.evidence:
            raise ValueError("evidence must include at least one source entry.")
        if not self.findings:
            raise ValueError("findings must include at least one finding.")
        if not self.backlog_items:
            raise ValueError("backlog_items must include at least one item.")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["evidence"] = [asdict(item) for item in self.evidence]
        payload["findings"] = [asdict(item) for item in self.findings]
        payload["backlog_items"] = [asdict(item) for item in self.backlog_items]
        payload["validation_commands"] = list(self.validation_commands)
        payload["notes"] = list(self.notes)
        return payload


def load_release_postmortem_config(*, config_path: Path | str) -> ReleasePostmortemConfig:
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    release_id = str(raw.get("release_id", "")).strip()
    output_root = str(raw.get("output_root", "")).strip() or (
        f"artifacts/release-postmortems/{release_id}"
    )
    release_tag = str(raw.get("release_tag", "")).strip() or None
    return ReleasePostmortemConfig(
        title=str(raw.get("title", "")).strip(),
        release_id=release_id,
        release_tag=release_tag,
        summary=str(raw.get("summary", "")).strip(),
        output_root=output_root,
        evidence=tuple(_load_evidence_items(raw.get("evidence", []))),
        findings=tuple(_load_findings(raw.get("finding", []))),
        backlog_items=tuple(_load_backlog_items(raw.get("backlog_item", []))),
        validation_commands=tuple(
            _coerce_string_list(raw.get("validation_commands", []), "validation_commands")
        ),
        notes=tuple(_coerce_string_list(raw.get("notes", []), "notes")),
    )


def _load_evidence_items(raw: object) -> list[ReleasePostmortemEvidenceConfig]:
    entries = _coerce_table_list(raw, "evidence")
    return [
        ReleasePostmortemEvidenceConfig(
            label=str(entry.get("label", "")).strip(),
            kind=str(entry.get("kind", "")).strip(),
            path=str(entry.get("path", "")).strip(),
            description=str(entry.get("description", "")).strip(),
        )
        for entry in entries
    ]


def _load_findings(raw: object) -> list[ReleasePostmortemFindingConfig]:
    entries = _coerce_table_list(raw, "finding")
    return [
        ReleasePostmortemFindingConfig(
            area=str(entry.get("area", "")).strip(),
            outcome=str(entry.get("outcome", "")).strip(),
            title=str(entry.get("title", "")).strip(),
            detail=str(entry.get("detail", "")).strip(),
            evidence_labels=tuple(
                _coerce_string_list(entry.get("evidence", []), "finding.evidence")
            ),
            related_issues=tuple(
                _coerce_string_list(entry.get("related_issues", []), "finding.related_issues")
            ),
        )
        for entry in entries
    ]


def _load_backlog_items(raw: object) -> list[ReleaseBacklogItemConfig]:
    entries = _coerce_table_list(raw, "backlog_item")
    return [
        ReleaseBacklogItemConfig(
            title=str(entry.get("title", "")).strip(),
            priority=str(entry.get("priority", "")).strip(),
            disposition=str(entry.get("disposition", "")).strip(),
            area=str(entry.get("area", "")).strip(),
            rationale=str(entry.get("rationale", "")).strip(),
            related_issue=_coerce_optional_string(entry.get("related_issue")),
            dependencies=tuple(
                _coerce_string_list(entry.get("dependencies", []), "backlog_item.dependencies")
            ),
        )
        for entry in entries
    ]


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


def _coerce_table_list(raw: object, field_name: str) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{field_name} must be an array of tables.")
    values: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}[{index}] must be a table.")
        values.append({str(key): value for key, value in item.items()})
    return values


def _coerce_optional_string(raw: object) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("related_issue must be a string when provided.")
    stripped = raw.strip()
    return stripped or None


__all__ = [
    "ReleaseBacklogItemConfig",
    "ReleasePostmortemConfig",
    "ReleasePostmortemEvidenceConfig",
    "ReleasePostmortemFindingConfig",
    "SUPPORTED_RELEASE_POSTMORTEM_DISPOSITIONS",
    "SUPPORTED_RELEASE_POSTMORTEM_OUTCOMES",
    "SUPPORTED_RELEASE_POSTMORTEM_PRIORITIES",
    "load_release_postmortem_config",
    "normalize_release_postmortem_disposition",
    "normalize_release_postmortem_outcome",
    "normalize_release_postmortem_priority",
]
