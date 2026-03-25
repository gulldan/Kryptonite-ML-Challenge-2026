"""Machine-readable inventory and policy report for dataset resources."""

from __future__ import annotations

import json
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from kryptonite.deployment import resolve_project_path

InventoryStatus = Literal["approved", "conditional", "blocked"]

ALLOWED_STATUSES: tuple[InventoryStatus, ...] = ("approved", "conditional", "blocked")
STATUS_SORT_ORDER = {status: index for index, status in enumerate(ALLOWED_STATUSES)}


@dataclass(frozen=True, slots=True)
class DatasetInventorySource:
    id: str
    name: str
    kind: str
    status: InventoryStatus
    scopes: list[str]
    license: str
    access: str
    domain: str
    leakage_risk: str
    rationale: str
    source_urls: list[str] = field(default_factory=list)
    reference_paths: list[str] = field(default_factory=list)
    expected_paths: list[str] = field(default_factory=list)
    restrictions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DatasetInventoryPlan:
    notes: list[str]
    sources: list[DatasetInventorySource]


@dataclass(frozen=True, slots=True)
class InventoryPathCheck:
    configured_path: str
    resolved_path: str
    exists: bool
    path_type: str

    def to_dict(self) -> dict[str, object]:
        return {
            "configured_path": self.configured_path,
            "resolved_path": self.resolved_path,
            "exists": self.exists,
            "path_type": self.path_type,
        }


@dataclass(slots=True)
class DatasetInventoryEntry:
    source: DatasetInventorySource
    path_checks: list[InventoryPathCheck]

    @property
    def local_state(self) -> str:
        if not self.path_checks:
            return "not_tracked"
        existing_count = sum(1 for check in self.path_checks if check.exists)
        if existing_count == 0:
            return "missing"
        if existing_count == len(self.path_checks):
            return "present"
        return "partial"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.source.id,
            "name": self.source.name,
            "kind": self.source.kind,
            "status": self.source.status,
            "scopes": list(self.source.scopes),
            "license": self.source.license,
            "access": self.source.access,
            "domain": self.source.domain,
            "leakage_risk": self.source.leakage_risk,
            "rationale": self.source.rationale,
            "source_urls": list(self.source.source_urls),
            "reference_paths": list(self.source.reference_paths),
            "restrictions": list(self.source.restrictions),
            "notes": list(self.source.notes),
            "local_state": self.local_state,
            "path_checks": [check.to_dict() for check in self.path_checks],
        }


@dataclass(slots=True)
class DatasetInventoryReport:
    generated_at: str
    project_root: str
    plan_path: str | None
    notes: list[str]
    entries: list[DatasetInventoryEntry]

    @property
    def source_count(self) -> int:
        return len(self.entries)

    @property
    def status_counts(self) -> dict[str, int]:
        counts = Counter(entry.source.status for entry in self.entries)
        return {status: counts.get(status, 0) for status in ALLOWED_STATUSES}

    @property
    def local_state_counts(self) -> dict[str, int]:
        counts = Counter(entry.local_state for entry in self.entries)
        ordered_states = ("present", "partial", "missing", "not_tracked")
        return {state: counts.get(state, 0) for state in ordered_states if counts.get(state, 0)}

    @property
    def scope_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for entry in self.entries:
            counts.update(entry.source.scopes)
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at,
            "project_root": self.project_root,
            "plan_path": self.plan_path,
            "notes": list(self.notes),
            "source_count": self.source_count,
            "status_counts": self.status_counts,
            "local_state_counts": self.local_state_counts,
            "scope_counts": self.scope_counts,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class WrittenDatasetInventoryReport:
    output_root: str
    json_path: str
    markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
        }


def load_dataset_inventory_plan(path: Path | str) -> DatasetInventoryPlan:
    plan_path = Path(path)
    data = tomllib.loads(plan_path.read_text())
    sources_data = data.get("sources")
    if not isinstance(sources_data, list) or not sources_data:
        raise ValueError("Dataset inventory plan must define at least one [[sources]] entry.")

    sources: list[DatasetInventorySource] = []
    for source_data in sources_data:
        if not isinstance(source_data, dict):
            raise ValueError("Each inventory source entry must be a TOML table.")
        status = _require_literal(source_data, "status", ALLOWED_STATUSES)
        sources.append(
            DatasetInventorySource(
                id=_require_str(source_data, "id"),
                name=_require_str(source_data, "name"),
                kind=_require_str(source_data, "kind"),
                status=status,
                scopes=_require_str_list(source_data, "scopes"),
                license=_require_str(source_data, "license"),
                access=_require_str(source_data, "access"),
                domain=_require_str(source_data, "domain"),
                leakage_risk=_require_str(source_data, "leakage_risk"),
                rationale=_require_str(source_data, "rationale"),
                source_urls=_optional_str_list(source_data, "source_urls"),
                reference_paths=_optional_str_list(source_data, "reference_paths"),
                expected_paths=_optional_str_list(source_data, "expected_paths"),
                restrictions=_optional_str_list(source_data, "restrictions"),
                notes=_optional_str_list(source_data, "notes"),
            )
        )

    return DatasetInventoryPlan(
        notes=_optional_str_list(data, "notes"),
        sources=sources,
    )


def build_dataset_inventory_report(
    *,
    project_root: Path | str,
    plan: DatasetInventoryPlan,
    plan_path: Path | str | None = None,
) -> DatasetInventoryReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    resolved_plan_path = (
        str(resolve_project_path(str(project_root_path), str(plan_path)))
        if plan_path is not None
        else None
    )
    entries = [
        DatasetInventoryEntry(
            source=source,
            path_checks=[
                _build_path_check(project_root=project_root_path, configured_path=configured_path)
                for configured_path in source.expected_paths
            ],
        )
        for source in sorted(plan.sources, key=_sort_source)
    ]
    return DatasetInventoryReport(
        generated_at=_utc_now(),
        project_root=str(project_root_path),
        plan_path=resolved_plan_path,
        notes=list(plan.notes),
        entries=entries,
    )


def render_dataset_inventory_markdown(report: DatasetInventoryReport) -> str:
    lines = [
        "# Dataset Inventory Report",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Project root: `{report.project_root}`",
        f"- Plan path: `{report.plan_path or '-'}`",
        "",
    ]

    if report.notes:
        lines.extend(["## Policy Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
        lines.append("")

    lines.extend(
        [
            "## Overview",
            "",
            markdown_table(
                ["Metric", "Value"],
                [
                    ["Sources", str(report.source_count)],
                    ["Approved", str(report.status_counts.get("approved", 0))],
                    ["Conditional", str(report.status_counts.get("conditional", 0))],
                    ["Blocked", str(report.status_counts.get("blocked", 0))],
                    ["Present locally", str(report.local_state_counts.get("present", 0))],
                    ["Partially present locally", str(report.local_state_counts.get("partial", 0))],
                    ["Missing locally", str(report.local_state_counts.get("missing", 0))],
                    ["Scopes", _format_counts(report.scope_counts)],
                ],
            ),
            "",
            "## Source Matrix",
            "",
            markdown_table(
                ["Source", "Status", "Scopes", "Local state", "Kind"],
                [
                    [
                        entry.source.name,
                        entry.source.status,
                        ", ".join(entry.source.scopes),
                        entry.local_state,
                        entry.source.kind,
                    ]
                    for entry in report.entries
                ],
            ),
            "",
            "## Datasheets",
            "",
        ]
    )

    for entry in report.entries:
        lines.extend(_render_entry(entry))

    return "\n".join(lines).rstrip() + "\n"


def write_dataset_inventory_report(
    *,
    report: DatasetInventoryReport,
    output_root: Path | str,
) -> WrittenDatasetInventoryReport:
    output_root_path = resolve_project_path(report.project_root, str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)

    json_path = output_root_path / "dataset_inventory.json"
    markdown_path = output_root_path / "dataset_inventory.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_dataset_inventory_markdown(report))
    return WrittenDatasetInventoryReport(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |" for row in rows
    ]
    return "\n".join([header_row, separator_row, *body_rows])


def _render_entry(entry: DatasetInventoryEntry) -> list[str]:
    source = entry.source
    lines = [
        f"### {source.name}",
        "",
        f"- Id: `{source.id}`",
        f"- Status: `{source.status}`",
        f"- Kind: `{source.kind}`",
        f"- Scopes: `{', '.join(source.scopes)}`",
        f"- Local state: `{entry.local_state}`",
        f"- License / terms: {source.license}",
        f"- Access: {source.access}",
        f"- Domain fit: {source.domain}",
        f"- Leakage risk: {source.leakage_risk}",
        f"- Rationale: {source.rationale}",
    ]

    if source.restrictions:
        lines.append("- Restrictions:")
        lines.extend(f"  - {restriction}" for restriction in source.restrictions)

    if source.notes:
        lines.append("- Notes:")
        lines.extend(f"  - {note}" for note in source.notes)

    if source.source_urls:
        lines.append("- Source URLs:")
        lines.extend(f"  - {url}" for url in source.source_urls)

    if source.reference_paths:
        lines.append("- Repo references:")
        lines.extend(f"  - `{path}`" for path in source.reference_paths)

    if entry.path_checks:
        lines.append("- Expected paths:")
        lines.extend(
            "  - "
            f"`{check.configured_path}` -> `{check.path_type}` "
            f"({'present' if check.exists else 'missing'})"
            for check in entry.path_checks
        )

    lines.append("")
    return lines


def _build_path_check(*, project_root: Path, configured_path: str) -> InventoryPathCheck:
    resolved_path = resolve_project_path(str(project_root), configured_path)
    if resolved_path.is_dir():
        path_type = "dir"
    elif resolved_path.is_file():
        path_type = "file"
    else:
        path_type = "missing"
    return InventoryPathCheck(
        configured_path=configured_path,
        resolved_path=str(resolved_path),
        exists=resolved_path.exists(),
        path_type=path_type,
    )


def _sort_source(source: DatasetInventorySource) -> tuple[int, str, str]:
    return (
        STATUS_SORT_ORDER[source.status],
        ",".join(source.scopes),
        source.name.lower(),
    )


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Inventory field '{key}' is missing or invalid.")
    return value


def _require_str_list(data: dict[str, object], key: str) -> list[str]:
    values = _optional_str_list(data, key)
    if not values:
        raise ValueError(f"Inventory field '{key}' must contain at least one string.")
    return values


def _optional_str_list(data: dict[str, object], key: str) -> list[str]:
    value = data.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Inventory field '{key}' must be a list of strings.")
    return cast(list[str], list(value))


def _require_literal(
    data: dict[str, object],
    key: str,
    allowed_values: tuple[str, ...],
) -> InventoryStatus:
    value = _require_str(data, key)
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise ValueError(f"Inventory field '{key}' must be one of: {allowed}.")
    return cast(InventoryStatus, value)


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()
