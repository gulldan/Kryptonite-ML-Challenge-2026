"""Reproducible final family decision reports for export planning."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

FINAL_FAMILY_DECISION_JSON_NAME = "final_family_decision.json"
FINAL_FAMILY_DECISION_MARKDOWN_NAME = "final_family_decision.md"
SUPPORTED_FAMILY_OPTION_ROLES = frozenset({"production_student", "stretch_teacher", "alternative"})
SUPPORTED_FAMILY_OPTION_STATUSES = frozenset({"selected", "rejected", "deferred"})


@dataclass(frozen=True, slots=True)
class FinalFamilyOptionConfig:
    family_id: str
    label: str
    role: str
    status: str
    summary: str
    rationale: str
    export_readiness: str
    verification_report_path: str | None = None
    score_summary_path: str | None = None
    report_markdown_path: str | None = None
    checkpoint_path: str | None = None
    evidence_paths: tuple[str, ...] = ()
    rejected_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FinalFamilyDecisionConfig:
    title: str
    decision_id: str
    accepted_at: str
    summary: str
    context: str
    decision: str
    output_root: str
    selected_production_student: str
    selected_stretch_teacher: str
    decision_drivers: tuple[str, ...]
    next_step_issues: tuple[str, ...]
    options: tuple[FinalFamilyOptionConfig, ...]


@dataclass(frozen=True, slots=True)
class FinalFamilyMetrics:
    trial_count: int | None = None
    eer: float | None = None
    min_dcf: float | None = None
    mean_positive_score: float | None = None
    mean_negative_score: float | None = None
    score_gap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_count": self.trial_count,
            "eer": self.eer,
            "min_dcf": self.min_dcf,
            "mean_positive_score": self.mean_positive_score,
            "mean_negative_score": self.mean_negative_score,
            "score_gap": self.score_gap,
        }


@dataclass(frozen=True, slots=True)
class FinalFamilyOption:
    family_id: str
    label: str
    role: str
    status: str
    summary: str
    rationale: str
    export_readiness: str
    verification_report_path: str | None
    score_summary_path: str | None
    report_markdown_path: str | None
    checkpoint_path: str | None
    evidence_paths: tuple[str, ...]
    rejected_reason: str | None
    metrics: FinalFamilyMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "label": self.label,
            "role": self.role,
            "status": self.status,
            "summary": self.summary,
            "rationale": self.rationale,
            "export_readiness": self.export_readiness,
            "verification_report_path": self.verification_report_path,
            "score_summary_path": self.score_summary_path,
            "report_markdown_path": self.report_markdown_path,
            "checkpoint_path": self.checkpoint_path,
            "evidence_paths": list(self.evidence_paths),
            "rejected_reason": self.rejected_reason,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FinalFamilyDecisionReport:
    title: str
    decision_id: str
    accepted_at: str
    summary: str
    context: str
    decision: str
    output_root: str
    selected_production_student: FinalFamilyOption
    selected_stretch_teacher: FinalFamilyOption
    decision_drivers: tuple[str, ...]
    next_step_issues: tuple[str, ...]
    options: tuple[FinalFamilyOption, ...]

    @property
    def rejected_options(self) -> tuple[FinalFamilyOption, ...]:
        return tuple(option for option in self.options if option.status == "rejected")

    @property
    def deferred_options(self) -> tuple[FinalFamilyOption, ...]:
        return tuple(option for option in self.options if option.status == "deferred")

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "decision_id": self.decision_id,
            "accepted_at": self.accepted_at,
            "summary": self.summary,
            "context": self.context,
            "decision": self.decision,
            "output_root": self.output_root,
            "selected_production_student": self.selected_production_student.to_dict(),
            "selected_stretch_teacher": self.selected_stretch_teacher.to_dict(),
            "decision_drivers": list(self.decision_drivers),
            "next_step_issues": list(self.next_step_issues),
            "options": [option.to_dict() for option in self.options],
        }


@dataclass(frozen=True, slots=True)
class WrittenFinalFamilyDecision:
    output_root: str
    report_json_path: str
    report_markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
        }


def load_final_family_decision_config(*, config_path: Path | str) -> FinalFamilyDecisionConfig:
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    options = tuple(_load_option_config(item) for item in _require_list(raw, "option"))
    config = FinalFamilyDecisionConfig(
        title=_require_string(raw, "title"),
        decision_id=_require_string(raw, "decision_id"),
        accepted_at=_require_string(raw, "accepted_at"),
        summary=_require_string(raw, "summary"),
        context=_require_string(raw, "context"),
        decision=_require_string(raw, "decision"),
        output_root=_require_string(raw, "output_root"),
        selected_production_student=_require_string(raw, "selected_production_student"),
        selected_stretch_teacher=_require_string(raw, "selected_stretch_teacher"),
        decision_drivers=_require_string_tuple(raw.get("decision_drivers"), "decision_drivers"),
        next_step_issues=_require_string_tuple(raw.get("next_step_issues"), "next_step_issues"),
        options=options,
    )
    _validate_decision_config(config)
    return config


def build_final_family_decision(
    config: FinalFamilyDecisionConfig,
    *,
    project_root: Path | str = ".",
) -> FinalFamilyDecisionReport:
    resolved_root = Path(project_root)
    options = tuple(
        _build_option(option_config, project_root=resolved_root) for option_config in config.options
    )
    selected_production_student = _find_selected_option(
        options,
        family_id=config.selected_production_student,
        expected_role="production_student",
    )
    selected_stretch_teacher = _find_selected_option(
        options,
        family_id=config.selected_stretch_teacher,
        expected_role="stretch_teacher",
    )
    return FinalFamilyDecisionReport(
        title=config.title,
        decision_id=config.decision_id,
        accepted_at=config.accepted_at,
        summary=config.summary,
        context=config.context,
        decision=config.decision,
        output_root=config.output_root,
        selected_production_student=selected_production_student,
        selected_stretch_teacher=selected_stretch_teacher,
        decision_drivers=config.decision_drivers,
        next_step_issues=config.next_step_issues,
        options=options,
    )


def render_final_family_decision_markdown(report: FinalFamilyDecisionReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"- Decision id: `{report.decision_id}`",
        f"- Accepted at: `{report.accepted_at}`",
        f"- Production student family: `{report.selected_production_student.label}`",
        f"- Stretch teacher branch: `{report.selected_stretch_teacher.label}`",
        "",
        "## Context",
        "",
        report.context.strip(),
        "",
        "## Decision",
        "",
        report.decision.strip(),
        "",
        "## Decision Drivers",
        "",
    ]
    lines.extend(f"- {driver}" for driver in report.decision_drivers)
    lines.extend(
        [
            "",
            "## Option Matrix",
            "",
            "| Family | Role | Status | Export readiness | EER | MinDCF | Score gap | Trials |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for option in report.options:
        lines.append(
            "| "
            f"{option.label} | "
            f"{option.role} | "
            f"{option.status} | "
            f"{option.export_readiness} | "
            f"{_format_metric(option.metrics.eer)} | "
            f"{_format_metric(option.metrics.min_dcf)} | "
            f"{_format_metric(option.metrics.score_gap)} | "
            f"{_format_metric(option.metrics.trial_count)} |"
        )

    lines.extend(
        [
            "",
            "## Selected Production Student",
            "",
            _render_option_markdown(report.selected_production_student),
            "",
            "## Selected Stretch Teacher",
            "",
            _render_option_markdown(report.selected_stretch_teacher),
        ]
    )

    if report.rejected_options:
        lines.extend(["", "## Rejected Alternatives", ""])
        for option in report.rejected_options:
            lines.extend([_render_option_markdown(option), ""])

    if report.deferred_options:
        lines.extend(["## Deferred Options", ""])
        for option in report.deferred_options:
            lines.extend([_render_option_markdown(option), ""])

    lines.extend(["## Next Steps", ""])
    lines.extend(f"- `{issue}`" for issue in report.next_step_issues)
    return "\n".join(lines).rstrip() + "\n"


def write_final_family_decision(
    report: FinalFamilyDecisionReport,
    *,
    project_root: Path | str = ".",
) -> WrittenFinalFamilyDecision:
    resolved_output_root = Path(project_root) / report.output_root
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    json_path = resolved_output_root / FINAL_FAMILY_DECISION_JSON_NAME
    markdown_path = resolved_output_root / FINAL_FAMILY_DECISION_MARKDOWN_NAME
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_final_family_decision_markdown(report),
        encoding="utf-8",
    )
    return WrittenFinalFamilyDecision(
        output_root=str(resolved_output_root),
        report_json_path=str(json_path),
        report_markdown_path=str(markdown_path),
    )


def _load_option_config(raw: object) -> FinalFamilyOptionConfig:
    section = _require_mapping(raw, "option")
    return FinalFamilyOptionConfig(
        family_id=_require_string(section, "family_id"),
        label=_require_string(section, "label"),
        role=_normalize_role(_require_string(section, "role")),
        status=_normalize_status(_require_string(section, "status")),
        summary=_require_string(section, "summary"),
        rationale=_require_string(section, "rationale"),
        export_readiness=_require_string(section, "export_readiness"),
        verification_report_path=_optional_string(section.get("verification_report_path")),
        score_summary_path=_optional_string(section.get("score_summary_path")),
        report_markdown_path=_optional_string(section.get("report_markdown_path")),
        checkpoint_path=_optional_string(section.get("checkpoint_path")),
        evidence_paths=_require_string_tuple(section.get("evidence_paths", []), "evidence_paths"),
        rejected_reason=_optional_string(section.get("rejected_reason")),
    )


def _validate_decision_config(config: FinalFamilyDecisionConfig) -> None:
    if not config.options:
        raise ValueError("final family decision config must declare at least one [[option]].")
    option_ids = {option.family_id for option in config.options}
    if len(option_ids) != len(config.options):
        raise ValueError("final family decision config must use unique option family_id values.")
    if config.selected_production_student not in option_ids:
        raise ValueError("selected_production_student must match one declared option family_id.")
    if config.selected_stretch_teacher not in option_ids:
        raise ValueError("selected_stretch_teacher must match one declared option family_id.")


def _build_option(
    option_config: FinalFamilyOptionConfig,
    *,
    project_root: Path,
) -> FinalFamilyOption:
    verification_report_path = _resolve_optional_path(
        option_config.verification_report_path,
        project_root=project_root,
    )
    score_summary_path = _resolve_optional_path(
        option_config.score_summary_path,
        project_root=project_root,
    )
    report_markdown_path = _resolve_optional_path(
        option_config.report_markdown_path,
        project_root=project_root,
    )
    checkpoint_path = _resolve_optional_path(
        option_config.checkpoint_path,
        project_root=project_root,
    )
    evidence_paths = tuple(
        _resolve_required_path(path, project_root=project_root)
        for path in option_config.evidence_paths
    )
    metrics = _load_metrics(
        verification_report_path=verification_report_path,
        score_summary_path=score_summary_path,
        project_root=project_root,
    )
    return FinalFamilyOption(
        family_id=option_config.family_id,
        label=option_config.label,
        role=option_config.role,
        status=option_config.status,
        summary=option_config.summary,
        rationale=option_config.rationale,
        export_readiness=option_config.export_readiness,
        verification_report_path=verification_report_path,
        score_summary_path=score_summary_path,
        report_markdown_path=report_markdown_path,
        checkpoint_path=checkpoint_path,
        evidence_paths=evidence_paths,
        rejected_reason=option_config.rejected_reason,
        metrics=metrics,
    )


def _load_metrics(
    *,
    verification_report_path: str | None,
    score_summary_path: str | None,
    project_root: Path,
) -> FinalFamilyMetrics:
    trial_count: int | None = None
    eer: float | None = None
    min_dcf: float | None = None
    mean_positive_score: float | None = None
    mean_negative_score: float | None = None
    score_gap: float | None = None

    if verification_report_path is not None:
        payload = json.loads((project_root / verification_report_path).read_text(encoding="utf-8"))
        summary = _require_mapping(payload.get("summary"), "verification_report.summary")
        metrics = _require_mapping(summary.get("metrics"), "verification_report.summary.metrics")
        score_statistics = _require_mapping(
            summary.get("score_statistics"),
            "verification_report.summary.score_statistics",
        )
        trial_count = _optional_int(metrics.get("trial_count"))
        eer = _optional_float(metrics.get("eer"))
        min_dcf = _optional_float(metrics.get("min_dcf"))
        mean_positive_score = _optional_float(score_statistics.get("mean_positive_score"))
        mean_negative_score = _optional_float(score_statistics.get("mean_negative_score"))
        score_gap = _optional_float(score_statistics.get("score_gap"))

    if score_summary_path is not None:
        payload = json.loads((project_root / score_summary_path).read_text(encoding="utf-8"))
        trial_count = _optional_int(payload.get("trial_count"), fallback=trial_count)
        mean_positive_score = _optional_float(
            payload.get("mean_positive_score"),
            fallback=mean_positive_score,
        )
        mean_negative_score = _optional_float(
            payload.get("mean_negative_score"),
            fallback=mean_negative_score,
        )
        score_gap = _optional_float(payload.get("score_gap"), fallback=score_gap)

    return FinalFamilyMetrics(
        trial_count=trial_count,
        eer=eer,
        min_dcf=min_dcf,
        mean_positive_score=mean_positive_score,
        mean_negative_score=mean_negative_score,
        score_gap=score_gap,
    )


def _find_selected_option(
    options: tuple[FinalFamilyOption, ...],
    *,
    family_id: str,
    expected_role: str,
) -> FinalFamilyOption:
    for option in options:
        if option.family_id != family_id:
            continue
        if option.role != expected_role:
            raise ValueError(
                f"{family_id!r} is not tagged as {expected_role!r}; got {option.role!r}."
            )
        if option.status != "selected":
            raise ValueError(f"{family_id!r} must have status='selected'.")
        return option
    raise ValueError(f"Unknown selected option {family_id!r}.")


def _render_option_markdown(option: FinalFamilyOption) -> str:
    lines = [
        f"### {option.label}",
        "",
        f"- Role: `{option.role}`",
        f"- Status: `{option.status}`",
        f"- Export readiness: `{option.export_readiness}`",
        f"- Summary: {option.summary}",
        f"- Rationale: {option.rationale}",
    ]
    if option.rejected_reason:
        lines.append(f"- Rejected reason: {option.rejected_reason}")
    if option.metrics.trial_count is not None:
        lines.append(f"- Trials: `{option.metrics.trial_count}`")
    if option.metrics.eer is not None or option.metrics.min_dcf is not None:
        lines.append(
            "- Offline metrics: "
            f"EER `{_format_metric(option.metrics.eer)}`, "
            f"minDCF `{_format_metric(option.metrics.min_dcf)}`, "
            f"score gap `{_format_metric(option.metrics.score_gap)}`"
        )
    for label, path in (
        ("Verification report", option.verification_report_path),
        ("Score summary", option.score_summary_path),
        ("Markdown report", option.report_markdown_path),
        ("Checkpoint", option.checkpoint_path),
    ):
        if path is not None:
            lines.append(f"- {label}: `{path}`")
    if option.evidence_paths:
        lines.append(f"- Evidence paths: `{list(option.evidence_paths)}`")
    return "\n".join(lines)


def _resolve_optional_path(path: str | None, *, project_root: Path) -> str | None:
    if path is None:
        return None
    return _resolve_required_path(path, project_root=project_root)


def _resolve_required_path(path: str, *, project_root: Path) -> str:
    resolved = project_root / path
    if not resolved.exists():
        raise ValueError(f"Expected evidence path to exist: {path}")
    return path


def _normalize_role(value: str) -> str:
    if value not in SUPPORTED_FAMILY_OPTION_ROLES:
        raise ValueError(
            f"role must be one of {sorted(SUPPORTED_FAMILY_OPTION_ROLES)}, got {value!r}."
        )
    return value


def _normalize_status(value: str) -> str:
    if value not in SUPPORTED_FAMILY_OPTION_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(SUPPORTED_FAMILY_OPTION_STATUSES)}, got {value!r}."
        )
    return value


def _require_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a TOML table/object.")
    return cast(dict[str, Any], value)


def _require_list(raw: dict[str, Any], field_name: str) -> list[object]:
    value = raw.get(field_name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty TOML array.")
    return value


def _require_string(raw: dict[str, Any], field_name: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of strings.")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-empty string.")
        result.append(item.strip())
    return tuple(result)


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _optional_float(value: object, *, fallback: float | None = None) -> float | None:
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return float(value)
    return fallback


def _optional_int(value: object, *, fallback: int | None = None) -> int | None:
    if value is None:
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return fallback


def _format_metric(value: int | float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


__all__ = [
    "FINAL_FAMILY_DECISION_JSON_NAME",
    "FINAL_FAMILY_DECISION_MARKDOWN_NAME",
    "FinalFamilyDecisionConfig",
    "FinalFamilyDecisionReport",
    "FinalFamilyMetrics",
    "FinalFamilyOption",
    "FinalFamilyOptionConfig",
    "WrittenFinalFamilyDecision",
    "build_final_family_decision",
    "load_final_family_decision_config",
    "render_final_family_decision_markdown",
    "write_final_family_decision",
]
