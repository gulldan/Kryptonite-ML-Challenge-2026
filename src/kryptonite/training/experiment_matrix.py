"""Reproducible experiment-matrix reports for planning training and evaluation work."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

EXPERIMENT_MATRIX_JSON_NAME = "experiment_matrix.json"
EXPERIMENT_MATRIX_MARKDOWN_NAME = "experiment_matrix.md"
SUPPORTED_PRIORITY_LEVELS = frozenset({"P0", "P1", "P2", "P3"})
SUPPORTED_REPO_STATES = frozenset({"ready_gpu", "ready_offline", "deferred_stretch"})
READY_REPO_STATES = frozenset({"ready_gpu", "ready_offline"})


@dataclass(frozen=True, slots=True)
class BudgetRange:
    gpu_hours_low: float
    gpu_hours_high: float

    def to_dict(self) -> dict[str, float]:
        return {
            "gpu_hours_low": self.gpu_hours_low,
            "gpu_hours_high": self.gpu_hours_high,
        }

    def render(self) -> str:
        if self.gpu_hours_low == self.gpu_hours_high:
            return _format_number(self.gpu_hours_low)
        return f"{_format_number(self.gpu_hours_low)}-{_format_number(self.gpu_hours_high)}"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    sequence: int
    experiment_id: str
    label: str
    linear_issue: str
    family: str
    track: str
    priority: str
    repo_state: str
    summary: str
    expected_effect: str
    budget: BudgetRange
    depends_on: tuple[str, ...] = ()
    evidence_paths: tuple[str, ...] = ()
    command: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sequence": self.sequence,
            "experiment_id": self.experiment_id,
            "label": self.label,
            "linear_issue": self.linear_issue,
            "family": self.family,
            "track": self.track,
            "priority": self.priority,
            "repo_state": self.repo_state,
            "summary": self.summary,
            "expected_effect": self.expected_effect,
            "budget": self.budget.to_dict(),
            "depends_on": list(self.depends_on),
            "evidence_paths": list(self.evidence_paths),
            "notes": list(self.notes),
        }
        if self.command is not None:
            payload["command"] = self.command
        return payload


@dataclass(frozen=True, slots=True)
class ExperimentMatrixConfig:
    title: str
    matrix_id: str
    accepted_at: str
    summary: str
    context: str
    output_root: str
    assumptions: tuple[str, ...]
    validation_commands: tuple[str, ...]
    experiments: tuple[ExperimentConfig, ...]


@dataclass(frozen=True, slots=True)
class ExperimentMatrixReport:
    title: str
    matrix_id: str
    accepted_at: str
    summary: str
    context: str
    output_root: str
    assumptions: tuple[str, ...]
    validation_commands: tuple[str, ...]
    experiments: tuple[ExperimentConfig, ...]
    total_budget: BudgetRange
    ready_budget: BudgetRange
    deferred_budget: BudgetRange
    priority_budgets: dict[str, BudgetRange]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "matrix_id": self.matrix_id,
            "accepted_at": self.accepted_at,
            "summary": self.summary,
            "context": self.context,
            "output_root": self.output_root,
            "assumptions": list(self.assumptions),
            "validation_commands": list(self.validation_commands),
            "experiments": [experiment.to_dict() for experiment in self.experiments],
            "total_budget": self.total_budget.to_dict(),
            "ready_budget": self.ready_budget.to_dict(),
            "deferred_budget": self.deferred_budget.to_dict(),
            "priority_budgets": {
                priority: budget.to_dict()
                for priority, budget in sorted(self.priority_budgets.items())
            },
        }


@dataclass(frozen=True, slots=True)
class WrittenExperimentMatrix:
    output_root: str
    report_json_path: str
    report_markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
        }


def load_experiment_matrix_config(*, config_path: Path | str) -> ExperimentMatrixConfig:
    raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    experiments = tuple(_load_experiment_config(item) for item in _require_list(raw, "experiment"))
    config = ExperimentMatrixConfig(
        title=_require_string(raw, "title"),
        matrix_id=_require_string(raw, "matrix_id"),
        accepted_at=_require_string(raw, "accepted_at"),
        summary=_require_string(raw, "summary"),
        context=_require_string(raw, "context"),
        output_root=_require_string(raw, "output_root"),
        assumptions=_require_string_tuple(raw.get("assumptions"), "assumptions"),
        validation_commands=_require_string_tuple(
            raw.get("validation_commands"), "validation_commands"
        ),
        experiments=experiments,
    )
    _validate_experiment_matrix_config(config)
    return config


def build_experiment_matrix(
    config: ExperimentMatrixConfig,
    *,
    project_root: Path | str = ".",
) -> ExperimentMatrixReport:
    resolved_root = Path(project_root)
    _validate_experiment_dependencies(config.experiments)
    _validate_evidence_paths_exist(config.experiments, project_root=resolved_root)

    total_budget = _sum_budgets(config.experiments)
    ready_budget = _sum_budgets(
        experiment
        for experiment in config.experiments
        if experiment.repo_state in READY_REPO_STATES
    )
    deferred_budget = _sum_budgets(
        experiment
        for experiment in config.experiments
        if experiment.repo_state == "deferred_stretch"
    )
    priority_budgets = {
        priority: _sum_budgets(
            experiment for experiment in config.experiments if experiment.priority == priority
        )
        for priority in SUPPORTED_PRIORITY_LEVELS
    }

    return ExperimentMatrixReport(
        title=config.title,
        matrix_id=config.matrix_id,
        accepted_at=config.accepted_at,
        summary=config.summary,
        context=config.context,
        output_root=config.output_root,
        assumptions=config.assumptions,
        validation_commands=config.validation_commands,
        experiments=tuple(sorted(config.experiments, key=lambda experiment: experiment.sequence)),
        total_budget=total_budget,
        ready_budget=ready_budget,
        deferred_budget=deferred_budget,
        priority_budgets=priority_budgets,
    )


def render_experiment_matrix_markdown(report: ExperimentMatrixReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"- Matrix id: `{report.matrix_id}`",
        f"- Accepted at: `{report.accepted_at}`",
        f"- Ready-now budget: `{report.ready_budget.render()} GPU-hours`",
        f"- Deferred stretch budget: `{report.deferred_budget.render()} GPU-hours`",
        f"- Full plan budget: `{report.total_budget.render()} GPU-hours`",
        "",
        "## Summary",
        "",
        report.summary,
        "",
        "## Context",
        "",
        report.context.strip(),
        "",
        "## Planning Assumptions",
        "",
    ]

    for assumption in report.assumptions:
        lines.append(f"- {assumption}")

    lines.extend(
        [
            "",
            "## Priority Matrix",
            "",
            "| Seq | Experiment | Linear | Priority | Repo State | GPU-hours | Expected effect |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for experiment in report.experiments:
        lines.append(
            "| "
            f"{experiment.sequence} | "
            f"{experiment.label} | "
            f"`{experiment.linear_issue}` | "
            f"`{experiment.priority}` | "
            f"`{_render_repo_state(experiment.repo_state)}` | "
            f"`{experiment.budget.render()}` | "
            f"{experiment.expected_effect} |"
        )

    lines.extend(["", "## Priority Budget Rollup", ""])
    for priority in sorted(report.priority_budgets):
        lines.append(f"- `{priority}`: `{report.priority_budgets[priority].render()} GPU-hours`")

    lines.extend(["", "## Experiment Details", ""])
    for experiment in report.experiments:
        lines.extend(
            [
                f"### {experiment.sequence}. {experiment.label}",
                "",
                f"- Linear: `{experiment.linear_issue}`",
                f"- Family: `{experiment.family}`",
                f"- Track: `{experiment.track}`",
                f"- Priority: `{experiment.priority}`",
                f"- Repo state: `{_render_repo_state(experiment.repo_state)}`",
                f"- Budget: `{experiment.budget.render()} GPU-hours`",
                f"- Summary: {experiment.summary}",
                f"- Expected effect: {experiment.expected_effect}",
            ]
        )

        if experiment.depends_on:
            dependencies = ", ".join(f"`{dependency}`" for dependency in experiment.depends_on)
            lines.append(f"- Depends on: {dependencies}")
        else:
            lines.append("- Depends on: none inside this matrix")

        if experiment.evidence_paths:
            evidence = ", ".join(f"`{path}`" for path in experiment.evidence_paths)
            lines.append(f"- Evidence: {evidence}")

        if experiment.notes:
            lines.append("- Notes:")
            for note in experiment.notes:
                lines.append(f"  - {note}")

        if experiment.command is not None:
            lines.extend(["", "```bash", experiment.command.strip(), "```"])

        lines.append("")

    lines.extend(["## Validation", ""])
    for command in report.validation_commands:
        lines.extend(["```bash", command.strip(), "```", ""])

    return "\n".join(lines).rstrip() + "\n"


def write_experiment_matrix(
    report: ExperimentMatrixReport,
    *,
    project_root: Path | str = ".",
) -> WrittenExperimentMatrix:
    output_dir = Path(project_root) / report.output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / EXPERIMENT_MATRIX_JSON_NAME
    markdown_path = output_dir / EXPERIMENT_MATRIX_MARKDOWN_NAME
    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_experiment_matrix_markdown(report), encoding="utf-8")

    return WrittenExperimentMatrix(
        output_root=str(output_dir),
        report_json_path=str(json_path),
        report_markdown_path=str(markdown_path),
    )


def _load_experiment_config(raw: object) -> ExperimentConfig:
    item = _require_table(raw, name="experiment")
    budget = BudgetRange(
        gpu_hours_low=_require_float(item.get("gpu_hours_low"), "gpu_hours_low"),
        gpu_hours_high=_require_float(item.get("gpu_hours_high"), "gpu_hours_high"),
    )
    if budget.gpu_hours_low > budget.gpu_hours_high:
        raise ValueError("gpu_hours_low must be less than or equal to gpu_hours_high.")
    if budget.gpu_hours_low < 0:
        raise ValueError("gpu_hours_low must be non-negative.")

    command = item.get("command")
    if command is not None and not isinstance(command, str):
        raise ValueError("command must be a string when provided.")

    return ExperimentConfig(
        sequence=_require_int(item.get("sequence"), "sequence"),
        experiment_id=_require_string(item, "experiment_id"),
        label=_require_string(item, "label"),
        linear_issue=_require_string(item, "linear_issue"),
        family=_require_string(item, "family"),
        track=_require_string(item, "track"),
        priority=_require_string(item, "priority"),
        repo_state=_require_string(item, "repo_state"),
        summary=_require_string(item, "summary"),
        expected_effect=_require_string(item, "expected_effect"),
        budget=budget,
        depends_on=_require_string_tuple(item.get("depends_on"), "depends_on"),
        evidence_paths=_require_string_tuple(item.get("evidence_paths"), "evidence_paths"),
        command=command.strip() if isinstance(command, str) and command.strip() else None,
        notes=_require_string_tuple(item.get("notes"), "notes"),
    )


def _validate_experiment_matrix_config(config: ExperimentMatrixConfig) -> None:
    if not config.experiments:
        raise ValueError("At least one [[experiment]] entry is required.")

    seen_ids: set[str] = set()
    seen_sequences: set[int] = set()
    for experiment in config.experiments:
        if experiment.experiment_id in seen_ids:
            raise ValueError(f"Duplicate experiment_id: {experiment.experiment_id}")
        seen_ids.add(experiment.experiment_id)

        if experiment.sequence in seen_sequences:
            raise ValueError(f"Duplicate experiment sequence: {experiment.sequence}")
        seen_sequences.add(experiment.sequence)

        if experiment.priority not in SUPPORTED_PRIORITY_LEVELS:
            raise ValueError(
                f"Unsupported priority `{experiment.priority}` for `{experiment.experiment_id}`."
            )
        if experiment.repo_state not in SUPPORTED_REPO_STATES:
            raise ValueError(
                "Unsupported repo_state "
                f"`{experiment.repo_state}` for `{experiment.experiment_id}`."
            )


def _validate_experiment_dependencies(experiments: tuple[ExperimentConfig, ...]) -> None:
    known_ids = {experiment.experiment_id for experiment in experiments}
    for experiment in experiments:
        for dependency in experiment.depends_on:
            if dependency not in known_ids:
                raise ValueError(
                    f"Unknown dependency `{dependency}` for `{experiment.experiment_id}`."
                )


def _validate_evidence_paths_exist(
    experiments: tuple[ExperimentConfig, ...],
    *,
    project_root: Path,
) -> None:
    for experiment in experiments:
        for evidence_path in experiment.evidence_paths:
            resolved_path = project_root / evidence_path
            if not resolved_path.exists():
                raise ValueError(
                    "Evidence path "
                    f"`{evidence_path}` for `{experiment.experiment_id}` does not exist."
                )


def _sum_budgets(experiments: Iterable[ExperimentConfig]) -> BudgetRange:
    low = 0.0
    high = 0.0
    for experiment in experiments:
        low += experiment.budget.gpu_hours_low
        high += experiment.budget.gpu_hours_high
    return BudgetRange(gpu_hours_low=low, gpu_hours_high=high)


def _render_repo_state(repo_state: str) -> str:
    return repo_state.replace("_", " ")


def _format_number(value: float) -> str:
    text = f"{value:.1f}"
    if text.endswith(".0"):
        return text[:-2]
    return text


def _require_table(raw: object, *, name: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a TOML table.")
    return cast("dict[str, object]", raw)


def _require_list(raw: dict[str, object], name: str) -> list[object]:
    value = raw.get(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a TOML array.")
    return cast("list[object]", value)


def _require_string(raw: dict[str, object], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _require_string_tuple(raw: object, name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{name} must be an array of strings.")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} must contain only non-empty strings.")
        values.append(item.strip())
    return tuple(values)


def _require_int(raw: object, name: str) -> int:
    if not isinstance(raw, int):
        raise ValueError(f"{name} must be an integer.")
    return raw


def _require_float(raw: object, name: str) -> float:
    if not isinstance(raw, (int, float)):
        raise ValueError(f"{name} must be a number.")
    return float(raw)
