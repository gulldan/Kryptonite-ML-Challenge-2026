"""Render and write the internal verification-protocol snapshot."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VERIFICATION_PROTOCOL_JSON_NAME = "verification_protocol.json"
VERIFICATION_PROTOCOL_MARKDOWN_NAME = "verification_protocol.md"


@dataclass(frozen=True, slots=True)
class VerificationProtocolTrialSource:
    path: str
    exists: bool
    sha256: str | None
    trial_count: int
    positive_count: int
    negative_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationProtocolSliceDefinition:
    field_name: str
    title: str
    description: str
    required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationProtocolSliceObservation:
    field_name: str
    value_count: int
    sample_values: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sample_values"] = list(self.sample_values)
        return payload


@dataclass(frozen=True, slots=True)
class VerificationProtocolBundle:
    bundle_id: str
    stage: str
    family: str
    description: str
    metadata_path: str | None
    metadata_exists: bool
    metadata_sha256: str | None
    metadata_row_count: int
    speaker_count: int
    audio_count: int
    trial_sources: tuple[VerificationProtocolTrialSource, ...]
    available_trial_fields: tuple[str, ...]
    available_slice_fields: tuple[str, ...]
    missing_required_slice_fields: tuple[str, ...]
    slice_observations: tuple[VerificationProtocolSliceObservation, ...]
    notes: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "stage": self.stage,
            "family": self.family,
            "description": self.description,
            "metadata_path": self.metadata_path,
            "metadata_exists": self.metadata_exists,
            "metadata_sha256": self.metadata_sha256,
            "metadata_row_count": self.metadata_row_count,
            "speaker_count": self.speaker_count,
            "audio_count": self.audio_count,
            "trial_sources": [source.to_dict() for source in self.trial_sources],
            "available_trial_fields": list(self.available_trial_fields),
            "available_slice_fields": list(self.available_slice_fields),
            "missing_required_slice_fields": list(self.missing_required_slice_fields),
            "slice_observations": [item.to_dict() for item in self.slice_observations],
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class VerificationProtocolSummary:
    clean_bundle_count: int
    production_bundle_count: int
    covered_required_slice_fields: tuple[str, ...]
    missing_required_slice_fields: tuple[str, ...]
    warning_count: int
    is_complete: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["covered_required_slice_fields"] = list(self.covered_required_slice_fields)
        payload["missing_required_slice_fields"] = list(self.missing_required_slice_fields)
        return payload


@dataclass(frozen=True, slots=True)
class VerificationProtocolReport:
    title: str
    ticket_id: str
    protocol_id: str
    summary_text: str
    output_root: str
    source_config_path: str | None
    source_config_sha256: str | None
    required_slice_fields: tuple[str, ...]
    slice_definitions: tuple[VerificationProtocolSliceDefinition, ...]
    clean_bundles: tuple[VerificationProtocolBundle, ...]
    production_bundles: tuple[VerificationProtocolBundle, ...]
    validation_commands: tuple[str, ...]
    notes: tuple[str, ...]
    summary: VerificationProtocolSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "ticket_id": self.ticket_id,
            "protocol_id": self.protocol_id,
            "summary_text": self.summary_text,
            "output_root": self.output_root,
            "source_config_path": self.source_config_path,
            "source_config_sha256": self.source_config_sha256,
            "required_slice_fields": list(self.required_slice_fields),
            "slice_definitions": [item.to_dict() for item in self.slice_definitions],
            "clean_bundles": [item.to_dict() for item in self.clean_bundles],
            "production_bundles": [item.to_dict() for item in self.production_bundles],
            "validation_commands": list(self.validation_commands),
            "notes": list(self.notes),
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class WrittenVerificationProtocolReport:
    output_root: str
    report_json_path: str
    report_markdown_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def render_verification_protocol_markdown(report: VerificationProtocolReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"- Ticket: `{report.ticket_id}`",
        f"- Protocol id: `{report.protocol_id}`",
        f"- Output root: `{report.output_root}`",
        f"- Complete: `{report.summary.is_complete}`",
        "- Missing required slices: "
        f"`{', '.join(report.summary.missing_required_slice_fields) or 'none'}`",
        "",
        "## Summary",
        "",
        report.summary_text,
        "",
        "## Required Slices",
        "",
    ]
    for item in report.slice_definitions:
        marker = "required" if item.required else "optional"
        lines.extend(
            [
                f"### {item.title} (`{item.field_name}`)",
                "",
                f"- Status: `{marker}`",
                f"- Description: {item.description}",
                "",
            ]
        )
    lines.extend(_render_bundle_section("Clean Bundles", report.clean_bundles))
    lines.extend(_render_bundle_section("Production-Like Bundles", report.production_bundles))
    if report.validation_commands:
        lines.extend(["## Validation Commands", ""])
        lines.extend(f"- `{command}`" for command in report.validation_commands)
        lines.append("")
    if report.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
        lines.append("")
    return "\n".join(lines).rstrip()


def write_verification_protocol_report(
    report: VerificationProtocolReport,
    *,
    output_root: Path | str | None = None,
) -> WrittenVerificationProtocolReport:
    resolved_output_root = Path(report.output_root if output_root is None else output_root)
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    report_json_path = resolved_output_root / VERIFICATION_PROTOCOL_JSON_NAME
    report_markdown_path = resolved_output_root / VERIFICATION_PROTOCOL_MARKDOWN_NAME
    report_json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    report_markdown_path.write_text(render_verification_protocol_markdown(report) + "\n")
    return WrittenVerificationProtocolReport(
        output_root=str(resolved_output_root),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
    )


def _render_bundle_section(
    title: str,
    bundles: tuple[VerificationProtocolBundle, ...],
) -> list[str]:
    lines = [f"## {title}", ""]
    if not bundles:
        lines.extend(["- No bundles were configured.", ""])
        return lines
    for bundle in bundles:
        lines.extend(
            [
                f"### {bundle.bundle_id}",
                "",
                f"- Stage: `{bundle.stage}`",
                f"- Family: `{bundle.family}`",
                f"- Description: {bundle.description}",
                f"- Metadata: `{bundle.metadata_path or 'none'}`",
                f"- Metadata rows: `{bundle.metadata_row_count}`",
                f"- Speakers: `{bundle.speaker_count}`",
                f"- Audio rows: `{bundle.audio_count}`",
                f"- Available slices: `{', '.join(bundle.available_slice_fields) or 'none'}`",
                (
                    "- Missing required slices: "
                    f"`{', '.join(bundle.missing_required_slice_fields) or 'none'}`"
                ),
            ]
        )
        if bundle.trial_sources:
            lines.append("- Trial sources:")
            for source in bundle.trial_sources:
                lines.append(
                    "  "
                    + f"`{source.path}` trials=`{source.trial_count}` "
                    + f"pos=`{source.positive_count}` neg=`{source.negative_count}` "
                    + f"exists=`{source.exists}`"
                )
        if bundle.slice_observations:
            lines.append("- Slice observations:")
            for item in bundle.slice_observations:
                lines.append(
                    "  "
                    + f"`{item.field_name}` values=`{item.value_count}` "
                    + f"samples=`{', '.join(item.sample_values)}`"
                )
        if bundle.notes:
            lines.append("- Notes:")
            for note in bundle.notes:
                lines.append(f"  {note}")
        if bundle.warnings:
            lines.append("- Warnings:")
            for warning in bundle.warnings:
                lines.append(f"  {warning}")
        lines.append("")
    return lines


__all__ = [
    "VERIFICATION_PROTOCOL_JSON_NAME",
    "VERIFICATION_PROTOCOL_MARKDOWN_NAME",
    "VerificationProtocolBundle",
    "VerificationProtocolReport",
    "VerificationProtocolSliceDefinition",
    "VerificationProtocolSliceObservation",
    "VerificationProtocolSummary",
    "VerificationProtocolTrialSource",
    "WrittenVerificationProtocolReport",
    "render_verification_protocol_markdown",
    "write_verification_protocol_report",
]
