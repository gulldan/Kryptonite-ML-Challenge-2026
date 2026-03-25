"""Rendering and writing helpers for verification error analysis."""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    VERIFICATION_ERROR_ANALYSIS_JSON_NAME,
    VERIFICATION_ERROR_ANALYSIS_MARKDOWN_NAME,
    VerificationDomainFailure,
    VerificationErrorAnalysisReport,
    VerificationErrorExample,
    VerificationSpeakerConfusion,
    VerificationSpeakerFailure,
    WrittenVerificationErrorAnalysis,
)


def write_verification_error_analysis_report(
    report: VerificationErrorAnalysisReport,
    *,
    output_root: Path | str,
) -> WrittenVerificationErrorAnalysis:
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    report_json_path = output_path / VERIFICATION_ERROR_ANALYSIS_JSON_NAME
    report_markdown_path = output_path / VERIFICATION_ERROR_ANALYSIS_MARKDOWN_NAME
    report_json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path.write_text(
        render_verification_error_analysis_markdown(report),
        encoding="utf-8",
    )
    return WrittenVerificationErrorAnalysis(
        output_root=str(output_path),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
        summary=report.summary,
    )


def render_verification_error_analysis_markdown(report: VerificationErrorAnalysisReport) -> str:
    summary = report.summary
    lines = [
        "# Verification Error Analysis",
        "",
        "## Decision Summary",
        "",
        f"- Threshold source: `{summary.threshold_source}`",
        f"- Decision threshold: `{summary.decision_threshold}`",
        f"- Trials: `{summary.trial_count}`",
        f"- False accepts: `{summary.false_accept_count}` ({summary.false_accept_rate})",
        f"- False rejects: `{summary.false_reject_count}` ({summary.false_reject_rate})",
        f"- Total errors: `{summary.total_error_count}` ({summary.total_error_rate})",
    ]
    if report.priority_findings:
        lines.extend(["", "## Priority Weak Spots", ""])
        for item in report.priority_findings:
            lines.append(
                "- "
                f"**{item.title}**: {item.evidence} "
                f"(errors `{item.error_count}` / trials `{item.trial_count}`, "
                f"rate `{item.error_rate}`)"
            )
    else:
        lines.extend(["", "## Priority Weak Spots", "", "- No thresholded errors found."])

    lines.extend(_render_example_section("Hard False Accepts", report.hard_false_accepts))
    lines.extend(_render_example_section("Hard False Rejects", report.hard_false_rejects))
    lines.extend(_render_domain_failure_section(report.domain_failures))
    lines.extend(_render_speaker_confusion_section(report.speaker_confusions))
    lines.extend(_render_speaker_failure_section(report.speaker_failures))
    return "\n".join(lines).rstrip() + "\n"


def _render_example_section(
    title: str,
    examples: tuple[VerificationErrorExample, ...],
) -> list[str]:
    lines = ["", f"## {title}", ""]
    if not examples:
        lines.append("- None.")
        return lines
    for example in examples:
        speaker_fragment = _format_speaker_fragment(
            example.left_speaker_id,
            example.right_speaker_id,
        )
        slice_fragments = [
            fragment
            for fragment in (
                _format_named_fragment("dataset", example.dataset),
                _format_named_fragment("channel", example.channel),
                _format_named_fragment("role", example.role_pair),
                _format_named_fragment("duration", example.duration_bucket),
                _format_named_fragment("noise", example.noise_slice),
                _format_named_fragment("reverb", example.reverb_slice),
                _format_named_fragment("channel_slice", example.channel_slice),
                _format_named_fragment("distance", example.distance_slice),
                _format_named_fragment("silence", example.silence_slice),
            )
            if fragment is not None
        ]
        details = ", ".join(slice_fragments)
        lines.append(
            "- "
            f"`{example.left_id}` vs `{example.right_id}` "
            f"({speaker_fragment}); score `{example.score}`, margin `{example.margin}`"
            + ("" if not details else f"; {details}")
        )
    return lines


def _render_domain_failure_section(
    rows: tuple[VerificationDomainFailure, ...],
) -> list[str]:
    lines = ["", "## Domain Failures", ""]
    if not rows:
        lines.append("- No slice-aware failures were available.")
        return lines
    for row in rows[:10]:
        lines.append(
            "- "
            f"`{row.field_name}={row.field_value}`: "
            f"errors `{row.error_count}` / trials `{row.trial_count}` "
            f"(rate `{row.error_rate}`), "
            f"FA `{row.false_accept_count}`, FR `{row.false_reject_count}`, "
            f"mean margin `{row.mean_error_margin}`"
        )
    return lines


def _render_speaker_confusion_section(
    rows: tuple[VerificationSpeakerConfusion, ...],
) -> list[str]:
    lines = ["", "## Speaker Confusions", ""]
    if not rows:
        lines.append("- No recurrent false-accept speaker pairs were found.")
        return lines
    for row in rows:
        lines.append(
            "- "
            f"`{row.speaker_a}` vs `{row.speaker_b}`: "
            f"false accepts `{row.false_accept_count}` / "
            f"negative trials `{row.trial_count}` (rate `{row.false_accept_rate}`), "
            f"mean score `{row.mean_false_accept_score}`, "
            f"max score `{row.max_false_accept_score}`"
        )
    return lines


def _render_speaker_failure_section(
    rows: tuple[VerificationSpeakerFailure, ...],
) -> list[str]:
    lines = ["", "## Speaker Fragility", ""]
    if not rows:
        lines.append("- No recurrent false-reject speakers were found.")
        return lines
    for row in rows:
        lines.append(
            "- "
            f"`{row.speaker_id}`: false rejects `{row.false_reject_count}` / "
            f"positive trials `{row.positive_trial_count}` "
            f"(rate `{row.false_reject_rate}`), "
            f"mean score `{row.mean_false_reject_score}`, "
            f"min score `{row.min_false_reject_score}`"
        )
    return lines


def _format_speaker_fragment(left_speaker_id: str | None, right_speaker_id: str | None) -> str:
    if left_speaker_id and right_speaker_id:
        return f"{left_speaker_id} -> {right_speaker_id}"
    return left_speaker_id or right_speaker_id or "unknown speakers"


def _format_named_fragment(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return f"{name} `{value}`"


__all__ = [
    "render_verification_error_analysis_markdown",
    "write_verification_error_analysis_report",
]
