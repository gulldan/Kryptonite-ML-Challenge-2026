"""Build and write thresholded verification error-analysis artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .verification_data import (
    build_trial_item_index,
    resolve_trial_side_identifier,
)

VERIFICATION_ERROR_ANALYSIS_JSON_NAME = "verification_error_analysis.json"
VERIFICATION_ERROR_ANALYSIS_MARKDOWN_NAME = "verification_error_analysis.md"

_PAIR_FIELDS = {"dataset", "source_dataset", "channel", "device", "language", "split", "role"}
_DURATION_BUCKETS: tuple[tuple[float, float | None, str], ...] = (
    (0.0, 1.0, "lt_1s"),
    (1.0, 2.0, "1_to_2s"),
    (2.0, 4.0, "2_to_4s"),
    (4.0, 8.0, "4_to_8s"),
    (8.0, None, "8_plus_s"),
)


@dataclass(frozen=True, slots=True)
class VerificationErrorAnalysisSummary:
    threshold_source: str
    decision_threshold: float
    trial_count: int
    positive_count: int
    negative_count: int
    false_accept_count: int
    false_reject_count: int
    total_error_count: int
    false_accept_rate: float
    false_reject_rate: float
    total_error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationErrorExample:
    error_type: str
    score: float
    label: int
    margin: float
    left_id: str | None
    right_id: str | None
    left_speaker_id: str | None
    right_speaker_id: str | None
    dataset: str | None
    channel: str | None
    role_pair: str | None
    duration_bucket: str | None
    noise_slice: str | None
    reverb_slice: str | None
    channel_slice: str | None
    distance_slice: str | None
    silence_slice: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationDomainFailure:
    field_name: str
    field_value: str
    trial_count: int
    error_count: int
    false_accept_count: int
    false_reject_count: int
    error_rate: float
    mean_error_margin: float
    mean_error_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationSpeakerConfusion:
    speaker_a: str
    speaker_b: str
    trial_count: int
    false_accept_count: int
    false_accept_rate: float
    mean_false_accept_score: float
    max_false_accept_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationSpeakerFailure:
    speaker_id: str
    positive_trial_count: int
    false_reject_count: int
    false_reject_rate: float
    mean_false_reject_score: float
    min_false_reject_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationPriorityFinding:
    finding_type: str
    title: str
    evidence: str
    trial_count: int
    error_count: int
    error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VerificationErrorAnalysisReport:
    summary: VerificationErrorAnalysisSummary
    priority_findings: tuple[VerificationPriorityFinding, ...]
    hard_false_accepts: tuple[VerificationErrorExample, ...]
    hard_false_rejects: tuple[VerificationErrorExample, ...]
    domain_failures: tuple[VerificationDomainFailure, ...]
    speaker_confusions: tuple[VerificationSpeakerConfusion, ...]
    speaker_failures: tuple[VerificationSpeakerFailure, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "priority_findings": [item.to_dict() for item in self.priority_findings],
            "hard_false_accepts": [item.to_dict() for item in self.hard_false_accepts],
            "hard_false_rejects": [item.to_dict() for item in self.hard_false_rejects],
            "domain_failures": [item.to_dict() for item in self.domain_failures],
            "speaker_confusions": [item.to_dict() for item in self.speaker_confusions],
            "speaker_failures": [item.to_dict() for item in self.speaker_failures],
        }


@dataclass(frozen=True, slots=True)
class WrittenVerificationErrorAnalysis:
    output_root: str
    report_json_path: str
    report_markdown_path: str
    summary: VerificationErrorAnalysisSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_root": self.output_root,
            "report_json_path": self.report_json_path,
            "report_markdown_path": self.report_markdown_path,
            "summary": self.summary.to_dict(),
        }


def build_verification_error_analysis(
    score_rows: list[dict[str, Any]],
    *,
    decision_threshold: float,
    threshold_source: str,
    trial_rows: list[dict[str, Any]] | None = None,
    metadata_rows: list[dict[str, Any]] | None = None,
    slice_fields: tuple[str, ...] = (),
    max_examples_per_error: int = 10,
    max_domain_failures: int = 20,
    max_speaker_confusions: int = 10,
    max_speaker_failures: int = 10,
    max_priority_findings: int = 5,
) -> VerificationErrorAnalysisReport:
    if max_examples_per_error <= 0:
        raise ValueError("max_examples_per_error must be positive.")
    if not math.isfinite(decision_threshold):
        raise ValueError("decision_threshold must be finite.")

    metadata_index = {} if not metadata_rows else build_trial_item_index(metadata_rows)
    trial_lookup = _build_trial_lookup(trial_rows)
    score_records = _normalize_score_rows(score_rows)

    total_by_slice: dict[tuple[str, str], dict[str, int]] = {}
    error_by_slice: dict[tuple[str, str], dict[str, Any]] = {}
    negative_trials_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    false_accepts_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    positive_trials_by_speaker: dict[str, dict[str, Any]] = {}
    false_rejects_by_speaker: dict[str, dict[str, Any]] = {}
    false_accept_examples: list[VerificationErrorExample] = []
    false_reject_examples: list[VerificationErrorExample] = []
    false_accept_count = 0
    false_reject_count = 0

    for row_index, record in enumerate(score_records):
        raw_row = record["raw_row"]
        merged_row = _merge_trial_row(raw_row, trial_lookup, row_index=row_index)
        left_identifier = resolve_trial_side_identifier(merged_row, "left")
        right_identifier = resolve_trial_side_identifier(merged_row, "right")
        left_metadata = None if left_identifier is None else metadata_index.get(left_identifier)
        right_metadata = None if right_identifier is None else metadata_index.get(right_identifier)
        field_values = {
            field_name: _derive_slice_value(
                field_name,
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            )
            for field_name in slice_fields
        }
        for field_name, field_value in field_values.items():
            if field_value is None:
                continue
            bucket = total_by_slice.setdefault(
                (field_name, field_value),
                {"trial_count": 0},
            )
            bucket["trial_count"] += 1

        left_speaker_id = _resolve_speaker_id(
            merged_row=merged_row,
            metadata=left_metadata,
            side="left",
        )
        right_speaker_id = _resolve_speaker_id(
            merged_row=merged_row,
            metadata=right_metadata,
            side="right",
        )

        if record["label"] == 0:
            pair_key = _resolve_speaker_pair(left_speaker_id, right_speaker_id)
            if pair_key is not None:
                bucket = negative_trials_by_pair.setdefault(
                    pair_key,
                    {"trial_count": 0},
                )
                bucket["trial_count"] += 1

        speaker_id = _resolve_positive_speaker_id(
            label=record["label"],
            left_speaker_id=left_speaker_id,
            right_speaker_id=right_speaker_id,
        )
        if speaker_id is not None:
            bucket = positive_trials_by_speaker.setdefault(
                speaker_id,
                {"trial_count": 0},
            )
            bucket["trial_count"] += 1

        error_type = _classify_error(
            label=record["label"],
            score=record["score"],
            decision_threshold=decision_threshold,
        )
        if error_type is None:
            continue

        margin = (
            record["score"] - decision_threshold
            if error_type == "false_accept"
            else decision_threshold - record["score"]
        )
        example = VerificationErrorExample(
            error_type=error_type,
            score=round(record["score"], 6),
            label=record["label"],
            margin=round(margin, 6),
            left_id=left_identifier,
            right_id=right_identifier,
            left_speaker_id=left_speaker_id,
            right_speaker_id=right_speaker_id,
            dataset=_derive_slice_value(
                "dataset",
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            ),
            channel=_derive_slice_value(
                "channel",
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            ),
            role_pair=_derive_slice_value(
                "role_pair",
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            ),
            duration_bucket=_derive_slice_value(
                "duration_bucket",
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            ),
            noise_slice=_derive_slice_value(
                "noise_slice",
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            ),
            reverb_slice=_derive_slice_value(
                "reverb_slice",
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            ),
            channel_slice=_derive_slice_value(
                "channel_slice",
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            ),
            distance_slice=_derive_slice_value(
                "distance_slice",
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            ),
            silence_slice=_derive_slice_value(
                "silence_slice",
                left_metadata=left_metadata,
                right_metadata=right_metadata,
            ),
        )
        if error_type == "false_accept":
            false_accept_count += 1
            false_accept_examples.append(example)
            pair_key = _resolve_speaker_pair(left_speaker_id, right_speaker_id)
            if pair_key is not None:
                bucket = false_accepts_by_pair.setdefault(
                    pair_key,
                    {"count": 0, "scores": []},
                )
                bucket["count"] += 1
                bucket["scores"].append(record["score"])
        else:
            false_reject_count += 1
            false_reject_examples.append(example)
            if speaker_id is not None:
                bucket = false_rejects_by_speaker.setdefault(
                    speaker_id,
                    {"count": 0, "scores": []},
                )
                bucket["count"] += 1
                bucket["scores"].append(record["score"])

        for field_name, field_value in field_values.items():
            if field_value is None:
                continue
            bucket = error_by_slice.setdefault(
                (field_name, field_value),
                {
                    "error_count": 0.0,
                    "false_accept_count": 0.0,
                    "false_reject_count": 0.0,
                    "margins": [],
                    "scores": [],
                },
            )
            bucket["error_count"] += 1.0
            bucket[f"{error_type}_count"] += 1.0
            bucket["margins"].append(margin)
            bucket["scores"].append(record["score"])

    positive_count = sum(1 for row in score_records if row["label"] == 1)
    negative_count = len(score_records) - positive_count
    total_error_count = false_accept_count + false_reject_count
    summary = VerificationErrorAnalysisSummary(
        threshold_source=threshold_source,
        decision_threshold=round(decision_threshold, 6),
        trial_count=len(score_records),
        positive_count=positive_count,
        negative_count=negative_count,
        false_accept_count=false_accept_count,
        false_reject_count=false_reject_count,
        total_error_count=total_error_count,
        false_accept_rate=_safe_rate(false_accept_count, negative_count),
        false_reject_rate=_safe_rate(false_reject_count, positive_count),
        total_error_rate=_safe_rate(total_error_count, len(score_records)),
    )

    domain_failures = _build_domain_failures(
        total_by_slice=total_by_slice,
        error_by_slice=error_by_slice,
        limit=max_domain_failures,
    )
    speaker_confusions = _build_speaker_confusions(
        negative_trials_by_pair=negative_trials_by_pair,
        false_accepts_by_pair=false_accepts_by_pair,
        limit=max_speaker_confusions,
    )
    speaker_failures = _build_speaker_failures(
        positive_trials_by_speaker=positive_trials_by_speaker,
        false_rejects_by_speaker=false_rejects_by_speaker,
        limit=max_speaker_failures,
    )
    priority_findings = _build_priority_findings(
        domain_failures=domain_failures,
        speaker_confusions=speaker_confusions,
        speaker_failures=speaker_failures,
        limit=max_priority_findings,
    )

    return VerificationErrorAnalysisReport(
        summary=summary,
        priority_findings=tuple(priority_findings),
        hard_false_accepts=tuple(
            sorted(
                false_accept_examples,
                key=lambda item: (
                    -item.margin,
                    -item.score,
                    item.left_id or "",
                    item.right_id or "",
                ),
            )[:max_examples_per_error]
        ),
        hard_false_rejects=tuple(
            sorted(
                false_reject_examples,
                key=lambda item: (
                    -item.margin,
                    item.score,
                    item.left_id or "",
                    item.right_id or "",
                ),
            )[:max_examples_per_error]
        ),
        domain_failures=tuple(domain_failures),
        speaker_confusions=tuple(speaker_confusions),
        speaker_failures=tuple(speaker_failures),
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
            f"`{row.speaker_a}` vs `{row.speaker_b}`: false accepts `{row.false_accept_count}` / "
            f"negative trials `{row.trial_count}` (rate `{row.false_accept_rate}`), "
            f"mean score `{row.mean_false_accept_score}`, max score `{row.max_false_accept_score}`"
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
            f"`{row.speaker_id}`: false rejects `{row.false_reject_count}` / positive trials "
            f"`{row.positive_trial_count}` (rate `{row.false_reject_rate}`), "
            f"mean score `{row.mean_false_reject_score}`, min score `{row.min_false_reject_score}`"
        )
    return lines


def _build_domain_failures(
    *,
    total_by_slice: dict[tuple[str, str], dict[str, int]],
    error_by_slice: dict[tuple[str, str], dict[str, Any]],
    limit: int,
) -> list[VerificationDomainFailure]:
    rows: list[VerificationDomainFailure] = []
    for key, error_bucket in error_by_slice.items():
        total_bucket = total_by_slice.get(key)
        if total_bucket is None:
            continue
        field_name, field_value = key
        trial_count = int(total_bucket["trial_count"])
        error_count = int(error_bucket["error_count"])
        margins = [float(value) for value in error_bucket["margins"]]
        scores = [float(value) for value in error_bucket["scores"]]
        rows.append(
            VerificationDomainFailure(
                field_name=field_name,
                field_value=field_value,
                trial_count=trial_count,
                error_count=error_count,
                false_accept_count=int(error_bucket["false_accept_count"]),
                false_reject_count=int(error_bucket["false_reject_count"]),
                error_rate=_safe_rate(error_count, trial_count),
                mean_error_margin=round(sum(margins) / len(margins), 6),
                mean_error_score=round(sum(scores) / len(scores), 6),
            )
        )
    return sorted(
        rows,
        key=lambda item: (
            -item.error_rate,
            -item.error_count,
            -item.mean_error_margin,
            -item.trial_count,
            item.field_name,
            item.field_value,
        ),
    )[:limit]


def _build_speaker_confusions(
    *,
    negative_trials_by_pair: dict[tuple[str, str], dict[str, Any]],
    false_accepts_by_pair: dict[tuple[str, str], dict[str, Any]],
    limit: int,
) -> list[VerificationSpeakerConfusion]:
    rows: list[VerificationSpeakerConfusion] = []
    for pair_key, error_bucket in false_accepts_by_pair.items():
        total_bucket = negative_trials_by_pair.get(pair_key)
        if total_bucket is None:
            continue
        speaker_a, speaker_b = pair_key
        scores = [float(value) for value in error_bucket["scores"]]
        error_count = int(error_bucket["count"])
        trial_count = int(total_bucket["trial_count"])
        rows.append(
            VerificationSpeakerConfusion(
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                trial_count=trial_count,
                false_accept_count=error_count,
                false_accept_rate=_safe_rate(error_count, trial_count),
                mean_false_accept_score=round(sum(scores) / len(scores), 6),
                max_false_accept_score=round(max(scores), 6),
            )
        )
    return sorted(
        rows,
        key=lambda item: (
            -item.false_accept_rate,
            -item.false_accept_count,
            -item.max_false_accept_score,
            item.speaker_a,
            item.speaker_b,
        ),
    )[:limit]


def _build_speaker_failures(
    *,
    positive_trials_by_speaker: dict[str, dict[str, Any]],
    false_rejects_by_speaker: dict[str, dict[str, Any]],
    limit: int,
) -> list[VerificationSpeakerFailure]:
    rows: list[VerificationSpeakerFailure] = []
    for speaker_id, error_bucket in false_rejects_by_speaker.items():
        total_bucket = positive_trials_by_speaker.get(speaker_id)
        if total_bucket is None:
            continue
        scores = [float(value) for value in error_bucket["scores"]]
        error_count = int(error_bucket["count"])
        trial_count = int(total_bucket["trial_count"])
        rows.append(
            VerificationSpeakerFailure(
                speaker_id=speaker_id,
                positive_trial_count=trial_count,
                false_reject_count=error_count,
                false_reject_rate=_safe_rate(error_count, trial_count),
                mean_false_reject_score=round(sum(scores) / len(scores), 6),
                min_false_reject_score=round(min(scores), 6),
            )
        )
    return sorted(
        rows,
        key=lambda item: (
            -item.false_reject_rate,
            -item.false_reject_count,
            item.min_false_reject_score,
            item.speaker_id,
        ),
    )[:limit]


def _build_priority_findings(
    *,
    domain_failures: list[VerificationDomainFailure],
    speaker_confusions: list[VerificationSpeakerConfusion],
    speaker_failures: list[VerificationSpeakerFailure],
    limit: int,
) -> list[VerificationPriorityFinding]:
    candidates: list[tuple[tuple[float, float, float], VerificationPriorityFinding]] = []
    for row in domain_failures[:6]:
        candidates.append(
            (
                (row.error_rate, float(row.error_count), row.mean_error_margin),
                VerificationPriorityFinding(
                    finding_type="domain_failure",
                    title=f"Slice {row.field_name}={row.field_value}",
                    evidence=(
                        f"FA `{row.false_accept_count}` / FR `{row.false_reject_count}`, "
                        f"mean error margin `{row.mean_error_margin}`"
                    ),
                    trial_count=row.trial_count,
                    error_count=row.error_count,
                    error_rate=row.error_rate,
                ),
            )
        )
    for row in speaker_confusions[:4]:
        candidates.append(
            (
                (row.false_accept_rate, float(row.false_accept_count), row.max_false_accept_score),
                VerificationPriorityFinding(
                    finding_type="speaker_confusion",
                    title=f"Speaker confusion {row.speaker_a} vs {row.speaker_b}",
                    evidence=(
                        f"recurrent false accepts with mean score `{row.mean_false_accept_score}` "
                        f"and max score `{row.max_false_accept_score}`"
                    ),
                    trial_count=row.trial_count,
                    error_count=row.false_accept_count,
                    error_rate=row.false_accept_rate,
                ),
            )
        )
    for row in speaker_failures[:4]:
        candidates.append(
            (
                (row.false_reject_rate, float(row.false_reject_count), -row.min_false_reject_score),
                VerificationPriorityFinding(
                    finding_type="speaker_failure",
                    title=f"Speaker fragility {row.speaker_id}",
                    evidence=(
                        f"recurrent false rejects with mean score `{row.mean_false_reject_score}` "
                        f"and min score `{row.min_false_reject_score}`"
                    ),
                    trial_count=row.positive_trial_count,
                    error_count=row.false_reject_count,
                    error_rate=row.false_reject_rate,
                ),
            )
        )
    return [
        finding
        for _, finding in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        )[:limit]
    ]


def _normalize_score_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(score_rows, start=1):
        try:
            label = int(row["label"])
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid verification score row at index {index}: {row!r}") from exc
        if label not in {0, 1}:
            raise ValueError(f"Verification labels must be 0 or 1, got {label!r} at row {index}.")
        if not math.isfinite(score):
            raise ValueError(f"Verification score must be finite, got {score!r} at row {index}.")
        normalized.append({"raw_row": row, "label": label, "score": score})
    return normalized


def _build_trial_lookup(
    trial_rows: list[dict[str, Any]] | None,
) -> dict[tuple[str, str, int] | tuple[str, int], dict[str, Any]]:
    if not trial_rows:
        return {}
    lookup: dict[tuple[str, str, int] | tuple[str, int], dict[str, Any]] = {}
    for index, row in enumerate(trial_rows):
        left_identifier = resolve_trial_side_identifier(row, "left")
        right_identifier = resolve_trial_side_identifier(row, "right")
        label = int(row.get("label", -1))
        if left_identifier and right_identifier and label in {0, 1}:
            lookup[(left_identifier, right_identifier, label)] = row
        lookup[(f"index:{index}", label)] = row
    return lookup


def _merge_trial_row(
    raw_score_row: dict[str, Any],
    trial_lookup: dict[tuple[str, str, int] | tuple[str, int], dict[str, Any]],
    *,
    row_index: int,
) -> dict[str, Any]:
    if not trial_lookup:
        return raw_score_row
    left_identifier = resolve_trial_side_identifier(raw_score_row, "left")
    right_identifier = resolve_trial_side_identifier(raw_score_row, "right")
    label = int(raw_score_row.get("label", -1))
    matched_row = None
    if left_identifier and right_identifier and label in {0, 1}:
        matched_row = trial_lookup.get((left_identifier, right_identifier, label))
    if matched_row is None:
        matched_row = trial_lookup.get((f"index:{row_index}", label))
    if matched_row is None:
        return raw_score_row
    return {**matched_row, **raw_score_row}


def _classify_error(*, label: int, score: float, decision_threshold: float) -> str | None:
    if label == 0 and score >= decision_threshold:
        return "false_accept"
    if label == 1 and score < decision_threshold:
        return "false_reject"
    return None


def _resolve_speaker_id(
    *,
    merged_row: dict[str, Any],
    metadata: dict[str, Any] | None,
    side: str,
) -> str | None:
    candidates = [
        merged_row.get(f"{side}_speaker_id"),
        None if metadata is None else metadata.get("speaker_id"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = str(candidate).strip()
        if normalized:
            return normalized
    return None


def _resolve_positive_speaker_id(
    *,
    label: int,
    left_speaker_id: str | None,
    right_speaker_id: str | None,
) -> str | None:
    if label != 1:
        return None
    if left_speaker_id is not None and right_speaker_id is not None:
        if left_speaker_id == right_speaker_id:
            return left_speaker_id
        return None
    return left_speaker_id or right_speaker_id


def _resolve_speaker_pair(
    left_speaker_id: str | None,
    right_speaker_id: str | None,
) -> tuple[str, str] | None:
    if left_speaker_id is None or right_speaker_id is None:
        return None
    if left_speaker_id == right_speaker_id:
        return None
    speaker_a, speaker_b = sorted((left_speaker_id, right_speaker_id))
    return speaker_a, speaker_b


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _format_speaker_fragment(left_speaker_id: str | None, right_speaker_id: str | None) -> str:
    if left_speaker_id and right_speaker_id:
        return f"{left_speaker_id} -> {right_speaker_id}"
    return left_speaker_id or right_speaker_id or "unknown speakers"


def _format_named_fragment(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return f"{name} `{value}`"


def _derive_slice_value(
    field_name: str,
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if field_name == "noise_slice":
        return _derive_noise_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "reverb_slice":
        return _derive_reverb_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "channel_slice":
        return _derive_channel_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "distance_slice":
        return _derive_distance_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "silence_slice":
        return _derive_silence_slice(left_metadata=left_metadata, right_metadata=right_metadata)
    if field_name == "duration_bucket":
        duration_values = [
            _coerce_float_or_none(metadata.get("duration_seconds"))
            for metadata in (left_metadata, right_metadata)
            if metadata is not None
        ]
        filtered_values = [value for value in duration_values if value is not None]
        if not filtered_values:
            return "unknown"
        mean_duration = sum(filtered_values) / float(len(filtered_values))
        for start, stop, label in _DURATION_BUCKETS:
            if stop is None and mean_duration >= start:
                return label
            if stop is not None and start <= mean_duration < stop:
                return label
        return "unknown"
    if field_name == "role_pair":
        left_role = _coerce_label(None if left_metadata is None else left_metadata.get("role"))
        right_role = _coerce_label(None if right_metadata is None else right_metadata.get("role"))
        return f"{left_role}->{right_role}"
    if field_name.startswith("left_"):
        return _coerce_label(
            None if left_metadata is None else left_metadata.get(field_name.removeprefix("left_"))
        )
    if field_name.startswith("right_"):
        return _coerce_label(
            None
            if right_metadata is None
            else right_metadata.get(field_name.removeprefix("right_"))
        )
    if field_name.startswith("pair_"):
        field_name = field_name.removeprefix("pair_")
    if field_name in _PAIR_FIELDS:
        left_value = _coerce_label(None if left_metadata is None else left_metadata.get(field_name))
        right_value = _coerce_label(
            None if right_metadata is None else right_metadata.get(field_name)
        )
        if left_value == "unknown" and right_value == "unknown":
            return "unknown"
        if left_value == right_value:
            return left_value
        if left_value == "unknown":
            return right_value
        if right_value == "unknown":
            return left_value
        return "mixed"

    left_value = _coerce_label(None if left_metadata is None else left_metadata.get(field_name))
    right_value = _coerce_label(None if right_metadata is None else right_metadata.get(field_name))
    if left_value == right_value:
        return left_value
    if left_value == "unknown":
        return right_value
    if right_value == "unknown":
        return left_value
    return f"{left_value}|{right_value}"


def _derive_noise_slice(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if (
        _coerce_pair_label(
            left_metadata=left_metadata,
            right_metadata=right_metadata,
            field_name="corruption_family",
        )
        != "noise"
    ):
        return None
    category = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="corruption_category",
    )
    severity = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_severity",
    )
    return _join_slice_parts(category, severity)


def _derive_reverb_slice(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if (
        _coerce_pair_label(
            left_metadata=left_metadata,
            right_metadata=right_metadata,
            field_name="corruption_family",
        )
        != "reverb"
    ):
        return None
    direct = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="direct_condition",
    )
    rt60 = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="rt60_bucket",
    )
    field = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="field",
    )
    return _join_slice_parts(field, direct, rt60)


def _derive_channel_slice(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if (
        _coerce_pair_label(
            left_metadata=left_metadata,
            right_metadata=right_metadata,
            field_name="corruption_family",
        )
        != "codec"
    ):
        return None
    codec_family = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="codec_family",
    )
    suite_id = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_suite",
    )
    if codec_family != "channel" and "channel" not in suite_id:
        return None
    codec_name = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="codec_name",
    )
    severity = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_severity",
    )
    return _join_slice_parts(codec_family, codec_name, severity)


def _derive_distance_slice(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if (
        _coerce_pair_label(
            left_metadata=left_metadata,
            right_metadata=right_metadata,
            field_name="corruption_family",
        )
        != "distance"
    ):
        return None
    field = _coerce_pair_nested_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        nested_field_name="distance_field",
    )
    severity = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_severity",
    )
    return _join_slice_parts(field, severity)


def _derive_silence_slice(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
) -> str | None:
    if (
        _coerce_pair_label(
            left_metadata=left_metadata,
            right_metadata=right_metadata,
            field_name="corruption_family",
        )
        != "silence"
    ):
        return None
    severity = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_severity",
    )
    candidate = _coerce_pair_label(
        left_metadata=left_metadata,
        right_metadata=right_metadata,
        field_name="corruption_candidate_id",
    )
    return _join_slice_parts(severity, candidate)


def _coerce_pair_label(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
    field_name: str,
) -> str:
    return _merge_pair_labels(
        _coerce_label(None if left_metadata is None else left_metadata.get(field_name)),
        _coerce_label(None if right_metadata is None else right_metadata.get(field_name)),
    )


def _coerce_pair_nested_label(
    *,
    left_metadata: dict[str, Any] | None,
    right_metadata: dict[str, Any] | None,
    nested_field_name: str,
) -> str:
    return _merge_pair_labels(
        _coerce_label(_lookup_nested_metadata(left_metadata, nested_field_name)),
        _coerce_label(_lookup_nested_metadata(right_metadata, nested_field_name)),
    )


def _lookup_nested_metadata(
    metadata: dict[str, Any] | None,
    nested_field_name: str,
) -> Any:
    if metadata is None:
        return None
    container = metadata.get("corruption_metadata")
    if isinstance(container, dict):
        return container.get(nested_field_name)
    return None


def _merge_pair_labels(left_value: str, right_value: str) -> str:
    if left_value == "unknown" and right_value == "unknown":
        return "unknown"
    if left_value == right_value:
        return left_value
    if left_value == "unknown":
        return right_value
    if right_value == "unknown":
        return left_value
    return "mixed"


def _join_slice_parts(*parts: str) -> str | None:
    normalized = [part for part in parts if part and part != "unknown"]
    if not normalized:
        return None
    return "/".join(normalized)


def _coerce_label(value: Any) -> str:
    if value is None:
        return "unknown"
    normalized = str(value).strip()
    return normalized if normalized else "unknown"


def _coerce_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    return coerced if math.isfinite(coerced) else None


__all__ = [
    "VERIFICATION_ERROR_ANALYSIS_JSON_NAME",
    "VERIFICATION_ERROR_ANALYSIS_MARKDOWN_NAME",
    "VerificationDomainFailure",
    "VerificationErrorAnalysisReport",
    "VerificationErrorAnalysisSummary",
    "VerificationErrorExample",
    "VerificationPriorityFinding",
    "VerificationSpeakerConfusion",
    "VerificationSpeakerFailure",
    "WrittenVerificationErrorAnalysis",
    "build_verification_error_analysis",
    "render_verification_error_analysis_markdown",
    "write_verification_error_analysis_report",
]
