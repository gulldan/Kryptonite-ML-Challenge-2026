"""Markdown report rendering for speaker baseline training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from kryptonite.eval import (
    COHORT_SUMMARY_JSON_NAME,
    WrittenVerificationEvaluationReport,
)

from .baseline_config import BaselineProvenanceConfig

if TYPE_CHECKING:
    from .speaker_baseline import (
        EmbeddingExportSummary,
        ScoreSummary,
        TrainingSummary,
    )


def render_markdown_report(
    *,
    title: str,
    provenance: BaselineProvenanceConfig,
    training_summary: TrainingSummary,
    embedding_summary: EmbeddingExportSummary,
    score_summary: ScoreSummary,
    verification_report: WrittenVerificationEvaluationReport | None,
    output_root: Path,
    project_root: Path,
) -> str:
    final_epoch = training_summary.epochs[-1]
    relative_output_root = relative_to_project(output_root, project_root=project_root)
    relative_checkpoint = relative_to_project(
        Path(training_summary.checkpoint_path),
        project_root=project_root,
    )
    relative_embeddings = relative_to_project(
        Path(embedding_summary.embeddings_path),
        project_root=project_root,
    )
    relative_verification_report = None
    relative_error_analysis_report = None
    if verification_report is not None:
        relative_verification_report = relative_to_project(
            Path(verification_report.report_markdown_path),
            project_root=project_root,
        )
        if verification_report.error_analysis_markdown_path is not None:
            relative_error_analysis_report = relative_to_project(
                Path(verification_report.error_analysis_markdown_path),
                project_root=project_root,
            )
    lines = [
        f"# {title}",
        "",
        f"- Output root: `{relative_output_root}`",
        f"- Device: `{training_summary.device}`",
        f"- Train manifest: `{training_summary.train_manifest}`",
        f"- Dev manifest: `{training_summary.dev_manifest}`",
        f"- Ruleset: `{training_summary.provenance_ruleset}`",
        f"- Initialization: `{training_summary.provenance_initialization}`",
        f"- Speakers: `{training_summary.speaker_count}`",
        f"- Train rows: `{training_summary.train_row_count}`",
        f"- Dev rows: `{training_summary.dev_row_count}`",
    ]
    if provenance.teacher_resources or provenance.pretrained_resources or provenance.notes:
        lines.extend(
            [
                "",
                "## Provenance",
                "",
                f"- Teacher resources: `{list(provenance.teacher_resources)}`",
                f"- Pretrained resources: `{list(provenance.pretrained_resources)}`",
            ]
        )
        for note in provenance.notes:
            lines.append(f"- Note: {note}")

    lines.extend(
        [
            "",
            "## Training",
            "",
            f"- Epochs: `{len(training_summary.epochs)}`",
            f"- Final loss: `{final_epoch.mean_loss}`",
            f"- Final accuracy: `{final_epoch.accuracy}`",
            f"- Final learning rate: `{final_epoch.learning_rate}`",
            f"- Checkpoint: `{relative_checkpoint}`",
            "",
            "## Embeddings",
            "",
            f"- Utterances exported: `{embedding_summary.utterance_count}`",
            f"- Embedding dim: `{embedding_summary.embedding_dim}`",
            f"- Speaker count: `{embedding_summary.speaker_count}`",
            f"- Embeddings: `{relative_embeddings}`",
            "",
            "## Scores",
            "",
            f"- Scored trials: `{score_summary.trial_count}`",
            f"- Positive trials: `{score_summary.positive_count}`",
            f"- Negative trials: `{score_summary.negative_count}`",
            f"- Missing trial embeddings: `{score_summary.missing_embedding_count}`",
            f"- Mean positive score: `{score_summary.mean_positive_score}`",
            f"- Mean negative score: `{score_summary.mean_negative_score}`",
            f"- Score gap: `{score_summary.score_gap}`",
        ]
    )
    if verification_report is not None:
        metrics = verification_report.summary.metrics
        relative_slice_dashboard = relative_to_project(
            Path(verification_report.slice_dashboard_path),
            project_root=project_root,
        )
        lines.extend(
            [
                "",
                "## Verification Eval",
                "",
                f"- EER: `{metrics.eer}`",
                f"- EER threshold: `{metrics.eer_threshold}`",
                f"- MinDCF: `{metrics.min_dcf}`",
                f"- MinDCF threshold: `{metrics.min_dcf_threshold}`",
                f"- Eval report: `{relative_verification_report}`",
                f"- Slice dashboard: `{relative_slice_dashboard}`",
            ]
        )
        if relative_error_analysis_report is not None:
            lines.append(f"- Error analysis: `{relative_error_analysis_report}`")
    lines.extend(_render_cohort_bank_section(output_root=output_root, project_root=project_root))
    return "\n".join(lines) + "\n"


def relative_to_project(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _render_cohort_bank_section(*, output_root: Path, project_root: Path) -> list[str]:
    summary_path = output_root / COHORT_SUMMARY_JSON_NAME
    if not summary_path.is_file():
        return []

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []

    overlap = payload.get("overlapping_validation_speakers")
    if isinstance(overlap, list) and overlap:
        overlap_text = ", ".join(str(value) for value in overlap)
    else:
        overlap_text = "none"

    return [
        "",
        "## Cohort Bank",
        "",
        f"- Summary: `{relative_to_project(summary_path, project_root=project_root)}`",
        f"- Selected embeddings: `{payload.get('selected_row_count', '?')}`",
        f"- Selected speakers: `{payload.get('selected_speaker_count', '?')}`",
        f"- Trial-overlap fallback used: `{payload.get('trial_overlap_fallback_used', False)}`",
        f"- Validation speaker overlap: `{overlap_text}`",
    ]
