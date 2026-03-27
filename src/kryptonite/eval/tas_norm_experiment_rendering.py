"""Rendering helpers for reproducible TAS-norm experiment reports."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .tas_norm import TAS_NORM_MODEL_JSON_NAME, VERIFICATION_TAS_NORM_SCORES_JSONL_NAME
from .tas_norm_experiment_models import (
    TAS_NORM_EXPERIMENT_JSON_NAME,
    TAS_NORM_EXPERIMENT_MARKDOWN_NAME,
    VERIFICATION_AS_NORM_EVAL_SCORES_JSONL_NAME,
    BuiltTasNormExperiment,
    TasNormArtifactRef,
    TasNormExperimentReport,
    WrittenTasNormExperimentReport,
)


def render_tas_norm_experiment_markdown(report: TasNormExperimentReport) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"- Report id: `{report.report_id}`",
        f"- Candidate: `{report.candidate_label}`",
        f"- Decision: `{report.summary.decision}`",
        f"- Eval winner: `{report.summary.eval_winner}`",
        f"- Cohort bank built during run: `{_format_bool(report.cohort_built_during_run)}`",
        f"- Cohort bank: `{report.cohort_bank_output_root}`",
        f"- Cohort size: `{report.cohort_size}`",
        f"- Embedding dim: `{report.embedding_dim}`",
        f"- Top-k: requested `{report.top_k}`, effective `{report.effective_top_k}`",
        "",
        "## Summary",
        "",
        report.summary_text.strip(),
        "",
        "## Implementation Scope",
        "",
        report.implementation_scope,
        "",
        "## Split",
        "",
        f"- Eval fraction: `{report.split.eval_fraction}`",
        f"- Split seed: `{report.split.split_seed}`",
        (
            "- Train trials: "
            f"`{report.split.train_trial_count}` "
            f"(`{report.split.train_positive_count}` pos / "
            f"`{report.split.train_negative_count}` neg)"
        ),
        (
            "- Eval trials: "
            f"`{report.split.eval_trial_count}` "
            f"(`{report.split.eval_positive_count}` pos / "
            f"`{report.split.eval_negative_count}` neg)"
        ),
        (
            "- Same-speaker cohort candidates excluded during feature prep: "
            f"`{report.excluded_same_speaker_count}`"
        ),
        f"- Floored cohort std count: `{report.floored_std_count}`",
        "",
        "## Quality Snapshot",
        "",
        "| Method | Split | Trials | EER | MinDCF | Mean score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for method, split_label, metrics in (
        ("raw", "train", report.raw_train),
        ("raw", "eval", report.raw_eval),
        ("as-norm", "train", report.as_norm_train),
        ("as-norm", "eval", report.as_norm_eval),
        ("tas-norm", "train", report.tas_norm_train),
        ("tas-norm", "eval", report.tas_norm_eval),
    ):
        lines.append(
            "| "
            f"{method} | "
            f"{split_label} | "
            f"{metrics.trial_count} | "
            f"{metrics.eer:.6f} | "
            f"{metrics.min_dcf:.6f} | "
            f"{metrics.mean_score:.6f} |"
        )

    lines.extend(
        ["", "## Decision Checks", "", "| Check | Status | Detail |", "| --- | --- | --- |"]
    )
    for check in report.checks:
        status = "pass" if check.passed else "fail"
        lines.append(f"| {check.name} | `{status}` | {check.detail} |")

    lines.extend(
        [
            "",
            "## TAS Model",
            "",
            (
                "- Training loss: "
                f"`{report.model.training.initial_loss:.8f}` -> "
                f"`{report.model.training.best_loss:.8f}` "
                f"in `{report.model.training.steps_completed}` steps"
            ),
            f"- Converged: `{_format_bool(report.model.training.converged)}`",
            f"- Bias: `{report.model.bias:.8f}`",
            "",
            "| Feature | Weight | Offset | Scale |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, weight, offset, scale in zip(
        report.feature_names,
        report.model.weights,
        report.model.feature_offsets,
        report.model.feature_scales,
        strict=True,
    ):
        lines.append(f"| {name} | `{weight:.8f}` | `{offset:.8f}` | `{scale:.8f}` |")

    lines.extend(
        [
            "",
            "## Artifact Snapshot",
            "",
            "| Artifact | Exists | Kind | Configured path |",
            "| --- | --- | --- | --- |",
        ]
    )
    for artifact in report.artifacts:
        lines.append(_render_artifact_row(artifact))

    if report.validation_commands:
        lines.extend(["", "## Validation Commands", ""])
        lines.extend(f"- `{command}`" for command in report.validation_commands)
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines) + "\n"


def write_tas_norm_experiment_report(
    built: BuiltTasNormExperiment,
) -> WrittenTasNormExperimentReport:
    report = built.report
    output_root = Path(report.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    report_json_path = output_root / TAS_NORM_EXPERIMENT_JSON_NAME
    report_json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_markdown_path = output_root / TAS_NORM_EXPERIMENT_MARKDOWN_NAME
    report_markdown_path.write_text(
        render_tas_norm_experiment_markdown(report),
        encoding="utf-8",
    )

    as_norm_eval_scores_path = output_root / VERIFICATION_AS_NORM_EVAL_SCORES_JSONL_NAME
    as_norm_eval_scores_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in built.as_norm_eval_score_rows),
        encoding="utf-8",
    )
    tas_norm_eval_scores_path = output_root / VERIFICATION_TAS_NORM_SCORES_JSONL_NAME
    tas_norm_eval_scores_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in built.tas_norm_eval_score_rows),
        encoding="utf-8",
    )
    model_json_path = output_root / TAS_NORM_MODEL_JSON_NAME
    model_json_path.write_text(
        json.dumps(report.model.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_config_copy_path = None
    if report.source_config_path is not None:
        source_config_copy = output_root / "sources" / "tas_norm_experiment_config.toml"
        source_config_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report.source_config_path, source_config_copy)
        source_config_copy_path = str(source_config_copy)

    return WrittenTasNormExperimentReport(
        output_root=str(output_root),
        report_json_path=str(report_json_path),
        report_markdown_path=str(report_markdown_path),
        as_norm_eval_scores_path=str(as_norm_eval_scores_path),
        tas_norm_eval_scores_path=str(tas_norm_eval_scores_path),
        model_json_path=str(model_json_path),
        source_config_copy_path=source_config_copy_path,
        summary=report.summary,
    )


def _render_artifact_row(artifact: TasNormArtifactRef) -> str:
    exists = "yes" if artifact.exists else "no"
    configured_path = artifact.configured_path or "-"
    return f"| {artifact.label} | `{exists}` | `{artifact.kind}` | `{configured_path}` |"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


__all__ = ["render_tas_norm_experiment_markdown", "write_tas_norm_experiment_report"]
