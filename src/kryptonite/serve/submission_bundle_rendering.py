"""Render and write submission/release bundle summary artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .submission_bundle_models import (
    SUBMISSION_BUNDLE_JSON_NAME,
    SUBMISSION_BUNDLE_MARKDOWN_NAME,
    SUBMISSION_BUNDLE_README_NAME,
    SubmissionBundleReport,
    WrittenSubmissionBundle,
)


def write_submission_bundle(
    report: SubmissionBundleReport,
    *,
    create_archive: bool = True,
) -> WrittenSubmissionBundle:
    output_root = Path(report.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    readme_path = output_root / SUBMISSION_BUNDLE_README_NAME
    json_path = output_root / SUBMISSION_BUNDLE_JSON_NAME
    markdown_path = output_root / SUBMISSION_BUNDLE_MARKDOWN_NAME

    rendered_readme = render_submission_bundle_readme(report)
    readme_path.write_text(rendered_readme + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_submission_bundle_markdown(report) + "\n",
        encoding="utf-8",
    )

    archive_path: str | None = None
    if create_archive:
        archive_root = shutil.make_archive(
            base_name=str(output_root),
            format="gztar",
            root_dir=str(output_root.parent),
            base_dir=output_root.name,
        )
        archive_path = str(Path(archive_root))

    return WrittenSubmissionBundle(
        output_root=str(output_root),
        readme_path=str(readme_path),
        report_json_path=str(json_path),
        report_markdown_path=str(markdown_path),
        archive_path=archive_path,
        summary=report.summary,
    )


def render_submission_bundle_readme(report: SubmissionBundleReport) -> str:
    lines = [f"# {report.title}", ""]
    if report.summary_text:
        lines.extend([report.summary_text, ""])
    lines.extend(
        [
            "## Bundle Summary",
            "",
            f"- Bundle id: `{report.bundle_id}`",
            f"- Mode: `{report.bundle_mode}`",
            f"- Model version: `{report.summary.model_version}`",
            f"- Structural stub: `{str(report.summary.structural_stub).lower()}`",
            f"- Config files: `{report.summary.config_count}`",
            f"- Benchmark artifacts: `{report.summary.benchmark_artifact_count}`",
            f"- Checkpoints: `{report.summary.checkpoint_count}`",
            (
                "- Threshold calibration included: "
                f"`{str(report.summary.threshold_calibration_included).lower()}`"
            ),
            (f"- TensorRT plan included: `{str(report.summary.tensorrt_plan_included).lower()}`"),
            (
                "- Triton repository included: "
                f"`{str(report.summary.triton_repository_included).lower()}`"
            ),
            f"- Demo assets included: `{str(report.summary.demo_assets_included).lower()}`",
            "",
            "## Suggested Validation",
            "",
            "1. Sync the repo-local environment on the target machine:",
            "",
            "```bash",
            "uv sync --dev --group train --group tracking",
            "```",
            "",
            "2. Run strict inference smoke against the staged configs/artifacts:",
            "",
            "```bash",
            "uv run python scripts/infer_smoke.py \\",
            "  --config configs/deployment/infer.toml \\",
            "  --require-artifacts",
            "```",
            "",
            "3. If the bundle includes the Triton repository, launch it and probe "
            "the generated request:",
            "",
            "```bash",
            "uv run python scripts/triton_infer_smoke.py \\",
            "  --repository-root artifacts/triton-model-repository \\",
            "  --model-name kryptonite_encoder \\",
            "  --server-url http://127.0.0.1:8000",
            "```",
        ]
    )
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    lines.extend(["", "## Staged Artifacts", ""])
    if report.source_config_artifact is not None:
        lines.append(
            "- bundle_config: "
            f"`{report.source_config_artifact.staged_path}` "
            f"(sha256 `{report.source_config_artifact.sha256}`)"
        )
    for artifact in report.artifacts:
        lines.append(
            f"- {artifact.kind}: `{artifact.staged_path}` "
            f"(type `{artifact.path_type}`, files `{artifact.file_count}`, "
            f"sha256 `{artifact.sha256}`)"
        )
    return "\n".join(lines)


def render_submission_bundle_markdown(report: SubmissionBundleReport) -> str:
    lines = [f"# {report.title}", ""]
    if report.summary_text:
        lines.extend([report.summary_text, ""])
    lines.extend(
        [
            "## Summary",
            "",
            f"- Bundle id: `{report.bundle_id}`",
            f"- Mode: `{report.bundle_mode}`",
            f"- Model version: `{report.summary.model_version}`",
            f"- Source artifacts: `{report.summary.source_artifact_count}`",
            "",
            _markdown_table(
                headers=["Kind", "Path", "Type", "Files", "SHA256"],
                rows=[
                    [
                        artifact.kind,
                        artifact.staged_path,
                        artifact.path_type,
                        str(artifact.file_count),
                        artifact.sha256,
                    ]
                    for artifact in report.artifacts
                ],
            ),
        ]
    )
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    if report.notes:
        lines.extend(["", "## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
    return "\n".join(lines)


def _markdown_table(*, headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


__all__ = [
    "render_submission_bundle_markdown",
    "render_submission_bundle_readme",
    "write_submission_bundle",
]
