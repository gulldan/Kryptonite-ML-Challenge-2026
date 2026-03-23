"""Artifact writing helpers for audio-quality reports."""

from __future__ import annotations

import json
from pathlib import Path

from kryptonite.deployment import resolve_project_path

from .models import DatasetAudioQualityReport, WrittenDatasetAudioQualityReport
from .report import render_dataset_audio_quality_markdown


def write_dataset_audio_quality_report(
    *,
    report: DatasetAudioQualityReport,
    output_root: Path | str,
) -> WrittenDatasetAudioQualityReport:
    output_root_path = resolve_project_path(report.project_root, str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)

    json_path = output_root_path / "dataset_audio_quality.json"
    markdown_path = output_root_path / "dataset_audio_quality.md"
    rows_path = output_root_path / "dataset_audio_quality_rows.jsonl"
    flagged_rows_path = output_root_path / "dataset_audio_quality_flagged_rows.jsonl"

    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_dataset_audio_quality_markdown(report))
    _write_jsonl_rows(rows_path, (record.to_dict() for record in report.records))
    _write_jsonl_rows(
        flagged_rows_path,
        (record.to_dict() for record in report.records if record.quality_flags),
    )
    return WrittenDatasetAudioQualityReport(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        rows_path=str(rows_path),
        flagged_rows_path=str(flagged_rows_path),
    )


def _write_jsonl_rows(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
