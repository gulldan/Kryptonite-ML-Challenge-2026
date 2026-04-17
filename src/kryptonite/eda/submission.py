"""Submission-format validation for retrieval outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def validate_submission(
    *,
    template_csv: Path | str,
    submission_csv: Path | str,
    k: int = 10,
    max_errors: int = 100,
) -> dict[str, Any]:
    """Validate challenge submission rows, path order, and neighbour lists."""

    if k <= 0:
        raise ValueError("k must be positive.")
    template = pl.read_csv(template_csv)
    submission = pl.read_csv(submission_csv)
    errors: list[str] = []

    if "filepath" not in template.columns:
        errors.append("template_csv must contain filepath column")
    if "filepath" not in submission.columns:
        errors.append("submission_csv must contain filepath column")
    if "neighbours" not in submission.columns:
        errors.append("submission_csv must contain neighbours column")
    if errors:
        return _report(False, template, submission, errors)

    if template.height != submission.height:
        errors.append(
            "row count mismatch: "
            f"template has {template.height}, submission has {submission.height}"
        )

    template_paths = template.get_column("filepath").cast(pl.Utf8).to_list()
    submitted_paths = submission.get_column("filepath").cast(pl.Utf8).to_list()
    if template_paths != submitted_paths:
        errors.append("filepath order/content differs from template")

    neighbour_values = submission.get_column("neighbours").cast(pl.Utf8).fill_null("").to_list()
    row_count = min(len(neighbour_values), len(template_paths))
    invalid_rows = 0
    for row_index in range(row_count):
        row_errors = _validate_neighbour_cell(
            neighbour_values[row_index],
            row_index=row_index,
            row_count=template.height,
            k=k,
        )
        if row_errors:
            invalid_rows += 1
            if len(errors) < max_errors:
                errors.extend(row_errors)
    passed = not errors
    report = _report(passed, template, submission, errors[:max_errors])
    report["invalid_row_count"] = invalid_rows
    report["k"] = k
    return report


def _validate_neighbour_cell(
    value: str,
    *,
    row_index: int,
    row_count: int,
    k: int,
) -> list[str]:
    prefix = f"row {row_index + 1}"
    if value.strip() == "":
        return [f"{prefix}: neighbours is empty"]
    raw_parts = [part.strip() for part in value.split(",")]
    errors: list[str] = []
    if len(raw_parts) != k:
        errors.append(f"{prefix}: expected {k} neighbours, got {len(raw_parts)}")
    if any(part == "" for part in raw_parts):
        errors.append(f"{prefix}: neighbours contains an empty value")
        return errors
    try:
        neighbours = [int(part) for part in raw_parts]
    except ValueError:
        errors.append(f"{prefix}: neighbours contains a non-integer value")
        return errors
    if len(set(neighbours)) != len(neighbours):
        errors.append(f"{prefix}: neighbours contains duplicate indices")
    if row_index in neighbours:
        errors.append(f"{prefix}: neighbours contains self-match index {row_index}")
    out_of_range = [value for value in neighbours if value < 0 or value >= row_count]
    if out_of_range:
        errors.append(f"{prefix}: neighbours contains out-of-range indices {out_of_range[:5]}")
    return errors


def _report(
    passed: bool,
    template: pl.DataFrame,
    submission: pl.DataFrame,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "passed": passed,
        "template_row_count": int(template.height),
        "submission_row_count": int(submission.height),
        "error_count": len(errors),
        "errors": errors,
    }
