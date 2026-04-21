#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate speaker-retrieval submission.csv.")
    parser.add_argument(
        "--template-csv", required=True, help="CSV with the expected filepath order."
    )
    parser.add_argument("--submission-csv", required=True, help="Submission CSV to validate.")
    parser.add_argument("--output-json", default="", help="Optional path for validation report.")
    parser.add_argument("--k", type=int, default=10, help="Expected neighbour count per row.")
    return parser.parse_args()


def validate_submission(template_csv: Path, submission_csv: Path, k: int = 10) -> dict[str, Any]:
    template = pd.read_csv(template_csv)
    submission = pd.read_csv(submission_csv)
    errors: list[str] = []

    if "filepath" not in template.columns:
        errors.append("template_csv must contain filepath column")
    if list(submission.columns) != ["filepath", "neighbours"]:
        errors.append(
            "submission_csv columns must be exactly ['filepath', 'neighbours'], "
            f"got {list(submission.columns)}"
        )
    if errors:
        return _report(template, submission, errors)

    template_paths = template["filepath"].astype(str).tolist()
    submission_paths = submission["filepath"].astype(str).tolist()
    if len(template_paths) != len(submission_paths):
        errors.append(
            f"row count mismatch: template has {len(template_paths)}, "
            f"submission has {len(submission_paths)}"
        )
    if template_paths != submission_paths:
        errors.append("filepath order/content differs from template")

    row_count = len(template_paths)
    invalid_rows = 0
    for row_idx, value in enumerate(submission.get("neighbours", pd.Series(dtype=str)).astype(str)):
        row_errors = _validate_neighbours(value, row_idx=row_idx, row_count=row_count, k=k)
        if row_errors:
            invalid_rows += 1
            if len(errors) < 100:
                errors.extend(row_errors)

    report = _report(template, submission, errors[:100])
    report["invalid_row_count"] = invalid_rows
    report["k"] = int(k)
    return report


def _validate_neighbours(value: str, *, row_idx: int, row_count: int, k: int) -> list[str]:
    prefix = f"row {row_idx + 1}"
    if not value.strip() or value == "nan":
        return [f"{prefix}: neighbours is empty"]
    parts = [part.strip() for part in value.split(",")]
    errors: list[str] = []
    if len(parts) != k:
        errors.append(f"{prefix}: expected {k} neighbours, got {len(parts)}")
    if any(part == "" for part in parts):
        errors.append(f"{prefix}: neighbours contains an empty value")
        return errors
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        errors.append(f"{prefix}: neighbours contains a non-integer value")
        return errors
    if len(set(numbers)) != len(numbers):
        errors.append(f"{prefix}: neighbours contains duplicate indices")
    if row_idx in numbers:
        errors.append(f"{prefix}: neighbours contains self-match index {row_idx}")
    out_of_range = [number for number in numbers if number < 0 or number >= row_count]
    if out_of_range:
        errors.append(f"{prefix}: neighbours contains out-of-range indices {out_of_range[:5]}")
    return errors


def _report(template: pd.DataFrame, submission: pd.DataFrame, errors: list[str]) -> dict[str, Any]:
    return {
        "passed": not errors,
        "template_row_count": int(len(template)),
        "submission_row_count": int(len(submission)),
        "error_count": int(len(errors)),
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    report = validate_submission(
        template_csv=Path(args.template_csv),
        submission_csv=Path(args.submission_csv),
        k=int(args.k),
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
