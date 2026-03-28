"""Build a teacher-vs-student robust-dev comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.eval import (
    build_teacher_student_robust_dev_report,
    load_teacher_student_robust_dev_config,
    write_teacher_student_robust_dev_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config describing the robust-dev comparison.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Repository root used to resolve relative run roots and output paths.",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="CLI output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_teacher_student_robust_dev_config(config_path=args.config)
    report = build_teacher_student_robust_dev_report(
        config,
        config_path=args.config,
        project_root=args.project_root,
    )
    written = write_teacher_student_robust_dev_report(
        report,
        project_root=args.project_root,
    )

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
        return

    lines = [
        "Teacher/student robust-dev evaluation complete",
        f"Title: {report.title}",
        f"Ticket: {report.ticket_id}",
        f"Teacher candidate: {report.summary.teacher_candidate_id}",
        f"Best quality candidate: {report.summary.best_quality_candidate_id}",
        f"Output root: {written.output_root}",
        f"JSON: {written.report_json_path}",
        f"Markdown: {written.report_markdown_path}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
