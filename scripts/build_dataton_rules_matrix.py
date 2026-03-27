"""Generate the Dataton rules matrix and risk register from the policy plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.data.rules_matrix import (
    build_rules_matrix_report,
    load_rules_matrix_plan,
    write_rules_matrix_report,
)
from kryptonite.project import get_project_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/data-inventory/dataton-rules-matrix.toml"),
        help="Path to the Dataton rules-matrix TOML plan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/dataton-rules-matrix"),
        help="Directory where the JSON and Markdown report should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_rules_matrix_plan(args.plan)
    report = build_rules_matrix_report(
        project_root=get_project_layout().root,
        plan=plan,
        plan_path=args.plan,
    )
    written = write_rules_matrix_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "item_count": report.item_count,
                "decision_counts": report.decision_counts,
                "confidence_counts": report.confidence_counts,
                "category_counts": report.category_counts,
                "open_question_count": report.open_question_count,
                "risk_severity_counts": report.risk_severity_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
