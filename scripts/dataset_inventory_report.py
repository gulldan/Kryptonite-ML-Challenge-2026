"""Generate a reproducible dataset inventory report from the policy plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.data.inventory import (
    build_dataset_inventory_report,
    load_dataset_inventory_plan,
    write_dataset_inventory_report,
)
from kryptonite.project import get_project_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/data-inventory/allowed-sources.toml"),
        help="Path to the dataset inventory TOML plan.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports/dataset-inventory"),
        help="Directory where the JSON and Markdown report should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_dataset_inventory_plan(args.plan)
    report = build_dataset_inventory_report(
        project_root=get_project_layout().root,
        plan=plan,
        plan_path=args.plan,
    )
    written = write_dataset_inventory_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "source_count": report.source_count,
                "status_counts": report.status_counts,
                "local_state_counts": report.local_state_counts,
                "scope_counts": report.scope_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
