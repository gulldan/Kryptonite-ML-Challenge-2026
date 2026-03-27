"""Build a reproducible final family decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.eval.final_family_decision import (
    build_final_family_decision,
    load_final_family_decision_config,
    write_final_family_decision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config describing the final family decision.",
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
    config = load_final_family_decision_config(config_path=args.config)
    report = build_final_family_decision(config)
    written = write_final_family_decision(report)

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
        return

    lines = [
        "Final family decision complete",
        f"Title: {report.title}",
        f"Decision id: {report.decision_id}",
        f"Selected production student: {report.selected_production_student.label}",
        f"Selected stretch teacher: {report.selected_stretch_teacher.label}",
        f"Output root: {written.output_root}",
        f"JSON: {written.report_json_path}",
        f"Markdown: {written.report_markdown_path}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
