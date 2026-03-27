"""Build a reproducible experiment-matrix report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.training.experiment_matrix import (
    build_experiment_matrix,
    load_experiment_matrix_config,
    write_experiment_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config describing the experiment matrix.",
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
    config = load_experiment_matrix_config(config_path=args.config)
    report = build_experiment_matrix(config)
    written = write_experiment_matrix(report)

    if args.output == "json":
        print(
            json.dumps(
                {"report": report.to_dict(), "written": written.to_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        return

    lines = [
        "Experiment matrix complete",
        f"Title: {report.title}",
        f"Matrix id: {report.matrix_id}",
        f"Ready budget: {report.ready_budget.render()} GPU-hours",
        f"Deferred stretch budget: {report.deferred_budget.render()} GPU-hours",
        f"Output root: {written.output_root}",
        f"JSON: {written.report_json_path}",
        f"Markdown: {written.report_markdown_path}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
