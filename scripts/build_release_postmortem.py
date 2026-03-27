"""Build a reproducible release postmortem and backlog report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.serve import (
    build_release_postmortem,
    load_release_postmortem_config,
    write_release_postmortem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config describing the postmortem findings and backlog.",
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
    config = load_release_postmortem_config(config_path=args.config)
    report = build_release_postmortem(config, config_path=args.config)
    written = write_release_postmortem(report)

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
        return

    lines = [
        "Release postmortem complete",
        f"Title: {report.title}",
        f"Release id: {report.release_id}",
        f"Output root: {written.output_root}",
        f"JSON: {written.report_json_path}",
        f"Markdown: {written.report_markdown_path}",
        f"Next-iteration items: {report.summary.next_iteration_count}",
        f"De-scoped items: {report.summary.de_scoped_count}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
