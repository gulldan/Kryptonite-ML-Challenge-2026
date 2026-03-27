"""Build a self-contained submission/release bundle from frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.serve import (
    build_submission_bundle,
    load_submission_bundle_config,
    write_submission_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config describing the bundle contents.",
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
    config = load_submission_bundle_config(config_path=args.config)
    report = build_submission_bundle(config, config_path=args.config)
    written = write_submission_bundle(report, create_archive=config.create_archive)

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
        return

    lines = [
        "Submission bundle complete",
        f"Title: {report.title}",
        f"Bundle id: {report.bundle_id}",
        f"Mode: {report.bundle_mode}",
        f"Output root: {written.output_root}",
        f"README: {written.readme_path}",
        f"JSON: {written.report_json_path}",
        f"Markdown: {written.report_markdown_path}",
        f"Release freeze JSON: {written.release_freeze_json_path}",
        f"Release freeze Markdown: {written.release_freeze_markdown_path}",
    ]
    if written.archive_path is not None:
        lines.append(f"Archive: {written.archive_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
