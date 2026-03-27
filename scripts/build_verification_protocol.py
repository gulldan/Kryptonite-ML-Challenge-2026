"""Build the internal verification-protocol snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.eval import (
    build_verification_protocol_report,
    load_verification_protocol_config,
    write_verification_protocol_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config describing the verification protocol.",
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
    config = load_verification_protocol_config(config_path=args.config)
    report = build_verification_protocol_report(config, config_path=args.config)
    written = write_verification_protocol_report(report)

    if args.output == "json":
        print(json.dumps(written.to_dict(), indent=2, sort_keys=True))
        return

    lines = [
        "Verification protocol snapshot complete",
        f"Title: {report.title}",
        f"Ticket: {report.ticket_id}",
        f"Protocol id: {report.protocol_id}",
        f"Clean bundles: {report.summary.clean_bundle_count}",
        f"Production-like bundles: {report.summary.production_bundle_count}",
        "Missing required slices: "
        f"{', '.join(report.summary.missing_required_slice_fields) or 'none'}",
        f"Output root: {written.output_root}",
        f"JSON: {written.report_json_path}",
        f"Markdown: {written.report_markdown_path}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
