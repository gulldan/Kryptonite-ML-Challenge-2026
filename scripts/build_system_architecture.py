"""Build a reproducible repository-level system architecture snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.system_architecture import (
    DEFAULT_SYSTEM_ARCHITECTURE_OUTPUT_ROOT,
    build_system_architecture_contract,
    write_system_architecture_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the TOML config that anchors the architecture snapshot.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_SYSTEM_ARCHITECTURE_OUTPUT_ROOT,
        help="Directory where the generated architecture snapshot should be written.",
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
    config = load_project_config(config_path=args.config)
    contract = build_system_architecture_contract(config, output_root=args.output_root)
    written = write_system_architecture_contract(contract)

    if args.output == "json":
        print(
            json.dumps(
                {"contract": contract.to_dict(), "written": written.to_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        return

    lines = [
        "System architecture snapshot complete",
        f"Title: {contract.title}",
        f"Decision id: {contract.decision_id}",
        f"Stages: {len(contract.stages)}",
        f"Output root: {written.output_root}",
        f"JSON: {written.report_json_path}",
        f"Markdown: {written.report_markdown_path}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
