"""Assemble deterministic far-field presets into preview artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.data.far_field_bank import (
    build_far_field_bank,
    load_far_field_bank_plan,
    write_far_field_bank_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="Path to the base TOML config.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/corruption/far-field-bank.toml"),
        help="Far-field bank TOML plan with deterministic distance presets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/corruptions/far-field-bank"),
        help="Directory where preview audio, kernels, manifests, and reports should be written.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in dotted.key=value form. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(config_path=args.config, overrides=args.override)
    plan = load_far_field_bank_plan(args.plan)
    report = build_far_field_bank(
        project_root=config.paths.project_root,
        output_root=args.output_dir,
        plan=plan,
        plan_path=args.plan,
    )
    written = write_far_field_bank_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "summary": report.summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
