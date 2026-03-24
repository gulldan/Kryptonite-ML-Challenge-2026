"""Assemble and normalize a reproducible RIR and room-simulation bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.data.normalization import AudioNormalizationPolicy
from kryptonite.data.rir_bank import (
    build_rir_bank,
    load_rir_bank_plan,
    write_rir_bank_report,
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
        default=Path("configs/corruption/rir-bank.toml"),
        help="RIR-bank TOML plan with source roots, room-size hints, and analysis ranges.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/corruptions/rir-bank"),
        help="Directory where normalized audio, manifests, and reports should be written.",
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
    plan = load_rir_bank_plan(args.plan)
    report = build_rir_bank(
        project_root=config.paths.project_root,
        dataset_root=config.paths.dataset_root,
        output_root=args.output_dir,
        plan=plan,
        plan_path=args.plan,
        policy=AudioNormalizationPolicy.from_config(config.normalization),
    )
    written = write_rir_bank_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "summary": report.summary.to_dict(),
                "sources": [source.to_dict() for source in report.sources],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
