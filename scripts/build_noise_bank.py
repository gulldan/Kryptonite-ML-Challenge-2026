"""Assemble and normalize an additive noise bank from approved corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.data.noise_bank import (
    build_noise_bank,
    load_noise_bank_plan,
    write_noise_bank_report,
)
from kryptonite.data.normalization import AudioNormalizationPolicy


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
        default=Path("configs/corruption/noise-bank.toml"),
        help="Noise-bank TOML plan with source roots and classification rules.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/corruptions/noise-bank"),
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
    plan = load_noise_bank_plan(args.plan)
    report = build_noise_bank(
        project_root=config.paths.project_root,
        dataset_root=config.paths.dataset_root,
        output_root=args.output_dir,
        plan=plan,
        plan_path=args.plan,
        policy=AudioNormalizationPolicy.from_config(config.normalization),
    )
    written = write_noise_bank_report(report=report, output_root=args.output_dir)
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
