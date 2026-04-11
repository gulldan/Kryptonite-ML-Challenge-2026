"""Build JSONL training manifests from participant speaker-disjoint CSV splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.data.participant_manifests import build_participant_training_manifests


def main() -> None:
    args = _parse_args()
    result = build_participant_training_manifests(
        train_split_csv=Path(args.train_split_csv),
        dev_split_csv=Path(args.dev_split_csv),
        output_dir=Path(args.output_dir),
        project_root=Path(args.project_root).resolve(),
        dataset_name=args.dataset_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-split-csv",
        default="artifacts/baseline_fixed_participants/train_split.csv",
    )
    parser.add_argument(
        "--dev-split-csv",
        default="artifacts/baseline_fixed_participants/val_split.csv",
    )
    parser.add_argument("--output-dir", default="artifacts/manifests/participants_fixed")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-name", default="participants_fixed")
    return parser.parse_args()


if __name__ == "__main__":
    main()
