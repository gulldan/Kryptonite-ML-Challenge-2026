"""Validate a Kryptonite retrieval submission CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.eda import validate_submission


def main() -> None:
    args = _parse_args()
    report = validate_submission(
        template_csv=args.template_csv,
        submission_csv=args.submission_csv,
        k=args.k,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"passed={report['passed']} errors={report['error_count']}")
    print(f"Wrote {output_json}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-csv",
        default="datasets/Для участников/test_public.csv",
        help="Expected filepath template CSV.",
    )
    parser.add_argument("--submission-csv", required=True, help="Submission CSV to validate.")
    parser.add_argument(
        "--output-json",
        default="artifacts/eda/participants/submission_validation_report.json",
    )
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main()
