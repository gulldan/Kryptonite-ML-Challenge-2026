"""Materialize a validated reference submission byte-for-byte for a fixed dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from kryptonite.eda.submission import validate_submission


def main() -> None:
    args = _parse_args()
    reference_csv = Path(args.reference_csv)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)

    validation = validate_submission(
        template_csv=Path(args.template_csv),
        submission_csv=reference_csv,
        k=args.k,
    )
    if args.require_valid and not validation["passed"]:
        raise RuntimeError(f"Reference submission failed validation: {validation['errors'][:5]}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference_csv, output_csv)
    reference_sha256 = _sha256(reference_csv)
    output_sha256 = _sha256(output_csv)
    byte_identical = reference_sha256 == output_sha256
    if args.require_identical and not byte_identical:
        raise RuntimeError(
            f"Copied submission is not byte-identical: {reference_sha256} != {output_sha256}"
        )

    payload: dict[str, Any] = {
        "reference_csv": str(reference_csv),
        "output_csv": str(output_csv),
        "template_csv": args.template_csv,
        "k": args.k,
        "reference_sha256": reference_sha256,
        "output_sha256": output_sha256,
        "byte_identical": byte_identical,
        "validation": validation,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-csv",
        default="artifacts/backbone_public/campp/default_model_submission.csv",
    )
    parser.add_argument(
        "--template-csv",
        default="datasets/Для участников/test_public.csv",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--require-valid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-identical",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
