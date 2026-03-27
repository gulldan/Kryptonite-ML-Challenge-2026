"""Write a reproducible export-boundary contract report from config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.serve.export_boundary import (
    build_export_boundary_contract,
    render_export_boundary_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="Path to the project config used to derive the export boundary.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv file for config secret resolution.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in dotted.key=value form. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/export-boundary"),
        help="Directory where the JSON and Markdown reports should be written.",
    )
    parser.add_argument(
        "--inferencer-backend",
        default="feature_statistics",
        help="Logical runtime backend paired with the export boundary.",
    )
    parser.add_argument(
        "--stage",
        default="demo",
        help="Default chunking stage assumed by the runtime contract.",
    )
    parser.add_argument(
        "--embedding-mode",
        default="mean_std",
        help="Runtime embedding mode used to derive a placeholder embedding dimension.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Optional explicit embedding dimension override for the output tensor contract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(
        config_path=args.config,
        overrides=args.override,
        env_file=args.env_file,
    )
    contract = build_export_boundary_contract(
        config=config,
        inferencer_backend=args.inferencer_backend,
        embedding_stage=args.stage,
        embedding_mode=args.embedding_mode,
        embedding_dim=args.embedding_dim,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "export_boundary.json"
    markdown_path = output_dir / "export_boundary.md"

    json_path.write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_export_boundary_markdown(contract) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "markdown_path": str(markdown_path),
                "boundary": contract.boundary,
                "input_name": contract.input_tensor.name,
                "output_name": contract.output_tensor.name,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
