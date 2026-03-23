"""Validate data-manifest rows against the unified Kryptonite schema contract."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.config import load_project_config
from kryptonite.data.validation import build_manifest_validation_report

app = typer.Typer(add_completion=False, help=__doc__)

CONFIG_OPTION = typer.Option(
    Path("configs/base.toml"),
    "--config",
    help="Base project config that defines project and manifests roots.",
)
MANIFESTS_ROOT_OPTION = typer.Option(
    None,
    "--manifests-root",
    help="Override the manifests root from config.",
)
STRICT_OPTION = typer.Option(
    True,
    "--strict/--no-strict",
    help="Exit with a non-zero status when invalid rows are found.",
)


@app.command()
def main(
    config: Path = CONFIG_OPTION,
    manifests_root: str | None = MANIFESTS_ROOT_OPTION,
    strict: bool = STRICT_OPTION,
) -> None:
    project_config = load_project_config(config_path=config)
    report = build_manifest_validation_report(
        project_root=project_config.paths.project_root,
        manifests_root=manifests_root or project_config.paths.manifests_root,
    )
    typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if strict and not report.passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
