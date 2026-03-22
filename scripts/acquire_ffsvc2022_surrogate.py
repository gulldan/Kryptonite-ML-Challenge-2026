"""Acquire the server-only FFSVC 2022 surrogate dataset bundle."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.data.acquisition import (
    acquire_plan,
    load_acquisition_plan,
    render_acquisition_report,
)
from kryptonite.project import get_project_layout

app = typer.Typer(add_completion=False, help=__doc__)

PLAN_OPTION = typer.Option(
    Path("configs/data-acquisition/ffsvc2022-surrogate.toml"),
    "--plan",
    help="Path to the FFSVC 2022 surrogate acquisition plan.",
)
EXECUTE_OPTION = typer.Option(
    False,
    "--execute",
    help="Download and extract the configured artifacts.",
)
OUTPUT_OPTION = typer.Option(
    "text",
    "--output",
    help="Output format: text or json.",
    case_sensitive=False,
)


@app.command()
def main(
    plan: Path = PLAN_OPTION,
    execute: bool = EXECUTE_OPTION,
    output: str = OUTPUT_OPTION,
) -> None:
    acquisition_plan = load_acquisition_plan(plan)
    report = acquire_plan(
        project_root=get_project_layout().root,
        plan=acquisition_plan,
        execute=execute,
    )

    if output == "json":
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    elif output == "text":
        typer.echo(render_acquisition_report(report))
    else:
        raise typer.BadParameter("output must be one of: text, json")

    raise typer.Exit(code=0 if report.passed else 1)


if __name__ == "__main__":
    app()
