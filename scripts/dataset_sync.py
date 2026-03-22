"""Synchronize dataset payloads to gpu-server and capture a readiness report."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.data.sync import (
    load_sync_plan,
    render_sync_report,
    run_data_sync,
    write_sync_report,
)
from kryptonite.project import get_project_layout

app = typer.Typer(add_completion=False, help=__doc__)

PLAN_OPTION = typer.Option(
    Path("configs/data-sync/gpu-server.toml"),
    "--plan",
    help="Path to the dataset sync plan.",
)
EXECUTE_OPTION = typer.Option(
    False,
    "--execute",
    help="Run rsync before the final remote inventory check.",
)
OUTPUT_OPTION = typer.Option(
    "text",
    "--output",
    help="Output format: text or json.",
    case_sensitive=False,
)
SAMPLE_LIMIT_OPTION = typer.Option(
    10,
    "--sample-limit",
    min=1,
    help="How many sample files to retain per payload in the report.",
)


@app.command()
def main(
    plan: Path = PLAN_OPTION,
    execute: bool = EXECUTE_OPTION,
    output: str = OUTPUT_OPTION,
    sample_limit: int = SAMPLE_LIMIT_OPTION,
) -> None:
    sync_plan = load_sync_plan(plan)
    report = run_data_sync(
        project_root=get_project_layout().root,
        plan=sync_plan,
        execute=execute,
        sample_limit=sample_limit,
    )
    report = write_sync_report(
        report=report,
        plan=sync_plan,
        project_root=get_project_layout().root,
    )

    if output == "json":
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    elif output == "text":
        typer.echo(render_sync_report(report))
    else:
        raise typer.BadParameter("output must be one of: text, json")

    raise typer.Exit(code=0 if report.passed else 1)


if __name__ == "__main__":
    app()
