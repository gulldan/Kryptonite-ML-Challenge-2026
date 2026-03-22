"""Smoke-check the training environment imports expected by the project."""

from __future__ import annotations

import json

import typer

from kryptonite.training.environment import (
    build_training_environment_report,
    render_training_environment_report,
)

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    require_gpu: bool = typer.Option(
        False,
        "--require-gpu",
        help="Fail unless the GPU/TensorRT stack is importable too.",
    ),
    output: str = typer.Option(
        "text",
        "--output",
        help="Output format: text or json.",
        case_sensitive=False,
    ),
) -> None:
    report = build_training_environment_report(require_gpu=require_gpu)
    if output == "json":
        typer.echo(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    elif output == "text":
        typer.echo(render_training_environment_report(report))
    else:
        raise typer.BadParameter("output must be one of: text, json")

    raise typer.Exit(code=0 if report.passed else 1)


if __name__ == "__main__":
    app()
