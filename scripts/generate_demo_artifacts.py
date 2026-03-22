"""Generate a tiny reproducible demo dataset, manifests, and model bundle."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.config import load_project_config
from kryptonite.demo_artifacts import generate_demo_artifacts

app = typer.Typer(add_completion=False, help=__doc__)

CONFIG_OPTION = typer.Option(
    Path("configs/deployment/infer.toml"),
    "--config",
    help="Path to the deployment profile that defines artifact roots.",
)
ENV_FILE_OPTION = typer.Option(
    Path(".env"),
    "--env-file",
    help="Optional dotenv file with secrets.",
)
OVERRIDE_OPTION = typer.Option(
    None,
    "--override",
    help="Config override in dotted.key=value form. Can be passed multiple times.",
)


@app.command()
def main(
    config: Path = CONFIG_OPTION,
    env_file: Path = ENV_FILE_OPTION,
    override: list[str] | None = OVERRIDE_OPTION,
) -> None:
    project_config = load_project_config(
        config_path=config,
        overrides=override or [],
        env_file=env_file,
    )
    artifacts = generate_demo_artifacts(config=project_config)
    typer.echo(json.dumps(artifacts.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
