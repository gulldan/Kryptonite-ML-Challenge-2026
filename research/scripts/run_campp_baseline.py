"""Train the CAM++ baseline, export dev embeddings, and write cosine scores."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.training import load_campp_baseline_config, run_campp_baseline

app = typer.Typer(add_completion=False, help=__doc__)

CONFIG_OPTION = typer.Option(
    Path("research/configs/training/campp-baseline.toml"),
    "--config",
    help="Path to the CAM++ baseline TOML config.",
)
ENV_FILE_OPTION = typer.Option(
    Path(".env"),
    "--env-file",
    help="Optional dotenv file with secrets.",
)
PROJECT_OVERRIDE_OPTION = typer.Option(
    None,
    "--project-override",
    help="Extra base ProjectConfig override in dotted.key=value form. Can be repeated.",
)
DEVICE_OPTION = typer.Option(
    None,
    "--device",
    help="Optional device override. Defaults to the project runtime.device setting.",
)
OUTPUT_OPTION = typer.Option(
    "text",
    "--output",
    help="Output format: text or json.",
    case_sensitive=False,
)


@app.command()
def main(
    config: Path = CONFIG_OPTION,
    env_file: Path = ENV_FILE_OPTION,
    project_override: list[str] | None = PROJECT_OVERRIDE_OPTION,
    device: str | None = DEVICE_OPTION,
    output: str = OUTPUT_OPTION,
) -> None:
    baseline = load_campp_baseline_config(
        config_path=config,
        env_file=env_file,
        project_overrides=project_override or [],
    )
    artifacts = run_campp_baseline(
        baseline,
        config_path=config,
        device_override=device,
    )

    if output == "json":
        typer.echo(json.dumps(artifacts.to_dict(), indent=2, sort_keys=True))
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text, json")

    final_epoch = artifacts.training_summary.epochs[-1]
    typer.echo(
        "\n".join(
            [
                "CAM++ baseline run complete",
                f"Output root: {artifacts.output_root}",
                f"Checkpoint: {artifacts.checkpoint_path}",
                f"Embeddings: {artifacts.embeddings_path}",
                f"Scores: {artifacts.scores_path}",
                f"Report: {artifacts.report_path}",
                f"Final train loss: {final_epoch.mean_loss}",
                f"Final train accuracy: {final_epoch.accuracy}",
                f"Score gap: {artifacts.score_summary.score_gap}",
            ]
        )
    )


if __name__ == "__main__":
    app()
