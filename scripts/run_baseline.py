"""Train a speaker baseline for any supported model architecture."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)

MODEL_OPTION = typer.Option(
    ...,
    "--model",
    help="Model architecture: campp, eres2netv2.",
    case_sensitive=False,
)
CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the baseline TOML config for the chosen model.",
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

SUPPORTED_MODELS = ("campp", "eres2netv2")


@app.command()
def main(
    model: str = MODEL_OPTION,
    config: Path = CONFIG_OPTION,
    env_file: Path = ENV_FILE_OPTION,
    project_override: list[str] | None = PROJECT_OVERRIDE_OPTION,
    device: str | None = DEVICE_OPTION,
    output: str = OUTPUT_OPTION,
) -> None:
    normalized_model = model.strip().lower()
    overrides = project_override or []

    if normalized_model == "campp":
        from kryptonite.training import load_campp_baseline_config, run_campp_baseline

        baseline = load_campp_baseline_config(
            config_path=config, env_file=env_file, project_overrides=overrides
        )
        artifacts = run_campp_baseline(baseline, config_path=config, device_override=device)
    elif normalized_model == "eres2netv2":
        from kryptonite.training import load_eres2netv2_baseline_config, run_eres2netv2_baseline

        baseline = load_eres2netv2_baseline_config(
            config_path=config, env_file=env_file, project_overrides=overrides
        )
        artifacts = run_eres2netv2_baseline(baseline, config_path=config, device_override=device)
    else:
        raise typer.BadParameter(
            f"Unknown model {model!r}. Supported: {', '.join(SUPPORTED_MODELS)}"
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
                f"{model} baseline run complete",
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
