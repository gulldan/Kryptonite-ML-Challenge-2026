"""Fine-tune ERes2NetV2 from an existing checkpoint with baseline training machinery."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import typer

from kryptonite.models.eres2netv2 import load_eres2netv2_encoder_from_checkpoint
from kryptonite.training.baseline_pipeline import run_speaker_baseline
from kryptonite.training.eres2netv2 import load_eres2netv2_baseline_config
from kryptonite.training.eres2netv2.pipeline import REPORT_FILE_NAME
from kryptonite.training.speaker_baseline import resolve_device

app = typer.Typer(add_completion=False, help=__doc__)

CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the ERes2NetV2 fine-tune TOML config.",
)
INIT_CHECKPOINT_OPTION = typer.Option(
    ...,
    "--init-checkpoint",
    help="Path to the source ERes2NetV2 encoder checkpoint.",
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
    init_checkpoint: Path = INIT_CHECKPOINT_OPTION,
    env_file: Path = ENV_FILE_OPTION,
    project_override: list[str] | None = PROJECT_OVERRIDE_OPTION,
    device: str | None = DEVICE_OPTION,
    output: str = OUTPUT_OPTION,
) -> None:
    overrides = project_override or []
    baseline = load_eres2netv2_baseline_config(
        config_path=config,
        env_file=env_file,
        project_overrides=overrides,
    )
    resolved_device = resolve_device(device or baseline.project.runtime.device)
    checkpoint_path, checkpoint_model_config, encoder = load_eres2netv2_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=init_checkpoint,
    )
    if checkpoint_model_config != baseline.model:
        typer.echo(
            "warning: config model section differs from checkpoint model_config; "
            "using checkpoint encoder/config for fine-tune.",
            err=True,
        )
    encoder = encoder.to(resolved_device)
    artifacts = run_speaker_baseline(
        baseline,
        encoder=encoder,
        embedding_size=checkpoint_model_config.embedding_size,
        model_config_dict=asdict(checkpoint_model_config),
        baseline_name="ERes2NetV2 pseudo-label fine-tune",
        report_file_name=REPORT_FILE_NAME,
        embedding_source="eres2netv2_pseudo_finetune",
        tracker_kind="eres2netv2-pseudo-finetune",
        config_path=config,
        device=resolved_device,
    )
    payload = artifacts.to_dict()
    payload["init_checkpoint"] = str(checkpoint_path)
    if output == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text, json")
    final_epoch = artifacts.training_summary.epochs[-1]
    typer.echo(
        "\n".join(
            [
                "ERes2NetV2 pseudo-label fine-tune complete",
                f"Init checkpoint: {checkpoint_path}",
                f"Output root: {artifacts.output_root}",
                f"Checkpoint: {artifacts.checkpoint_path}",
                f"Final train loss: {final_epoch.mean_loss}",
                f"Final train accuracy: {final_epoch.accuracy}",
                f"Score gap: {artifacts.score_summary.score_gap}",
            ]
        )
    )


if __name__ == "__main__":
    app()
