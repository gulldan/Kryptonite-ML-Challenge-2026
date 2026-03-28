"""Run CAM++ teacher-student distillation.

The checked-in recipe keeps the student family fixed to CAM++ and reuses the
stage-3 curriculum as the warm-start baseline. A frozen PEFT teacher supplies
embedding and pairwise-score supervision on the same waveform crop that feeds
the student Fbank frontend.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer

from kryptonite.training import load_campp_distillation_config, run_campp_distillation

app = typer.Typer(add_completion=False, help=__doc__)

_CONFIG = typer.Option(
    Path("configs/training/campp-distillation.toml"),
    "--config",
    help="Path to the CAM++ distillation TOML config file.",
    show_default=True,
)
_ENV_FILE = typer.Option(
    Path(".env"),
    "--env-file",
    help="Optional .env file with secrets (e.g. tracking credentials).",
    show_default=True,
)
_PROJECT_OVERRIDE = typer.Option(
    None,
    "--project-override",
    help="Override a base project config key, e.g. 'training.max_epochs=2'. Repeatable.",
)
_DEVICE = typer.Option(
    None,
    "--device",
    help="Force device: 'cpu', 'cuda', 'mps'. Defaults to the config/runtime value.",
)
_STUDENT_CHECKPOINT = typer.Option(
    None,
    "--student-checkpoint",
    help="Override the student warm-start checkpoint with a checkpoint file or completed run dir.",
)
_COMPARISON_CHECKPOINT = typer.Option(
    None,
    "--comparison-checkpoint",
    help="Optional explicit baseline-reference checkpoint. Defaults to --student-checkpoint.",
)
_TEACHER_CHECKPOINT = typer.Option(
    None,
    "--teacher-checkpoint",
    help="Override the teacher PEFT checkpoint with a checkpoint dir or completed run dir.",
)
_OUTPUT = typer.Option(
    "text",
    "--output",
    help="Output format: 'text' (default) or 'json'.",
)


@app.command()
def main(
    config: Path = _CONFIG,
    env_file: Path = _ENV_FILE,
    project_override: list[str] | None = _PROJECT_OVERRIDE,
    device: str | None = _DEVICE,
    student_checkpoint: Path | None = _STUDENT_CHECKPOINT,
    comparison_checkpoint: Path | None = _COMPARISON_CHECKPOINT,
    teacher_checkpoint: Path | None = _TEACHER_CHECKPOINT,
    output: str = _OUTPUT,
) -> None:
    distillation_config = load_campp_distillation_config(
        config_path=config,
        env_file=env_file,
        project_overrides=project_override or [],
    )
    if student_checkpoint is not None:
        distillation_config = replace(
            distillation_config,
            student=replace(
                distillation_config.student,
                checkpoint=student_checkpoint.as_posix(),
                comparison_checkpoint=(
                    distillation_config.student.comparison_checkpoint
                    if comparison_checkpoint is None
                    else comparison_checkpoint.as_posix()
                ),
            ),
        )
    elif comparison_checkpoint is not None:
        distillation_config = replace(
            distillation_config,
            student=replace(
                distillation_config.student,
                comparison_checkpoint=comparison_checkpoint.as_posix(),
            ),
        )
    if teacher_checkpoint is not None:
        distillation_config = replace(
            distillation_config,
            teacher=replace(
                distillation_config.teacher,
                checkpoint=teacher_checkpoint.as_posix(),
            ),
        )
    artifacts = run_campp_distillation(
        distillation_config,
        config_path=config,
        device_override=device,
    )

    if output == "json":
        typer.echo(json.dumps(artifacts.to_dict(), indent=2, sort_keys=True))
        return

    ts = artifacts.training_summary
    final_epoch = artifacts.distillation_epochs[-1]
    comparison = artifacts.comparison

    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("  CAM++ distillation complete")
    typer.echo("=" * 60)
    typer.echo(f"  Checkpoint   : {artifacts.checkpoint_path}")
    typer.echo(f"  Device       : {ts.device}")
    typer.echo(f"  Speakers     : {ts.speaker_count}")
    typer.echo(f"  Train rows   : {ts.train_row_count}")
    typer.echo(f"  Epochs       : {len(artifacts.distillation_epochs)}")
    typer.echo(f"  Final loss   : {final_epoch.mean_loss}")
    typer.echo(f"  Class loss   : {final_epoch.mean_classification_loss}")
    typer.echo(f"  Embed loss   : {final_epoch.mean_embedding_loss}")
    typer.echo(f"  Score loss   : {final_epoch.mean_score_loss}")
    typer.echo(f"  Distilled EER: {comparison.distilled_eer}")
    typer.echo(f"  Baseline EER : {comparison.baseline_eer}")
    typer.echo(f"  EER delta    : {comparison.eer_delta}")
    typer.echo(f"  MinDCF delta : {comparison.min_dcf_delta}")
    typer.echo(f"  Report       : {artifacts.report_path}")
    typer.echo(f"  Comparison   : {artifacts.comparison_markdown_path}")
    typer.echo("")


if __name__ == "__main__":
    app()
