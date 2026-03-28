"""Run CAM++ clean/corrupted consistency fine-tuning.

The checked-in recipe reuses the existing stage-3 CAM++ checkpoint as the warm
start and applies an additional clean/corrupted invariance objective on aligned
utterance crops. Every run also re-evaluates the baseline checkpoint on the
same clean dev manifest and writes a robust-dev ablation report.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer

from kryptonite.training import load_campp_consistency_config, run_campp_consistency

app = typer.Typer(add_completion=False, help=__doc__)

_CONFIG = typer.Option(
    Path("configs/training/campp-consistency.toml"),
    "--config",
    help="Path to the CAM++ consistency TOML config file.",
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
    help="Override the stage-3 warm-start checkpoint with a checkpoint file or completed run dir.",
)
_COMPARISON_CHECKPOINT = typer.Option(
    None,
    "--comparison-checkpoint",
    help="Optional explicit baseline-reference checkpoint. Defaults to --student-checkpoint.",
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
    output: str = _OUTPUT,
) -> None:
    consistency_config = load_campp_consistency_config(
        config_path=config,
        env_file=env_file,
        project_overrides=project_override or [],
    )
    if student_checkpoint is not None:
        consistency_config = replace(
            consistency_config,
            student=replace(
                consistency_config.student,
                checkpoint=student_checkpoint.as_posix(),
                comparison_checkpoint=(
                    consistency_config.student.comparison_checkpoint
                    if comparison_checkpoint is None
                    else comparison_checkpoint.as_posix()
                ),
            ),
        )
    elif comparison_checkpoint is not None:
        consistency_config = replace(
            consistency_config,
            student=replace(
                consistency_config.student,
                comparison_checkpoint=comparison_checkpoint.as_posix(),
            ),
        )
    artifacts = run_campp_consistency(
        consistency_config,
        config_path=config,
        device_override=device,
    )

    if output == "json":
        typer.echo(json.dumps(artifacts.to_dict(), indent=2, sort_keys=True))
        return

    ts = artifacts.training_summary
    final_epoch = artifacts.consistency_epochs[-1]
    comparison = artifacts.comparison

    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("  CAM++ consistency fine-tuning complete")
    typer.echo("=" * 60)
    typer.echo(f"  Checkpoint   : {artifacts.checkpoint_path}")
    typer.echo(f"  Device       : {ts.device}")
    typer.echo(f"  Speakers     : {ts.speaker_count}")
    typer.echo(f"  Train rows   : {ts.train_row_count}")
    typer.echo(f"  Epochs       : {len(artifacts.consistency_epochs)}")
    typer.echo(f"  Final loss   : {final_epoch.mean_loss}")
    typer.echo(f"  Clean cls    : {final_epoch.mean_clean_classification_loss}")
    typer.echo(f"  Corr cls     : {final_epoch.mean_corrupted_classification_loss}")
    typer.echo(f"  Embed loss   : {final_epoch.mean_embedding_loss}")
    typer.echo(f"  Score loss   : {final_epoch.mean_score_loss}")
    typer.echo(f"  Paired ratio : {final_epoch.paired_ratio}")
    typer.echo(f"  Consist. EER : {comparison.consistency_eer}")
    typer.echo(f"  Baseline EER : {comparison.baseline_eer}")
    typer.echo(f"  EER delta    : {comparison.eer_delta}")
    typer.echo(f"  MinDCF delta : {comparison.min_dcf_delta}")
    typer.echo(f"  Report       : {artifacts.report_path}")
    typer.echo(f"  Comparison   : {artifacts.comparison_markdown_path}")
    if artifacts.robust_dev_ablation_markdown_path is not None:
        typer.echo(f"  Robust dev   : {artifacts.robust_dev_ablation_markdown_path}")
    typer.echo("")


if __name__ == "__main__":
    app()
