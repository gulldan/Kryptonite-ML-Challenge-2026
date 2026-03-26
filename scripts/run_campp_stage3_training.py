"""Run CAM++ stage-3 large-margin fine-tuning.

Stage-3 requires a completed stage-2 checkpoint (campp_stage2_encoder.pt) specified in the
[stage3] config section. The model is warm-initialised from that checkpoint and then trained with:
  - target-like augmentation policy (configured via project overrides)
  - longer fixed crops than stage-2
  - linear large-margin schedule over the early fine-tuning epochs

Example usage (dry-run with demo data after a compatible stage-2 smoke run):
  uv run python scripts/run_campp_stage3_training.py \
      --config configs/training/campp-stage3-smoke.toml \
      --stage2-checkpoint artifacts/baselines/campp-stage2/<run-id> \
      --device cpu

Production run on gpu-server:
  uv run python scripts/run_campp_stage3_training.py \
      --config configs/training/campp-stage3.toml \
      --stage2-checkpoint \
      /mnt/storage/Kryptonite-ML-Challenge-2026/artifacts/baselines/campp-stage2/<run-id>
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer

from kryptonite.training.campp import load_campp_stage3_config, run_campp_stage3

app = typer.Typer(add_completion=False, help=__doc__)

_CONFIG = typer.Option(
    Path("configs/training/campp-stage3.toml"),
    "--config",
    help="Path to the stage-3 TOML config file.",
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
    help="Override a base config key, e.g. 'training.max_epochs=2'. Repeatable.",
)
_DEVICE = typer.Option(
    None,
    "--device",
    help="Force device: 'cpu', 'cuda', 'mps'. Defaults to auto-detect.",
)
_STAGE2_CHECKPOINT = typer.Option(
    None,
    "--stage2-checkpoint",
    help=(
        "Override the stage-2 warm-start source with a checkpoint file or completed run directory."
    ),
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
    stage2_checkpoint: Path | None = _STAGE2_CHECKPOINT,
    output: str = _OUTPUT,
) -> None:
    stage3_config = load_campp_stage3_config(
        config_path=config,
        env_file=env_file,
        project_overrides=project_override or [],
    )
    if stage2_checkpoint is not None:
        stage3_config = replace(
            stage3_config,
            stage3=replace(
                stage3_config.stage3,
                stage2_checkpoint=stage2_checkpoint.as_posix(),
            ),
        )
    artifacts = run_campp_stage3(
        stage3_config,
        config_path=config,
        device_override=device,
    )

    if output == "json":
        typer.echo(json.dumps(artifacts.to_dict(), indent=2, sort_keys=True))
        return

    ts = artifacts.training_summary
    final = ts.epochs[-1]
    ss = artifacts.score_summary
    vr = artifacts.verification_report

    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("  CAM++ Stage-3 complete")
    typer.echo("=" * 60)
    typer.echo(f"  Checkpoint   : {artifacts.checkpoint_path}")
    typer.echo(f"  Device       : {ts.device}")
    typer.echo(f"  Speakers     : {ts.speaker_count}")
    typer.echo(f"  Train rows   : {ts.train_row_count}")
    typer.echo(f"  Epochs       : {len(ts.epochs)}")
    typer.echo(f"  Final loss   : {final.mean_loss}")
    typer.echo(f"  Final acc    : {final.accuracy}")
    typer.echo(f"  Score gap    : {ss.score_gap}")
    if vr is not None:
        metrics = vr.summary.metrics
        typer.echo(f"  EER          : {metrics.eer}")
        typer.echo(f"  MinDCF       : {metrics.min_dcf}")
    typer.echo(f"  Report       : {artifacts.report_path}")
    typer.echo("")


if __name__ == "__main__":
    app()
