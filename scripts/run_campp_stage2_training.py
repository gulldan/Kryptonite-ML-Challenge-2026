"""Run CAM++ stage-2 heavy multi-condition training, export dev embeddings, write scores.

Stage-2 requires a stage-1 checkpoint (campp_stage1_encoder.pt) specified in the
[stage2] config section.  The model is warm-initialised from that checkpoint and
then trained with:
  - Heavy augmentation from epoch 1 (corruption bank, multi-severity, no warmup)
  - Hard-negative speaker mining every N epochs
  - Short-utterance curriculum (1.5 s → 4.0 s across three training phases)

Example usage (dry-run with demo data):
  uv run python scripts/run_campp_stage2_training.py \\
      --config configs/training/campp-stage2.toml \\
      --project-override 'training.max_epochs=1' \\
      --project-override 'runtime.num_workers=0' \\
      --device cpu

Production run on gpu-server (after approval):
  uv run python scripts/run_campp_stage2_training.py \\
      --config configs/training/campp-stage2.toml
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.training.campp import load_campp_stage2_config, run_campp_stage2

app = typer.Typer(add_completion=False, help=__doc__)

_CONFIG = typer.Option(
    Path("configs/training/campp-stage2.toml"),
    "--config",
    help="Path to the stage-2 TOML config file.",
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
    output: str = _OUTPUT,
) -> None:
    stage2_config = load_campp_stage2_config(
        config_path=config,
        env_file=env_file,
        project_overrides=project_override or [],
    )
    artifacts = run_campp_stage2(
        stage2_config,
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
    typer.echo("  CAM++ Stage-2 complete")
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
        m = vr.summary.metrics
        typer.echo(f"  EER          : {m.eer}")
        typer.echo(f"  MinDCF       : {m.min_dcf}")
    typer.echo(f"  Report       : {artifacts.report_path}")
    typer.echo("")


if __name__ == "__main__":
    app()
