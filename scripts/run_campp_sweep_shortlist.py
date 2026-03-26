"""Run the bounded CAM++ stage-3 hyperparameter sweep shortlist."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.training.campp import (
    load_campp_sweep_shortlist_config,
    run_campp_sweep_shortlist,
)

app = typer.Typer(add_completion=False, help=__doc__)

_CONFIG = typer.Option(
    Path("configs/training/campp-stage3-sweep-shortlist.toml"),
    "--config",
    help="Path to the sweep-shortlist TOML config file.",
    show_default=True,
)
_ENV_FILE = typer.Option(
    Path(".env"),
    "--env-file",
    help="Optional .env file with secrets (e.g. tracking credentials).",
    show_default=True,
)
_DEVICE = typer.Option(
    None,
    "--device",
    help="Force device: 'cpu', 'cuda', 'mps'. Defaults to the shortlist/base config.",
)
_STAGE2_CHECKPOINT = typer.Option(
    None,
    "--stage2-checkpoint",
    help="Override the shared stage-2 warm-start source for all shortlist candidates.",
)
_CANDIDATE = typer.Option(
    None,
    "--candidate",
    help="Run only the named shortlist candidate. Repeatable.",
)
_CANDIDATE_LIMIT = typer.Option(
    None,
    "--candidate-limit",
    help="Only execute the first N configured candidates after filtering.",
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
    device: str | None = _DEVICE,
    stage2_checkpoint: Path | None = _STAGE2_CHECKPOINT,
    candidate: list[str] | None = _CANDIDATE,
    candidate_limit: int | None = _CANDIDATE_LIMIT,
    output: str = _OUTPUT,
) -> None:
    shortlist_config = load_campp_sweep_shortlist_config(config_path=config)
    artifacts = run_campp_sweep_shortlist(
        shortlist_config,
        config_path=config,
        env_file=env_file,
        device_override=device,
        stage2_checkpoint=stage2_checkpoint,
        candidate_ids=(None if not candidate else tuple(candidate)),
        candidate_limit=candidate_limit,
    )

    if output == "json":
        typer.echo(json.dumps(artifacts.to_dict(), indent=2, sort_keys=True))
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text, json")

    winner = artifacts.candidates[0]
    typer.echo("")
    typer.echo("=" * 72)
    typer.echo("  CAM++ Stage-3 shortlist complete")
    typer.echo("=" * 72)
    typer.echo(f"  Shortlist root : {artifacts.output_root}")
    typer.echo(f"  Report         : {artifacts.report_markdown_path}")
    typer.echo(f"  Winner         : {winner.candidate_id}")
    typer.echo(f"  Score          : {winner.selection_score}")
    typer.echo(f"  Weighted EER   : {winner.weighted_eer}")
    typer.echo(f"  Weighted MinDCF: {winner.weighted_min_dcf}")
    typer.echo(f"  Robust EER     : {winner.robust_eer}")
    typer.echo(f"  Clean EER      : {winner.clean_eer}")
    typer.echo("")
    typer.echo("  Leaderboard")
    for candidate_result in artifacts.candidates:
        typer.echo(
            "  "
            f"{candidate_result.rank}. {candidate_result.candidate_id} "
            f"(score={candidate_result.selection_score}, "
            f"robust_eer={candidate_result.robust_eer}, "
            f"clean_eer={candidate_result.clean_eer})"
        )
    typer.echo("")


if __name__ == "__main__":
    app()
