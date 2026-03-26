"""Select the final CAM++ stage-3 candidate from a shortlist report.

This post-shortlist step keeps the shortlist ranking objective as the source of
truth and optionally evaluates uniform checkpoint averages over the top-ranked,
checkpoint-compatible candidates.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.training.campp import (
    load_campp_model_selection_config,
    run_campp_model_selection,
)

app = typer.Typer(add_completion=False, help=__doc__)

_CONFIG = typer.Option(
    Path("configs/training/campp-stage3-model-selection.toml"),
    "--config",
    help="Path to the model-selection TOML config file.",
    show_default=True,
)
_ENV_FILE = typer.Option(
    Path(".env"),
    "--env-file",
    help="Optional .env file with secrets used by the base stage-3 config.",
    show_default=True,
)
_DEVICE = typer.Option(
    None,
    "--device",
    help="Force device for averaged-variant evaluation: 'cpu', 'cuda', 'mps'.",
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
    output: str = _OUTPUT,
) -> None:
    selection_config = load_campp_model_selection_config(config_path=config)
    artifacts = run_campp_model_selection(
        selection_config,
        config_path=config,
        env_file=env_file,
        device_override=device,
    )

    if output == "json":
        typer.echo(json.dumps(artifacts.to_dict(), indent=2, sort_keys=True))
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text, json")

    winner = artifacts.variants[0]
    typer.echo("")
    typer.echo("=" * 72)
    typer.echo("  CAM++ model selection complete")
    typer.echo("=" * 72)
    typer.echo(f"  Output root      : {artifacts.output_root}")
    typer.echo(f"  Report           : {artifacts.report_markdown_path}")
    typer.echo(f"  Winner variant   : {winner.variant_id}")
    typer.echo(f"  Winner checkpoint: {artifacts.final_checkpoint_path}")
    typer.echo(f"  Selection score  : {winner.selection_score}")
    typer.echo(f"  Weighted EER     : {winner.weighted_eer}")
    typer.echo(f"  Weighted MinDCF  : {winner.weighted_min_dcf}")
    typer.echo(f"  Robust EER       : {winner.robust_eer}")
    typer.echo(f"  Clean EER        : {winner.clean_eer}")
    typer.echo("")
    typer.echo("  Ranked variants")
    for variant in artifacts.variants:
        typer.echo(
            "  "
            f"{variant.rank}. {variant.variant_id} "
            f"(score={variant.selection_score}, averaged={variant.uses_checkpoint_averaging})"
        )
    typer.echo("")


if __name__ == "__main__":
    app()
