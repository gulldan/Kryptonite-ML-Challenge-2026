"""Prepare manifests and speaker-disjoint splits for the FFSVC 2022 surrogate bundle."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.config import load_project_config
from kryptonite.data.ffsvc import prepare_ffsvc2022_surrogate

app = typer.Typer(add_completion=False, help=__doc__)

CONFIG_OPTION = typer.Option(
    Path("configs/base.toml"),
    "--config",
    help="Base project config that defines dataset and manifests roots.",
)
DEV_SPEAKERS_OPTION = typer.Option(
    6,
    "--dev-speakers",
    min=1,
    help="How many speakers to hold out for the deterministic dev split.",
)
SEED_OPTION = typer.Option(
    42,
    "--seed",
    help="Seed used to shuffle speakers before the split.",
)


@app.command()
def main(
    config: Path = CONFIG_OPTION,
    dev_speakers: int = DEV_SPEAKERS_OPTION,
    seed: int = SEED_OPTION,
) -> None:
    project_config = load_project_config(config_path=config)
    artifacts = prepare_ffsvc2022_surrogate(
        project_root=project_config.paths.project_root,
        dataset_root=project_config.paths.dataset_root,
        manifests_root=project_config.paths.manifests_root,
        dev_speaker_count=dev_speakers,
        seed=seed,
    )
    typer.echo(json.dumps(artifacts.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
