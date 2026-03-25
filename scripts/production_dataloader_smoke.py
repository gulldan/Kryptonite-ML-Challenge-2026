"""Smoke-check the production training dataloader over a manifest-backed train split."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer

from kryptonite.config import load_project_config
from kryptonite.deployment import resolve_project_path
from kryptonite.training.manifest_speaker_data import build_speaker_index, load_manifest_rows
from kryptonite.training.production_dataloader import build_production_train_dataloader

app = typer.Typer(add_completion=False, help=__doc__)

TRAIN_MANIFEST_OPTION = typer.Option(..., "--train-manifest", help="Train manifest JSONL.")
CONFIG_OPTION = typer.Option(
    Path("configs/deployment/train.toml"),
    "--config",
    help="Project/base config used to build the loader.",
)
ENV_FILE_OPTION = typer.Option(
    Path(".env"),
    "--env-file",
    help="Optional dotenv file with secrets.",
)
OVERRIDE_OPTION = typer.Option(
    None,
    "--override",
    help="Config override in dotted.key=value form. Can be passed multiple times.",
)
BATCHES_OPTION = typer.Option(4, "--batches", min=1, help="How many batches to inspect.")
OUTPUT_OPTION = typer.Option(
    "text",
    "--output",
    case_sensitive=False,
    help="Output format: text or json.",
)


@app.command()
def main(
    train_manifest: Path = TRAIN_MANIFEST_OPTION,
    config: Path = CONFIG_OPTION,
    env_file: Path = ENV_FILE_OPTION,
    override: list[str] | None = OVERRIDE_OPTION,
    batches: int = BATCHES_OPTION,
    output: str = OUTPUT_OPTION,
) -> None:
    project = load_project_config(
        config_path=config,
        overrides=override or [],
        env_file=env_file,
    )
    project_root = resolve_project_path(project.paths.project_root, ".")
    rows = load_manifest_rows(train_manifest, project_root=project_root)
    speaker_to_index = build_speaker_index(rows)
    dataset, sampler, loader = build_production_train_dataloader(
        rows=rows,
        speaker_to_index=speaker_to_index,
        project=project,
        total_epochs=project.training.max_epochs,
        pin_memory=False,
    )
    dataset.set_epoch(0)
    sampler.set_epoch(0)

    observed_speakers: Counter[str] = Counter()
    observed_intensities: Counter[str] = Counter()
    batch_shapes: list[list[int]] = []
    clean_counts: list[int] = []
    crop_seconds: list[float | None] = []

    for batch_index, batch in enumerate(loader):
        if batch_index >= batches:
            break
        observed_speakers.update(batch.speaker_ids)
        observed_intensities.update(
            intensity for intensity in batch.recipe_intensities if intensity is not None
        )
        batch_shapes.append(list(batch.features.shape))
        clean_counts.append(int(batch.clean_sample_mask.sum().item()))
        crop_seconds.extend(batch.crop_seconds)

    summary = {
        "train_manifest": str(train_manifest),
        "speaker_count": len(speaker_to_index),
        "row_count": len(rows),
        "inspected_batches": len(batch_shapes),
        "batch_shapes": batch_shapes,
        "clean_counts": clean_counts,
        "unique_crop_seconds": sorted({value for value in crop_seconds if value is not None}),
        "observed_speakers": dict(sorted(observed_speakers.items())),
        "observed_intensities": dict(sorted(observed_intensities.items())),
        "sampler_state": sampler.state_dict(),
    }

    if output == "json":
        typer.echo(json.dumps(summary, indent=2, sort_keys=True))
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text, json")

    typer.echo("Production dataloader smoke")
    typer.echo(f"Train manifest: {summary['train_manifest']}")
    typer.echo(f"Rows / speakers: {summary['row_count']} / {summary['speaker_count']}")
    typer.echo(f"Inspected batches: {summary['inspected_batches']}")
    typer.echo(f"Batch shapes: {summary['batch_shapes']}")
    typer.echo(f"Clean counts: {summary['clean_counts']}")
    typer.echo(f"Unique crop seconds: {summary['unique_crop_seconds']}")
    typer.echo(f"Observed speakers: {summary['observed_speakers']}")
    typer.echo(f"Observed intensities: {summary['observed_intensities']}")
    typer.echo(f"Sampler state: {summary['sampler_state']}")


if __name__ == "__main__":
    app()
