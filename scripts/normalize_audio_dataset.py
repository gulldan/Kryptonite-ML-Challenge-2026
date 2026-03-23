"""Normalize manifests-backed audio into a deterministic 16 kHz mono bundle."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.config import load_project_config
from kryptonite.data.normalization import AudioNormalizationPolicy, normalize_audio_manifest_bundle

app = typer.Typer(add_completion=False, help=__doc__)

CONFIG_OPTION = typer.Option(
    Path("configs/base.toml"),
    "--config",
    help="Base project config that defines project and dataset roots.",
)
SOURCE_MANIFESTS_ROOT_OPTION = typer.Option(
    "artifacts/manifests/ffsvc2022-surrogate",
    "--source-manifests-root",
    help="Source manifests root with active manifests and optional quarantine/trials.",
)
OUTPUT_ROOT_OPTION = typer.Option(
    "artifacts/preprocessed/ffsvc2022-surrogate",
    "--output-root",
    help="Output root for normalized audio, rewritten manifests, and reports.",
)
OVERRIDE_OPTION = typer.Option(
    None,
    "--override",
    help="Config override in dotted.key=value form. Can be passed multiple times.",
)


@app.command()
def main(
    config: Path = CONFIG_OPTION,
    source_manifests_root: str = SOURCE_MANIFESTS_ROOT_OPTION,
    output_root: str = OUTPUT_ROOT_OPTION,
    override: list[str] | None = OVERRIDE_OPTION,
) -> None:
    project_config = load_project_config(config_path=config, overrides=override or [])
    summary = normalize_audio_manifest_bundle(
        project_root=project_config.paths.project_root,
        dataset_root=project_config.paths.dataset_root,
        source_manifests_root=source_manifests_root,
        output_root=output_root,
        policy=AudioNormalizationPolicy.from_config(project_config.normalization),
    )
    typer.echo(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
