"""Smoke-check the training environment imports expected by the project."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.config import ProjectConfig, load_project_config
from kryptonite.deployment import (
    ArtifactReport,
    ArtifactSpec,
    build_artifact_report,
    render_artifact_report,
)
from kryptonite.training.environment import (
    build_training_environment_report,
    render_training_environment_report,
)

app = typer.Typer(add_completion=False, help=__doc__)

REQUIRE_GPU_OPTION = typer.Option(
    False,
    "--require-gpu",
    help="Fail unless the GPU/TensorRT stack is importable too.",
)
CONFIG_OPTION = typer.Option(
    Path("configs/base.toml"),
    "--config",
    help="Path to the active training/deployment config.",
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
REQUIRE_ARTIFACTS_OPTION = typer.Option(
    False,
    "--require-artifacts",
    help="Fail unless dataset/manifests paths are present for target-machine runs.",
)
OUTPUT_OPTION = typer.Option(
    "text",
    "--output",
    help="Output format: text or json.",
    case_sensitive=False,
)


def build_training_artifact_report(*, config: ProjectConfig, strict: bool) -> ArtifactReport:
    specs = [
        ArtifactSpec(
            name="dataset_root",
            configured_path=config.paths.dataset_root,
            path_type="dir",
            required_on_target=True,
            require_non_empty=True,
            description="Local training datasets.",
        ),
        ArtifactSpec(
            name="manifests_root",
            configured_path=config.paths.manifests_root,
            path_type="dir",
            required_on_target=False,
            description="Generated train/dev manifests.",
        ),
        ArtifactSpec(
            name="artifacts_root",
            configured_path=config.paths.artifacts_root,
            path_type="dir",
            required_on_target=False,
            description="Experiment outputs.",
        ),
        ArtifactSpec(
            name="cache_root",
            configured_path=config.paths.cache_root,
            path_type="dir",
            required_on_target=False,
            description="Feature and model caches.",
        ),
    ]
    return build_artifact_report(
        scope="training",
        strict=strict,
        project_root=config.paths.project_root,
        specs=specs,
    )


@app.command()
def main(
    require_gpu: bool = REQUIRE_GPU_OPTION,
    config: Path = CONFIG_OPTION,
    env_file: Path = ENV_FILE_OPTION,
    override: list[str] | None = OVERRIDE_OPTION,
    require_artifacts: bool = REQUIRE_ARTIFACTS_OPTION,
    output: str = OUTPUT_OPTION,
) -> None:
    config_payload = load_project_config(
        config_path=config,
        overrides=override or [],
        env_file=env_file,
    )
    strict_artifacts = require_artifacts or config_payload.deployment.require_artifacts
    report = build_training_environment_report(require_gpu=require_gpu)
    artifact_report = build_training_artifact_report(config=config_payload, strict=strict_artifacts)

    if output == "json":
        typer.echo(
            json.dumps(
                {
                    "environment": report.to_dict(),
                    "artifacts": artifact_report.to_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif output == "text":
        typer.echo(
            "\n\n".join(
                (
                    render_training_environment_report(report),
                    render_artifact_report(artifact_report),
                )
            )
        )
    else:
        raise typer.BadParameter("output must be one of: text, json")

    raise typer.Exit(code=0 if report.passed and artifact_report.passed else 1)


if __name__ == "__main__":
    app()
