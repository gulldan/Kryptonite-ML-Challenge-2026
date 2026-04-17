"""Fine-tune CAM++ from an existing official or local checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import typer

from kryptonite.models.campp.checkpoint import (
    load_campp_checkpoint_payload,
    load_campp_encoder_from_checkpoint,
)
from kryptonite.training.baseline_pipeline import run_speaker_baseline
from kryptonite.training.campp import load_campp_baseline_config
from kryptonite.training.campp.pipeline import REPORT_FILE_NAME
from kryptonite.training.speaker_baseline import resolve_device
from kryptonite.training.trainable_scope import apply_encoder_trainable_scope

app = typer.Typer(add_completion=False, help=__doc__)

CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the CAM++ fine-tune TOML config.",
)
INIT_CHECKPOINT_OPTION = typer.Option(
    ...,
    "--init-checkpoint",
    help="Path to the source CAM++ encoder checkpoint.",
)
ENV_FILE_OPTION = typer.Option(
    Path(".env"),
    "--env-file",
    help="Optional dotenv file with secrets.",
)
PROJECT_OVERRIDE_OPTION = typer.Option(
    None,
    "--project-override",
    help="Extra base ProjectConfig override in dotted.key=value form. Can be repeated.",
)
DEVICE_OPTION = typer.Option(
    None,
    "--device",
    help="Optional device override. Defaults to the project runtime.device setting.",
)
INIT_CLASSIFIER_OPTION = typer.Option(
    False,
    "--init-classifier-from-checkpoint",
    help="Restore classifier weights and speaker index from the init checkpoint.",
)
OUTPUT_OPTION = typer.Option(
    "text",
    "--output",
    help="Output format: text or json.",
    case_sensitive=False,
)
ENCODER_TRAINABLE_SCOPE_OPTION = typer.Option(
    "all",
    "--encoder-trainable-scope",
    help=(
        "Encoder parameter scope to optimize: all or batchnorm-affine. "
        "batchnorm-affine freezes the encoder except BatchNorm affine tensors; "
        "BatchNorm running stats still update during training."
    ),
)


@app.command()
def main(
    config: Path = CONFIG_OPTION,
    init_checkpoint: Path = INIT_CHECKPOINT_OPTION,
    env_file: Path = ENV_FILE_OPTION,
    project_override: list[str] | None = PROJECT_OVERRIDE_OPTION,
    device: str | None = DEVICE_OPTION,
    init_classifier_from_checkpoint: bool = INIT_CLASSIFIER_OPTION,
    encoder_trainable_scope: str = ENCODER_TRAINABLE_SCOPE_OPTION,
    output: str = OUTPUT_OPTION,
) -> None:
    overrides = project_override or []
    baseline = load_campp_baseline_config(
        config_path=config,
        env_file=env_file,
        project_overrides=overrides,
    )
    resolved_device = resolve_device(device or baseline.project.runtime.device)
    checkpoint_path, checkpoint_model_config, encoder = load_campp_encoder_from_checkpoint(
        torch=torch,
        checkpoint_path=init_checkpoint,
    )
    if checkpoint_model_config != baseline.model:
        typer.echo(
            "warning: config model section differs from checkpoint model_config; "
            "using checkpoint encoder/config for fine-tune.",
            err=True,
        )
    try:
        trainable_scope_summary = apply_encoder_trainable_scope(
            encoder,
            scope=encoder_trainable_scope,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(
        "encoder trainable scope: "
        f"{trainable_scope_summary.scope}; "
        f"{trainable_scope_summary.trainable_parameters}/"
        f"{trainable_scope_summary.total_parameters} encoder params "
        f"({trainable_scope_summary.trainable_fraction:.4%}); "
        f"batchnorm_modules={trainable_scope_summary.batchnorm_module_count}",
        err=True,
    )
    encoder = encoder.to(resolved_device)
    classifier_state_dict, classifier_speaker_to_index = _load_classifier_init(
        checkpoint_path=checkpoint_path,
        enabled=init_classifier_from_checkpoint,
    )
    artifacts = run_speaker_baseline(
        baseline,
        encoder=encoder,
        embedding_size=checkpoint_model_config.embedding_size,
        model_config_dict=asdict(checkpoint_model_config),
        baseline_name="CAM++ official-frontend fine-tune",
        report_file_name=REPORT_FILE_NAME,
        embedding_source="campp_official_finetune",
        tracker_kind="campp-official-finetune",
        config_path=config,
        device=resolved_device,
        classifier_state_dict=classifier_state_dict,
        classifier_speaker_to_index=classifier_speaker_to_index,
    )
    payload = artifacts.to_dict()
    payload["init_checkpoint"] = str(checkpoint_path)
    payload["encoder_trainable_scope"] = trainable_scope_summary.to_dict()
    trainable_scope_summary_path = Path(artifacts.output_root) / "encoder_trainable_scope.json"
    trainable_scope_summary_path.write_text(
        json.dumps(trainable_scope_summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["encoder_trainable_scope_path"] = str(trainable_scope_summary_path)
    if output == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text, json")
    final_epoch = artifacts.training_summary.epochs[-1]
    typer.echo(
        "\n".join(
            [
                "CAM++ official-frontend fine-tune complete",
                f"Init checkpoint: {checkpoint_path}",
                f"Init classifier: {init_classifier_from_checkpoint}",
                f"Encoder trainable scope: {trainable_scope_summary.scope}",
                f"Feature frontend: {baseline.project.features.frontend}",
                f"Output root: {artifacts.output_root}",
                f"Checkpoint: {artifacts.checkpoint_path}",
                f"Final train loss: {final_epoch.mean_loss}",
                f"Final train accuracy: {final_epoch.accuracy}",
                f"Score gap: {artifacts.score_summary.score_gap}",
            ]
        )
    )


def _load_classifier_init(
    *,
    checkpoint_path: Path,
    enabled: bool,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, int] | None]:
    if not enabled:
        return None, None
    payload = load_campp_checkpoint_payload(torch=torch, checkpoint_path=checkpoint_path)
    classifier_state = payload.get("classifier_state_dict")
    speaker_to_index = payload.get("speaker_to_index")
    if not isinstance(classifier_state, dict) or not isinstance(speaker_to_index, dict):
        raise typer.BadParameter(
            "init checkpoint does not contain classifier_state_dict and speaker_to_index"
        )
    return dict(classifier_state), {
        str(speaker_id): int(index) for speaker_id, index in speaker_to_index.items()
    }


if __name__ == "__main__":
    app()
