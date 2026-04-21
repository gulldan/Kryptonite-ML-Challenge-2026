"""Continue a teacher PEFT run from a saved checkpoint, optionally after LoRA merge."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from kryptonite.training.teacher_peft import (
    load_teacher_checkpoint_payload,
    load_teacher_peft_config,
    load_teacher_peft_encoder_from_checkpoint,
    merge_teacher_lora_backbone,
    run_teacher_peft,
)
from kryptonite.training.teacher_peft.model import prepare_teacher_backbone_for_training

app = typer.Typer(add_completion=False, help=__doc__)

CONFIG_OPTION = typer.Option(..., "--config", help="Path to the stage config.")
INIT_CHECKPOINT_OPTION = typer.Option(
    ...,
    "--init-checkpoint",
    help="Checkpoint dir or run dir from a previous teacher PEFT stage.",
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
MERGE_LORA_OPTION = typer.Option(
    False,
    "--merge-lora",
    help="Merge LoRA adapters into the backbone before continuing training.",
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


@app.command()
def main(
    config: Path = CONFIG_OPTION,
    init_checkpoint: Path = INIT_CHECKPOINT_OPTION,
    env_file: Path = ENV_FILE_OPTION,
    project_override: list[str] | None = PROJECT_OVERRIDE_OPTION,
    device: str | None = DEVICE_OPTION,
    merge_lora: bool = MERGE_LORA_OPTION,
    init_classifier_from_checkpoint: bool = INIT_CLASSIFIER_OPTION,
    output: str = OUTPUT_OPTION,
) -> None:
    teacher = load_teacher_peft_config(
        config_path=config,
        env_file=env_file,
        project_overrides=project_override or [],
    )
    token = teacher.project.resolved_secrets.get("huggingface_hub_token")
    checkpoint_dir, metadata, feature_extractor, encoder = (
        load_teacher_peft_encoder_from_checkpoint(
            checkpoint_path=init_checkpoint,
            project_root=teacher.project.paths.project_root,
            token=token,
            trainable=not merge_lora and not teacher.model.freeze_feature_encoder,
        )
    )
    if merge_lora:
        encoder = merge_teacher_lora_backbone(encoder)
    prepare_teacher_backbone_for_training(
        backbone=encoder.backbone,
        model_config=teacher.model,
        peft_only=False,
    )
    classifier_state_dict = None
    classifier_speaker_to_index = None
    if init_classifier_from_checkpoint:
        _, _, payload = load_teacher_checkpoint_payload(
            checkpoint_path=checkpoint_dir,
            project_root=teacher.project.paths.project_root,
        )
        raw_classifier_state = payload.get("classifier_state_dict")
        raw_speaker_to_index = payload.get("speaker_to_index")
        if not isinstance(raw_classifier_state, dict) or not isinstance(raw_speaker_to_index, dict):
            raise typer.BadParameter(
                "init checkpoint does not contain classifier_state_dict and speaker_to_index"
            )
        classifier_state_dict = dict(raw_classifier_state)
        classifier_speaker_to_index = {
            str(speaker_id): int(index) for speaker_id, index in raw_speaker_to_index.items()
        }
    artifacts = run_teacher_peft(
        teacher,
        config_path=config,
        device_override=device,
        feature_extractor_override=feature_extractor,
        encoder_override=encoder,
        classifier_state_dict=classifier_state_dict,
        classifier_speaker_to_index=classifier_speaker_to_index,
    )
    payload = artifacts.to_dict()
    payload["init_checkpoint"] = str(checkpoint_dir)
    payload["merge_lora"] = merge_lora
    payload["source_metadata"] = metadata

    if output == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text, json")
    final_epoch = artifacts.training_summary.epochs[-1]
    typer.echo(
        "\n".join(
            [
                "Teacher PEFT fine-tune complete",
                f"Init checkpoint: {checkpoint_dir}",
                f"Merged LoRA: {merge_lora}",
                f"Output root: {artifacts.output_root}",
                f"Checkpoint dir: {artifacts.checkpoint_path}",
                f"Final train loss: {final_epoch.mean_loss}",
                f"Final train accuracy: {final_epoch.accuracy}",
                f"Score gap: {artifacts.score_summary.score_gap}",
            ]
        )
    )


if __name__ == "__main__":
    app()
