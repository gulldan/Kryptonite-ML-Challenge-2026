"""Fine-tune official 3D-Speaker ERes2Net from a raw or local training checkpoint."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import typer

from kryptonite.training.baseline_config import (
    BaselineDataConfig,
    BaselineObjectiveConfig,
    BaselineOptimizationConfig,
    BaselineProvenanceConfig,
)
from kryptonite.training.baseline_pipeline import run_speaker_baseline
from kryptonite.training.config_helpers import load_baseline_toml_sections
from kryptonite.training.speaker_baseline import resolve_device

app = typer.Typer(add_completion=False, help=__doc__)

CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    help="Path to the official ERes2Net fine-tune TOML config.",
)
INIT_CHECKPOINT_OPTION = typer.Option(
    ...,
    "--init-checkpoint",
    help="Path to the source official ERes2Net checkpoint.",
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
OUTPUT_OPTION = typer.Option(
    "text",
    "--output",
    help="Output format: text or json.",
    case_sensitive=False,
)


@dataclass(frozen=True, slots=True)
class OfficialERes2NetModelConfig:
    speakerlab_root: str = "/tmp/3D-Speaker"
    feat_dim: int = 80
    embedding_size: int = 512
    m_channels: int = 64
    model_id: str = "iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k"

    def __post_init__(self) -> None:
        if self.feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        if self.embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if self.m_channels <= 0:
            raise ValueError("m_channels must be positive")
        if not self.speakerlab_root.strip():
            raise ValueError("speakerlab_root must not be empty")


@dataclass(frozen=True, slots=True)
class OfficialERes2NetBaselineConfig:
    base_config_path: str
    project_overrides: tuple[str, ...]
    project: Any
    data: BaselineDataConfig
    model: OfficialERes2NetModelConfig
    objective: BaselineObjectiveConfig
    optimization: BaselineOptimizationConfig
    provenance: BaselineProvenanceConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_config_path": self.base_config_path,
            "project_overrides": list(self.project_overrides),
            "project": self.project.to_dict(mask_secrets=True),
            "data": asdict(self.data),
            "model": asdict(self.model),
            "objective": asdict(self.objective),
            "optimization": asdict(self.optimization),
            "provenance": self.provenance.to_dict(),
        }


def load_official_eres2net_baseline_config(
    *,
    config_path: Path | str,
    env_file: Path | str | None = None,
    project_overrides: list[str] | None = None,
) -> OfficialERes2NetBaselineConfig:
    sections = load_baseline_toml_sections(
        config_path=config_path,
        env_file=env_file,
        project_overrides=project_overrides,
        data_defaults={
            "train_manifest": "artifacts/manifests/demo_manifest.jsonl",
            "dev_manifest": "artifacts/manifests/demo_manifest.jsonl",
            "output_root": "artifacts/baselines/official-3dspeaker-eres2net",
            "trials_manifest": None,
            "checkpoint_name": "official_3dspeaker_eres2net_encoder.pt",
            "generate_demo_artifacts_if_missing": True,
            "max_train_rows": None,
            "max_dev_rows": None,
        },
    )
    return OfficialERes2NetBaselineConfig(
        base_config_path=sections.base_config_path,
        project_overrides=sections.project_overrides,
        project=sections.project,
        data=sections.data,
        model=OfficialERes2NetModelConfig(**sections.model_section),
        objective=sections.objective,
        optimization=sections.optimization,
        provenance=sections.provenance,
    )


@app.command()
def main(
    config: Path = CONFIG_OPTION,
    init_checkpoint: Path = INIT_CHECKPOINT_OPTION,
    env_file: Path = ENV_FILE_OPTION,
    project_override: list[str] | None = PROJECT_OVERRIDE_OPTION,
    device: str | None = DEVICE_OPTION,
    output: str = OUTPUT_OPTION,
) -> None:
    baseline = load_official_eres2net_baseline_config(
        config_path=config,
        env_file=env_file,
        project_overrides=project_override or [],
    )
    resolved_device = resolve_device(device or baseline.project.runtime.device)
    checkpoint_path, encoder = load_official_eres2net_encoder_from_checkpoint(
        checkpoint_path=init_checkpoint,
        model_config=baseline.model,
    )
    encoder = encoder.to(resolved_device)
    artifacts = run_speaker_baseline(
        baseline,
        encoder=encoder,
        embedding_size=baseline.model.embedding_size,
        model_config_dict=asdict(baseline.model),
        baseline_name="official 3D-Speaker ERes2Net fine-tune",
        report_file_name="official_3dspeaker_eres2net_report.md",
        embedding_source="official_3dspeaker_eres2net_finetune",
        tracker_kind="official-3dspeaker-eres2net-finetune",
        config_path=config,
        device=resolved_device,
    )
    payload = artifacts.to_dict()
    payload["init_checkpoint"] = str(checkpoint_path)
    if output == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    if output != "text":
        raise typer.BadParameter("output must be one of: text or json")
    final_epoch = artifacts.training_summary.epochs[-1]
    typer.echo(
        "\n".join(
            [
                "official 3D-Speaker ERes2Net fine-tune complete",
                f"Init checkpoint: {checkpoint_path}",
                f"SpeakerLab root: {baseline.model.speakerlab_root}",
                f"Output root: {artifacts.output_root}",
                f"Checkpoint: {artifacts.checkpoint_path}",
                f"Final train loss: {final_epoch.mean_loss}",
                f"Final train accuracy: {final_epoch.accuracy}",
                f"Score gap: {artifacts.score_summary.score_gap}",
            ]
        )
    )


def load_official_eres2net_encoder_from_checkpoint(
    *,
    checkpoint_path: Path,
    model_config: OfficialERes2NetModelConfig,
) -> tuple[Path, Any]:
    speakerlab_root = Path(model_config.speakerlab_root)
    if str(speakerlab_root) not in sys.path:
        sys.path.insert(0, str(speakerlab_root))
    eres_module = importlib.import_module("speakerlab.models.eres2net.ERes2Net")
    model = eres_module.ERes2Net(
        feat_dim=model_config.feat_dim,
        embedding_size=model_config.embedding_size,
        m_channels=model_config.m_channels,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_model_state_dict(payload)
    model.load_state_dict(state_dict)
    return checkpoint_path, model


def _extract_model_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state dict payload, got {type(state_dict)!r}.")
    return state_dict


if __name__ == "__main__":
    app()
