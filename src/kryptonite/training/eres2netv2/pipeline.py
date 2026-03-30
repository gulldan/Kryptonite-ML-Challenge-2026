"""End-to-end ERes2NetV2 baseline training, embedding export, and cosine scoring."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from kryptonite.models import ERes2NetV2Encoder

from ..baseline_pipeline import run_speaker_baseline
from ..speaker_baseline import SpeakerBaselineRunArtifacts, resolve_device
from .config import ERes2NetV2BaselineConfig

REPORT_FILE_NAME = "eres2netv2_baseline_report.md"
ERes2NetV2RunArtifacts = SpeakerBaselineRunArtifacts


def run_eres2netv2_baseline(
    config: ERes2NetV2BaselineConfig,
    *,
    config_path: Path | str,
    device_override: str | None = None,
) -> ERes2NetV2RunArtifacts:
    device = resolve_device(device_override or config.project.runtime.device)
    encoder = ERes2NetV2Encoder(config.model).to(device)
    return run_speaker_baseline(
        config,
        encoder=encoder,
        embedding_size=config.model.embedding_size,
        model_config_dict=asdict(config.model),
        baseline_name="ERes2NetV2",
        report_file_name=REPORT_FILE_NAME,
        embedding_source="eres2netv2_baseline",
        tracker_kind="eres2netv2-baseline",
        config_path=config_path,
        device=device,
    )
