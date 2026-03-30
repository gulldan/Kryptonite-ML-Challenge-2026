"""End-to-end CAM++ baseline training, embedding export, and cosine scoring."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from kryptonite.models import CAMPPlusEncoder

from ..baseline_pipeline import run_speaker_baseline
from ..speaker_baseline import SpeakerBaselineRunArtifacts, resolve_device
from .config import CAMPPlusBaselineConfig

REPORT_FILE_NAME = "campp_baseline_report.md"
CAMPPlusRunArtifacts = SpeakerBaselineRunArtifacts


def run_campp_baseline(
    config: CAMPPlusBaselineConfig,
    *,
    config_path: Path | str,
    device_override: str | None = None,
) -> CAMPPlusRunArtifacts:
    device = resolve_device(device_override or config.project.runtime.device)
    encoder = CAMPPlusEncoder(config.model).to(device)
    return run_speaker_baseline(
        config,
        encoder=encoder,
        embedding_size=config.model.embedding_size,
        model_config_dict=asdict(config.model),
        baseline_name="CAM++",
        report_file_name=REPORT_FILE_NAME,
        embedding_source="campp_baseline",
        tracker_kind="campp-baseline",
        config_path=config_path,
        device=device,
    )
