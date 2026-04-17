"""Typed project configuration with override and dotenv support."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config_sections import (
    AugmentationFamilyWeightsConfig,
    AugmentationSchedulerConfig,
    BackendsConfig,
    ChunkingConfig,
    DeploymentConfig,
    ExportConfig,
    FeatureCacheConfig,
    FeaturesConfig,
    NormalizationConfig,
    PathsConfig,
    ReproducibilityConfig,
    RuntimeConfig,
    SecretRefs,
    SilenceAugmentationConfig,
    TelemetryConfig,
    TrackingConfig,
    TrainingConfig,
    VADConfig,
)


@dataclass(slots=True)
class ProjectConfig:
    paths: PathsConfig
    runtime: RuntimeConfig
    reproducibility: ReproducibilityConfig
    training: TrainingConfig
    backends: BackendsConfig
    export: ExportConfig
    tracking: TrackingConfig
    telemetry: TelemetryConfig
    normalization: NormalizationConfig
    vad: VADConfig
    silence_augmentation: SilenceAugmentationConfig
    augmentation_scheduler: AugmentationSchedulerConfig
    features: FeaturesConfig
    feature_cache: FeatureCacheConfig
    chunking: ChunkingConfig
    secrets: SecretRefs
    deployment: DeploymentConfig
    resolved_secrets: dict[str, str | None] = field(default_factory=dict)

    def to_dict(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        payload = {
            "paths": asdict(self.paths),
            "runtime": asdict(self.runtime),
            "reproducibility": asdict(self.reproducibility),
            "training": asdict(self.training),
            "backends": asdict(self.backends),
            "export": asdict(self.export),
            "tracking": asdict(self.tracking),
            "telemetry": asdict(self.telemetry),
            "normalization": asdict(self.normalization),
            "vad": asdict(self.vad),
            "silence_augmentation": asdict(self.silence_augmentation),
            "augmentation_scheduler": asdict(self.augmentation_scheduler),
            "features": asdict(self.features),
            "feature_cache": asdict(self.feature_cache),
            "chunking": asdict(self.chunking),
            "secrets": asdict(self.secrets),
            "deployment": asdict(self.deployment),
            "resolved_secrets": dict(self.resolved_secrets),
        }
        if mask_secrets:
            payload["resolved_secrets"] = {
                key: mask_secret(value) for key, value in self.resolved_secrets.items()
            }
        return payload


def load_project_config(
    *,
    config_path: Path | str,
    overrides: list[str] | None = None,
    env_file: Path | str | None = None,
) -> ProjectConfig:
    config_file = Path(config_path)
    data = tomllib.loads(config_file.read_text())
    for override in overrides or []:
        apply_override(data, override)

    env = dict(os.environ)
    if env_file is not None:
        env.update(load_dotenv(Path(env_file)))

    config = ProjectConfig(
        paths=PathsConfig(**require_section(data, "paths")),
        runtime=RuntimeConfig(**require_section(data, "runtime")),
        reproducibility=ReproducibilityConfig(**require_section(data, "reproducibility")),
        training=TrainingConfig(**require_section(data, "training")),
        backends=BackendsConfig(**require_section(data, "backends")),
        export=ExportConfig(**require_section(data, "export")),
        tracking=TrackingConfig(**require_section(data, "tracking")),
        telemetry=TelemetryConfig(
            **optional_section(
                data,
                "telemetry",
                {
                    "enabled": True,
                    "structured_logs": True,
                    "metrics_enabled": True,
                    "metrics_path": "/metrics",
                },
            )
        ),
        normalization=NormalizationConfig(
            **optional_section(
                data,
                "normalization",
                {
                    "target_sample_rate_hz": 16000,
                    "target_channels": 1,
                    "output_format": "wav",
                    "output_pcm_bits_per_sample": 16,
                    "peak_headroom_db": 1.0,
                    "dc_offset_threshold": 0.01,
                    "clipped_sample_threshold": 0.999,
                    "loudness_mode": "none",
                    "target_loudness_dbfs": -27.0,
                    "max_loudness_gain_db": 20.0,
                    "max_loudness_attenuation_db": 12.0,
                },
            )
        ),
        vad=VADConfig(
            **optional_section(
                data,
                "vad",
                {
                    "mode": "none",
                    "backend": "silero_vad_v6_onnx",
                    "provider": "auto",
                    "min_output_duration_seconds": 1.0,
                    "min_retained_ratio": 0.4,
                },
            )
        ),
        silence_augmentation=SilenceAugmentationConfig(
            **optional_section(
                data,
                "silence_augmentation",
                {
                    "enabled": False,
                    "max_leading_padding_seconds": 0.0,
                    "max_trailing_padding_seconds": 0.0,
                    "max_inserted_pauses": 0,
                    "min_inserted_pause_seconds": 0.08,
                    "max_inserted_pause_seconds": 0.25,
                    "pause_ratio_min": 1.0,
                    "pause_ratio_max": 1.0,
                    "min_detected_pause_seconds": 0.08,
                    "max_perturbed_pause_seconds": 0.6,
                    "analysis_frame_ms": 20.0,
                    "silence_threshold_dbfs": -45.0,
                },
            )
        ),
        augmentation_scheduler=_load_augmentation_scheduler_config(data),
        features=FeaturesConfig(
            **optional_section(
                data,
                "features",
                {
                    "frontend": "local",
                    "sample_rate_hz": 16000,
                    "num_mel_bins": 80,
                    "frame_length_ms": 25.0,
                    "frame_shift_ms": 10.0,
                    "fft_size": 512,
                    "window_type": "hann",
                    "f_min_hz": 20.0,
                    "f_max_hz": None,
                    "power": 2.0,
                    "log_offset": 1e-6,
                    "pad_end": True,
                    "cmvn_mode": "none",
                    "cmvn_window_frames": 300,
                    "output_dtype": "float32",
                },
            )
        ),
        feature_cache=FeatureCacheConfig(
            **optional_section(
                data,
                "feature_cache",
                {
                    "namespace": "fbank-v1",
                    "train_policy": "precompute_cpu",
                    "dev_policy": "optional",
                    "infer_policy": "runtime",
                    "benchmark_device": "auto",
                    "benchmark_warmup_iterations": 1,
                    "benchmark_iterations": 3,
                },
            )
        ),
        chunking=ChunkingConfig(
            **optional_section(
                data,
                "chunking",
                {
                    "train_min_crop_seconds": 1.0,
                    "train_max_crop_seconds": 4.0,
                    "train_num_crops": 1,
                    "train_short_utterance_policy": "repeat_pad",
                    "eval_max_full_utterance_seconds": 4.0,
                    "eval_chunk_seconds": 4.0,
                    "eval_chunk_overlap_seconds": 1.0,
                    "eval_pooling": "mean",
                    "demo_max_full_utterance_seconds": 4.0,
                    "demo_chunk_seconds": 4.0,
                    "demo_chunk_overlap_seconds": 1.0,
                    "demo_pooling": "mean",
                },
            )
        ),
        secrets=SecretRefs(**require_section(data, "secrets")),
        deployment=DeploymentConfig(
            **optional_section(
                data,
                "deployment",
                {
                    "model_bundle_root": "artifacts/model-bundle",
                    "demo_subset_root": "artifacts/demo-subset",
                    "enrollment_cache_root": "artifacts/enrollment-cache",
                    "require_artifacts": False,
                },
            )
        ),
    )
    config.resolved_secrets = {
        key: env.get(env_var_name) for key, env_var_name in asdict(config.secrets).items()
    }
    return config


def _load_augmentation_scheduler_config(data: dict[str, Any]) -> AugmentationSchedulerConfig:
    scheduler_section = optional_section(
        data,
        "augmentation_scheduler",
        {
            "enabled": False,
            "warmup_epochs": 2,
            "ramp_epochs": 3,
            "max_augmentations_per_sample": 2,
            "clean_probability_start": 0.7,
            "clean_probability_end": 0.25,
            "light_probability_start": 0.25,
            "light_probability_end": 0.3,
            "medium_probability_start": 0.05,
            "medium_probability_end": 0.25,
            "heavy_probability_start": 0.0,
            "heavy_probability_end": 0.2,
            "family_weights": {
                "noise": 1.0,
                "reverb": 1.0,
                "distance": 0.9,
                "codec": 0.8,
                "silence": 0.6,
            },
        },
    )
    family_weights_data = scheduler_section.get("family_weights", {})
    if not isinstance(family_weights_data, dict):
        raise ValueError(
            "Config section 'augmentation_scheduler.family_weights' is missing or invalid."
        )
    scheduler_values = dict(scheduler_section)
    scheduler_values["family_weights"] = AugmentationFamilyWeightsConfig(**family_weights_data)
    return AugmentationSchedulerConfig(**scheduler_values)


def require_section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' is missing or invalid.")
    return value


def optional_section(data: dict[str, Any], name: str, default: dict[str, Any]) -> dict[str, Any]:
    value = data.get(name)
    if value is None:
        return dict(default)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' is missing or invalid.")
    return {**default, **value}


def apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override '{override}'. Expected dotted.key=value.")

    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.split(".")
    cursor: dict[str, Any] = data
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if next_value is None:
            next_value = {}
            cursor[key] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot override nested key '{dotted_key}'.")
        cursor = next_value
    cursor[keys[-1]] = parse_override_value(raw_value)


def parse_override_value(raw_value: str) -> Any:
    value = raw_value.strip()
    try:
        return tomllib.loads(f"override = {value}")["override"]
    except tomllib.TOMLDecodeError:
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    result: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        if not key or not _:
            raise ValueError(f"Invalid dotenv line: {raw_line}")
        result[key.strip()] = value.strip().strip("'").strip('"')
    return result


def mask_secret(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
